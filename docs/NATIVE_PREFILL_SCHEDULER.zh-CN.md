# Native 多 Token Prefill 调度器

[English](NATIVE_PREFILL_SCHEDULER.md)

完整的分阶段实现、资格、MMZ admission 和发布门禁见
[S128→S32→S16→S1 闭环工作流](NATIVE_PREFILL_CLOSURE_WORKFLOW.zh-CN.md)。

## 目标

MiniCPM 当前仍逐 token 送入新增 prompt。Resident K/V、固定前缀快照和
prompt-only head 抑制已经避免了无效重放和词表投影，但每个剩余新增 prompt token
仍需执行一次 transformer handle。要继续降低 TTFT，必须提供 native 多 token
transformer 产物。

目标策略固定为：

```text
S128 -> S32 -> S16 -> strict S1 tail
```

`S<N>` 表示一次已验收调用连续消费 `N` 个 prompt token，并发布后续块或 decode
所需的全部 K/V 行。它不是另一套模型，也不改变 context 容量。

## 调度契约

运行时用上述固定顺序贪心覆盖
`[resident_prefix_tokens, prompt_tokens)`；调用方传入宽度的顺序不能改变策略。
S1 必须始终存在，保证任意尾部都能精确覆盖。若区间从 position 0 开始，其中一次
S1 是必须先执行的 startup bootstrap；例如 433 个 token 会执行
`1xS1(startup) + 3xS128 + 1xS32 + 1xS16`。resident 前缀为 643、prompt 结束于 810
时，拆成 `1xS128 + 1xS32 + 7xS1`。

`app/src/minicpm_prefill_schedule.py` 已实现该 fail-closed 调度器。每次请求报告会
记录 schema、绝对区间、启用宽度、各宽度次数、总调用数和紧凑 segment。当前
已验收 Release **只启用 S1**，因此单独加入调度器不代表 TTFT 已经提升。

`app/src/minicpm_prefill_runtime.py` 是 release activation 与调度器之间的运行注册表。
板端 CLI 可以在启动时加载 v3 manifest：

```bash
./app/chat.sh \
  --prefill-activation-manifest work/prefill/activation.json \
  --available-bytes "$MMZ_AVAILABLE" \
  --base-resident-bytes "$BASE_RESIDENT" \
  --reserve-bytes 268435456
```

manifest 与三项 live MMZ 数值必须同时提供；不提供 manifest 时精确保持 strict S1
默认路线。顶层 S1 anchor 无效会在任何模型 handle 执行前终止启动。报告区分
`qualified_widths` 与实际可执行的 `enabled_widths`，只有后者会进入
`prefill_schedule.enabled_widths`。merged runtime 现在已经具备可注入的 typed
wide-handler 边界，并精确校验 width/context/model index、ready descriptor 与
publisher ABI。但当前 Release 没有任何通过完整门禁的宽块 OM，因此 production
handler 注册表仍为空，也没有 CLI 注入开关。即使 S16/S32/S128 通过 release
qualification，也只会报告为 handler-unavailable，实际集合仍是 `[1]`；绝不会
挂着宽块标签暗中执行 S1。

该边界目前只由 fake transport 测试驱动：一次宽块事务先生成 `W` 行 embedding、
`W×context` mask 和 `W` 行绝对位置 RoPE；用 opcode 9 把 canonical decode 的
K/V `[0,start)` 镜像到 wide handle；只执行一次 wide model；再用 opcode 6 仅把
`[start,start+W)` 发布回 canonical decode。最后一行 hidden 必须按完全相等的
字节 descriptor 直链 head。prepare、copy、execute 或 publish 任一步失败都会
丢弃整个 resident 进程；只有重建新 session 后才能回退 S1。fake 测试只证明调用
顺序、字节范围和 fail-closed 状态机，不证明 OM 数值或 Hi3403 性能。

canonical cache 只有 `context-1` 个可写行，而 opcode 6 无法部分发布 wide tensor。
因此运行时会在任何 model execute 前预检整份计划：最终 position `context-1` 会
重规划为 strict S1，由既有 S1 契约安全地省略这行不再被消费的 K/V scatter。

固定前缀 snapshot 位置同样是调度硬边界。planner 可以在显式边界后重新从最大宽度
开始，但任何 wide segment 都不能跨界；因此 Agent 首次请求会在精确 fixed-token
位置创建 snapshot，后续请求才能安全恢复。

## 产物准入条件

每个块宽度只有在对应 context 的精确产物全部通过以下门禁后才能启用：

1. OM 与 build manifest 的哈希绑定；
2. 输入/输出 descriptor 与物理 stride 校验；
3. 每个公开输出 cosine 严格大于 `0.98`；
4. 绝对位置 mask 与 RoPE byte-exact；
5. 24 层全部 `N` 行 K/V byte-exact 发布；
6. block-to-block 以及 prefill-to-S1/decode handoff；
7. Hi3403 上 greedy token、EOS、中英文和 context 边界门禁；
8. 相比下一档已验收小块确有 TTFT 收益。

物理角色槽位固定，不能从相同 tensor 大小猜测：输入 embedding/mask/RoPE/K/V
固定为 `0/1/2/3/4`，输出 K/V/hidden 固定为 `0/1/2`。qualification 会按
`(context,width,start)` 独立重建每个 capture 的 FP32 mask hash，同时验证可见
prefix、遮蔽 future 区间和 `context-1` current-token sentinel。

准入按 context 隔离：ctx1024 的 S16 不能授权 ctx4096 或 ctx8192。证据缺失、过期
或不匹配时，该宽度自动禁用并落到下一档。

`pico_minicpm5.prefill_blocks` 已把这套规则实现成机器门禁。steady S16 必须覆盖
`1,15,16,31,32,255,256,643,context-16`；S32 覆盖
`1,31,32,127,128,643,context-32`；S128 覆盖
`1,127,128,511,512,643,context-128`。position 0 保持 strict-S1 startup 量化域。
K、V publisher 各为连续 `[1,48,16,128]` FP32，由 opcode 6 转成逻辑 FP16
resident cache；最终 hidden 为 `[1,1536,1,1]` FP32。每个 capture 都要绑定物理
descriptor、mask、RoPE 和原始输出哈希，并证明每个角色 768 行 K/V 完整、handoff、
token exact 和板端通过。生成可发布激活的 v3 资格记录：

```bash
pico-minicpm5 qualify-prefill-block-release \
  --evidence work/s16/release-evidence.json \
  --out work/s16/qualification.json
```

精度阈值由策略固定，CLI 和 evidence JSON 均不能把它调低。较短的
`qualify-prefill-block` 仅保留为开发期 v2 兼容入口，不能激活发布。详见
[Native prefill 发布资格 v4](NATIVE_PREFILL_RELEASE_QUALIFICATION.zh-CN.md)。

## Native 数据路径

一个块消费连续 `N` 个绝对位置的 embedding、resident packed K/V 前缀、因果
mask 行和对应 RoPE 行。24 层图共同计算这段序列，并发布：

- 最后一个 prompt position 所需的末层 hidden；
- 每个 transformer 层的 `N` 行 K 和 `N` 行 V；
- 精确前移 `N` 个位置的 resident cache extent。

query row `j` 的 mask 暴露绝对前缀 `[0,start+j)`，最后一个 context column 是
current-token sentinel。wide 图必须在内部把同一块先前生成的 K/V 动态追加到对应
绝对 slot，后续 row 才能按该 mask 消费；host 不做这一步 repack。publisher 的
row `j` 固定回写 canonical 绝对行 `start+j`。

Prompt-only head 抑制仍然成立：块内部不执行词表投影，只在最后一个已知 prompt
position 执行。块不能只发布最后一行 K/V，不能注入 reference hidden，也不能把
host 可见 output staging 当作隐式层间 bridge。

## 激活顺序

1. **先做 S16：** 闭环一个 byte-exact native 块，再在上述完整 steady 位置矩阵
   验证它到现有 S1/decode 路径的 handoff；
   position 0 继续走 strict S1。
2. **再做 S32：** S16 验收后才组合或编译专用族；相同输入下与连续两次 S16 对比。
3. **最后 S128：** 与四次 S32、八次 S16 对比，并覆盖 resident-prefix 与
   context-rebase 产生的非宽度对齐起点。
4. 启用全部合格宽度，执行 Agent 端到端 TTFT、token-exact 和长会话门禁；strict
   S1 始终保留独立回归路径。

本地 native bring-up 已验证 S16 input RMSNorm、RMSNorm→M16 QKV/RoPE、
dynamic C4096 H2 KV resident witness bridge，以及 attention→layer-tail
private-TEMP splice。原先 15 个 full-state、1 个 causal-tail 和 1 个 merge 的
17-invocation attention schedule 也已合成一只三公开输入的 C4096 OM：32 个 cache
Slice 全在图内，event SSA `32076/32076`，libinstsim cosine `0.9998773`。不过这只
OM 仍从完整 Q/K/V cache 边界开始，尚未把当前 S16 的双 K/V 动态 append
接入；24 层复制和板端 handoff 也未完成。这些阶段性证据支持优先选择 S16，
并不是可发布产物。

## 性能统计

必须同时报告 transformer 调用数和墙钟时间。将 433 次 S1 调用规划成 6 次块调用
只是调用数结论，不等于时延结论；S16/S32/S128 的执行成本必须实测。报告至少包含
cold/resident-prefix TTFT、各块执行耗时、cache 发布耗时、fallback 次数、生成
token 时延和端到端总耗时。
