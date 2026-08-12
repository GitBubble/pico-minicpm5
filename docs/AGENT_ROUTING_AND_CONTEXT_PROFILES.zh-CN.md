# Agent 路由与运行时 Context Profile 设计

[English](AGENT_ROUTING_AND_CONTEXT_PROFILES.md)

状态：设计已确认；Iteration 1 以本文为实现基线。

## 1. 目标

Hi3403 应用不能把每个请求都无条件交给语言模型。命令处理、权限、工具执行、
结果展示和 MiniCPM5 推理必须解耦；同一份应用源码通过严格校验的 runtime
profile 支持多个静态编译 context。

设计以板端实际成本为依据：列目录约 3 ms；当前 ctx1024 三句柄版本中，仅处理
prompt 的 transformer/KV position 约 85.2 ms，生成阶段的 transformer/head
position 约 106.6 ms；重放系统提示、工具定义和历史会主导首 token 时间；K/V
存储和 attention 工作量随编译 context 增长。

MiniCPM5 只负责语言理解、规划、推理和总结。命令、参数范围、权限、路径边界及
已经完备的结构化结果展示由确定性代码负责。

## 2. 路由流水线

每个请求生成一个可审计的决策：

```text
LOCAL_COMMAND       本地 REPL 命令，不发送给模型
DIRECT_TOOL         确定性工具调用，工具结果就是最终答案
TOOL_THEN_MODEL     工具提供事实，MiniCPM 负责总结或推理
MODEL_ONLY          不需要工具定义和工具执行
PLAN_AND_APPROVE    模型提出步骤，宿主逐项校验和授权
CLARIFY             缺少的选择会实质改变操作，必须澄清
```

返回策略独立于工具：

```text
DIRECT_RAW
DIRECT_FORMATTED
MODEL_SUMMARIZE
MODEL_REASON
CONFIRM_BEFORE_EXECUTE
```

例如：

```json
{
  "mode": "DIRECT_TOOL",
  "confidence": 0.99,
  "tool_calls": [
    {
      "name": "list_directory",
      "arguments": {"path": ".", "max_entries": 10}
    }
  ],
  "response_policy": "DIRECT_FORMATTED",
  "schema_profile": "none",
  "permission": "automatic",
  "reason": "explicit directory listing request"
}
```

### 2.1 路由层级

1. `/help`、`/context`、`/max`、`/clear`、`/quit` 等本地命令不调用工具和模型。
2. 明确的列出、显示、读取、搜索、status 请求调用只读工具并直接展示结果。
3. 包含总结、解释、比较、诊断、建议的请求先取事实，再将紧凑证据交给模型。
4. 含糊或多步骤任务使用 MiniCPM5 原生 XML 工具协议规划，宿主校验每次调用。
5. 写入、shell 及未来网络工具继续遵守显式权限；路由不能扩大授权。

## 3. 工具注册与渐进披露

每个工具声明副作用、权限、结果类型、输出预算、超时和是否允许直接展示。
工具按组注册：

```text
filesystem-read  list_directory, read_file, search_text
git-read         git_status 及未来 diff/log
filesystem-write write_file
shell            run_shell
```

模型只看到相关工具组。`MODEL_ONLY` 不注入工具 schema，直接路由也完全不渲染
schema。工具结果由类型元数据、紧凑预览和稳定 result_id 组成；大结果分页读取，
不整段复制进上下文。

## 4. Runtime Profile 合同

Context 不是一个孤立整数。Runtime profile 必须同时绑定：

- 编译 context 与 past length；
- prefill 窗口：position-zero bootstrap 产物的编译 context。当 prefill 句柄
  继承自更小 context 的已资格化 profile 时（混合 prefill 窗口合同），它可以
  小于 capacity；
- decode、position-zero/prefill、head 产物；
- packed K/V 几何和 runtime descriptor 数量；
- transformer K、V、hidden 公开输出槽索引；
- 默认和最大生成长度；
- 压缩阈值和预留预算；
- chat/agent/tool 能力；
- 工具轮数与输出预算；
- 产物资格状态和数值阈值。

Profile 在进程启动时选择：

```bash
./app/chat.sh --profile ctx128
./app/agent.sh --profile ctx1024
./app/agent.sh --profile ctx4096
./app/agent.sh --profile ctx8192
```

也可直接指定文件：

```bash
./app/agent.sh --profile /opt/pico-minicpm5/profiles/ctx4096.json
```

命令行 `--profile` 优先于可选的 `PICO_PROFILE`，后者优先于安装默认值；初始
安装默认保持 ctx1024。

### 4.1 最终能力矩阵

| Profile | Chat | Agent | 定位 |
|---|---:|---:|---|
| ctx128 | 支持 | **不支持** | 短对话/补全、低内存、冒烟 |
| ctx1024 | 支持 | 支持 | 默认本地 Agent |
| ctx4096 | 支持 | 支持 | 文档、代码、多步任务 |
| ctx8192 | 支持 | 支持 | 超长上下文和长会话 |

执行 `agent.sh --profile ctx128` 必须在加载模型句柄前失败，并提示改用
`chat.sh --profile ctx128`。

各 profile 的资格状态：ctx1024 与 ctx4096 为 `qualified`（ctx4096 走混合
prefill 窗口合同，由 `release/contexts/ctx4096.qualification.json` 把关）；
ctx128 与 ctx8192 保持 `pending`。ctx8192 的候选证据已入库
（`release/contexts/ctx8192.qualification.json`），严格 EOS 序列门 FAIL，
因此仍需 `--allow-unqualified-profile`。

### 4.2 静态 ABI 与内存

24 层、每层 2 个 KV head、head_dim=128、FP16 cache 时，单个 packed K 或 V：

```text
48 * (context - 1) * 128 * 2 bytes
```

| Context | 单个 K/V | K+V |
|---:|---:|---:|
| 128 | 1,560,576 B（约 1.49 MiB） | 约 2.98 MiB |
| 1024 | 12,570,624 B（约 12 MiB） | 约 24 MiB |
| 4096 | 50,319,360 B（约 48 MiB） | 约 96 MiB |
| 8192 | 100,651,008 B（约 96 MiB） | 约 192 MiB |

加载器必须 fail-closed：decode 的 attention mask 宽度与 K/V past length 对齐
profile capacity，prefill 句柄的 mask 宽度与 K/V past length 对齐声明的
`prefill_window`，runtime `--context` 与 profile 一致；发布 profile 还必须校验
产物 hash。禁止静默截断、根据文件名猜合同。跨 profile 复用 OM 仅允许显式声明
的混合 prefill 窗口合同：扩展 context profile 可以继承冻结的已资格化 ctx1024
`models/prefill.om` 作为 position-zero bootstrap，并在
`release/contexts/<profile>.qualification.json` 记录中按 hash 绑定。position 0
之后的 prompt token 由 decode 句柄上的 S1/native-prefill 规划器摄入，因此窗口
从不限制 prompt 长度——capacity 才限制。ABI/hash 相同时 tokenizer、embedding 和
vocabulary head 可以共享。

已发布 profile 绑定互不重复且完整覆盖 0、1、2 的 K/V/hidden 槽位。这用
已验证的编译期 ABI 取代 runtime KV 特征化，移除四次启动 execute。显式 legacy
模型路径可省略该合同并使用动态探测；release profile 不得依靠推断。

### 4.3 生成长度

Context capacity 与回答策略分离。Profile 提供 `default_max_new`、
`max_new_limit` 和 `reserve_tokens`。`/max` 同时显示配置范围、硬件范围和本轮可用
预算；ctx8192 不意味着默认回答 8191 tokens。

## 5. Hi3403 Context 执行策略

只增加 context 而不改变 prompt 执行会让 Agent 更慢。已知 prompt position 跳过
词表 head 后仍约为 85.2 ms，串行处理 4096 或 8192 个位置约需 5.8 或 11.6
分钟。因此 Agent profile 必须具备：

1. 会话级 resident K/V 追加，而不是每轮从 position 0 重放；
2. 固定 system/tool 前缀 K/V 快照；
3. 明确工具结果直接返回，不调用模型；
4. 紧凑、带类型、可分页的工具证据；
5. 接近 profile 阈值时执行 context rebase；
6. 分开报告 route、tool、prompt/prefill、decode 和总耗时。

Native compiler 按固定最大块优先策略增加真正多 token prefill：
`S128 -> S32 -> S16 -> S1 tail`。Fail-closed 调度器和逐请求 telemetry 已实现；
在宽块通过对应 context 的数值与板端门禁前，已验收 Release 只启用 S1。实现顺序
先闭环 S16，再扩 S32、S128。Builder 由 context 和 sequence length 参数化，
不为每个 context 复制应用源码。详见
[native prefill 合同](NATIVE_PREFILL_SCHEDULER.zh-CN.md)。

当前实现为每个工具 schema 懒创建固定前缀快照。Executor 的通用 opcode 保存/恢复
显式 resident-input 范围；MiniCPM adapter 把 prefix token 数转换成两只 packed
KV 输入中 96 个连续 channel range。每只快照上限 64 MiB、总预算 128 MiB、最多
8 只，超限时 fail closed。该功能已通过 Hi3403 板端 A/B 并在 Agent 中默认开启：
137-token 前缀恢复耗时 `1.76 ms`，生成 token ID 与文本完全一致，同一条 32-token
请求由 `26.97 s` 降至 `12.56 s`（降低 `53.4%`）。设置
`FIXED_PREFIX_SNAPSHOTS=0` 可保留全量重放诊断路径。

## 6. Context Rebase

会话达到 profile 的 `compact_at_tokens` 时，host 确定性保留固定前缀、当前交换、
最近对话和工具产物引用，删除旧的原始工具输出，并按 `reserve_tokens` 重建紧凑
transcript。该过程不额外调用模型，也不会放宽工具权限。`/clear` 仍是显式全量
对话重置，但已验证的固定前缀快照可继续保留。

ctx1024 Hi3403 板端门禁先通过 12 轮不调用模型的直接工具请求积累历史，再执行一次
短模型回答。A/B 两轮均把 `2808` 个 prompt token 确定性压到 `810`，压缩全部 12
个旧轮次，并生成完全相同的 `[18655, 4569, EOS]`（`测试通过`）。重复运行在
`7.15 ms` 内恢复 643-token resident 前缀，仅执行 167 个新增 prompt token，总
耗时由 `69.45 s` 降至 `14.61 s`（降低 `79.0%`，即 `4.75x`）。这完成了 rebase
溢出保护及其与固定前缀快照协作的板端资格测试，但不替代未来长 context OM 门禁。

运行时还会在最后一个已知 prompt position 之前跳过词表 head 和 argmax。同一组
token-exact 板测中，首次请求 812 个 position 有 809 个跳过，resident 前缀请求
169 个 position 有 166 个跳过；3 个真正生成 position 保持完整执行。端到端耗时
分别降低 `19.89%` 和 `19.59%`，`phase_ms[].head_skipped` 记录每次决策。

第一阶段不支持会话内热切 context。切换需要加载 handle、重新分配 K/V、迁移
cache layout、校验 mask/RoPE 并重新做数值门禁。未来热迁移必须显式执行，并与
全量回放参考做 token-exact 对照。

## 7. 权限与终端安全

直接路由不改变权限。只读工具可自动执行；`write_file` 和 `run_shell` 仍对精确
参数逐次授权。路径限制在 workspace，拒绝 symlink 逃逸，不可信工具结果在终端
展示前转义控制字符。

## 8. 可观测性与性能目标

每个请求记录：

```text
route_mode, route_reason, route_ms, tool_ms, model_called,
prompt_tokens_new, prompt_tokens_replayed, prefix_cache_hit,
prefix_snapshot_hit, prefix_snapshot_created, prefix_snapshot_restore_ms,
context_rebased, context_tokens_before, context_tokens_after,
context_turns_compacted,
prefill_schedule.{enabled_widths,counts,invocation_count,segments},
prefill_ms, decode_ms, generated_tokens, time_to_first_token_ms, total_ms
```

初始常驻目标：

| 请求 | 目标 |
|---|---:|
| 本地命令 | <10 ms |
| 直接目录/git status | <50 ms |
| 直接文件/搜索窗口 | <200 ms |
| 工具后简短模型总结 | 前缀/KV 优化后 3～10 s |

终端对直接路由明确显示 `model skipped`，并把 prompt 处理时间和输出速度分开。

## 9. 迭代计划

1. **Iteration 1（已实现）：** 发布本文；实现 profile 加载与 ctx128 Chat-only 门禁；
   实现 fail-closed 确定性只读直达路由；增加 route/tool 耗时证据。
2. **Iteration 2（已实现）：** 渐进披露只读/写入/shell 工具组，保留最近 16 个
   typed、有界、可分页结果引用。
3. **Iteration 3（已实现并通过板端门禁）：** 会话 live K/V 与固定 system/tool
   前缀快照已在 Agent 中默认启用，均通过 token-exact 板端 A/B；固定前缀恢复耗时
   `1.76 ms`，重复请求耗时降低 `53.4%`。
4. **Iteration 4（已实现并通过板端门禁）：** context rebase 在两轮板端测试中都
   将 12 个旧工具轮次由 `2808` token 压到 `810`，保持输出 token 完全一致，并与
   643-token resident 前缀恢复安全协作；随后 prompt-only head 抑制在不改变输出
   的情况下将首次与 resident 请求耗时均降低约 `19.7%`。
5. **Iteration 5（控制面已实现）：** 运行时已记录并校验严格的
   `S128 -> S32 -> S16 -> S1` 调度，且只启用通过门禁的宽度。先完成 S16
   端到端闭环，再为各 context 验收 S32、S128；当前交付行为保持 strict S1。

每个发布 context 都必须通过 descriptor 校验、public-output cosine 严格大于 0.98、
greedy token、边界位置、EOS、context overflow、板端加载和性能报告。
