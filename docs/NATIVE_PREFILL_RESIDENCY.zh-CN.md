# Native Prefill residency 与 MMZ 闭环

[English](NATIVE_PREFILL_RESIDENCY.md)

本文只解决 S16/S32/S128 发布时的模型常驻、KV 所有权和切换合同。数值资格仍以
[`NATIVE_PREFILL_CLOSURE_WORKFLOW.zh-CN.md`](NATIVE_PREFILL_CLOSURE_WORKFLOW.zh-CN.md)
为准。本文中的状态含义如下：

- **PASS：** 已有源码合同或落盘的 Hi3403/SS928 实测证据；
- **CANDIDATE：** 现有接口可以实现，但还没有相应的真实宽块 OM 板端证据；
- **BLOCKED：** 当前实现或公开运行时接口不能满足，不能用于发布激活。

## 结论

发布路线不应让三份约 `687 MB` 的 24 层 S16/S32/S128 OM 与现有三个基础
handle 同时常驻。

推荐分两步实施：

1. **bring-up 路线（CANDIDATE）：** decode 与 head 始终常驻；position 0 完成后
   卸载 `prefill.om`，只保留一个可替换的 lazy-wide slot。按需加载 S128、S32、
   S16，canonical KV 始终由 decode handle 持有。宽块执行前只把已用前缀复制到
   当前宽块，宽块输出再通过 opcode 6 发布回 canonical KV。此路线用于分别闭合
   三个宽度和测量切换成本。
2. **Agent 发布路线（推荐，CANDIDATE）：** 做一个静态 S128 carrier，在同一 OM
   内以 `valid_len ∈ {16,32,128}` 选择宽度专用入口，并共享 24 层权重。它不是
   ACL 动态 shape；descriptor 始终是 S128 最大物理形状。native 分支必须跳过无效
   M16 group，使 S16 只执行一个 group。若 runtime-scalar ingress、分支数值或 PMU
   门未通过，继续使用 lazy-wide 路线，不得把该 carrier 标为 PASS。

“把三个 GraphDef 放进一个 OM，再由 runtime 选择 entry”不是当前可用方案。
当前 ACL 只按 `model_id` 执行一个 descriptor，没有 graph/entry selector；现有
`PackPicoOm` 也只组装一个 item region。多层 merged OM 是一条合成执行图，不是
三个可独立调用的入口。

## 证据清单

| 结论 | 状态 | 证据 |
|---|---|---|
| 当前 executor 为每个模型独立分配 OM、input、output MMZ buffer | PASS | `app/native/pico_persistent_acl_executor.c`: `load_om_source()`、`create_dataset()` |
| 模型只在进程启动时加载、进程清理时卸载 | PASS | 同文件 `main()`、`load_model()`、`destroy_model()`；协议没有 load/unload opcode |
| ACL 支持 load/unload 一整个 model handle | PASS（API） | SDK `svp_acl_mdl_load_from_mem()` / `svp_acl_mdl_unload()`；尚无 lazy-wide 板端实现 |
| ACL 没有 graph/entry 选择参数 | PASS（源码检查） | `svp_acl_mdl_execute(model_id,input,output)`；本 SDK `svp_acl_mdl.h` 无 graph selector |
| MiniCPM ATC 产物是静态 input shape | PASS | `src/pico_minicpm5/compiler/atc.py` 固定传 `--input_shape=` |
| SDK 的动态接口只覆盖 batch/HW/total-T，不是任意 sequence dim | PASS（接口检查） | `svp_acl_mdl_set_dynamic_batch_size()`、`set_dynamic_hw_size()`、`set_total_t()` |
| PICO ND 输入不能直接依赖当前 ATC dynamic-batch 开关 | PASS（配置检查） | 当前 ATC `atc_param_conf.json` 中 `dynamic_batch_size` 对 `InputFormat=ND` 禁用 |
| 固定 trip native branch/loop 已有局部数值证据 | PASS（组件） | integration 的 `pico_minicpm_prefill_steady_loop_scale.py` |
| runtime `valid_len` 从模型输入进入 native scalar/branch | BLOCKED | 尚无真实 S128 carrier、descriptor、libinstsim 与板端资格 |
| output→input chain 与 FP32→FP16 resident scatter | PASS（本地协议） | executor opcode 4 / 6；`make -C app/native contract-check` PASS |
| input→input cache copy | PASS（本地协议） | executor opcode 9；96-record wire/越界/no-partial 回归通过；板端 byte-exact 待测 |
| 跨模型 shared input binding | BLOCKED | `create_dataset()` 仍独立 malloc；尚无引用计数/owner 合同 |
| resident snapshot/restore | PASS（当前同模型合同） | executor opcode 7 / 8；snapshot 绑定原 `model_index` |

## MMZ 下界

以下数字来自已落盘的 3-handle 产物和 2026-08-02 板端资格记录。OM source
bytes 是可靠的**分配下界**：当前 `load_om_source()` 先按文件大小执行
`svp_acl_rt_malloc_cached()`，之后 runtime 还会分配 input/output 和可能的内部
资源。因此“下界超过 MMZ”可以直接判定失败；“下界未超过”只能进入板端 admission，
不能直接判 PASS。

| 项目 | 字节 | GiB |
|---|---:|---:|
| `decode.om` | 686,997,372 | 0.640 |
| `prefill.om` | 686,999,901 | 0.640 |
| `head_flat.om` | 202,651,666 | 0.189 |
| 当前基础三 handle | 1,576,648,939 | 1.468 |
| 一个代表性 24L 宽块 OM | 687,076,012 | 0.640 |
| 板端 clean/post-exit MMZ | 2,896,191,488 | 2.697 |

由此得到：

| 组合 | source-byte 下界 | 下界余量 | 判定 |
|---|---:|---:|---|
| base + S16 + S32 + S128 | 3.388 GiB | **-707.3 MiB** | **BLOCKED，尚未计算 IO 就已超池** |
| base + 一只 wide | 2.108 GiB | 603.2 MiB | CANDIDATE，必须实测 |
| base + 一只 wide + 256 MiB reserve | 2.358 GiB | 347.2 MiB | CANDIDATE，仍未含 runtime 内部资源 |
| decode + head + 一只 wide（替换 p0 prefill） | 1.468 GiB | 1,258.3 MiB | 推荐 lazy-wide 常驻集合 |

C4096 的 canonical FP16 cache 为：

```text
one kind = 48 * (4096 - 1) * 128 * 2 = 50,319,360 B
K + V    = 100,638,720 B = 95.98 MiB
```

如果 decode 与 wide 各自分配 cache，至少再付一份约 `96 MiB`。shared binding
若通过 descriptor/stride/ownership 门，可以省掉这份内存；在它通过前 activation
必须按两份 cache admission。

## 四个方案的判定

### 1. 单 OM、多 graph/多 entry 共享权重

**当前判定：BLOCKED。**

PICO item 区可描述 `ModelDef → GraphDef → OpDef`，但当前执行接口只接受一个
`model_id`，descriptor 也只有一组 inputs/outputs。`PackPicoOm.build_om()` 只接收
一份 `param_head + instr + item_region`。现有 `merge_minicpm5_containers.py` 合成的是
一条执行图；它没有产出可供 ACL 选择的 S16/S32/S128 entry table。

可行替代是**单 graph 的显式 dispatcher**：

```text
static S128 descriptor
        |
        +-- valid_len=16  -> one M16 group
        +-- valid_len=32  -> two M16 groups
        `-- valid_len=128 -> eight M16 groups
```

三个分支允许共用相同 weight source offsets，但每个宽度的 instruction/quant
常量可以独立。该方案必须证明：

- 不能用当前 per-width qualification ABI（开发 v2、发布 v3）的三个 W-specific
  exact publisher ABI 冒充一个 descriptor；carrier 需要后续 schema 显式区分
  `physical_width=128` 与
  `valid_width=16/32/128`，并绑定 opcode6-v2 source stride 或 compact publisher；
- `valid_len` 是真实 runtime input，不是 emitter 写死的 immediate；
- 只接受 `16/32/128`，其他值 fail closed；
- skipped group 没有 DLD、Cube/Vector、DSTR 和 KV publication；
- S16/S32/S128 各自满足完整数值、token、handoff 与性能门；
- PMU/trace 中 executed M16 groups 精确为 `1/2/8`，且板端延迟满足
  `T16 < T32 < T128`。权重带宽可形成固定底噪，所以不预设线性 1:2:8。

### 2. dynamic sequence / valid_len

**动态 tensor shape：BLOCKED；静态 S128 + valid_len：CANDIDATE。**

不能把 ACL 的 dynamic batch/HW 接口解释成任意 sequence 动态形状。宽块输入、
mask、RoPE、K/V publisher 都应保持 S128 最大物理 descriptor，runtime 只填入有效
范围。native 程序在第一条宽度相关 DLD/compute 之前读取 `valid_len` 并分支。

防止 S16 实际跑成 S128 的红绿门：

| 门 | GREEN | RED |
|---|---|---|
| instruction trace | 只访问 group 0 | group 1..7 有任何有效 DLD/compute/DSTR |
| publisher | 只写 `48×16×128` K/V | 发布或 scatter 了 128 行 |
| invalid rows | poison 后不影响 hidden/下一 S1 token | poison 改变输出 |
| wall/PMU | `T16 < T32 < T128`，执行 group 为 1/2/8 | 三档耗时/计数相同 |

### 3. 按请求 load/unload 一只宽块 OM

**当前实现：BLOCKED；运行时 API 与内存下界：CANDIDATE。**

现 executor 已有完整 `load_model()` / `destroy_model()`，但只在启动/退出调用。
建议增加一个**manifest 预注册**的 lazy slot，而不是让请求携带任意路径：

```text
LOAD_WIDTH(width, expected_sha256)
  -> wait until no execute is in flight
  -> unload current lazy slot and free its datasets/OM memory
  -> load the pre-qualified path for width
  -> return exact descriptor, load_ms and model generation

UNLOAD_WIDTH(generation)
  -> reject stale generation
  -> invalidate slot-bound snapshots/copies
  -> unload model and report unload_ms
```

落盘证据只提供两组整体 ready 时间：约 `904.5 MB / 52 handles` 为
`3.91 s` 和 `4.36 s`；3-handle demo UI 显示约 `1.576 GB / 3 handles` 为
`6.4 s`。没有单独 `687 MB` merged OM 的 load/unload 实测，因此不能据此给出
切换 PASS。每个宽度的发布性能必须包含 load、prefix copy、execute 和 unload：

```text
T(load W) + T(copy prefix) + T(execute W) + T(unload W)
    < T(the already-qualified lower-width route)
```

对 30 次 `S128→S32→S16→none` 循环记录 p50/p95；每次卸载后 MMZ 必须返回基线，
model-id 不得泄漏，下一次 strict S1 token 必须一致。若切换成本使不等式失败，
该宽度不能通过 activation，即使裸 `execute` 很快。

### 4. input-to-input copy 与 shared cache binding

**copy 本地合同已实现；先完成板端资格，再考虑 alias。**

canonical KV 永远是 decode handle 的 input 3/4。宽块只拥有临时镜像：

1. 宽块执行前，把 decode `[0,start)` 的 K/V 行复制到当前 wide handle；
2. 宽块输出连续 channel-major FP32 `[48,W,128]`；
3. opcode 6 只 scatter 新的 `[start,start+W)` 行回 decode；
4. 切换到另一宽度时，从 decode 再复制前缀，不从旧 wide 复制；
5. strict S1 和 session snapshot 永远只读写 decode canonical cache。

当前 input-to-input record 包含：

```text
destination_model, destination_input, destination_offset,
source_model, source_input, source_offset, length, flags=0
```

executor 会先 drain 并验证全部 record，再进行任何 memcpy；同一底层 buffer 重叠
使用 `memmove`，跨 buffer 使用 `memcpy`；cached source 先 invalidate，destination
最后统一 flush。协议显式不提供 model-table generation guard，进程/phase 更换后
caller 必须重建 record。对 packed
channel-major cache，每个 K/V channel 单独复制：

```text
row_bytes      = 128 * 2
channel_stride = (context - 1) * row_bytes
offset(c)      = c * channel_stride
length(c)      = start * row_bytes
records        = 48 K + 48 V = 96
```

shared binding 的下一阶段可以让 decode 与 wide dataset 的 cache slot 指向同一块
`svp_acl_rt_malloc_cached()` 地址，但当前 `destroy_dataset()` 会无条件 free 每个
dataset buffer，直接 alias 会 double-free。必须先引入引用计数/owner flag，并验证
两边的 exact size、default stride、cached flush 语义和 unload 顺序。未通过这些门前
不得把两个 handle 声明为 shared residency group。

## Snapshot 方案

现有 snapshot 保存在 executor 的普通 host heap，绑定同一 `model_index`；它不是
跨模型 cache copy。其限制为单 snapshot `64 MiB`、全部 snapshot `128 MiB`。

- 643-token Agent prefix 的 K+V 为约 `15.1 MiB`，当前单 snapshot 路径可用；
- 单 snapshot 最多容纳 `2730` 个 C4096 token 的 K+V；
- 完整 C4096 K 与 V 各约 `48 MiB`，应使用一对 snapshot id（K 一只、V 一只），
  合计约 `96 MiB`；
- C8192 完整 K+V 约 `192 MiB`，超过当前总预算，当前合同下 **BLOCKED**。

一对 snapshot 必须作为一个逻辑事务：先完整校验两只 id、model generation、ranges
与 checksum，再恢复 K 和 V；中途失败则 invalidate resident-token metadata，不能
执行模型。lazy slot 卸载时清除所有绑定到该 slot generation 的 snapshot。推荐只
snapshot canonical decode cache，wide 临时镜像不做 snapshot。

## 发布红绿矩阵

| 测试 | GREEN | RED / fallback |
|---|---|---|
| MMZ lower bound | base + active-wide + IO + reserve 小于实测 available | 任一估算缺项或超池 → S1 |
| 30 次 lazy switch | p95 有界、无 model-id/MMZ 增长 | 任一泄漏或 500004 → 禁用 lazy-wide |
| descriptor | hash、input/output size、stride、context、width 全绑定 | 任一漂移 → 不加载 |
| input copy | 与 full-payload baseline 的宽块输出 byte-exact | 不一致 → strict S1 |
| opcode 6 publication | 所有 `24×2×W` 行 exact，未改其他行 | 任一越界/旧行变化 → strict S1 |
| block→block | S128→S32→S16 与单宽度/reference 一致 | cache owner 不一致 → strict S1 |
| block→S1 | 下一 strict-S1 hidden、K/V、token exact | 任一 token/cos 门失败 → strict S1 |
| snapshot | restore 后同请求 token exact；poison 未使用行无影响 | partial K/V restore → reset session |
| universal carrier | group 1/2/8，三档各自数值与 PMU PASS | runtime scalar/branch 未证 → lazy-wide |
| request performance | 含 load/copy/unload 后仍优于下级路线 | 只看裸 execute 的收益不算 PASS |

## 推荐实施顺序

1. 串行构建、资格化 S16、S32、S128；此阶段不要求同时常驻。
2. 用已实现的 input-to-input copy，在固定两 handle 上做 full-payload byte-exact 对照。
3. 增加 manifest-pinned lazy slot，实测 30 次 load/unload、MMZ、model-id 与切换耗时。
4. 用 lazy slot 跑完整 `S128→S32→S16→S1`，canonical decode cache + snapshot
   全门通过，作为 bring-up 可回退实现。
5. 实现静态 S128 + `valid_len` dispatcher，复用权重；完成 runtime-scalar、分支
   trace、PMU、三宽数值和板端 token 门。
6. universal carrier 通过后替代 lazy slot；shared cache alias 作为独立优化，失败时
   保留 input-copy，不阻断正确性发布。

## 本地合同复现

```bash
cd release_work/pico-minicpm5
make -C app/native contract-check
```

2026-08-11 的输出为：

```json
{"schema":"pico.persistent_acl_executor.protocol.v1","byte_order":"little","model_limit":128,"io_limit":64,"resident_scatter_f32_to_f16":true,"resident_scatter_record_bytes":48,"resident_scatter_validate_all":true,"resident_scatter_flush_failure_fatal":true,"resident_input_snapshots":true,"resident_input_copy":true,"resident_input_copy_opcode":9,"resident_input_copy_record_bytes":44,"resident_input_copy_generation_guard":false,"self_test":true,"model_execution":false}
```

这只证明协议的本地编码、自测、scatter、snapshot 与 input-copy 能力；
`resident_input_copy_generation_guard=false` 要求 phase/process 更换后重建 record，
`model_execution=false`
明确表示它不能替代真实宽块 OM 的 libinstsim/Hi3403 门。
