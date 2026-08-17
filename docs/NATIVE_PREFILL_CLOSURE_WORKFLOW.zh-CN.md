# Native Prefill S128→S32→S16→S1 闭环工作流

[English](NATIVE_PREFILL_CLOSURE_WORKFLOW.md)

## 两种顺序不可混淆

- **运行时选择顺序：** 若区间从 position 0 开始，先执行一次 strict S1 bootstrap；
  position 1 起按 `S128 → S32 → S16 → strict S1` 最大块优先且不做 padding。
- **实现与验收顺序：** `S16 → S32 → S128`。S32 必须与 `2×S16` 对照，
  S128 必须与 `4×S32` 和 `8×S16` 对照。

S1 始终是独立、已验收的回退路径。没有真实宽块 OM、证据不完整、哈希漂移或
MMZ 不足时，运行时只能报告该宽度被禁用，不能生成一个“已启用”的占位产物。

## 当前状态（2026-08-12）

| 工作包 | 状态 | 下一门禁 |
|---|---|---|
| 最大块优先调度器 | PASS（S1-only） | 由资格记录提供可启用宽度 |
| M16 Q/K/V + RoPE producer | 组件 PASS | 已与 input RMSNorm 同 OM resident 连接 |
| S16 input RMSNorm→M16 QKV/RoPE | 组件 PASS（private TEMP） | 动态 K/V append |
| layer0 RMSNorm→QKV/RoPE→双 K/V dynamic append | 组件 PASS（bounded witness） | 汇总 24 层 publisher slice |
| 双 K/V append→C256 causal-tail attention | synthetic PASS；real 4/4 FAIL | 收敛真实域 panel online-state 数值 |
| C4096 16-state static attention | 组件 PASS（single OM） | 动态 append→full-C4096 same-OM join |
| S16 attention 后 layer tail | 组件 PASS（public boundary） | 已由 resident splice 覆盖 |
| S16 attention→o_proj resident splice | 组件 PASS（private TEMP） | input RMSNorm |
| S16 input RMSNorm | 组件 PASS（public boundary） | 已由 RMSNorm→QKV splice 覆盖 |
| dynamic C4096 H2 KV resident bridge | 组件 PASS | 真实 calibration + full-C4096/B16 same-OM join |
| full-C4096/B16 layer0 boundary | BLOCKED（H=0.956954；ABI V0/K1/H2） | 修复 attention 精度与 K/V 槽顺序 |
| S16 单层 / 24 层 | BLOCKED | full-C4096/B16 数值、全层 handoff |
| S32 / S128 | NOT STARTED | 等 S16 全门禁通过 |
| 多行 resident KV scatter | 本地契约 PASS | 真实宽块 descriptor/板端验证 |
| canonical KV input→input copy | 本地协议 PASS | 两 handle 板端 byte-exact |
| 多宽度同时 residency | BLOCKED | 未来 carrier schema 或 lazy-wide admission |

阶段性 M1 layer-tail 只能证明算子族和拓扑来源，不能作为 S16 数值证据。

最新真实门已生成 97-tile S16 tail OM：event SSA `1880/1880`，libinstsim cosine
`0.9994710106`，无 deadloop；OM SHA256 为
`5007dfd19f7ea970d4bfd2913b5df131cc306dfadc82eb828771a202f5a9c839`。它仍使用
public attention input，因而只关闭 bounded tail execution；下一门是
`splice_s16_attention_to_o_proj_resident`，不能据此声称单层或 24 层完成。

该 resident splice 随后也通过了真实门：attention 不再作为公开 I/O，bridge 的
`61` 个访问全部属于 private TEMP，公开 INOUT 为 `0`，唯一 Report 是
`next_hidden`；event SSA `4092/4092`，libinstsim cosine `0.99969908`，无
deadloop。OM SHA256 为
`65de2000252cdcd584864f8b0033ba813d0c309dfa0de1ee5d23a22704c8c434`，
qualification SHA256 为
`4a7147f60df8dc8227e5bc103b4c5af775abeb0a9eb5d631e2667315f6009126`。
这关闭的是 attention→tail 的同图 resident 边，不是完整单层；后续 input RMSNorm
门已由下述独立证据关闭。

input RMSNorm 门也已通过：event SSA `142/142`，libinstsim cosine
`0.9999759688`，无 deadloop 或 continue-event warning。OM SHA256 为
`ecedc2ddd1247f981e043cbf893b2269f149627e58bbf7ed54644047f1275c8f`，
qualification SHA256 为
`165b0ba421a89173d91ec726fa5c4d216c713f8ad46d8d3551a3215929b3aeb1`。
该 public RMSNorm 边界随后也已消除：same-OM
`input RMSNorm → M16 QKV/RoPE` 只保留 hidden 与 RoPE 公开输入，normalized hidden
和 QKV activation 全部留在 private TEMP，公开中间 INOUT 为 `0`。event SSA 为
`540/540`，Q/K/V cosine 为 `0.9999167/0.9999570/0.9999097`；OM SHA256 为
`ab56ded310b7f7aaf2c376b4e64c5a47852e802ab808fc3d20557317bd90d01d`。

下一项独立 component 门也已通过：layer0
`input RMSNorm → M16 QKV/RoPE → 双 K/V dynamic append` 在同一只 OM 内覆盖绝对
起点 `0,1,31,4080`，四个位置均无 deadloop，event SSA `940/940`。公开 ABI 精确为
21 个输入（hidden、RoPE cosine/sine、K/V cache 和 16 个绝对 position scalar）与
4 个输出（Q、compact K、compact V、一个 bounded packed witness）。K/V 输出如实只
表示 layer0 publisher slice `[1,2,16,128]`，即最终发布形状 `[1,48,16,128]` 的
channel `[0,2)`；完整 C4096 cache 没有跨过 Report 边界。OM SHA256 为
`6f31c8284ecee4809eda6692fccbc20f7a6208e1b48a6aff7b4220f9ccea5294`，
ONNX SHA256 为
`c91308a83df6cfc4f4f5732cd8f04214b0143860ff7e6b9d5e151524a8c65b48`，
qualification SHA256 为
`a9007f1bc58d383e8a637969fb7c9a5bec30cce134c69ed37a4f86af60c03862`。

| 绝对起点 | Q cosine | K publisher cosine | V publisher cosine |
|---:|---:|---:|---:|
| 0 | `0.9999173161` | `0.9999574123` | `0.9999096573` |
| 1 | `0.9999173215` | `0.9999573036` | `0.9999096573` |
| 31 | `0.9999172380` | `0.9999573519` | `0.9999096573` |
| 4080 | `0.9999169568` | `0.9999568232` | `0.9999096573` |

固定 sentinel 行的 resident-FP16 byte-exact 检查只作诊断；资格门采用已记录的
FP32 sentinel cosine/max-error 阈值，不会把该 byte mismatch 越级写成发布通过。
该产物仍只是 bounded component witness：`attention_ready`、
`single_layer_ready`、`all24_pack_ready`、`all_24_layers_ready`、
`release_runtime_eligible` 和 `production_ready` 全部为 false。

独立终审还通过了一个 bounded synthetic 门：同一只 OM 内已闭合
`双 K/V dynamic append → 单个 C256 causal-tail attention consumer`。物理 ABI 为
21 个用户输入、1 个公开 Report（`attention_context`）、0 次公开中间 INOUT 写入和
0 个公开中间 Report。Gather 后的 K/V panel 先经过 private Neg→Neg canonical
materialization seam，再进入真实 QK/AV consumer；没有 host cache repack 或公开
bridge。libinstsim 无 deadloop，cosine `0.9966713753`，相对 reference 的最大绝对
误差 `0.0073841021`，event SSA `2568/2568`。no-update 与 no-sentinel 负控均明显
差于 reference：两者的 cosine/max-error 分别为
`0.9594016617`/`0.0220721886` 与 `0.9824994984`/`0.0106568751`，从而把结果同时
绑定到动态更新和 sentinel 行。OM SHA256 为
`90f757be0f2771eae1b1f4108279f1337f94e137f48746f05b15c2600c7ca35d`；
qualification SHA256 为
`1c82512d6246e95a147cf550996817722cb9ffaedea76b710fe4029906f783e8`。

**资格边界（禁止越级）：** `synthetic_domain=true`、
`real_model_activation_calibration=false`、`held_out_real_snapshot=false`；因此
`dynamic_full_c4096_attention_ready=false`、`b16_dynamic_ready=false`、
`single_layer_ready=false`、`all24_layers_ready=false`、
`production_ready=false`。该组件不是 S16 发布产物。

同一张图现已使用内容绑定的 3 行真实模型 calibration 编译，并在 3 个 calibration
样本和 1 个完全隔离的 held-out 样本上执行。执行 OM SHA256 为
`70f15965725b3dd2ac430af3c49b1c7fc88edbf16ea581afc3f9b269c3327362`。
四次执行均完成且输出 finite/non-zero，但四次都未通过数值门；cosine/max-error
分别为 `0.902962/0.135358`、`0.935589/0.117447`、
`0.930731/0.163486`、`0.912596/0.107355`。独立终审的 execution
qualification SHA256 为
`4c944cfb2f413c723304fc1c76ee3823a66914eed7d511e542f58649c83c0c7c`。
它证明证据链完整，但 owner numeric、single-layer、full-C4096、24 层、release 和
production readiness 仍全部为 false。CPU 上叠加 input/QK-K/AV-V factor 的简化
量化仿真仍有至少 `0.999645` cosine，因此当前证据不能把根因归到某个 S8 factor
或输入 absmax；最早未证边界仍是 panel online-state 计算，当前只用单 Report
诊断逐层收敛。

第一层诊断现已 fail-closed 收敛。score 输出 descriptor 是 dense FP32
`[1,16,16,32]`/32768B；descriptor-native NCHW 与唯一仍符合 descriptor 的
head/query transpose 都不能解释误差，W4 与 NC1HWC0 则被物理字节契约排除。
独立编译的 terminal Neg→Neg 变体与原 score 的 raw 输出逐字节相同
（`8982ab50...`），cosine 仍为 `-0.0709976621`，最大误差仍为
`67.1187148`。因此 descriptor/layout、原 Nop/Report seam 与 terminal
materialization 都已排除；首个可证坏语义边界是 softmax 前已物化的 panel0
scaled score，但现有证据仍不能继续归因到 MatMul 或 K 量化。layout 与 Neg→Neg
execution qualification SHA256 分别为
`e3f0c99a0456fc3a92c7dc71934fa91c8a3cf6eb6fecc894104a2daad0c9bb27`
和 `43821356cd6529b9a9500dcb42bf78a1a0d84d3247ccda2a68e6bce203e5bb49`。

随后执行的 post-Gather 串行二分进一步把边界压缩了一层。直接发布完整
`packed_attention_panel0 [1,32,32,128]` 的 owner 变体数值为绿色：cosine
`0.9999994531`、最大误差 `0.015625`。下一只仅增加 K-half Crop 的
`key_panel0_packed_slice [1,16,32,128]` 变体则立即失败：cosine
`-0.0126486001`、最大误差 `265.21875`；OM 与 execution qualification SHA256
分别为 `9b198f7582ffcb2fa7ff52564044d204d1852fd743a23fa808e7b0a877d4de28`
和 `f6e7372474ae1e7b8a95d8cdf1b80c1c9c9b022ea8803f7fc375c658e9fe571f`。
物理对照显示这不是一个可以直接归因给 Crop 的实验：绿色终端图让 Gather
直接输出 FP32；增加 Crop 后，mapper 把 Gather 和 Crop 都调度为私有 S16，最后
才由 Report 转为 FP32。因此当前首坏区间应写成
`Gather private-S16 materialization → K-half Crop`，而不是“Crop 已证实有错”。
串行门禁已在此停止，`key_panel0`、raw QK、S32 和 S128 均未继续执行；下一门
必须用独立 S16-materialization A/B 区分 Gather 私有量化域与 Crop/Report 边界。

完整 16-state C4096 attention reduction 也已经编成一个静态 same-OM 图：公开输入仅
Q/full-K/full-V，32 个 cache Slice 留在图内，16 个 C256 state 与 15 次 merge
全部私有；event SSA `32076/32076`，libinstsim cosine `0.9998773019`，OM SHA256
为 `d8cad715b344334b923026b9c26bccd20e696e01b170ddd1f0e77bce2fc86dd6`。
这证明的是 full-cache attention，不是 dynamic append。

layer0 full-C4096 boundary join 也已执行并通过独立结构终审，但它只是 blocked
characterization，不是 qualification。主 OM SHA256 为
`4717e207dc536dbe758bcd3da30ba219b88eb9296a90e6e1ec389139a016d048`，
global event SSA `39685/39685`。K/V publisher 的 cosine 仍为
`0.999958`/`0.999910`，但 `next_hidden` 只有 `0.956954`。独立四输出
diagnostic OM
`ffe6da001272c6f21fdee4aea028da1cb203b3fa898bffacb0b3c77bc5ca793f`
测得 attention-context cosine `0.995993`；把该硬件 attention 送入同一 tail
reference 后，hidden cosine 为 `0.999496`。因此当前问题是 attention 误差经
o_proj/MLP 放大，而不是 private tail seam 读错。主 OM 的物理输出槽还实际为
`V0/K1/H2`，不符合发布契约 `K0/V1/H2`。当前不存在 release qualification，
boundary、single-layer、24 层、runtime、production readiness 全部为 false。

四样本 fulljoin reference dataset 现已物化并完成闭集审计。manifest SHA256 为
`b93cfbae41dad32ce35499b4a3ef826bbc5bbe8b83b59cc9b91449ff234df044`；
134/134 个文件、113,855,374B 全部登记，84 个 raw input 与 20 个 reference
array 均重哈希，21 个 calibration image-list 都恰有 3 行且 held-out 行数为 0。
CPU/ORT 完整单层 reference 的最低 cosine 仍为 `0.999999999998609`；内容绑定的
recipe SHA256 为
`9c20840249757c44e1f2ae043cf1c48a32c6d09bbe78a29ef72d5435ba52f8c5`。
审计只把 `dataset_materialized_and_audited` 置为 true；owner ATC、owner numeric、
single-layer、24 层、release 与 production 仍全部为 false，并明确记录三行
calibration 不保证修复数值问题。

因此完整单层的首个阻塞已精确收敛为：先关闭上述 real-C256
`Gather-S16 → Crop` 边界，修复并验收真实域 attention 精度，纠正
K/V 物理输出槽顺序，再用绑定的真实 calibration 与 held-out 样本重跑
full-C4096/B16 same-OM boundary。只有这些门都通过，才能连接已通过的
RMSNorm/QKV 与 resident tail，并把 24 个如实的 layer slice 汇总为
`[1,48,16,128]`。
bounded H2 resident bridge 也已证明基础 Scatter/Gather 执行契约：position `0,1,31,4095`
均 no-deadloop，selected/witness cosine 约 `0.99999995`，非目标行 byte-exact，
event SSA `102/102`；OM SHA256 为
`76eeedd497b5e3cbfad74b5c10392cf3e031e50972eea7c2ae8a12f7fedb990d`。
两个 bounded component 都不能替代真实模型 calibration 或 dynamic full-C4096/B16
attention boundary 门禁。

工作流下一步先关闭真实域 attention 精度与 full-C4096/B16 boundary join，再把
已经通过的 RMSNorm/QKV、attention 与 tail 接成完整单层，然后才复制到 24 层、
验证 S16→S1/decode handoff，以及板端 token、边界和性能。任何失败都停在 S16，
禁止用 host repack 或公开 `maximum/sum/av` 边界冒充 same-OM 单层；S32/S128
尚未开始。

## Workflow A：S16 首个可执行闭环

1. 已用真实 M16 attention context 构建并验证
   `attention → o_proj → residual → post-RMSNorm → gate/up → SwiGLU → down → residual`
   的同图 private-TEMP splice，并闭合 input RMSNorm→Q/K/V/RoPE 与静态 16-state
   attention；bounded synthetic C256 append→consumer 已通过，下一步先完成真实
   activation calibration，再关闭 full-C4096/B16 dynamic append→attention boundary，
   最后合成完整 single-layer same-OM 路径。
2. 单层覆盖 steady 绝对起点 `1, 15, 16, context-16`，并追加 tile/bank 边界和
   非对齐起点 `643`；position 0 继续使用 strict S1。
3. 每个公开 hidden/K/V 严格 `cosine > 0.98`；mask、RoPE 和全部
   `48×16` K/V 行必须按原始字节证据绑定。
4. 复制到 24 层，禁止 reference hidden 注入；验证层间 hidden、24 层全部 K/V
   以及 S16→S1/decode handoff。
5. Hi3403 上跑英文、中文、EOS、context 边界和 greedy token exact；S16 墙钟时间
   必须严格小于 16 次 S1。

任一门失败即停在 S16，不构建发布资格的 S32/S128。

## Workflow B：S32 与 S128

S32 使用独立宽度产物和证据，在完全相同的输入、绝对位置与 cache 前缀下，与
连续两次已验收 S16 比较：最终 hidden、全部 K/V 行、下一次 S1 token、EOS 和
板端输出必须一致。S32 通过后，S128 同时对比 `4×S32` 与 `8×S16`。

最低位置矩阵：

| 宽度 | 必测起点 |
|---|---|
| S16 | `1, 15, 16, 31, 32, 255, 256, 643, context-16` |
| S32 | `1, 31, 32, 127, 128, 643, context-32` |
| S128 | `1, 127, 128, 511, 512, 643, context-128` |

position 0 继续使用 strict S1/prefill bootstrap；宽块首版只允许 `steady` phase。
只有单独的 startup 量化证据通过后，才可改变该规则。

## Workflow C：物理 ABI 与 resident cache

宽块 publisher 的物理 ABI 固定为连续 channel-major FP32：

```text
K, V: [1, 48, W, 128] FP32
```

executor opcode 6 一次处理每通道 `W×128` 元素，按 RNE 转成 decode resident
cache 的 FP16 行。runtime 会检查 source output 精确字节数、destination cache
descriptor、绝对 offset 和 context-1 边界。每个增量只原子发布到 canonical decode
cache；其他临时 wide handle 在执行前通过 opcode 9 从 canonical cache 重建前缀。

宽块执行前，opcode 9 用 96 条固定记录将 canonical decode K/V 的有效前缀复制到
目标 handle。executor 会先 drain、校验全部记录并 invalidate 全部 source，之后才
执行 copy 和 destination flush；坏记录不会留下半拷贝。协议索引只在当前 executor
model table 生命周期内有效，进程或 phase 更换后必须重建记录。

逻辑 cache ABI 仍是 FP16。qualification 同时记录 publisher FP32 与 resident FP16，
禁止用一个模糊的 `dtype` 字段混淆两者。

## Workflow D：资格激活与 MMZ admission

`app/src/minicpm_prefill_activation.py` 负责发布前的第二道门：

- 只接受 release qualification v4；development-only v2 与历史 v3 明确拒绝激活；
- 读取真实 OM、build manifest、runner、executor 和 ready descriptor；
- 拒绝绝对路径、`..`、symlink、size/SHA 漂移；
- 真实读取并复验所有 capture、workload、性能测量 artifact，以及完整的 baseline
  qualification + OM 链（S16←S1、S32←S16、S128←S32/S16）；
- 从绑定的 clean-board MMZ before/after 观测推导 `admission_bytes`，并要求与
  activation manifest 完全一致；
- 每个宽度独立计费，重复 residency group 直接禁用；
- 任一宽度失败时只禁用该宽度，strict S1 永远保留。

Activation manifest 必须显式绑定板端实际常驻、完成 content qualification 的 S1
OM/build/runner/executor/descriptor identity，布尔标志不再有效。所有宽块 baseline
链必须解析到同一个 anchor；调用方 `base_resident_bytes` 还必须不小于 S1 实测
residency。顶层 S1 anchor 不合法时整体拒绝 activation；anchor 不一致或 base 少报时
禁用宽块，但保留已验证 S1 路线。

三份独立 24 层 S16/S32/S128 OM 不得默认同时常驻。串行构建和验收可以继续，
但完整运行时路线必须满足以下条件之一：

1. clean-board 实测证明 base models、独立宽块及安全 reserve 能通过 MMZ
   admission；或
2. 未来静态 carrier 由新 qualification schema 绑定单一最大 descriptor、
   `valid_width` 分支和一次真实 residency，不能复用当前 per-width v4 ABI。

具体 MMZ 下界、lazy-wide bring-up、canonical KV/input-copy 和静态 S128
`valid_len` 发布方案见
[Native Prefill residency 与 MMZ 闭环](NATIVE_PREFILL_RESIDENCY.zh-CN.md)。
v4 evidence index 与 CLI 详见
[Native prefill 发布资格 v4](NATIVE_PREFILL_RELEASE_QUALIFICATION.zh-CN.md)。

验证 activation manifest：

```bash
python3 app/src/minicpm_prefill_activation.py \
  --manifest work/prefill/activation.json \
  --deployment-root /opt/pico-minicpm5 \
  --context 4096 \
  --available-bytes "$MMZ_AVAILABLE" \
  --base-resident-bytes "$BASE_RESIDENT" \
  --reserve-bytes 268435456 \
  --output work/prefill/activation-report.json
```

报告中的 `enabled_widths` 才能传给调度器。构建成功或 qualification 文件存在本身
都不能激活宽度。

板端应用启动参数与上面的验证器保持同一组 live 输入：
`--prefill-activation-manifest`、`--available-bytes`、
`--base-resident-bytes`、`--reserve-bytes`。`minicpm_prefill_runtime.py` 会先执行
`load_activation`，再与本进程注册的 typed 宽块 handler 求交。typed dispatcher 与
fake transport 闭环现在已覆盖精确 descriptor/publisher 绑定、opcode 9 前缀恢复、
一次 wide execute、opcode 6 canonical 行发布、中间块 no-head 和最终 hidden
handoff。任一步失败都会毒化并终止整个 resident session，未重建前不能继续伪装成
S1。production registry 仍为空且没有 CLI 注入入口，因此资格通过的宽度只出现在
`qualified_widths`，不会进入调度器的 `enabled_widths`；生产遥测仍明确记录 S1。

mask row `j` 暴露绝对 K/V 前缀 `[0,start+j)`，并以最后一个 context column 作为
current-token sentinel。真实合格的 wide 图必须在内部把同一块先前的 K/V 动态追加
到绝对 slot，后续 row 才能消费。运行时在任何 model execute 前预检整份计划；由于
v1 opcode 6 不能越过 `context-1` cache，最终 context position 强制重规划为 S1。
这些只是 fake 验证的软件契约，不代表真实 wide OM 已存在。

## 自动门禁

release 工作区：

```bash
cd release_work/pico-minicpm5
PYTHONPATH=src ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_prefill_schedule.py \
  tests/test_prefill_runtime.py \
  tests/test_prefill_wide_dispatch.py \
  tests/test_prefill_blocks.py \
  tests/test_prefill_activation.py \
  tests/test_board_repl.py
make -C app/native contract-check
```

native compiler 工作区先运行 S16 closure contract，再运行新产物的 ATC、libinstsim
和板端门。每一级报告必须保存输入、输出、OM、build manifest、descriptor、runner、
executor 的 SHA256；没有落盘证据的 PASS 不进入下一级。

修复前的 single-layer blocked qualification SHA256 为
`0deaa519316a9741fbdc17e5beeaba0aac5d7511bc97ff8f38d2b44201eb0fbf`；
它记录 `owner_atc_compile_attempted=false`，因为 resident bridge 的语义执行门先于
完整单层编译，不能生成 partial OM 伪装进展。该记录作为历史失败证据保留；当前
重跑必须引用已修复 bridge 的新 OM/audit hash。

## 最终完成定义

只有以下条件同时满足，才可将路线标为闭环：

- S16、S32、S128 各自完整 24 层、数值、token、边界、板端和性能门全部 PASS；
- 发布运行时实测选择顺序为 `S128→S32→S16→S1`，任意尾部无 padding；
- canonical scatter、input-copy、snapshot restore、block-to-block、block-to-S1
  handoff 均 exact；
- clean-board MMZ admission 和异常回落已验证；
- Agent/OpenClaw 长 prompt TTFT 有同轮 strict-S1 对照，且生成 token 不回退。
