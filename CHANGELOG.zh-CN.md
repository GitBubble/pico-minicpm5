# 变更日志

[English](CHANGELOG.md)

## 0.2.0 - 2026-08-17

### 执行器可复现，且每一档都变快了

- 板端执行器现绑定 `cef4edb2…`，由本仓库源码用受认可的
  `aarch64-mix210-linux-gcc` 7.3.0 工具链构建，并在写清单之前验证过逐字节一致
  （`docs/EXECUTOR_BUILD.zh-CN.md`）。`v0.1.0` 钉的那个二进制，其随附源码无法
  重建。
- 它在多次 execute 之间保留 workspace 输入而不是反复重写，因此每个 decode 步省下
  一次完整的 workspace 写入。三档在同一次会话中实测：ctx1024
  `9.46 → 9.96 tok/s`（+5.2%）、ctx4096 `6.53 → 7.81 tok/s`（+19.7%）、ctx8192
  `4.59 → 6.03 tok/s`（+31.6%）。48 个贪心 oracle token 与各自已验收基线
  逐个一致。
- 节省与保留的 workspace 成正比——24.6 / 98.3 / 196.6 MiB 对应 5.5 / 27.6 /
  54.9 ms——三点共线，这才使它成为机理而非巧合。
- 构成 TTFT 的 prompt 送入成本现在三档皆有实测：每 prompt token `79.49` /
  `106.28` / `144.02` ms。ctx1024 的数字与一次独立冷 prefill 实测相差 `0.10%`。

### ctx4096 以混合 prefill 窗口契约转正

- Runtime profile 携带 `context.prefill_window`；ctx4096 与 ctx8192 的 position 0
  在冻结的 ctx1024 `prefill.om` 上引导，并共享 `head_flat.om`，逐字节相同。直接
  实测：两者的 position-0 transformer 时间相差 `0.39 ms`。
- ctx4096 为 `qualified`；ctx8192 因 donor 零扩展标定保持 `pending`，中文 oracle、
  内存包络与长 prompt 三项仍未闭合。

### 一条量错了东西的门

- ctx8192 一直带着 `eos: FAIL_STRICT_SEQUENCE_MISMATCH`，而它对照的序列从未被
  追溯到参考模型。用固定 checkpoint 以 float64 重新推导后，参考模型写下一个句号
  然后停止——这正是 ctx8192 的输出，而 ctx1024 与 ctx4096 并非如此。那条"期望"
  是从第一个跑出来的工件记录下来的。见 `release/contexts/strict-eos-oracle.md`。

### 文档

- 两个首页上的板端 agent 真实会话录制，自包含动画 SVG：`1.9 ms` 返回的工具调用、
  一次上下文 rebase、`9.74 tok/s` 的生成。等待段按 `4.5x` 播放且倍率写在画面上，
  每帧右下角是板子自己的墙钟。
- `docs/QUANTIZATION_CONTRACT.zh-CN.md`：`Clip` 如何为 ATC 的 IFMR 量程搜索封顶、
  position 0 为何需要自己的标定 family（是 layer-0 的 MLP 分支而非 attention——
  attention 那个解释被明确记为已证伪），以及一条因自身证据推翻而撤回的规则。
- `release/perf/` 新增 TTFT、逐上下文相位拆解、被取代的并轨前数字，以及每项的
  证据哈希。

## 0.1.0 - 2026-08-09

### Runtime 增量刷新 - 2026-08-10

- 在 `app/` 中补齐中英文板端应用、runtime 源码与 executor C/Makefile。
- 启用 resident packed K/V scatter 和 byte-exact RoPE/embedding 快速准备，
  不改变三只已验收 OM 哈希。
- ctx1024 板端性能提升到 `9.42–9.48 token/s`，保持 48/48 token
  一致、EOS 和中文路径通过。
- 新增常驻 stdin REPL（`/help`、`/reset`、`/quit`），多次输入复用
  三个已加载句柄；无参数执行 `app/chat.sh` 直接进入 REPL。
- REPL 改为逐 token 流式显示，初始回答上限从 32 提升到 128 token，
  新增 `/max N` 动态调整和明确的上限提示。
- 固定 system/tool 前缀 resident 快照通过 Hi3403 token-exact A/B 后在 Agent 中
  默认启用；恢复 137-token 前缀耗时 `1.76 ms`，重复 32-token 请求由
  `26.97 s` 降至 `12.56 s`（降低 `53.4%`）。
- 确定性 context rebase 通过 Hi3403 长会话门禁：两轮都把 12 个旧工具轮次由
  `2808` token 压到 `810` 并返回同一组 `[18655, 4569, EOS]`；重复运行命中
  643-token 前缀后提速 `4.75x`。
- 最后一个已知输入 token 之前跳过词表 head 和 argmax。token-exact 板端 A/B
  将首次长 prompt 延迟降低 `19.89%`（`86.70→69.45 s`），resident 重复请求
  降低 `19.59%`（`18.17→14.61 s`）。
- 新增未来 `S128 -> S32 -> S16 -> strict S1 tail` 路线的 fail-closed
  native-prefill 调度器，记录绝对区间、各宽度次数和总调用数。在每个 context 的
  宽块产物通过完整数值与板端门禁前，已验收 bundle 仍保持 S1-only。
- 新增 `qualify-prefill-block`，固定严格 `>0.98` 策略，并绑定 S16/S32/S128
  的 OM 血缘、K/V 全行发布、绝对位置 capture、prefill-to-decode handoff、
  token exact 和板端证据。
- 启动 S16→S32→S128 实现/验收与 S128→S32→S16→S1 运行时选择的闭环
  workflow；S16 input RMSNorm、RMSNorm→QKV/RoPE、C4096 attention 和
  attention→layer-tail 已分别通过同图执行门，bounded synthetic C256
  append→attention 门已通过，但 4 个真实 calibration/held-out 执行全部未过数值门；
  full-C4096/B16 same-OM 也仍被 hidden 精度与 K/V 物理槽顺序阻断。
- 新增 fail-closed prefill activation/MMZ admission 契约：真实校验 OM、build
  manifest、qualification 哈希和物理 publisher ABI，宽度证据或内存不合格时只回退
  strict S1。
- resident KV scatter 已泛化为连续 W 行 FP32→FP16 RNE，并只原子发布到
  canonical decode cache；其他 wide handle 用 opcode 9 从 canonical cache 重建前缀。
  完整 S16 OM 通过前仍不在发布运行时启用宽块。
- 新增 resident input→input copy opcode 9：96 条 channel-wise K/V 前缀记录会先
  全量校验、source invalidate，再执行 copy 与 destination flush；runtime 已提供
  canonical decode cache→wide handle helper，真实宽块未验收前仍不激活。
- S16 attention→layer-tail 的同图 resident splice 已通过真实 libinstsim 门：
  bridge 仅使用 private TEMP、公开 INOUT 为 0，event SSA `4092/4092`，cosine
  `0.99969908`。
- S16 input RMSNorm 独立真实门以 event SSA `142/142`、cosine
  `0.9999759688` 通过；当前推进完整 single-layer same-OM join，尚未宣称完整单层
  或 24 层。
- S16 `input RMSNorm→M16 QKV/RoPE` 同 OM resident splice 已通过：中间
  normalized hidden/QKV activation 只有 private TEMP、公开 INOUT 为 0，event SSA
  `540/540`，Q/K/V cosine 分别为 `0.9999167/0.9999570/0.9999097`。
- layer0 `RMSNorm→QKV/RoPE→双 K/V dynamic append` component 已独立验收：绝对
  起点 `0,1,31,4080` 均无 deadloop，event SSA `940/940`，各位置 Q/K/V publisher
  cosine 均高于 `0.9999`。其 21-input/4-output ABI 只发布如实的
  `[1,2,16,128]` layer slice（最终 publisher 为 `[1,48,16,128]`），完整 C4096
  cache 没有 Report。OM/ONNX/qualification SHA256 分别为
  `6f31c8284ecee4809eda6692fccbc20f7a6208e1b48a6aff7b4220f9ccea5294`、
  `c91308a83df6cfc4f4f5732cd8f04214b0143860ff7e6b9d5e151524a8c65b48`、
  `a9007f1bc58d383e8a637969fb7c9a5bec30cce134c69ed37a4f86af60c03862`。
  sentinel FP16 byte-exact 状态仅作诊断；attention、single-layer、
  all24-pack/all-24-layer、release-runtime 与 production readiness 均仍为 false，
  该 publisher-only 组件本身不证明 attention consumer。
- 16 个 C256 online-attention state（第 16 支为 causal tail）及其 merge 已合成
  一只 C4096 OM；为避开 mapper 的 32-input 上限，公开 ABI 固定为
  Q/full-K/full-V 三输入，32 个 cache Slice 留在图内且无需 host repack。
  libinstsim 完整执行，cosine
  `0.9998773`、event SSA `32076/32076`、公开 intermediate 为 0；日志中的 6 个
  large-Jump warning 作为非致命执行警告保留。该门尚未包含动态 16 行 K/V
  append，因而不代表完整 S16 单层或 24 层已经就绪。
- 完整单层 preflight 将首个硬阻断收敛到 dynamic C4096 H2 KV resident bridge：
  根因是 Scatter/Gather 重复 VA epoch；仅 NOP Gather 的 ACTVA/ALLOCBG 后，四个
  必测位置全部执行，cosine 约 `0.99999995`、非目标行 byte-exact、SSA `102/102`。
  完整 full-C4096/B16 attention same-OM join 仍保持 fail-closed。
- bounded synthetic `双 K/V dynamic append→C256 causal-tail attention` 门已通过
  独立终审。OM 具有 21 个用户输入、1 个公开 Report、0 个公开中间结果，真实 QK/AV
  consumer 前使用 private Neg→Neg materialization seam。libinstsim cosine
  `0.9966713753`、最大绝对误差 `0.0073841021`，无 deadloop，event SSA
  `2568/2568`；no-update 与 no-sentinel 负控均差于 reference。OM 与 qualification
  SHA256 分别为
  `90f757be0f2771eae1b1f4108279f1337f94e137f48746f05b15c2600c7ca35d` 和
  `1c82512d6246e95a147cf550996817722cb9ffaedea76b710fe4029906f783e8`。
  资格记录醒目标明 `synthetic_domain=true`、
  `real_model_activation_calibration=false`、`held_out_real_snapshot=false`；
  full-C4096/B16、single-layer、all-24-layer 与 production readiness 仍为 false。
  当前首阻塞为真实 activation calibration + full-C4096 boundary join；S32/S128
  尚未开始。
- 同一 bounded C256 图已用 3 行内容绑定的真实模型 calibration 编译，并执行 3 个
  calibration 样本和 1 个隔离 held-out 样本。四次均执行完成，但 cosine 分别仅为
  `0.902962`、`0.935589`、`0.930731`、`0.912596`，全部未过数值门。执行 OM 为
  `70f15965725b...3327362`，独立终审 execution qualification 为
  `4c944cfb2f...c0c7c`；证据完整，但 owner numeric、full-C4096、single-layer、
  24 层、release 与 production readiness 仍全部为 false。
- real C256 的首坏边界已继续收窄且未越级归因：物理 layout 排除了
  NCHW/head-query 重解释、W4 与 NC1HWC0；独立 terminal Neg→Neg 与原 score
  raw 逐字节相同，cosine 仍为 `-0.070998`。当前首个可证坏值是 softmax 前已物化
  的 panel0 scaled score，尚不能指定为 MatMul 或 K 量化根因。
- post-Gather 串行二分继续得到一绿一红：完整 packed panel 终端 cosine
  `0.9999994531`，而 K-half Crop 终端仅 `-0.0126486001`（最大误差
  `265.21875`）。物理 dataflow 同时从 Gather-FP32 变成 Gather-S16/Crop-S16，
  所以首坏区间记为 `Gather private-S16 materialization → K-half Crop`，不把
  Crop 单独冒充为根因；`key_panel0`、raw QK、S32/S128 均按门禁停止。
- layer0 full-C4096 boundary join 已完成 blocked characterization 与独立终审。
  主 OM `4717e207...d048` 的 event SSA 为 `39685/39685`，K/V cosine 为
  `0.999958`/`0.999910`，但 `next_hidden` 只有 `0.956954`。诊断 OM
  `ffe6da00...793f` 单独测得 attention cosine `0.995993`，该硬件 attention 经
  tail 后为 `0.999496`，证明是 attention 误差被放大，而非 private tail seam
  失配。主图物理 ABI 还是 `V0/K1/H2`，不符合 `K0/V1/H2`；未签发 qualification，
  S32/S128 仍未开始。
- 四样本 fulljoin reference dataset 已物化并完成闭集审计：manifest
  `b93cfbae...f044`，134 个文件/113,855,374B，21 个 calibration image-list
  各 3 行且不含 held-out；CPU/ORT 完整单层最低 cosine
  `0.999999999998609`。当前只有 dataset readiness 为 true，owner ATC/numeric
  与全部上层 readiness 仍为 false。
- opcode 6 改为事务式发布：全部 scatter record drain/校验、全部 source
  invalidate 后才转换，destination 全部 flush 后才 ACK；flush 失败会终止
  executor，避免继续使用可能失配的 resident cache。
- strict-S1 发布锚点升级为双路 v4 契约：position 0 bootstrap OM、steady
  canonical decode OM、实际 head OM、embedding、两套 descriptor、imported protocol
  runner 和 executor 分别绑定；token-exact 证据哈希 little-endian uint32 精确 token-ID
  序列，无需 tokenizer identity。只要提供 activation manifest，即使没有 wide handler，
  创建进程前也会重新哈希完整实际路线与每只已注册 wide OM；文档明确要求可信只读部署
  树并记录路径重新打开的残余竞态，不宣称具备 inherited-fd 安全性。
- resident snapshot restore 会先校验全部 range，再写入并 flush 每个 cached
  destination；native 或 host 任一恢复失败都会 poison 并销毁整套 resident session。
- executor 源码、Makefile 与 demo 不再作为独立 Asset 重复发布。

- 首次开源从固定 MiniCPM5-1B checkpoint 到真实权重单层 ONNX、24 层打包
  prefill/decode ONNX、外部 ATC 编译及可复现三句柄 Release manifest 的流水线。
- 记录 Hi3403 ctx1024 验收，不在源码中分发权重、私有 SDK 或板端二进制。
- 引入 `runtime-capture.v1` 血缘证据，把每次 libinstsim/Hi3403 执行与严格的
  transformer/head 评分绑定，并约束 head 使用同 position hidden 和全零 residual。
