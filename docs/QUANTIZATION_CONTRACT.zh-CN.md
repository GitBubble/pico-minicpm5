# 量化契约

[English](QUANTIZATION_CONTRACT.md)

本文记录三个 OM 句柄的 int8/int16 量化如何被决定：标定由谁执行、图内的 `Clip`
节点对它做了什么、以及为什么同一张图要构建两份标定 family。

出处说明：代码事实引用本仓库文件。激活测量由浮点参考 dump(`reference capture`,
六个 position × 24 层）重新计算得到；第 2 节引用的 ATC 记录来自整合 monorepo
中的一个 ATC 工作目录——两者都不随本仓库分发，因此这些数字无法仅凭 release
检出重新推导。

## 1. 谁在做量化

ATC 自己。`src/pico_minicpm5/compiler/atc.py` 是唯一能产出真实 `.om` 的后端
(另有一个供 CI 使用的 `FakeCompiler` 桩，可用 `--backend fake` 选择)，其
`compile()` 只对 ATC 二进制做一次 `subprocess.run`。流水线交给 ATC 的是一张
浮点 ONNX 图，外加每个输入一份 `--image_list` 标定语料；ATC 在该语料上观察全图，
内部跑 IFMR 量程搜索，把 scale/offset 烘进 `.om`。每个 `.om` 旁边出现的
`calibration_param.txt` 是 ATC 的**输出**而非开发者编写的输入——厂商的
`atc_param_conf.json` 里没有任何键能消费这个名字的文件。

由此得到几条应当明确写下的结论：

- PICO ATC 是 IFMR-only。它**无法**送入 QDQ、AMCT 或 QAT 的 scale——算子表里
  根本没有 `AscendQuant` / `QuantizeLinear`。
- ATC 本身确实提供了一条外部 scale 通道 `--gfpq_param_file`，但
  `AtcCompiler.command()` 构造的是固定 argv，没有留出钩子。要送入外部量化参数
  需要改这里的代码，而不是引入新工具。
- 数据通路模式由 `--compile_mode` 选择，而非独立开关。厂商枚举把两个取值命名为
  `Low-bandwidth`(0)与 `High-precision`(1)；本项目称其为 A8W8 与 A16W8,
  即 8-bit 或 16-bit 激活配 8-bit 权重。支持这一命名的工件证据是间接的：一个
  `compile_mode=1` 的构建记录为 `calc_data_type: S16` 配 `weight { S8 }`，而
  **另一个模型**的 `compile_mode=0` 构建记录为 `S8`/`S8`。不存在同一张图上的
  受控 A/B。
- `atc.py` 以默认构造参数形式取 `compile_mode: int = 1`，且没有 CLI 开关暴露它，
  因此这条流水线构建的每个 OM 都走 A16W8 模式。release 仓库不携带 `.om`，也不
  携带构建日志，所以这是代码路径的性质，而不是 release 检出能验证的事实。

## 2. Clip 对量化器做了什么

预设 `Clip` 不是元数据，也不是建议。`_insert_clip`
(`src/pico_minicpm5/onnx/layer.py`)插入的是真实的 ONNX `Clip` 节点，并把该
张量的**每一个**消费者改接到钳位后的值，因此推理时的钳位是无条件的。由此可以
直接得到两条性质，而一条被广泛复述的推论则**不成立**。

**有证据：ATC 采用 Clip 的常数边界作为其输出的量化器量程。** 在一份 layer 0 的
ATC 记录中，`in_rmsnorm_clip` 的 `±sqrt(1536)` 边界产生了 `q_proj` 输入量化
scale `52.2430191`、offset `0`，而 `52.2430191` 正是该边界的步长倒数在 float32
下的结果：

```
step   = float32(2 * float32(sqrt(1536))) / 4095   # 4095 级,[-2048, 2047]
1/step = 52.243019104   ->  打印为 52.2430191
```

注意这里是 float32 下的**步长倒数**，不是普通除法：`4095 / (2 × 39.191837)`
等于 `52.2430219`，与记录值差一个 ULP。同样的更正适用于 attention mask：其记录值
`63.9843712` 是 `float32(1/(64/4095))`，而非 `4095/64 = 63.984375`。

**有证据：对称 Clip 使反量化 offset 恰好为 0。** 已在同一记录的 `q_proj`、
`k_proj`、`v_proj`、`gate_proj`、`up_proj` 上确认。

**未证实：Clip 只能收紧。** 上游笔记把规则写成
`min(推断值, Clip 边界)`。但唯一被拿来当作其证明的工件恰恰否定了它：该张量实测
最大分量是 `37.8636`，若按 `min()` 应选 `37.86` 并得到 scale `54.10`；而记录里
是边界给出的 `52.2430191`。工件说明的是**边界赢了**，而非**两者取小者赢了**。
比数据更松的边界是否会撑大发射量程，尚未被测试——第 5 节正取决于这个开放问题。

但有一点无论如何都是确定的：钳位是真实的运行时算子，因此标定与推理看到同一个分布——会
溢出的值在两遍里都被钳，而不是只在其中一遍里被悄悄回绕。

## 3. 两级 Clip

**可证明锚点。** 每个 `ExtendRMSNorm` 之后，图会钳到 `±sqrt(H)`。
`sqrt(1536) = 39.191835884530846`，在图中以 float32 `39.191837310791016` 存储。
这是数学边界而非经验值：RMSNorm 使每行 RMS 为 1，故 `Σx² = H`，任一分量都不可能
超过 `sqrt(H)`。两个 family 都会无条件发射它。在全部 24 层 × 六个参考 position
上实测到的最大分量是 `37.8636`(layer 19,position 0)，因此锚点实际上从不钳位
任何东西——它的存在是为了向 ATC **声明**量程。

**经验族预设。** `configs/calibration/{prefill,decode}-clips.json` 携带逐层、
逐张量的实测边界。这一级是 `family` 参数在图上改变的唯一东西。

与 `sqrt(H)` 相差在 `1e-5` 以内的预设边界会被丢弃——但只在 `normed` 与
`post_normed` 上（`onnx/layer.py`)。这个容差是必要的：预设存的是 float32 值，
与 `sqrt(1536)` 相距 `1.4e-6`，若按严格相等判断则一个都丢不掉。其他张量上取到
锚点值的边界，仍会变成真实的 Clip 节点。

## 4. 为什么要有两个 family

两个 OM 是同一张图。`family` 只通过一扇门进入图——加载哪份 clip 预设；只通过
一行进入标定——prefill **只**取参考的 position 0,decode **只**取 position
`>= 1`。已发行的 ctx1024 两个二进制出自同一张图，体积上只差 2,529 字节（共 687 MB），
但内容的差异不止于此（见第 5 节）。

它们不能合并，因为 position 0 的激活不是略有不同，而是**高出若干数量级**。由
浮点参考重新计算，layer 0(绝对最大值；倍数在取整前计算):

| L0 张量 | position 0 | position 1 | 倍数 |
|---|---:|---:|---:|
| hidden(输入） | 0.1025 | 0.1191 | 0.86 |
| attn_residual | 1.1073 | 1.1462 | 0.97 |
| swiglu | 28.54 | 0.6262 | **45.6×** |
| down_out | 84.93 | 2.878 | **29.5×** |
| next_hidden | 86.04 | 4.024 | 21.4× |

预设边界记录着同一件事：两个 family 之间同名而取值不同的 64 个边界中，prefill
在全部 64 个上更松、没有一个更紧，极值出现在 L4 `down_out`(`4000.0` 对 `7.5`)
与 L5/L6/L8 `attn_residual`(`4000.0` 对 `18.0`)。这些被抬高的边界**无一例外**
落在残差载体或 MLP 分支张量上，没有一个是 `v_cur`、`context_grouped` 或 `o_out`。
合并标定会把一个宽出两个数量级的量程交给 decode，直接摧毁它的分辨率。而完全
去掉 prefill family，会逐字节复现历史上的整模失败。

### 机理在 MLP，不在 attention

很容易用 attention 来解释：position 0 时 KV cache 为空，softmax 只能关注当前
token，无法做平均。前半句完全正确，而且比需要的更强——
`context_grouped == broadcast(v_cur)`，在**全部** 24 层上最大绝对差都是 `0.0`。
**但这个解释仍然是错的**，因为 `v_cur` 本身在 position 0 就更小（pos0/pos1
绝对最大值：L1 `0.089`、L4 `0.238`、L7 `0.318`)。两个效应互相抵消：在全部 24 层
上，attention 输出的 pos0/pos1 倍数，`context_grouped` 跨 `0.28`–`1.27`,
`o_out` 跨 `0.47`–`2.83`。最大的一个是 L20 的 `o_out`,`2.8×`——远不及预设边界
所要求的 10×–500×，而且中位数低于 1。

真实发生的是:**position 0 在 layer 0 的 MLP 分支引爆**，在 layer 4 再次引爆，
此后由残差流把量级一路带下去。layer 0 MLP 之前的一切都正常（hidden `0.86×`、
attn_residual `0.97×`)，而 SwiGLU 输出一步跳升 `45.6×`。从 layer 1 起，
`hidden` 与 `attn_residual` 只是继承这个载体（L1 `21.4×`、L4 `15.6×`、
L7 `448×`、L23 `13.9×`)，而各层的 attention 分支始终接近 1×——因为
`ExtendRMSNorm` 是尺度不变的，且 norm 的 gamma 已折进 q/k/v/gate/up 投影。
layer 4 实测的 position-0 值（`swiglu 1018.8`、`down_out 2777.0`)正是该层携带
`4000.0` 边界的原因。

MiniCPM5-1B **没有** `scale_emb`、`scale_depth` 或任何 muP 因子，其 config 就是
标准的 `LlamaForCausalLM`。layer 0 两个锚定 norm 前面那个 `×32` 前置缩放是本
流水线自己的已验收选择，不是模型属性。

## 5. 那 33 个冗余的 prefill clip，以及一个开放问题

prefill 预设比 decode 多产生 33 个有效 `Clip` 节点：15 个在 `normed`、18 个在
`post_normed`，边界一律为 `50.0`，且全部落在锚点 Clip 已经以 `39.1918` 产出的
张量上。此外，两个 family 在 12 个共有的锚定键上也不同，其中 11 个是 prefill
停在同样失效的 `50.0`，而 decode 收紧到 `27.0`–`33.0`。

它们**结构上是冗余的**。在 gamma 折进后继投影的恒等归一化下，`RMS(out) = 1`
精确成立，因此即便删掉该张量上的每一个 Clip，也没有任何分量能超过 `sqrt(H)`;
实测最大为 `37.86`。

它们在发射 OM 中的代价小而有界：两个已发行二进制携带**同为 851** 条层记录，
说明 ATC 把每个额外 Clip 融进了锚点已有的那一层，而非新增一层。2,529 字节的
大小差**全部**落在 net-def item 区（33 个增长块的差值之和恰好等于 2,529)。
两个 OM 并非在其他方面完全相同——它们还相差 562,052 字节 param-head 与 301 字节
指令流——但那些源自 64 个不同的**活**边界，不是这 33 个。

**尚未定论的是它们会不会改变量化器。** 若第 2 节那条未证实的 `min()` 规则成立，
`39.19` 锚点之上的 `50.0` 就是空操作；若 ATC 采用的是最后一个 Clip 的边界，
prefill 就会在锚点声明 `39.19` 的地方声明 `50.0`——量程宽 28%，在
`normed`/`post_normed` 上约损失 0.35 bit 分辨率。现有证据无法区分两者：两个
family 之间有 6 条投影量化记录不同（`L9_q_proj`、`L0_k_proj`、`L9_k_proj`、
`L4_gate_proj`、`L23_gate_proj`、`L4_up_proj`;`v_proj` 24 条全同)，而这同样
可以由那 64 个不同的活边界解释。

能了结此事的实验很小：去掉这 33 个边界重建一层 prefill，再 diff 发射出的
`calibration_param.txt`。记录相同则证明 `min()`;`normed`/`post_normed` 上的
scale 发生变化则证明"最后一个 Clip 说了算"，并使删除它们成为一次免费的精度
改善。在此之前，本节记录的是一个开放问题，不是结论。

`tests/test_clip_contract.py` 钉住第 3–5 节背后的**预设级**不变量，使再生不会
悄悄发射一个低于锚点的 prefill 边界，或一个比 prefill 更松的 decode 边界。它
不能、也没有钉住 ATC 的行为。
