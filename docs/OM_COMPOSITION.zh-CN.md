# OM 图级组合合同

[English](OM_COMPOSITION.md)

发布路线先组合 ONNX 图，再一次编译结果，让图编译器统一负责内存、调度、
TaskInfo 和物理 hidden bridge。

## 图规则

1. 私有 node、tensor、initializer 增加 `L{index}_` 前缀。
2. `hidden`、`attention_mask`、`rope_r` 保持公共/共享名称。
3. 第 `i>0` 层的 hidden 输入连接 `L{i-1}_next_hidden`。
4. 24 层 K/V cache 分别打包，并沿 channel axis 1 Slice 到各层。
5. current K/V 按 layer-major、KV-head-major 顺序 Concat。
6. 只发布最终 hidden 和两个打包 current-row tensor。

24 层公共 ABI 固定为 5 输入/3 输出。ATC 的 Hi3403 descriptor 会额外合成两个
runtime input，它们不是图输入。

## External data 与量化域

真实 24 层 ONNX 超过 protobuf 2 GiB 限制；单一 external blob 的 offset 还可能
越过 signed-32-bit。因此每个大 initializer 使用独立文件且 offset 为 0，并在
ATC 工作目录创建相对链接。

打包输入用于绕过输入数量限制；打包输出会形成共享量化域。prefill 的公开 V
cosine 为 `0.996646`，但低能量 L2 slice 为 `0.971119`。发布必须同时报告聚合
门禁和逐层诊断。若策略要求每层都大于 0.98，应保留打包输入并拆分输出量化域。

## 为什么不默认二进制 post-link

二进制链接必须同时闭合参数 heap/累计权重表、绝对分支 PC、END 生命周期、
TaskInfo、Item/Net IO、workspace 和物理层间 bridge。模型能加载/执行并不能证明
L1+ 使用了自己的权重。任何未来 linker 都必须先通过 N=1 identity、N=2 liveness
和参数隔离，才可进入 24 层门禁。
