# pico-minicpm5 v0.2.0

[English](RELEASE_NOTES.md)

三个 `ctx1024` OM 与 `v0.1.0` 逐字节相同。下面每一个吞吐数字的变化都来自执行器，
不来自任何模型重编。

## 执行器现在可由本仓库源码复现

`v0.1.0` 钉的执行器无法由随附源码重建。本版钉的是 `cef4edb2…`，由
`app/native/pico_persistent_acl_executor.c` 用受认可的
`aarch64-mix210-linux-gcc` 7.3.0 工具链构建。配方见
[`docs/EXECUTOR_BUILD.zh-CN.md`](../../docs/EXECUTOR_BUILD.zh-CN.md)；在写这份
清单之前，它对着**已提交的源码**重跑过一次，逐字节复现出同一个哈希。

新执行器在多次 execute 之间保留 workspace 输入而不是反复重写，因此每个 decode
步省下一次完整的 workspace 写入。节省因此随上下文增长。三档里有两档折算出相同
的避免带宽；ctx1024 比按比例的拟合少省一些，这与它最没有空间摊薄固定的单次
execute 开销相符：

| 上下文 | 保留的 workspace | transformer 节省 | 折合带宽 |
|---|---:|---:|---:|
| ctx1024 | 24.6 MiB | 5.5 ms | 4.66 GB/s |
| ctx4096 | 98.3 MiB | 27.6 ms | 3.74 GB/s |
| ctx8192 | 196.6 MiB | 54.9 ms | 3.76 GB/s |

## 实测：一次板端会话，三档同场

| Profile | 每 token p50 | token/s | 相对 v0.1.0 路径 | prompt 送入 |
|---|---:|---:|---:|---:|
| ctx1024 | 100.40 ms | **9.96** | +5.2% | 79.49 ms/token |
| ctx4096 | 127.96 ms | **7.81** | +19.7% | 106.28 ms/token |
| ctx8192 | 165.71 ms | **6.03** | +31.6% | 144.02 ms/token |

三档的 48 个贪心 oracle token 与各自已验收基线**逐个一致**。ctx1024 的送入数字
与一次独立的冷 prefill 实测相差 `0.10%`。

## ctx4096 转正；ctx8192 仍为候选

`ctx4096` 通过数值门（position 4095 最低公开输出 cosine `0.9908`、板端尾部与
模拟器逐字节一致、48/48 贪心 token、边界 fail-closed），作为 **qualified**
profile 发布。它的 decode OM 是 Release 资产；`prefill.om` 与 `head_flat.om`
就是冻结的 `v0.1.0` 文件，在每个上下文之间逐字节共享——这就是混合 prefill 窗口
契约，现在已在一次会话中于三个宽度上完成板端验证。

`ctx8192` 保持 **pending**。它的公开输出过门、EOS 判定现已为 `PASS`，但标定是
donor 零扩展而非原生，且中文 oracle、内存包络、长 prompt 三项仍未闭合。

## 一条量错了东西的门

`ctx8192` 一直带着 `eos: FAIL_STRICT_SEQUENCE_MISMATCH`，而它对照的那个序列
从未被追溯到参考模型。用固定 checkpoint 以 float64 重新推导后，参考模型写下一个
句号然后停止——这恰恰是 `ctx8192` 的输出，而 `ctx1024` 与 `ctx4096` 并非如此。
那条"期望"是从第一个跑出来的工件记录下来的。详见
[`release/contexts/strict-eos-oracle.md`](../contexts/strict-eos-oracle.md)。

这并不说明另外两档有缺陷：它们的 48 token oracle 全过，而参考模型在那一步也只
以 `0.31` 个 logit 的优势选择句号。

## 本版还包含

- 项目首页与 app 首页上的板端 agent 真实会话录制，渲染为自包含动画 SVG。等待段
  按 `4.5x` 播放且压缩倍率写在画面上，每一帧右下角是板子自己的墙钟。
- [`docs/QUANTIZATION_CONTRACT.zh-CN.md`](../../docs/QUANTIZATION_CONTRACT.zh-CN.md)：
  `Clip` 对 ATC 的 IFMR 量程搜索做了什么、position 0 为什么需要自己的标定 family
  （是 layer-0 的 MLP 分支，不是 attention），以及一条被广泛复述、却被它自己的
  证据推翻因而撤回的规则。
- [`release/perf/`](../perf/README.md)：性能板现在载有 TTFT、逐上下文相位拆解、
  被取代的并轨前数字，以及每一项背后的证据哈希。

## 已知限制

长 prompt 的 TTFT 很差：prompt token 仍然逐个送入，512 token 的 prompt 在
ctx1024 上需要 `40.7 s`，在 ctx4096 上需要 `54.4 s`。能摊薄这笔
开销的宽块 prefill 路径不在本版中——目前没有任何宽块通过数值门。
