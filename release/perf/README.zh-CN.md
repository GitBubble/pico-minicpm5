# 性能板

[English](README.md)

[`perf-board.json`](perf-board.json) 的人工摘要；两者不一致时以 JSON 为准。
表头数据来自 2026-08-17 的一次板端会话，三个 profile 在 retain-input 执行器
（`cef4edb2…`）上背靠背跑完；被它取代的旧数字一并保留，以便随时看清差值。
不在本仓库内的数字，都按其证据文件的 sha256 绑定，与数值门的做法一致。

目标平台：Hi3403 / V101，SS928 级开发板。

## Decode 的五个相位

一个 decode 步由五个宿主可观测的相位组成：**prepare**（embedding 行、attention
mask、RoPE 矩阵）、**transformer**（常驻的 24 层句柄）、**kv**（把 packed K/V
发布进 canonical 常驻缓存）、**head**（词表 head 句柄）与 **argmax**。稳态取
position `>= 1` 的中位数；position 0 跑在 prefill 句柄上，单独列出。吞吐由
`1000 / total p50` 推导，证据文件里并不存这个值。

## 稳态 decode，p50 毫秒

| Profile | prepare | transformer | kv | head | argmax | **合计** | token/s | 相对旧值 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctx1024 | 1.32 | 77.12 | 0.95 | 19.90 | 1.01 | **100.40** | 9.96 | +5.2% | 已验收 |
| ctx4096 | 1.38 | 102.13 | 2.66 | 20.69 | 1.00 | **127.96** | 7.81 | +19.7% | 已验收 |
| ctx8192 | 1.36 | 139.88 | 2.67 | 20.70 | 0.99 | **165.71** | 6.03 | +31.6% | pending |

head 与 argmax 两个相位在三档之间是平的——它们看不到 KV 窗口。上下文的代价
几乎全部落在 transformer 相位上。

### 收益从哪里来

统一执行器在多次 execute 之间保留 workspace 输入，而不是反复重写，因此一个
decode 步省下一次完整的 workspace 写入。这预测出的节省应与上下文成正比：

| Profile | 保留的 workspace | transformer 节省 | 折合避免的带宽 |
|---|---:|---:|---:|
| ctx1024 | 24.6 MiB | 82.66 → 77.12 = 5.54 ms | 4.66 GB/s |
| ctx4096 | 98.3 MiB | 129.70 → 102.13 = 27.57 ms | 3.74 GB/s |
| ctx8192 | 196.6 MiB | 194.75 → 139.88 = 54.87 ms | 3.76 GB/s |

过这三点的比例拟合落在 `0.28 ms/MiB`。ctx4096 与 ctx8192 都在拟合的 `0.6%`
以内；ctx1024 比拟合少省 `24%`，这正是固定的单次 execute 开销在最小上下文上
最难摊薄的样子。趋势解释了为什么上下文越长收益越大——但这不是三点共线。

### position 0 在三档上是同一个句柄

| Profile | pos-0 transformer | pos-0 合计 | 跳过 prompt head 后 |
|---|---:|---:|---:|
| ctx1024 | 80.59 | 103.67 | 82.77 |
| ctx4096 | 86.40 | 109.52 | 88.11 |
| ctx8192 | 86.01 | 109.15 | 87.74 |

ctx4096 与 ctx8192 相差 `0.39 ms`，因为它们的 position 0 引导在**同一个**冻结的
ctx1024 `prefill.om` 上，逐字节相同。这是混合 prefill 窗口契约的直接测量。

## 尾部位置

| Profile | 尾部 position | 合计 |
|---|---:|---:|
| ctx4096 | 4095 | 327.91 ms |
| ctx8192 | 8191 | 479.95 ms（旧路径 574.97） |

尾部步不计入稳态。每次都是模型加载后立即执行的单步，因此无法把首步预热与
"在整窗上做 attention"的代价分开；这两步里 head 与 argmax 没有变化，所以膨胀
来自 attention 与预热。

## 执行器模式 A/B（ctx8192，16 个生成 token）

五种模式产生逐字节相同的 position-1 原始输出和同一组 16 个 token id，因此这张
表只比较成本。

| 模式 | transformer | argmax | **合计 p50** | token/s | p50 变化 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| cached | 194.54 | 1.00 | **219.30** | 4.56 | — | 基线 |
| nocache | 171.25 | 25.21 | **222.34** | 4.50 | +1.4% 更慢 | 否决 |
| per-model-mixed | 171.30 | 1.00 | **198.57** | 5.04 | −9.5% 更快 | 被取代 |
| zero-once | 139.85 | 0.99 | **167.11** | 5.98 | −23.8% 更快 | **采用** |
| promptskip + zero-once | 139.87 | 0.99 | **167.28** | 5.98 | −23.7% 更快 | **采用** |

nocache 那一臂最有启发：取消缓存让 transformer 快 `23 ms`、却让 argmax 慢
`24 ms`，总账是亏的。per-model-mixed 把两者拆开——transformer 不缓存、argmax
缓存——把收益落袋。zero-once 执行器再把 transformer 压下 `31 ms`。promptskip
在稳态上刻意是平的，它的收益只出现在 prompt 位置。

`zero-once` 就是后来成为发行版 retain-input 执行器（`cef4edb2…`）的那个模式。
这里的 `167.11 ms` 与表头的 `165.71 ms` 是同一套配置在两次会话中的测量，差值
是运行间波动，不是变化。

## 首 token 时间

TTFT 是从请求开始到第一个生成 token 之间的墙钟时间，也就是全部 prompt 送入
步骤之和。position 0 跑在 prefill 句柄上，position `1..N-1` 在 decode 句柄上
**逐个 token** 执行，head 与 argmax 只在最后一个 prompt 位置各跑一次。模型加载
不计入。

因此三个 OM 都参与，但次数不同：prefill 与 head 各执行一次，decode 句柄执行
`N-1` 次。prompt 稍长之后，TTFT 几乎全是 decode 句柄的时间——这也是宽块规划器
（S16/S32/S128）瞄准 decode 侧、而不是为每个上下文单造一个 prefill 二进制的
原因。

```
ttft ≈ position_zero_total + (N − 1) × decode_step_total
```

这个模型在每一个实测点上的误差都在 **0.12%** 以内：

| 案例 | 模型 | 实测 | 误差 |
|---|---:|---:|---:|
| ctx128 bucket，121 token | 11509.6 ms | 11495.3 ms | +0.12% |
| ctx4096，12 token | 1795.8 ms | 1796.2 ms | −0.02% |
| ctx8192 旧路径，12 token | 2511.7 ms | 2511.3 ms | +0.02% |
| ctx8192 zero-once，12 token | 1931.3 ms | 1931.9 ms | −0.03% |

### prompt 送入成本，在发行执行器上实测

每 token 的送入成本就是 TTFT 的构成单元。它等于稳态步减去 head 与 argmax，
因为运行时在非末位的 prompt 位置会跳过这两者：

| Profile | 每 prompt token | 47 token | 128 token | 512 token |
|---|---:|---:|---:|---:|
| ctx1024 | **79.49 ms** | 3.77 s | 10.20 s | 40.73 s |
| ctx4096 | **106.28 ms** | 5.00 s | 13.61 s | 54.42 s |
| ctx8192 | **144.02 ms** | 6.73 s | 18.40 s | 73.70 s |

每 token 的数字是实测，总计是上面那个模型算出来的。ctx1024 的 `79.49 ms` 与一次
独立冷 prefill 实测的 `79.570 ms` 相差 `0.10%`——正是这条交叉验证让模型可用。

## 门禁绑定

一个性能数字，只有和同一件工件通过的数值门并列时才可以发布。吞吐永远不能替代
cosine 或 token 门。

| Profile | 门禁记录 | 判定 | 最低公开输出 cosine |
|---|---|---|---:|
| ctx1024 | `release/v0.1.0/qualification.json` | PASS | 0.996646 |
| ctx4096 | `release/contexts/ctx4096.qualification.json` | PASS | 0.990820 |
| ctx8192 | `release/contexts/ctx8192.qualification.json` | CANDIDATE_CALIBRATION_NOT_NATIVE | 0.986076 |

ctx8192 的吞吐真实且可复现，而该 profile 仍是 `pending`——因为它的标定是 donor
零扩展而非原生，且中文 oracle、内存包络与长 prompt 三项仍未闭合。它的 EOS 门
是通过的：以重新推导的 FP64 参考衡量，三档里只有它与参考逐 token 一致
（[原因](../contexts/strict-eos-oracle.zh-CN.md)）。使用时请加
`--allow-unqualified-profile`，仅限开发。

## 哪些没有测

**TTFT。** 没有任何已发布 profile 做过端到端的长 prompt 实测：三档的每 token
送入成本都有实测，但在 ctx1024/ctx4096/ctx8192 上真正跑到首 token 的最长 prompt
是 **47 个 token**。命中常驻前缀、固定前缀快照、宽块 prefill 这三种情况，磁盘上
都没有记录。

**三个只有散文的数字。** 810-token 冷请求由 `86.70 s` 降至 `69.45 s`、常驻重复
请求由 `18.17 s` 降至 `14.61 s`、以及 643-token 前缀命中达到 `14.61 s`——这三个
**没有可检索的逐步记录**。它们在板上跑出来后只被总结成了文字。现在放在
`perf-board.json` 的 `ttft.unbound_prose_claims` 下；没有任何门依赖它们，补救办法
是把三者重跑一遍并把报告落盘。

**其余：**

- 逐层或逐 kernel 的 NPU 拆解——只存在那五个宿主可观测相位。
- 尾部位置的 KV 快照水化字节数与耗时——没有记录这个字段。
- 首步预热与尾部位置代价的分离——尾部跑只执行一步。
- ctx128——没有板端性能实测。
- ctx1024 或 ctx4096 上的执行器模式 A/B——五模式对比只在 ctx8192 上做过。
- 混合契约下超过 prefill 窗口的 prompt——所有已记录的跑分都用 2–13 token 的
  prompt。
