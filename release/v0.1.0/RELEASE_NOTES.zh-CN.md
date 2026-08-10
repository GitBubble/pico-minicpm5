# pico-minicpm5 v0.1.0

[English](RELEASE_NOTES.md)

本版本开源 MiniCPM5-1B → ONNX → 24 层打包 PICO OM 的可复现流程，并记录
SS928 ctx1024 三句柄产物合同。合格产物的大小和 SHA256 位于
`release-manifest.json`；模型产物通过 GitHub Release 分发，不嵌入源码归档。

默认编译路线是图级组合后分别调用 ATC 编译 prefill/decode；二进制 OM post-link
不是生产路线。portable qualification 保存 raw hash、公开 tensor cosine、greedy
token 和性能证据，同时移除板端地址。

本次 v0.1.0 增量刷新不改变三只 OM：新的 resident-K/V runtime 将
packed K/V 保留在板端，直接完成 FP32 到 FP16 cache scatter，并优化
C4 embedding 与稀疏 RoPE 准备。板端性能由 `8.20–8.60 token/s` 提升到
`9.42–9.48 token/s`（`105.5–106.1 ms/token`），同时保持 48/48 token 一致、
EOS 和中文路径通过。

板端应用源码、executor C 和 Makefile 统一归档到 `app/`。预编译
executor 只放入 runtime 包的 `app/bin/`；不再发布 executor 源码、
Makefile、二进制和 `chat.sh` 的独立重复 Asset。
