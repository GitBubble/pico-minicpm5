# 变更日志

[English](CHANGELOG.md)

## 0.1.0 - 2026-08-09

### Runtime 增量刷新 - 2026-08-10

- 在 `app/` 中补齐中英文板端应用、runtime 源码与 executor C/Makefile。
- 启用 resident packed K/V scatter 和 byte-exact RoPE/embedding 快速准备，
  不改变三只已验收 OM 哈希。
- ctx1024 板端性能提升到 `9.42–9.48 token/s`，保持 48/48 token
  一致、EOS 和中文路径通过。
- 新增常驻 stdin REPL（`/help`、`/reset`、`/quit`），多次输入复用
  三个已加载句柄；无参数执行 `app/chat.sh` 直接进入 REPL。
- executor 源码、Makefile 与 demo 不再作为独立 Asset 重复发布。

- 首次开源从固定 MiniCPM5-1B checkpoint 到真实权重单层 ONNX、24 层打包
  prefill/decode ONNX、外部 ATC 编译及可复现三句柄 Release manifest 的流水线。
- 记录 SS928 ctx1024 验收，不在源码中分发权重、私有 SDK 或板端二进制。
- 引入 `runtime-capture.v1` 血缘证据，把每次 libinstsim/SS928 执行与严格的
  transformer/head 评分绑定，并约束 head 使用同 position hidden 和全零 residual。
