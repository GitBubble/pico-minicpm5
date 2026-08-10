# 变更日志

[English](CHANGELOG.md)

## 0.1.0 - 2026-08-09

- 首次开源从固定 MiniCPM5-1B checkpoint 到真实权重单层 ONNX、24 层打包
  prefill/decode ONNX、外部 ATC 编译及可复现三句柄 Release manifest 的流水线。
- 记录 SS928 ctx1024 验收，不在源码中分发权重、私有 SDK 或板端二进制。
- 引入 `runtime-capture.v1` 血缘证据，把每次 libinstsim/SS928 执行与严格的
  transformer/head 评分绑定，并约束 head 使用同 position hidden 和全零 residual。
