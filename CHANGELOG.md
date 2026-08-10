# Changelog

## 0.1.0 - 2026-08-09

- Initial open-source pipeline from pinned MiniCPM5-1B checkpoint to real-weight
  layer ONNX, packed 24-layer prefill/decode ONNX, external ATC compilation and
  reproducible three-handle release manifests.
- Captures the accepted SS928 ctx1024 qualification without redistributing
  weights, proprietary SDK components or board binaries.
- Documents the `runtime-capture.v1` lineage step required between each
  libinstsim/SS928 execution and strict transformer/head scoring, including
  same-position hidden and canonical zero-residual binding for the head.
