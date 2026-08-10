# Contributing

[中文](CONTRIBUTING.zh-CN.md)

Keep changes reproducible and separable from private toolchains.

1. Add a tiny-fixture unit test for graph or manifest changes.
2. Run `pytest` and `pico-minicpm5 release source --check-only`.
3. Never commit weights, ONNX external data, OM files, SDKs, shared objects,
   board binaries, image lists, raw board tensors or credentials.
4. Describe compiler changes by graph/ABI/numeric contracts. The default OM
   merge path is graph-level composition; experimental binary linking must
   remain explicitly opt-in and fail closed.
