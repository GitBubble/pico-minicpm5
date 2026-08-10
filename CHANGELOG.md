# Changelog

[中文](CHANGELOG.zh-CN.md)

## 0.1.0 - 2026-08-09

### Runtime refresh - 2026-08-10

- Added the complete bilingual SS928 board application under `app/`, including
  `chat.sh`, runtime sources and the executor C/Makefile build path.
- Enabled resident packed K/V scatter and byte-exact fast RoPE/embedding
  preparation without changing the three accepted OM hashes.
- Improved measured ctx1024 throughput to `9.42–9.48 token/s` with 48/48
  greedy tokens exact, EOS and Chinese prompts passing.
- Consolidated the Release layout: executor source, Makefile and demo are no
  longer duplicated as standalone assets.
- Added a resident stdin REPL (`/help`, `/reset`, `/quit`) so repeated prompts
  reuse the three loaded handles; no-argument `app/chat.sh` enters it directly.

- Initial open-source pipeline from pinned MiniCPM5-1B checkpoint to real-weight
  layer ONNX, packed 24-layer prefill/decode ONNX, external ATC compilation and
  reproducible three-handle release manifests.
- Captures the accepted SS928 ctx1024 qualification without redistributing
  weights, proprietary SDK components or board binaries.
- Documents the `runtime-capture.v1` lineage step required between each
  libinstsim/SS928 execution and strict transformer/head scoring, including
  same-position hidden and canonical zero-residual binding for the head.
