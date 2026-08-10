# Model provenance

[中文](MODEL_PROVENANCE.zh-CN.md)

- Repository: `openbmb/MiniCPM5-1B`
- Revision: `4e9de7a0778dc1c362e983e6858f0e77542cbdca`
- Architecture: standard `LlamaForCausalLM`
- Checkpoint shard SHA256:
  `7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d`
- Checkpoint shard size: `2,161,290,912` bytes
- Weight payload: BF16, `2,161,265,664` bytes
- Tokenizer SHA256:
  `3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81`
- Derived FP16 embedding SHA256:
  `5a93b589f0c5920021c95e04327c0771da2721d8eec2dd7ac1b283aa0d3b7df5`
- Model-card license declaration observed for the pinned revision: Apache-2.0

The project downloads the checkpoint directly through `hf download`; it does
not mirror or vendor model files. `model verify` rejects a different revision,
geometry, symbol table, shard size, config/index hash or safetensors header.

The ONNX graphs, OM files and exported token embedding are derived from the
checkpoint. Release manifests retain that relationship instead of presenting
them as ordinary compiler-only binaries.
