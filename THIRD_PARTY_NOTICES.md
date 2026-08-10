# Third-party notices

| Component | Use | Distribution policy |
|---|---|---|
| OpenBMB MiniCPM5-1B | Checkpoint and tokenizer | Downloaded directly by the user at a pinned revision; absent from source archives |
| Hugging Face `hf` CLI | Reproducible checkpoint download | Executed as an external tool |
| ONNX / NumPy / safetensors | Graph construction and checkpoint access | Python dependencies under their respective licenses |
| Transformers / PyTorch | Float reference capture | Optional Python dependencies; no package bytes vendored |
| Vendor ATC/DDK/libinstsim | PICO compilation and local simulation | User-supplied; never included in public source or CI artifacts |
| `libsvp_custom.so` | `ExtendRMSNorm` compile/runtime registration | User-built or user-supplied; never included in this repository |

Generated OM and embedding artifacts contain or derive from model parameters.
Release manifests therefore record them as `derived-model` artifacts even
when the upstream model card declares Apache-2.0.
