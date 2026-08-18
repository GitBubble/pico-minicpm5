# 第三方声明

[English](THIRD_PARTY_NOTICES.md)

| 组件 | 用途 | 分发策略 |
|---|---|---|
| OpenBMB MiniCPM5-1B | checkpoint 与 tokenizer | 用户按固定 revision 直接下载，源码归档不包含 |
| Hugging Face `hf` CLI | 可复现下载 | 作为外部工具执行 |
| ONNX / NumPy / safetensors | 构图与 checkpoint 访问 | 遵循各自许可证的 Python 依赖 |
| Transformers / PyTorch | 浮点参考 | 可选依赖，不 vendoring |
| 厂商 ATC/DDK/libinstsim | 编译与仿真 | 用户提供，不进入公开源码或 CI |
| `libsvp_custom.so` | 注册 `ExtendRMSNorm` | 用户构建或提供，本仓库不分发 |
| `app/lib/libsvp_acl.so` 及同目录库 | 板端执行器运行时 | 取自 `SS928V100_SDK_V2.0.2.2`，见 `app/lib/README.zh-CN.md` |

OM 与 embedding 包含或派生自模型参数，因此 Release manifest 将其标为
`derived-model`，不把它们当作普通的纯编译器二进制。
