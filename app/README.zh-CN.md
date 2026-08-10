# 在 SS928 板端直接运行预编译 Demo

[English](README.md)

本目录是板端用户入口。以下步骤假设 GitHub `v0.1.0` Release 中的文件已经
复制到 `/root/minicpm5_gate_3handle`。直接运行不需要重新导出 ONNX、不需要
调用 ATC，也不需要在板端安装本项目的 host 构建包。

## 板端目录结构

```text
/root/minicpm5_gate_3handle/
├── app/chat.sh
├── models/{prefill.om,decode.om,head_flat.om}
├── assets/{token_embedding.f16.bin,tokenizer.json}
├── bin/pico_persistent_acl_executor.resident.aarch64
└── src/{merged_board_server.py,pico_minicpm5_split_board_runner.py,
         probe_om_execute_latency.py,qualify_minicpm_greedy_chain.py}
```

有授权的 SS928 运行库默认位于 `/root/pico_default_smoke/lib`。它们由板端
SDK 环境提供，开源仓库和 Release 不会重新分发这些动态库。

## 直接运行

```bash
cd /root/minicpm5_gate_3handle
chmod +x app/chat.sh bin/pico_persistent_acl_executor.resident.aarch64

# 英文续写
PROMPT='The capital of France is' MAX_NEW=16 sh app/chat.sh

# 中文生成
PROMPT='请用一句话解释什么是神经网络。' MAX_NEW=32 sh app/chat.sh

# 算术与 EOS 路径
PROMPT='1+1 equals' MAX_NEW=16 sh app/chat.sh
```

`chat.sh` 支持下列环境变量，脚本名之后的额外参数会继续传给板端 server：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `GATE` | `/root/minicpm5_gate_3handle` | 部署根目录 |
| `LIB` | `/root/pico_default_smoke/lib` | 板端运行库目录 |
| `TOKENIZERS` | `/opt/pico-minicpm5/venv/lib/python3.10/site-packages` | Python 包目录 |
| `PROMPT` | `The capital of France is` | 输入提示词 |
| `MAX_NEW` | `24` | 最大生成 token 数 |

## 快速排障

```bash
cd /root/minicpm5_gate_3handle
sha256sum -c SHA256SUMS
test -r "${LIB:-/root/pico_default_smoke/lib}/libsvp_acl.so" || \
  ls "${LIB:-/root/pico_default_smoke/lib}"
python3 -c 'import tokenizers; print(tokenizers.__version__)'
```

如果 `tokenizers` 安装在其他位置，通过 `TOKENIZERS` 指定对应的
`site-packages`。若动态库加载失败，通过 `LIB` 指向匹配当前板端 SDK 的运行库
目录。Release 自带的 executor 是 AArch64 二进制；如需按特定工具链重编译，
runtime 包的 `native/` 中同时提供 C++ 源码和 Makefile。
