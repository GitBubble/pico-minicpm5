# 在 SS928 板端直接运行预编译 Demo

[English](README.md)

本目录是板端用户入口。以下步骤假设 GitHub `v0.1.0` Release 中的文件已经
复制到 `/opt/pico-minicpm5`。直接运行不需要重新导出 ONNX、不需要
调用 ATC，也不需要在板端安装本项目的 host 构建包。

## 板端目录结构

```text
/opt/pico-minicpm5/
├── app/
│   ├── chat.sh
│   ├── bin/pico_persistent_acl_executor.aarch64
│   ├── native/{Makefile,pico_persistent_acl_executor.c}
│   └── src/{merged_board_server.py,pico_minicpm5_split_board_runner.py,
│            probe_om_execute_latency.py,qualify_minicpm_greedy_chain.py}
├── models/{prefill.om,decode.om,head_flat.om}
└── assets/{token_embedding.f16.bin,tokenizer.json}
```

有授权的 SS928 运行库默认位于 `/root/pico_default_smoke/lib`。它们由板端
SDK 环境提供，开源仓库和 Release 不会重新分发这些动态库。

## 直接运行

```bash
cd /opt/pico-minicpm5
chmod +x app/chat.sh app/bin/pico_persistent_acl_executor.aarch64

# 直接进入常驻 REPL，三个模型句柄只加载一次
./app/chat.sh
```

```text
MiniCPM5 REPL ready. Commands: /help, /max N, /reset, /quit
You> 请用一句话解释什么是神经网络。
MiniCPM> ...
You> /quit
```

REPL 中每次输入都开始一个新的 ctx1024 逻辑序列，但模型句柄、
executor 进程和板端缓冲区保持常驻，避免每个问题重新加载模型的约
10 秒开销。文本会随 token 生成逐步显示。默认回答上限是 128 token，
`/max N` 可在不重启模型的情况下查看或调整。ctx1024 下 `N` 可为
1–1023，实际可生成长度还会扣除输入 prompt 占用的
token。`/reset` 会在可选 JSON 报告中标记新的 transcript。当前是
独立轮次的文本续写 REPL，不会自动拼接 chat-template 多轮历史。

单次非交互执行：

```bash
./app/chat.sh --prompt 'The capital of France is' --max-new 16

# 中文生成
./app/chat.sh --prompt '请用一句话解释什么是神经网络。' --max-new 32

# 算术与 EOS 路径
./app/chat.sh --prompt '1+1 equals' --max-new 16
```

`chat.sh` 支持下列环境变量，脚本名之后的额外参数会继续传给板端 server：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PICO_MINICPM5_ROOT` | `app/` 的上级目录 | 部署根目录 |
| `PICO_RUNTIME_LIB` | 自动探测 | 板端运行库目录 |
| `PYTHON` | 自动探测 | Python 可执行文件 |
| `TOKENIZERS` | 空 | 可选的额外 `site-packages` 路径 |
| `PROMPT` | 未设置 | 可选单次 prompt；未设置时进入 REPL |
| `MAX_NEW` | `128` | 初始最大生成 token 数 |

运行库依次探测 `/root/pico_default_smoke/lib` 和 `/opt/ss928-runtime/lib`；
Python 依次探测 `$PICO_MINICPM5_ROOT/venv/bin/python` 和 `python3`。

## 快速排障

```bash
cd /opt/pico-minicpm5
sha256sum -c SHA256SUMS
test -r "${PICO_RUNTIME_LIB:-/opt/ss928-runtime/lib}/libsvp_acl.so" || \
  ls "${PICO_RUNTIME_LIB:-/opt/ss928-runtime/lib}"
python3 -c 'import tokenizers; print(tokenizers.__version__)'
```

如果 `tokenizers` 安装在其他位置，通过 `TOKENIZERS` 指定对应的
`site-packages`。若动态库加载失败，通过 `PICO_RUNTIME_LIB` 指向匹配当前
板端 SDK 的运行库目录。Release 自带的 executor 是 AArch64 二进制；源码与
Makefile 统一归档在 `app/native/`：

```bash
cd /opt/pico-minicpm5/app/native
make SDK_ROOT=/path/to/sdk/smp/a55_linux/mpp/out CC=aarch64-mix210-linux-gcc
```

优化后 ctx1024 路径的板端性能为 `105.5–106.1 ms/token`，即
`9.42–9.48 token/s`，并保持 48/48 greedy token 一致。
