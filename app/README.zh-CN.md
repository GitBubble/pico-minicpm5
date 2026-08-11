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
│   ├── agent.sh
│   ├── bin/pico_persistent_acl_executor.aarch64
│   ├── native/{Makefile,pico_persistent_acl_executor.c}
│   └── src/{merged_board_server.py,minicpm_agent.py,
│            pico_minicpm5_split_board_runner.py,probe_om_execute_latency.py,
│            qualify_minicpm_greedy_chain.py}
├── models/{prefill.om,decode.om,head_flat.om}
└── assets/{token_embedding.f16.bin,tokenizer.json}
```

有授权的 SS928 运行库默认位于 `/root/pico_default_smoke/lib`。它们由板端
SDK 环境提供，开源仓库和 Release 不会重新分发这些动态库。

## 直接运行

```bash
cd /opt/pico-minicpm5
chmod +x app/chat.sh app/agent.sh app/bin/pico_persistent_acl_executor.aarch64

# MiniCPM5 官方无工具 chat template
./app/chat.sh

# 原生工具调用 Agent，三个模型句柄只加载一次
./app/agent.sh
```

```text
        /\_/\
       ( o.o )    MiniCPM 5
        > ^ <     SS928 local AI
     ctx1024 · resident KV · streaming

⠹ Loading three resident model handles  6.4s
✓ ready · loaded 3 handles · ctx1024 · 10.2s
Agent ready · /help · /tools · /think on|off · /context · /clear · /quit
You ❯ 读取 README.md 的前 20 行并概括项目用途。
⠴ Planning  0.8s
⚙ read_file(path='README.md', start_line='1', end_line='20')
✓ read_file: 1: # pico-minicpm5
MiniCPM ✦ ...
You ❯ /quit
```

两个板端应用都默认使用 `ctx1024`。`chat.sh` 使用 MiniCPM5 官方无工具 chat
template，并保留多轮历史直至 `/clear`；`agent.sh` 再加入下面描述的原生工具
协议。工具定义放在
`<tools>` 中，模型原生生成 `<function>/<param>` XML，执行结果通过
`<tool_response>` 回填；没有自定义另一套工具协议。Agent 会保留当前会话和
工具历史，`/clear` 可清空。模型句柄、executor 和板端缓冲区持续常驻，避免
每个问题重新加载约 10 秒。

内置工具为 `list_directory`、`read_file`、`search_text`、`git_status`、
`write_file` 和 `run_shell`。前四项自动执行；文件写入和 shell 每次都弹出
`Allow once? [y/N]`，默认拒绝。所有文件工具被限制在启动时的工作目录内，可用
`--workspace PATH` 显式指定边界。`/tools`、`/permissions` 和 `/context` 分别
显示工具、权限和 token 预算。

Agent 的 thinking 默认关闭。可用 `./app/agent.sh --thinking` 或
`THINKING=1 ./app/agent.sh` 在启动时开启；常驻会话中，`/think` 查看状态，
`/think on` 与 `/think off` 可控制下一次生成而不重新加载三只模型。Thinking
token 与工具定义、历史和最终回答共同占用 ctx1024 预算。

模型加载和首 token 等待时会显示带耗时的动态状态，最终回答逐 token 流式
显示。默认回答上限是 128 token，
`/max N` 可在不重启模型的情况下查看或调整。ctx1024 下 `N` 可为
1–1023，实际可生成长度还会扣除输入 prompt 占用的
token、工具定义、会话历史和工具结果。上下文不足时 Agent 会先清理较早轮次，
仍不足则明确要求 `/clear`。每个用户请求默认最多 4 轮工具交互。

颜色和动画默认只在交互式终端开启；重定向、管道和日志输出自动保持为稳定的
纯文本。可使用 `NO_COLOR=1 ./app/agent.sh` 关闭颜色，或添加
`--no-spinner` 关闭动画。`--color always|never|auto` 和环境变量
`PICO_MINICPM5_COLOR` 可显式控制颜色策略。REPL 默认隐藏 executor 的底层加载
日志，调试时可添加 `--verbose-executor` 恢复显示。

单次非交互执行：

```bash
./app/chat.sh --prompt 'The capital of France is' --max-new 16

# 中文生成
./app/chat.sh --prompt '请用一句话解释什么是神经网络。' --max-new 32

# 算术与 EOS 路径
./app/chat.sh --prompt '1+1 equals' --max-new 16
```

这些 `--prompt` 命令保留裸文本续写兼容路径。无参数运行时，`chat.sh` 进入官方
无工具对话 REPL，`agent.sh` 进入原生工具调用 Agent，两个入口互不改变对方的
默认行为。显式执行 `chat.sh --interactive` 可进入旧的裸文本 REPL。

两个 REPL 均使用 UTF-8 安全的追加式增量解码。跨 token 的汉字会等待完整后再
显示，不会因为临时的 `�` 字符而停止刷新或在结束时整段重放。

两个脚本都支持下列环境变量，脚本名之后的额外参数会继续传给板端 server：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PICO_MINICPM5_ROOT` | `app/` 的上级目录 | 部署根目录 |
| `PICO_RUNTIME_LIB` | 自动探测 | 板端运行库目录 |
| `PYTHON` | 自动探测 | Python 可执行文件 |
| `TOKENIZERS` | 空 | 可选的额外 `site-packages` 路径 |
| `PROMPT` | 未设置 | 可选单次 prompt；未设置时进入 REPL |
| `MAX_NEW` | `128` | 初始最大生成 token 数 |
| `THINKING` | `0` | `agent.sh` 启动 thinking：`0/1`、`off/on`、`false/true` |
| `PICO_MINICPM5_COLOR` | `auto` | `auto`、`always` 或 `never` |
| `NO_COLOR` | 未设置 | 设置后在 auto 模式关闭 ANSI 颜色 |

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
