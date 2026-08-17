# 在 Hi3403 板端直接运行预编译 Demo

[English](README.md)

<img src="../docs/media/board-chat.svg" alt="MiniCPM5-1B 在 Hi3403 板上回答两个问题，9.9 token/s" width="100%">

录自本目录的 `chat.sh`，真实板端会话。等待段按 `2.4x` 播放，右下角是板子
自己的墙钟。

本目录是板端用户入口。以下步骤假设 `/opt/pico-minicpm5` 下已经装配好一套
部署——运行时归档取自 `v0.2.0` Release，模型文件取自 `v0.1.0`，装配方式见项目
README。直接运行不需要重新导出 ONNX、不需要
调用 ATC，也不需要在板端安装本项目的 host 构建包。

## 板端目录结构

```text
/opt/pico-minicpm5/
├── app/
│   ├── chat.sh
│   ├── agent.sh
│   ├── bin/pico_persistent_acl_executor.aarch64
│   ├── native/{Makefile,pico_persistent_acl_executor.c}
│   ├── profiles/{ctx128,ctx1024,ctx4096,ctx8192}.json
│   └── src/{merged_board_server.py,minicpm_agent.py,
│            minicpm_profile.py,
│            pico_minicpm5_split_board_runner.py,probe_om_execute_latency.py,
│            qualify_minicpm_greedy_chain.py}
├── models/{prefill.om,decode.om,head_flat.om}          # 已验收 ctx1024
├── models/ctx128/{prefill.om,decode.om}                # 验收后放入
├── models/ctx4096/decode.om                            # 已验收；prefill 共享 models/prefill.om
├── models/ctx8192/decode.om                            # pending（标定非原生）；prefill 同上共享
└── assets/{token_embedding.f16.bin,tokenizer.json}
```

有授权的 Hi3403 运行库默认位于 `/root/pico_default_smoke/lib`。它们由板端
SDK 环境提供，开源仓库和 Release 不会重新分发这些动态库。

## 直接运行

```bash
cd /opt/pico-minicpm5
chmod +x app/chat.sh app/agent.sh app/bin/pico_persistent_acl_executor.aarch64

# MiniCPM5 官方无工具 chat template
./app/chat.sh

# 原生工具调用 Agent，三个模型句柄只加载一次
./app/agent.sh

# 显式选择 profile；ctx128 仅支持 Chat
./app/chat.sh --profile ctx128
./app/agent.sh --profile ctx4096
```

```text
        /\_/\
       ( o.o )    HiAgent
        > ^ <     Hi3403 端侧 AI
     本地运行 · 隐私安全 · 实时响应

⠹ Loading three resident model handles  6.4s
✓ ready · loaded 3 handles · ctx1024 · 7.4s
Agent ready · /help · /tools · /think on|off · /context · /clear · /quit
You ❯ 读取 README.md 的前 20 行并概括项目用途。
⠴ Planning  0.8s
⚙ read_file(path='README.md', start_line='1', end_line='20')
✓ read_file: 1: # pico-minicpm5
MiniCPM ✦ ...
You ❯ /quit
```

两个板端应用默认使用已验收的 `ctx1024` profile。`chat.sh` 使用 MiniCPM5 官方无工具 chat
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

`/help` 会按 Linux 命令行风格列出命令的语法、参数范围与作用域；使用
`/help COMMAND`（例如 `/help max`）可查看单个命令的详细说明。

| 命令 | 用途与使用范围 |
|---|---|
| `/help [COMMAND]` | 列出全部本地命令，或查看一个命令的详细帮助。 |
| `/profile` | 显示当前 runtime profile、context 和能力；切换需要重启。 |
| `/tools` | 只显示已注册的原生工具，不执行工具。 |
| `/permissions` | 显示哪些工具自动执行、哪些工具需要逐次授权。 |
| `/think [on\|off]` | 查看或切换后续 Agent 生成的 thinking。 |
| `/context` | 显示当前 profile 的 prompt 用量，包含工具定义和会话历史。 |
| `/clear` | 清空对话和工具历史但不重载模型句柄；`/reset` 是别名。 |
| `/max [N]` | 查看或设置 profile 配置的回答上限，实际仍受剩余上下文限制。 |
| `/quit` | 关闭常驻会话；`/exit` 和 Ctrl-D 等价。 |

Agent 已知配置的 workspace 根目录，并以 `path='.'` 表示该目录，因此应主动
调用工具检查，而不是向用户追问当前路径。明确的当前目录、目录列举、文件行窗口、
字面搜索和 Git 状态请求会直接路由到只读工具并显示 `model skipped`；总结、
转换与一般工具选择仍由模型完成。当前已验收
ctx1024 profile 的单次工具结果
限制为 800 字符，目录、文件与搜索默认窗口也相应缩小；需要更多信息时应缩小范围
或继续分页调用。

模型原生请求使用渐进式工具披露。明确只读任务只注入只读 schema；显式写入或
命令意图才增加对应的权限工具，模糊开发任务保留全量工具作为 fail-safe。成功
结果携带类型、本地引用、截断状态和下一 offset；`read_result_page` 可读取后续有界
分页而不重新执行原工具，会话保留最近 16 个结果。

Chat 和 Agent 默认复用 token-exact 的会话 resident K/V，后续轮次只执行新增
prompt token；`/clear` 会同时清空对话和 resident-prefix 元数据。可用
`REUSE_SESSION_KV=0 ./app/chat.sh` 或 `REUSE_SESSION_KV=0 ./app/agent.sh`
回到全量重放，便于对照诊断。板端两轮 Agent A/B 中，复用组与重放组的输出 token
完全一致，第二轮命中 134-token 前缀，耗时由 `94.94 s` 降至 `80.78 s`。类似
“第二行有什么作用”且已有上文证据的追问不会再注入无关工具 schema；修改和 shell
意图仍保持 fail-closed 的权限工具披露。

当前源码还提供按 schema 懒创建的固定 system/tool 前缀快照。新版 executor 通过
通用 resident-input snapshot/restore opcode 只保存实际使用的 K/V 行；切换工具
schema 或执行 `/clear` 后可在板内恢复，不把 cache 传回 Python。该路径已通过
板端 token-exact A/B 并在 Agent 中默认开启：恢复 137-token 前缀仅耗时
`1.76 ms`，同一条 32-token 请求由 `26.97 s` 降至 `12.56 s`（降低 `53.4%`），
生成 token ID 与文本完全一致。仅在全量重放诊断时使用
`FIXED_PREFIX_SNAPSHOTS=0 ./app/agent.sh` 关闭。

Agent 在达到 profile 的 `compact_at_tokens` 时执行确定性 context rebase：原始
工具输出被替换为带类型的本地引用，保留当前交换和最近对话，并按
`reserve_tokens` 留出回答空间。报告会记录压缩前后 token 数及被压缩轮数；若当前
交换本身仍无法放入 context，则 fail closed 并要求缩短输入或 `/clear`。Hi3403
长会话板端 A/B 的两轮都将 12 个旧工具轮次由 `2808` token 确定性压到 `810`，
并生成完全相同的 `[18655, 4569, EOS]`。重复运行命中 643-token resident 前缀，
总耗时由 `69.45 s` 降至 `14.61 s`（`4.75x`）。

已知 prompt token 不再执行词表 head：最后一个 prompt position 之前只运行
transformer 和 K/V 更新，因为下一个输入 token 已经确定；最后一个 prompt position
及所有生成 token 仍执行 head 和 argmax。板端输出 token 完全不变，同一首次长
prompt 由 `86.70 s` 降至 `69.45 s`（降低 `19.89%`），resident-prefix 重复请求由
`18.17 s` 降至 `14.61 s`（降低 `19.59%`）。逐 position 报告以
`head_skipped` 标记该路径。

请求报告还会包含 fail-closed 的 `prefill_schedule`。固定策略是
`S128 -> S32 -> S16 -> strict S1 tail`，但当前已验收 bundle 只启用 S1。不能
因为目录里出现某个 OM 就选择宽块；每个 context 的产物必须先通过 descriptor、
公开输出 cosine `>0.98`、K/V 发布、prefill-to-decode handoff、token-exact 和
Hi3403 板端门禁。

运维人员可在应用启动时传入 `--prefill-activation-manifest`，并同时提供 live
`--available-bytes`、`--base-resident-bytes`、`--reserve-bytes`，复验可选的
release-v4 activation；四项必须成组出现。`/profile` 和 JSON 报告会同时显示资格
状态与实际可执行宽度。typed dispatcher 已通过 fake transport 契约测试，但当前
Release 没有注册任何 production wide handler：尚无完整宽块 OM 通过发布门禁，
也没有 CLI 注入入口。因此即便 S16/S32/S128 已通过资格，也不会进入调度器，执行
仍保持 strict S1，绝不会伪造宽块标签。
Release-v4 token-exact 证据会绑定实际 head OM 与 embedding；启动时还会在 spawn
紧前重新哈希它们、两只 S1 route OM、imported protocol runner、executor、两套
descriptor 和已注册 wide OM。部署树必须可信，并在整个进程生命周期保持只读/不可变；
该基于路径的 preflight 不宣称能以 inherited-fd handoff 抵抗主动写入者。详见
[Native prefill 发布资格](../docs/NATIVE_PREFILL_RELEASE_QUALIFICATION.zh-CN.md)。

Agent 的 thinking 默认关闭。可用 `./app/agent.sh --thinking` 或
`THINKING=1 ./app/agent.sh` 在启动时开启；常驻会话中，`/think` 查看状态，
`/think on` 与 `/think off` 可控制下一次生成而不重新加载三只模型。Thinking
token 与工具定义、历史和最终回答共同占用 ctx1024 预算。

模型加载和首 token 等待时会显示带耗时的动态状态，最终回答逐 token 流式
显示。ctx1024 的默认回答上限是 128 token，`/max N` 可在不重启模型的情况下
查看或调整当前 profile 的限制。ctx1024 下 `N` 可为
1–1023，实际可生成长度还会扣除输入 prompt 占用的
token、工具定义、会话历史和工具结果。上下文不足时 Agent 会先清理较早轮次，
仍不足则明确要求 `/clear`。每个用户请求默认最多 4 轮工具交互。

颜色和动画默认只在交互式终端开启。首次加载模型时，HiAgent 会左右巡视并
眨眼，MiniCPM 使用更柔和的眨眼动画；动画与原有加载等待并行，不增加启动时间。
后续规划和生成继续使用单行状态指示，避免覆盖已有对话。重定向、管道和日志输出自动保持为
稳定的纯文本。可使用 `NO_COLOR=1 ./app/agent.sh` 关闭颜色，或添加
`--no-spinner` 关闭动画。`--color always|never|auto` 和环境变量
`PICO_MINICPM5_COLOR` 可显式控制颜色策略。REPL 默认隐藏 executor 的底层加载
日志，调试时可添加 `--verbose-executor` 恢复显示。

executor 先启动三只 OM 加载，同时解析 tokenizer，将两项独立的冷启动
开销重叠。已验收 runtime profile 还固定 transformer 输出槽为 K=0、V=1、
hidden=2，因而移除原先四次 execute、约 0.8 秒的 KV 启动探测。ctx1024 在
Hi3403 上完整生成冒烟实测加载为 `7.4 s`，优于只做重叠后的 `8.2–8.6 s`
和原始的 `10.8–11.5 s`。未声明可信槽位契约的 legacy 调用仍保留动态探测回退。

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
显示，不会因为临时的 `�` 字符而停止刷新或在结束时整段重放。输入行由 GNU
readline 处理 UTF-8 编辑，彩色 prompt 的转义序列标记为零宽，退格可完整删除到
行首。

两个脚本都支持下列环境变量，脚本名之后的额外参数会继续传给板端 server：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `PICO_MINICPM5_ROOT` | `app/` 的上级目录 | 部署根目录 |
| `PICO_RUNTIME_LIB` | 自动探测 | 板端运行库目录 |
| `PYTHON` | 自动探测 | Python 可执行文件 |
| `TOKENIZERS` | 空 | 可选的额外 `site-packages` 路径 |
| `PROMPT` | 未设置 | 可选单次 prompt；未设置时进入 REPL |
| `PICO_PROFILE` | `ctx1024` | 加载模型前选择的 runtime profile |
| `MAX_NEW` | profile 默认值 | 可选的初始最大生成 token 数 |
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

完整 runtime profile 与混合路由契约见源码仓库中的
[Agent 路由与运行时 Context Profile 设计](https://github.com/GitBubble/pico-minicpm5/blob/main/docs/AGENT_ROUTING_AND_CONTEXT_PROFILES.zh-CN.md)。
ctx128 明确只支持 Chat；ctx4096/ctx8192 在对应 OM 完成 descriptor、数值
（`>0.98`）和板端门禁前保持 pending，受控开发测试必须显式添加
`--allow-unqualified-profile`。

在 `v0.2.0` 的执行器上，ctx1024 profile 实测 `100.40 ms/token`，即
`9.96 token/s`，48/48 greedy token 保持一致；ctx4096 为 `7.81 token/s`。
逐相位数字见[性能板](../release/perf/README.zh-CN.md)。
