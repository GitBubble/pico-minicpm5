# MiniCPM5 服务接入 OpenClaw：普通用户使用指南

本文面向已经安装 OpenClaw、希望使用本地 MiniCPM5 服务进行对话的用户。主流程默认是安全且稳定性更高的纯文字模式；工具调用单独列为实验功能。

## 先确认你使用的是哪一类发行包

当前公开的 `pico-minicpm5 v0.1.0` 是 `ctx1024` 三句柄版本。它低于 OpenClaw 对本地模型要求的 4096 token 上下文下限，而且发行包中尚未包含 OpenAI service、`chat_template.jinja` 和可发布的 JSONL runner。因此：

- `v0.1.0/ctx1024` 可以继续通过发行包里的 `chat.sh` 使用；
- `v0.1.0/ctx1024` 不能直接配置成 OpenClaw provider；
- 本文的 OpenClaw 操作适用于已经部署好的兼容服务，例如当前 C4096 split-runner 开发预览，或者后续明确标注为 OpenClaw-ready 的发行包；
- 如果你的包中没有本文“文件检查”列出的组件，请停止操作，不要用其他上下文或其他 runner 冒充。

> 截至本文编写时，公开 GitHub Release 中还没有可供普通用户下载的 OpenClaw-ready Asset。本文是“已部署服务的使用指南”和后续发行契约，不是当前 `v0.1.0` 的可执行安装承诺。没有运营方提供的 SHA 绑定预览包时，请继续使用 `chat.sh`，等待正式发行。

当前兼容状态如下。

| 路径 | 状态 | 说明 |
|---|---|---|
| `ctx1024` 三句柄 + `chat.sh` | 已发布 | 原生文字对话，不经过 OpenClaw。 |
| C4096 split-runner + OpenAI 兼容服务 + OpenClaw 文字模式 | 开发预览 | 存在明确 JSONL handoff，但 runner 仍标记 `production_ready: false`。 |
| C8192 native OM + OpenAI service + OpenClaw 文字/Agent | **已闭环（2026-08-27 板端）** | native OM → JSONL runner → `/v1/chat/completions`（非流式、SSE、HTTP 层工具往返）→ OpenClaw 2026.7.1 `infer`（逐字 `pong`）与 `agent --local`（真实中文回答）全链在板上通过；契约超时须为 1800 s。 |
| C16384 native OM + OpenAI service + OpenClaw 文字/Agent | **已闭环（2026-08-27 板端）** | 同一链路在 `RELEASED_BOARD_VERIFIED` 的 ctx16384 decode OM 上通过同组门禁；验收记录 `release/contexts/ctx16384.qualification.json`（候选：donor 零扩展标定）。 |
| 官方 Hugging Face MiniCPM5-1B + Host OpenClaw 工具调用 | 已验证 | 完成过真实工具执行、结果回灌和最终答案闭环，但不是 Hi3403 native OM 证据。 |
| Hi3403 上运行 OpenClaw + Host HF backend | 实验性 | 工具调用识别、一次内建工具执行和结果回灌已出现；`ENOMEM` 和重复工具循环阻断最终闭环。 |
| Hi3403 native OM + OpenClaw 工具调用 | 未执行 | 当前没有 native OM 工具调用资格证据。 |

## 使用拓扑

推荐把模型服务只绑定在本机回环地址：

```text
OpenClaw
   │ OpenAI-compatible HTTP，127.0.0.1:8000/v1
   ▼
pico_minicpm5_openai_service.py
   │ pico.minicpm5.runner.v1 JSONL
   ▼
Hi3403 native runner
   │ resident handles
   ▼
runner manifest 绑定的 OM 集合
```

当前 C4096 JSONL 开发路径使用多段 split OM。已经发布的 `ctx1024` 三句柄 `prefill.om + decode.om + head_flat.om` 不实现这条 JSONL 接口，不能直接套用本拓扑。

OpenClaw 和服务可以运行在同一台 Hi3403 设备上。也可以让 OpenClaw 运行在 PC 上，通过 SSH 隧道访问板端服务；后者见“跨机器使用”一节。

服务当前没有 HTTP 鉴权。不要把它直接监听到公网或局域网地址，也不要把 `--host` 改成 `0.0.0.0`。

## 前置条件

### OpenClaw

先确认 OpenClaw 已安装：

```bash
openclaw --version
```

当前验证过的版本身份是：

- Host 完整工具闭环：`OpenClaw 2026.6.1 (2e08f0f)`；
- Linux ARM64 配置生成器固定：OpenClaw `2026.7.1`、Node.js `24.15.0`。

不同 OpenClaw 版本对工具配置和 meta-tool 的处理可能不同。文字模式优先使用发行包指定的固定版本，不要直接升级到未知版本后继续沿用旧资格结论。

### 已部署服务需要的文件

以 `/opt/pico-minicpm5` 为例，OpenClaw-ready 包至少应提供：

```text
/opt/pico-minicpm5/
├── assets/
│   ├── tokenizer.json
│   └── chat_template.jinja
├── service/
│   ├── pico_minicpm5_openai_service.py
│   └── pico_openclaw_minicpm5_config.py
├── models/
│   └── native runner manifest 列出的完整 OM 集合
├── bin/
│   └── pico_persistent_acl_executor
└── runtime/
    ├── native JSONL runner
    └── 与该 OM 组合匹配的配置或 manifest
```

运行前检查：

```bash
PICO_HOME=/opt/pico-minicpm5

test -x "$PICO_HOME/venv/bin/python"
test -f "$PICO_HOME/assets/tokenizer.json"
test -f "$PICO_HOME/assets/chat_template.jinja"
test -f "$PICO_HOME/service/pico_minicpm5_openai_service.py"
test -f "$PICO_HOME/service/pico_openclaw_minicpm5_config.py"
test -d "$PICO_HOME/models"
test -d "$PICO_HOME/runtime"
```

OpenClaw-ready 发行包必须带有 `SHA256SUMS` 或等价的逐文件哈希 manifest。缺少两者时应停止部署。带有 `SHA256SUMS` 时，在发行包根目录执行：

```bash
sha256sum -c SHA256SUMS
```

服务使用的 Python 环境至少需要 `tokenizers` 和 `jinja2`：

```bash
/opt/pico-minicpm5/venv/bin/python -c \
  'import jinja2, tokenizers; print("Python dependencies: OK")'
```

## 逐步接入：服务已经启动

本节假设 MiniCPM5 服务已经运行在 `127.0.0.1:8000`。如果服务尚未启动，请先看“服务部署者附录”。

OpenClaw 会注入 system prompt 和 Agent 上下文，Hi3403 首轮预填充可能明显慢于普通短对话。预览契约下面统一使用 3600 秒超时；正式发行应以 clean-board 实测和 release manifest 为准，不应承诺“五分钟内完成”。

### 1. 检查服务健康状态

```bash
SERVICE_ROOT=http://127.0.0.1:8000

curl -fsS --connect-timeout 5 --max-time 10 \
  "$SERVICE_ROOT/healthz" | python3 -m json.tool
```

正常响应类似：

```json
{
  "status": "ok",
  "model": "minicpm5-1b",
  "busy": false,
  "supportsTools": false,
  "context_window": 8192
}
```

这里要检查四件事：

1. `status` 必须是 `ok`；
2. `model` 必须是 `minicpm5-1b`；
3. 初次使用时 `busy` 应为 `false`；
4. `context_window` 必须等于实际 OM/runner 的上下文契约，并且至少为 4096。

再检查模型列表：

```bash
curl -fsS --connect-timeout 5 --max-time 10 \
  "$SERVICE_ROOT/v1/models" | python3 -m json.tool
```

返回的 `data` 中应包含模型 ID `minicpm5-1b`。

注意：`/healthz` 只返回 service 的内存状态，不会向 runner 发起探测，也不能证明 runner 子进程仍然可用。下一步的直接 chat completion 成功，才能证明 service 与 runner 完成了一次协议往返。若要进一步证明使用的是目标 Hi3403 OM，还必须在启动前验证 SHA 绑定的 deployment manifest、runner config、全部 OM 和 executor；当前 service 的健康响应及普通日志本身不提供这项证明。

### 2. 直接调用一次模型

在接入 OpenClaw 前，先绕过 OpenClaw 做最小请求：

```bash
curl -fsS --connect-timeout 5 --max-time 3600 \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5-1b",
    "messages": [
      {"role": "user", "content": "Reply exactly: pong"}
    ],
    "temperature": 0,
    "stream": false,
    "max_tokens": 8
  }' \
  "$SERVICE_ROOT/v1/chat/completions" | python3 -m json.tool
```

请求中的模型 ID 是 `minicpm5-1b`。成功响应应满足：

- HTTP 状态为 200；
- `object` 为 `chat.completion`；
- `choices[0].message.role` 为 `assistant`；
- `choices[0].message.content` 中出现 `pong`；
- `choices[0].finish_reason` 为 `stop` 或 `length`。

服务只支持确定性生成，`temperature` 必须为 `0`。

### 3. 创建隔离的 OpenClaw profile

不要直接覆盖已有的 `~/.openclaw/openclaw.json`。本文使用独立 profile `pico-minicpm`，它的配置和会话会放在 `~/.openclaw-pico-minicpm`。

```bash
PROFILE=pico-minicpm
PROFILE_DIR="$HOME/.openclaw-$PROFILE"
PICO_HOME=/opt/pico-minicpm5

mkdir -p "$PROFILE_DIR"
chmod 700 "$PROFILE_DIR"
```

从服务读取真实上下文：

```bash
CONTEXT_WINDOW="$(curl -fsS http://127.0.0.1:8000/healthz | python3 -c 'import json,sys; print(json.load(sys.stdin)["context_window"])')"
```

生成默认的纯文字配置：

```bash
"$PICO_HOME/venv/bin/python" \
  "$PICO_HOME/service/pico_openclaw_minicpm5_config.py" \
  --context-window "$CONTEXT_WINDOW" \
  --max-tokens 128 \
  --port 8000 \
  --timeout-seconds 3600 \
  --config-out "$PROFILE_DIR/openclaw.json" \
  --manifest-out "$PROFILE_DIR/pico-minicpm5.manifest.json" \
  >/dev/null
```

关键规则：

- `--context-window` 必须来自当前服务，不能填写 Hugging Face 模型卡中的架构上限；
- `--max-tokens` 不能超过 native runner 的生成上限；当前预览契约通常使用 128；
- `--port` 必须与服务端口一致；
- 普通用户不要添加 `--supports-tools`。

确认 OpenClaw 看到的是隔离配置：

```bash
openclaw --profile "$PROFILE" config file
```

预期路径为：

```text
~/.openclaw-pico-minicpm/openclaw.json
```

校验配置：

```bash
openclaw --profile "$PROFILE" config validate --json
openclaw --profile "$PROFILE" models list \
  --provider pico-minicpm --json
```

`config validate` 应返回有效配置；`models list` 中应出现完整模型引用：

```text
pico-minicpm/minicpm5-1b
```

`models list` 只证明 OpenClaw 读取了配置，不能替代前面的 HTTP 健康检查。

### 4. 运行第一轮 OpenClaw Agent

首次使用不需要 Gateway，直接使用 embedded/local 模式：

```bash
openclaw --profile "$PROFILE" agent --local \
  --session-id pico-minicpm5-first-run \
  --model pico-minicpm/minicpm5-1b \
  --message '你好，请用一句话介绍你自己。' \
  --thinking off \
  --timeout 3600 \
  --json
```

成功判据：

- 命令退出码为 0；
- JSON 中没有 provider、timeout 或 context 错误；
- 最终回复来自 `pico-minicpm/minicpm5-1b`；
- 服务 `/healthz` 在请求结束后恢复为 `busy: false`。

需要继续同一段对话时，复用相同的 `--session-id`：

```bash
openclaw --profile "$PROFILE" agent --local \
  --session-id pico-minicpm5-first-run \
  --model pico-minicpm/minicpm5-1b \
  --message '把上一句话缩短一半。' \
  --thinking off \
  --timeout 3600 \
  --json
```

需要全新会话时，换一个新的 session ID。不要通过增大 `contextWindow` 来规避会话过长；该值受 OM 的物理 KV cache 限制。

命令行 Agent 已通过后，也可以进入本地交互终端：

```bash
openclaw --profile "$PROFILE" chat --local \
  --session pico-minicpm5-tui \
  --thinking off
```

交互终端只是使用方式不同，仍然连接同一个 provider 和同一个单并发 native service。

### 5. 可选：直接跑一轮原始推理

如果只想验证 provider，而不需要 Agent 会话：

```bash
openclaw --profile "$PROFILE" infer model run --local \
  --model pico-minicpm/minicpm5-1b \
  --prompt 'Reply exactly: pong' \
  --json
```

## 使用 Gateway

先完成前面的 `agent --local` 门禁，再启用 Gateway。

终端 A：

```bash
PROFILE=pico-minicpm

openclaw --profile "$PROFILE" gateway run \
  --bind loopback \
  --port 18789
```

终端 B：

```bash
PROFILE=pico-minicpm

openclaw --profile "$PROFILE" health --json
openclaw --profile "$PROFILE" gateway health --json

openclaw --profile "$PROFILE" agent \
  --session-id pico-minicpm5-gateway \
  --model pico-minicpm/minicpm5-1b \
  --message '只回复 PICO_GATEWAY_OK' \
  --thinking off \
  --timeout 3600 \
  --json
```

不带 `--local` 的 `openclaw agent` 通过 Gateway 执行。Gateway 和 agent 必须使用同一个 `--profile`。

当前 Gateway 到 Hi3403 native OM 的完整发行门禁仍在补充中，因此预览版本应先以 local 模式作为主要验收路径。

## 跨机器使用：OpenClaw 在 PC，服务在 Hi3403

服务继续只监听板端 `127.0.0.1:8000`。在运行 OpenClaw 的 PC 上建立 SSH 隧道：

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:18080:127.0.0.1:8000 \
  BOARD_USER@BOARD_IP
```

保持这个终端运行。在 PC 的另一个终端检查：

```bash
curl -fsS http://127.0.0.1:18080/healthz | python3 -m json.tool
```

然后在 PC 上生成 OpenClaw profile，把端口改为 18080。配置生成器必须来自同一个 SHA 绑定预览包；先把它放到本机受控路径并检查存在：

```bash
PROFILE=pico-minicpm-remote
PROFILE_DIR="$HOME/.openclaw-$PROFILE"
GENERATOR="$HOME/.local/share/pico-minicpm5/pico_openclaw_minicpm5_config.py"
CONTEXT_WINDOW="$(curl -fsS http://127.0.0.1:18080/healthz | python3 -c 'import json,sys; print(json.load(sys.stdin)["context_window"])')"

mkdir -p "$PROFILE_DIR"
chmod 700 "$PROFILE_DIR"
test -f "$GENERATOR"

python3 "$GENERATOR" \
  --context-window "$CONTEXT_WINDOW" \
  --max-tokens 128 \
  --port 18080 \
  --timeout-seconds 3600 \
  --config-out "$PROFILE_DIR/openclaw.json" \
  --manifest-out "$PROFILE_DIR/pico-minicpm5.manifest.json" \
  >/dev/null
```

之后所有 OpenClaw 命令使用 `--profile pico-minicpm-remote`。

重要：工具由 OpenClaw 所在机器执行。若 OpenClaw 在 PC 上，工具不会自动在 Hi3403 板端执行。

## 实验功能：工具调用

普通用户当前不要启用工具调用。`release_tool_ready` 仍为 false，本指南故意不提供可复制的工具启用命令、`exec` 示例或宽泛工具 profile。只有未来发行 manifest 明确声明工具门禁通过，并同时给出精确 allowlist、审批策略、sandbox、工作目录限制和负例测试时，才能对外开放。

实现层面，service 和 OpenClaw 配置各有一个独立的工具 opt-in，二者必须由发行脚本成对生成。OpenClaw 开启工具而 service 未开启时会返回 `tools_not_supported`；反向只开启 service 时，OpenClaw 仍会按普通文字请求运行，并不会自动获得工具能力。

当前实验事实：

- 官方 Hugging Face MiniCPM5-1B 在 Host 上完成过真实 `exec` 工具调用、结果回灌和精确最终答案；
- 该 Host 证据不等于 Hi3403 native OM 工具资格；
- Hi3403 上 OpenClaw 2026.7.1 连接 Host HF backend 时，能识别工具调用，也完成过内建工具执行和结果回灌；这不是 native OM 证据；
- 但 `exec` 子进程曾因 `ENOMEM` 失败，minimal meta-tool 路径也出现过重复调用直至上下文溢出；
- native OM 的文字/Agent 闭环已在 ctx8192 与 ctx16384 上于 2026-08-27 板端通过（见上表）。service 层的工具往返（模型发出 `calculate` 调用、结果回灌、最终精确答案）在两个档位的 HTTP 直连下均已验证；
- 但经 OpenClaw 自身工具提示面的调用仍未收敛：1B 模型面对 OpenClaw 的 meta-tool 提示会原样复读指令或回答 `NO_REPLY`，不发出调用。OpenClaw 工具闭环对 1B 模型仍不成立，此前的结论维持不变。

生成器的工具配置会关闭 `strictMessageKeys` 以保留 `assistant.tool_calls` 和 `tool.tool_call_id`。同时，当前板端配置的 sandbox 是关闭状态；向不可信输入开放 `exec` 会允许模型在 OpenClaw 所在机器执行命令，风险很高。

因此，正式应用应继续使用文字模式。若确需 Agent 工具能力，并且所用预览包确实包含且已经验收原生 `agent.sh`，当前板端可优先使用该路径；现有公开 `v0.1.0` 只包含 `chat.sh`，不能假定存在 `agent.sh`。

## 服务部署者附录：当前 C4096 开发预览命令

本节不是 `v0.1.0` 普通用户路径。只有在已经获得 SHA 绑定的 C4096 split-runner 预览包时才可使用。

当前已知的 native JSONL handoff 形态如下。先根据预览包的 deployment manifest，把 `REPLACE_WITH_SHA_BOUND_RUN_DIR` 替换为实际目录名；不得凭目录时间或文件大小猜测候选。

```bash
export PICO_HOME=/opt/pico-minicpm5
export PICO_RUN="/opt/pico-minicpm5/split-runner-smoke/REPLACE_WITH_SHA_BOUND_RUN_DIR"
export PICO_PORT=8000

test -x "$PICO_HOME/venv/bin/python"
test -f "$PICO_RUN/bin/pico_minicpm5_split_board_runner.py"
test -f "$PICO_RUN/minicpm5_split_runner.candidate.json"
test -x "$PICO_RUN/bin/pico_persistent_acl_executor"

"$PICO_HOME/venv/bin/python" \
  "$PICO_RUN/bin/pico_minicpm5_split_board_runner.py" \
  --config "$PICO_RUN/minicpm5_split_runner.candidate.json" \
  --check-config

"$PICO_HOME/venv/bin/python" -u \
  "$PICO_HOME/service/pico_minicpm5_openai_service.py" \
  --tokenizer-json "$PICO_HOME/assets/tokenizer.json" \
  --chat-template "$PICO_HOME/assets/chat_template.jinja" \
  --context-window 4096 \
  --max-tokens 128 \
  --host 127.0.0.1 \
  --port "$PICO_PORT" \
  --runner-command \
    "$PICO_HOME/venv/bin/python" -u \
    "$PICO_RUN/bin/pico_minicpm5_split_board_runner.py" \
    --config "$PICO_RUN/minicpm5_split_runner.candidate.json" \
    --serve-jsonl \
    --persistent-executor "$PICO_RUN/bin/pico_persistent_acl_executor"
```

注意：

- `--runner-command` 必须是 service 的最后一个参数；其后的所有参数均属于 native runner；
- 当前普通用户契约禁止工具模式；未来若发行脚本启用工具 opt-in，它必须位于 `--runner-command` 之前；
- service、runner 与 OM 的 context 和 max-new 契约必须一致；
- 当前这条 split-runner 配置仍标记为 `production_ready: false`，只适合研发验证；
- 没有发行包 manifest 和 SHA 绑定时，不要把它包装成生产服务。

服务在前台运行，按 `Ctrl-C` 停止。正式部署时可再交给 systemd 或其他 supervisor 管理，但应先保留前台日志完成健康门禁。

## 常见问题

| 表现 | 原因 | 处理方法 |
|---|---|---|
| `connection refused` | 服务未启动、端口错误，或跨机器时仍使用目标机的 `127.0.0.1` | 先查 `/healthz`；跨机器使用 SSH 隧道。 |
| `/healthz` 正常但 OpenClaw 失败 | OpenClaw profile 未加载或 provider 配置错误 | 检查 `openclaw --profile ... config file` 和 `config validate --json`。 |
| HTTP 404 `model_not_found` | OpenAI 请求使用了错误模型 ID | HTTP API 使用 `minicpm5-1b`；OpenClaw 使用 `pico-minicpm/minicpm5-1b`。 |
| HTTP 400 `unsupported_temperature` | 请求设置了非零温度 | OpenAI 请求固定使用 `temperature: 0`。OpenClaw 命令中的 `--thinking off` 用于关闭 reasoning 请求，是另一项独立设置。 |
| HTTP 400 `context_length_exceeded` | 会话和生成长度超过真实 KV cache | 新建 session、缩短输入或减少 `max_tokens`；不要虚增 context。 |
| HTTP 400 `tools_not_supported` | OpenClaw 宣告工具能力，但 service 没有 `--enable-tools` | 回到文字模式，或在受控实验中同时打开两侧开关。 |
| HTTP 429 `server_busy` | native service 只有一个推理槽 | 等待当前请求结束；不要提高 OpenClaw 并发数。 |
| HTTP 502 `runner_protocol_error` | runner 退出、JSONL 不匹配或模型生成非法工具 XML | 检查 service 与 runner stderr，确认 bundle/manifest 匹配。 |
| `context_window must be at least 4096` | 试图把 ctx1024 release 接到 OpenClaw | 使用 `chat.sh`，或换成正式的 C4096/C8192 OpenClaw-ready 包。 |
| Agent 连续调用工具直到溢出 | 1B 模型与 OpenClaw meta-tool 路径未收敛 | 禁用工具，换新 session；正式版不要接受这条路径。 |
| Node/V8 `ENOMEM` | 板端 OpenClaw 子进程资源不足 | 禁用 OpenClaw 工具，改用原生 `agent.sh` 或在资源更充足的 Host 运行 OpenClaw。 |
| `models list` 显示 available 但请求失败 | 该命令只读取配置，没有证明服务可达 | 以 `/healthz`、`/v1/models` 和直接 chat completion 为准。 |

## 如何确认使用的是目标模型和目标 OM

至少保留以下信息：

1. OpenClaw `--version` 输出；
2. `openclaw --profile ... config file` 的路径；
3. `/healthz` 和 `/v1/models` 响应；
4. provider 配置中的 context、max tokens 和 base URL；
5. 启动前验证过的 deployment manifest、runner config、全部 OM 和 executor SHA256；
6. executor SHA256 和发行 manifest；
7. 一条直接 HTTP 请求和一条 `agent --local` 响应。

OpenAI 响应中的模型名称本身不能证明底层加载了哪个 OM。只有运行日志、manifest 和哈希交叉一致，才能证明 native backend 身份。

## 停止、重置和回滚

前台运行的 Gateway 或 service 使用 `Ctrl-C` 停止。

由于本文使用独立 profile，你原来的 OpenClaw 配置不会被覆盖。需要暂时停用 MiniCPM profile 时，可以重命名目录：

```bash
mv "$HOME/.openclaw-pico-minicpm" \
   "$HOME/.openclaw-pico-minicpm.disabled"
```

不要直接删除目录；其中可能包含会话和诊断信息。升级 OpenClaw、service、runner 或 OM 后，应重新执行本文的健康检查、直接 HTTP、local agent 和 Gateway 四层门禁。

## 发布就绪检查表

只有以下项目全部满足，发行包才可以宣称“普通用户安装 OpenClaw 后可直接使用”：

- [ ] 实际 context 至少为 4096，并与 OM/runner/config 完全一致；
- [ ] 发行包包含 tokenizer、chat template、OpenAI service、配置生成器和 production-ready JSONL runner；
- [ ] 所有 runtime 文件有 SHA256 manifest；
- [ ] clean-board `/healthz`、`/v1/models`、直接 chat completion 通过；
- [ ] OpenClaw 隔离 profile 的 `config validate` 和 `agent --local` 通过；
- [ ] Gateway 路径通过；
- [ ] 多轮会话和 cache isolation 通过；
- [ ] 工具模式若对外宣称支持，必须另外通过真实 native OM 工具执行、结果回灌、停止收敛和资源门禁；
- [ ] 安全文档明确服务只监听 loopback，工具执行位置和 sandbox 策略可审计。

在此之前，本文应保持“接入预览/已部署服务使用指南”的定位，而不是把 `v0.1.0` 描述为 OpenClaw-ready release。
