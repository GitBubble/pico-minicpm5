# MiniCPM5-1B 接入 OpenClaw

这个目录是一套可独立分发的 OpenClaw 适配源码，负责把 MiniCPM5 native
runner 的 JSONL 接口转换成 OpenAI-compatible HTTP 接口，并生成一个隔离、
仅访问本机服务的 OpenClaw profile。

它不包含模型权重、`.om`、芯片 SDK、驱动、板端执行器或 OpenClaw 本体。
这些组件必须由各自的合法来源单独安装。仅复制本目录不会得到一个可运行模型。

## 支持边界

- HTTP 接口：`GET /healthz`、`GET /v1/models`、
  `POST /v1/chat/completions`；
- runner 接口：`pico.minicpm5.runner.v1`，支持 Unix socket 或无 shell 的
  子进程 JSONL；
- 上下文：OpenClaw 配置要求实际编译合同至少为 4096 token；
- 并发：一个生成槽，忙时返回 HTTP 429；
- 采样：只支持 `temperature=0`；
- 工具调用：源码支持严格 XML → OpenAI `tool_calls` 转换，但默认关闭。

当前首先要闭环纯文字对话。工具模式仍需要对具体 OpenClaw 版本、native OM、
工具轮次和板端资源做单独验收；不要因为 HTTP 服务能启动就声称工具调用已发布。

## 目录

```text
app/openclaw/
├── bin/
│   ├── start.sh
│   ├── stop.sh
│   ├── doctor.sh
│   └── configure-openclaw.sh
├── config/
│   ├── runtime.example.json
│   └── runtime.command.example.json
├── src/
│   ├── openai_service.py
│   ├── config_generator.py
│   ├── merged_jsonl_runner.py
│   ├── merged_board_server.py
│   ├── pico_minicpm5_split_board_runner.py
│   ├── probe_om_execute_latency.py
│   ├── qualify_minicpm_greedy_chain.py
│   └── lifecycle.py
├── tests/
├── requirements.txt
├── SOURCE_SNAPSHOTS.json
└── README.zh-CN.md
```

## 1. 准备环境

确认设备上已经有 Python 3.10+ 和 OpenClaw：

```bash
python3 --version
openclaw --version
```

当前配置生成器固定的发布目标是 OpenClaw `2026.7.1`、Node.js `24.15.0`
（Linux ARM64）。其他版本可能仍能通过文字模式配置校验，但尤其不能沿用工具调用
结论；部署时应优先使用发行 manifest 指定的版本组合。

如果 OpenClaw 已安装但不在 `PATH`，给 doctor 指定绝对路径：

```bash
export PICO_OPENCLAW_BIN=/opt/openclaw/bin/openclaw
```

在本目录创建隔离 Python 环境：

```bash
cd /opt/pico-minicpm5/app/openclaw
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

生命周期脚本默认调用 `python3`。指定刚创建的环境：

```bash
export PICO_OPENCLAW_PYTHON=/opt/pico-minicpm5/app/openclaw/.venv/bin/python
```

不要把私有 PyPI token、SDK 路径或模型密钥写入本目录。

### 必须提供的外部文件

部署者需要另外准备：

```text
/opt/pico-minicpm5/
├── assets/
│   ├── tokenizer.json
│   ├── chat_template.jinja
│   └── token_embedding.f16.bin
├── bin/
│   └── pico_persistent_acl_executor…  # 发行包绑定的 zero-once 执行器
└── models/
    └── ...                             # 发行包绑定的 OM；本项目不附带
```

服务会拒绝 tokenizer/template 漂移。当前源码固定的官方文件 SHA-256 是：

```text
tokenizer.json       3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81
chat_template.jinja  7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c
```

模型 OM、runner manifest 和执行器还必须按对应发行包的 `SHA256SUMS` 校验；
上面两个哈希不能替代 native runtime 的完整身份绑定。

## 2. 配置 native runner

先复制示例，不要直接修改被 Git 跟踪的模板：

```bash
cd /opt/pico-minicpm5/app/openclaw
cp config/runtime.example.json config/runtime.json
chmod 600 config/runtime.json
```

`runtime.json` 只能使用绝对路径。默认、安全的 socket 模式如下：

```json
{
  "schema": "pico.minicpm5.openclaw-runtime.v1",
  "tokenizer_json": "/opt/pico-minicpm5/assets/tokenizer.json",
  "chat_template": "/opt/pico-minicpm5/assets/chat_template.jinja",
  "context_window": 8192,
  "max_tokens": 128,
  "host": "127.0.0.1",
  "port": 8000,
  "enable_tools": false,
  "runner": {
    "socket": "/run/pico-minicpm5/runner.sock"
  }
}
```

字段约束：

- `context_window` 必须等于 OM/KV runtime 的实际编译上限，不能填写模型卡上的
  理论上限；
- `max_tokens` 必须小于等于 runner 的生成上限；
- `host` 强制为 `127.0.0.1`。服务没有 HTTP 鉴权，配置成局域网或公网地址会被拒绝；
- `enable_tools` 初次部署保持 `false`；
- `runner` 必须且只能包含 `socket` 或 `command` 中的一种。

当前 C8192 三 OM 的 command 模板是
`config/runtime.command.example.json`。它直接启动本目录的
`src/merged_jsonl_runner.py`，并显式传入 runtime module、prefill/decode/head
三个 OM、embedding、executor 和 zero-once 合同。命令必须写成 JSON 数组：

这里的三 OM 是有意设计的混合上下文合同：long decode 使用 C8192，position-0
prefill 使用已资格化的 C1024，head 是两条路径共享的无上下文 head。不要为了让
文件名看起来一致而把 prefill/head 冒充为 C8192。

```json
"runner": {
  "command": [
    "/opt/pico-minicpm5/app/openclaw/.venv/bin/python",
    "/opt/pico-minicpm5/app/openclaw/src/merged_jsonl_runner.py",
    "--serve-jsonl",
    "--runtime-module",
    "/opt/pico-minicpm5/app/openclaw/src/merged_board_server.py",
    "--python-path",
    "/opt/pico-minicpm5/app/openclaw/src",
    "--persistent-executor",
    "/opt/pico-minicpm5/bin/pico_persistent_acl_executor.zero_once.v2_352.aarch64",
    "--decode-model",
    "/opt/pico-minicpm5/models/decode.ctx8192.om",
    "--prefill-model",
    "/opt/pico-minicpm5/models/prefill.ctx1024.om",
    "--head-model",
    "/opt/pico-minicpm5/models/head.om",
    "--library-path",
    "/root/pico_default_smoke/lib",
    "--library-path",
    "/opt/lib/npu",
    "--embedding",
    "/opt/pico-minicpm5/assets/token_embedding.f16.bin",
    "--context",
    "8192",
    "--max-new-limit",
    "128",
    "--decode-no-cache",
    "--characterize-decode-workspace-zero-once"
  ]
}
```

示例中的文件名是部署槽位，不是本目录附带的文件。必须替换为同一 C8192 发行
manifest 中的实际绝对路径，并逐一核对 SHA-256。zero-once 只能和经过资格验证的
C8192 executor/runtime/OM 组合使用；不要把两个 zero-once flag 套到 C4096、short
decode 或任意旧执行器。

上面的两个 `--library-path` 是当前板端已知路径；参数可重复。未来正式发行若把
运行库归档到 canonical 目录，应按 release manifest 同步替换，不能删除其中一个后
假定动态链接仍会成功。

`merged_board_server.py` 和它依赖的三个 Python 模块已随本目录打包。生命周期
检查会拒绝板端旧的 `merged_board_server_repl.py`，也会拒绝把 runtime module
指向 bundle 外部；这样可以避免旧签名看似启动、首个请求才失败。

这里不会经过 shell，不能使用 `~`、`$VAR`、管道、重定向或把整条命令写成
一个字符串。这是为了防止参数被二次解释。

## 3. 启动前检查

socket 模式需要先启动 native runner，并确认 socket 已经存在：

```bash
test -S /run/pico-minicpm5/runner.sock
```

执行 doctor：

```bash
cd /opt/pico-minicpm5/app/openclaw
PICO_OPENCLAW_PYTHON="$PICO_OPENCLAW_PYTHON" bin/doctor.sh
```

结果是 JSON。`status` 为 `ok` 才继续。它会检查：

- runtime JSON 的 schema 和安全约束；
- tokenizer、template；
- Python 依赖；
- runner socket 或可执行文件；
- 如果服务已启动，则验证 PID 所有权及 `/healthz` 合同。
- 传入 `--profile` 时，还会执行真实的 OpenClaw `config validate --json`；版本
  偏离固定版本会作为非阻断提示报告。

doctor 不执行推理，因此不能替代真实 chat 请求，也不能证明某组 OM 的数值精度。

## 4. 启动服务

```bash
cd /opt/pico-minicpm5/app/openclaw
bin/start.sh --wait-seconds 120
```

成功输出类似：

```json
{
  "status": "started",
  "pid": 1234,
  "health": {
    "status": "ok",
    "model": "minicpm5-1b",
    "supportsTools": false,
    "context_window": 8192
  }
}
```

默认状态目录为：

```text
~/.local/state/pico-minicpm5-openclaw/
```

其中包含 `service.json` 和 `service.log`，权限为当前用户私有。可以用环境变量
更换位置：

```bash
export PICO_OPENCLAW_STATE_DIR=/var/lib/pico-minicpm5/openclaw-state
```

再次检查运行态：

```bash
bin/doctor.sh --require-running
curl -fsS http://127.0.0.1:8000/healthz | python3 -m json.tool
curl -fsS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

如果启动失败，查看：

```bash
tail -n 100 ~/.local/state/pico-minicpm5-openclaw/service.log
```

## 5. 绕过 OpenClaw 做一次真实推理

先确认 service → runner 路径可用：

```bash
curl -fsS --connect-timeout 5 --max-time 3600 \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5-1b",
    "messages": [{"role": "user", "content": "只回复 PICO_HTTP_OK"}],
    "temperature": 0,
    "stream": false,
    "max_tokens": 16
  }' \
  http://127.0.0.1:8000/v1/chat/completions | python3 -m json.tool
```

必须得到 HTTP 200 和 assistant 内容。`/healthz` 成功但本请求失败，通常说明
runner 进程、socket、manifest 或 native 模型组合不匹配。

## 6. 生成隔离的 OpenClaw profile

不要覆盖日常使用的 `~/.openclaw/openclaw.json`。生成独立 profile：

```bash
cd /opt/pico-minicpm5/app/openclaw
bin/configure-openclaw.sh --profile pico-minicpm
```

它只写入：

```text
~/.openclaw-pico-minicpm/openclaw.json
~/.openclaw-pico-minicpm/pico-minicpm5.manifest.json
```

文件已存在时默认拒绝覆盖。确认 runtime 合同确实改变后才使用：

```bash
bin/configure-openclaw.sh --profile pico-minicpm --force
```

生成配置的关键安全默认值：

- provider 只访问 `http://127.0.0.1:8000/v1`；
- `models.mode=replace`，不混入其他远端模型；
- 关闭 pricing catalog 网络刷新；
- `maxConcurrent=1`；
- 纯文字模式不向 1B 模型注入工具 schema；
- 压缩保留量按实际本地上下文缩小，而不是沿用桌面大模型默认值。

校验 OpenClaw 配置：

```bash
PROFILE=pico-minicpm
openclaw --profile "$PROFILE" config file
openclaw --profile "$PROFILE" config validate --json
openclaw --profile "$PROFILE" models list --provider pico-minicpm --json
```

模型列表中必须出现：

```text
pico-minicpm/minicpm5-1b
```

## 7. 普通用户开始对话

先使用 local/embedded 模式，不启动 Gateway：

```bash
PROFILE=pico-minicpm
openclaw --profile "$PROFILE" agent --local \
  --session-id minicpm5-first-chat \
  --model pico-minicpm/minicpm5-1b \
  --message '你好，请用一句话介绍你自己。' \
  --thinking off \
  --timeout 3600 \
  --json
```

重复使用相同 `--session-id` 可以延续会话。为了控制上下文和定位问题，首次验收
应使用新的 session ID。

需要 Gateway 时，在一个终端启动：

```bash
openclaw --profile pico-minicpm gateway run \
  --bind loopback --port 18789
```

另一个终端调用：

```bash
openclaw --profile pico-minicpm agent \
  --session-id minicpm5-gateway-chat \
  --model pico-minicpm/minicpm5-1b \
  --message '只回复 PICO_GATEWAY_OK' \
  --thinking off \
  --timeout 3600 \
  --json
```

OpenClaw 注入的 system prompt 比直接 HTTP 请求长，板端首轮 prefill 也会更慢。

## 8. PC 上运行 OpenClaw、板端运行模型

服务仍然保持板端 `127.0.0.1:8000`。在 PC 建立 SSH 隧道：

```bash
ssh -N -L 18000:127.0.0.1:8000 root@BOARD_IP
```

在 PC 的 runtime JSON 副本中把 `port` 改为 `18000`，仅用于生成 PC profile；
不要用这份配置启动第二个模型服务。然后：

```bash
PICO_OPENCLAW_CONFIG=/absolute/path/to/runtime.pc.json \
  bin/configure-openclaw.sh --profile pico-minicpm-board

openclaw --profile pico-minicpm-board agent --local \
  --session-id board-through-ssh \
  --model pico-minicpm/minicpm5-1b \
  --message '只回复 SSH_OK' \
  --thinking off --timeout 3600 --json
```

SSH 断开后 provider 会不可达，这是预期行为。不要为省略隧道而把板端服务绑定到
`0.0.0.0`。

## 9. 实验性工具调用

只有纯文字 Gate 已通过，且当前 OM/runner/OpenClaw 版本有独立工具资格证据时，
才把 runtime JSON 改成：

```json
"enable_tools": true
```

随后重启服务并重新生成 profile：

```bash
bin/stop.sh
bin/start.sh --wait-seconds 120
bin/configure-openclaw.sh --profile pico-minicpm-tools --force
bin/doctor.sh --profile pico-minicpm-tools --require-running
```

注意：

- 服务只解析和验证模型产生的工具调用，不执行工具；执行权限归 OpenClaw；
- 默认生成的工具 profile 是最小表面，不会自动开放完整 shell/文件系统；
- MiniCPM5-1B 在某些 OpenClaw 版本的 meta-tool 间接调用中可能重复调用直到上下文
  溢出；
- 板端 Node 子进程还可能受到内存限制；
- 工具是否真正执行、结果是否回灌、模型是否停止并给出最终回答，三项都成功才算
  一次完整闭环。

生产环境还应在 OpenClaw 一侧设置工具 allowlist、工作目录、审批和超时，不能只靠
模型提示词限制权限。

## 10. 停止、升级与回滚

停止当前用户启动的服务：

```bash
cd /opt/pico-minicpm5/app/openclaw
bin/stop.sh
```

停止脚本会核对 PID 的命令行身份，不会在 PID 文件漂移时盲目杀进程。
它先让 service 正常退出，使 runner 收到 JSONL EOF 并执行 native cleanup；只有
超过等待窗口才会升级为进程组 `TERM/KILL`。

升级步骤：

1. 保存 `config/runtime.json` 和发行包 SHA manifest；
2. `bin/stop.sh`；
3. 替换适配源码；
4. 对照新版本示例检查 schema；
5. `bin/doctor.sh`；
6. `bin/start.sh`；
7. 重跑 HTTP、local agent、Gateway 三层 Gate；
8. 验证完成前保留旧目录，以便回滚。

删除隔离 profile 不会影响其他 OpenClaw profile：

```bash
rm -rf "$HOME/.openclaw-pico-minicpm"
```

执行前确认路径完全匹配，不要把变量为空的路径用于删除命令。

## 常见问题

### `runtime config not found`

复制示例为 `config/runtime.json`，或显式指定：

```bash
export PICO_OPENCLAW_CONFIG=/absolute/path/runtime.json
```

### `tokenizer.json hash drift` / `chat_template.jinja hash drift`

使用了不同 checkpoint 或文件被修改。不要绕过检查；重新取得发行包绑定的官方资产。

### `runner socket does not exist`

先启动 native runner。确认 socket 路径、用户权限及部署服务没有把 socket 放在另一个
mount namespace。

### HTTP 429 `server_busy`

服务只有一个推理槽。等待当前请求完成，不要提高 OpenClaw 并发。

### HTTP 400 `context_length_exceeded`

当前 system prompt、历史和 `max_tokens` 总和超过实际上下文。使用新 session、缩短输入，
或部署经过验证的更长上下文 OM；不要只修改 JSON 数字。

### HTTP 502 `runner_protocol_error`

runner 返回了错误、错误 request ID、非法 token ID，或者工具 XML 未通过严格校验。
查看 service log 和 native runner 日志，保持失败关闭，不要把错误响应伪装成 assistant 文本。

### OpenClaw 能列出模型但不能对话

`models list` 只验证配置。依次检查 `/healthz`、直接 HTTP chat、OpenClaw local agent，
在哪一层首次失败就定位哪一层。

### 工具执行后不断重复

这是工具轮次停止问题，不是把 `max_tokens` 调大就能修复。立即换新 session，关闭
`enable_tools` 回到文字模式，并保留 OpenClaw session JSONL、服务日志和 runner capture
用于定位。

## 发布验收清单

源码包的纯本地自检不需要 OM 或 SDK：

```bash
cd /opt/pico-minicpm5
python3 -m unittest discover -s app/openclaw/tests -v
```

它会校验安全配置、C8192 mixed-context/zero-once 参数、shell/Python 语法、bundle
内 runtime 依赖导入、七份源码快照 SHA，以及包中不存在模型或 SDK 二进制。独立
checkout 会跳过“与上游 monorepo byte-exact”一项，但仍执行 manifest 哈希检查。

准备把该适配交付给普通用户时，发行方至少应提供：

- 本目录源码和许可证；
- OpenClaw、Node、Python 依赖版本；
- tokenizer/template、runner、manifest、所有 OM、执行器的逐文件 SHA-256；
- 实际 `context_window`、`max_tokens` 和支持模式；
- clean-board 的 HTTP、OpenClaw local、Gateway 数值与时延记录；
- 如果宣称工具可用，至少一轮“调用 → 真执行 → `role=tool` 回灌 → 最终答案”的
  session 证据；
- 已知资源限制、回滚方法和安全边界。

缺少 native runtime 或上述身份绑定时，本目录只能作为适配源码使用，不能称为完整
OpenClaw-ready 模型发行包。
