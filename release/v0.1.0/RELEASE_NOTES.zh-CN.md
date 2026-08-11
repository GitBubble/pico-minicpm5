# pico-minicpm5 v0.1.0

[English](RELEASE_NOTES.md)

本版本开源 MiniCPM5-1B → ONNX → 24 层打包 PICO OM 的可复现流程，并记录
SS928 ctx1024 三句柄产物合同。合格产物的大小和 SHA256 位于
`release-manifest.json`；模型产物通过 GitHub Release 分发，不嵌入源码归档。

默认编译路线是图级组合后分别调用 ATC 编译 prefill/decode；二进制 OM post-link
不是生产路线。portable qualification 保存 raw hash、公开 tensor cosine、greedy
token 和性能证据，同时移除板端地址。

本次 v0.1.0 增量刷新不改变三只 OM：新的 resident-K/V runtime 将
packed K/V 保留在板端，直接完成 FP32 到 FP16 cache scatter，并优化
C4 embedding 与稀疏 RoPE 准备。板端性能由 `8.20–8.60 token/s` 提升到
`9.42–9.48 token/s`（`105.5–106.1 ms/token`），同时保持 48/48 token 一致、
EOS 和中文路径通过。

板端应用源码、executor C 和 Makefile 统一归档到 `app/`。预编译
executor 只放入 runtime 包的 `app/bin/`；不再发布 executor 源码、
Makefile、二进制和 `chat.sh` 的独立重复 Asset。

runtime 新增常驻 stdin REPL。无参数运行 `app/chat.sh` 只加载一次三个
模型句柄，随后可连续输入 prompt；内置 `/help`、`/reset` 和 `/quit`。
原有单次 `--prompt` 用法保持兼容。
REPL 回答现在会随 token 生成逐步显示，默认上限为 128 token，并支持
`/max N`。达到回答或 ctx 上限时会显式提示，不再静默截断。

原生 MiniCPM5 工具调用 Agent 由独立的 `app/agent.sh` 提供。运行时复用官方 chat template
的 `<tools>/<function>/<param>/<tool_response>` 合同，支持多轮工具回填、会话历史、
工作区边界和每次确认的写入/shell 权限。`app/chat.sh` 保持纯对话 REPL，单次
`--prompt` 路径继续保留。
聊天入口现使用官方无工具 chat template；UTF-8 安全增量解码会缓存尚未完整的
汉字 token，修复此前遇到拆分汉字时界面暂停并在结束后整段重放的问题。
Agent thinking 保持默认关闭，现可通过启动参数 `--thinking`、环境变量
`THINKING=1` 或常驻会话命令 `/think on|off` 配置。
彩色输入提示符现遵循 readline 的零宽转义合同，中文输入退格到行首时不会残留
第一个字符。
Agent 提示现明确将 `.` 绑定到配置的 workspace，要求直接使用文件工具而不是
追问用户当前路径；工具结果压缩到 800 字符，为 ctx1024 保留最终回答空间。
明确的列目录请求增加只读确定性路由，其余工具继续由模型原生选择。
`agent.sh` 新增 Linux 命令行风格的 `/help [COMMAND]`：总览会列出每个本地
命令、别名、权限作用域和数值范围，`/help max` 等主题帮助则显示详细语法与
实际限制。

## Agent 路由与 Context Profile — 2026-08-11

应用现将本地命令、确定性直接工具、工具后模型总结和纯模型请求分离。明确的列
目录请求走 `DIRECT_TOOL`，直接展示类型化结果并显示 `model skipped`，不会为该
请求注入工具 schema、重放 prompt 或调用 MiniCPM。板端实测工具执行约
`4.3 ms`，常驻请求总耗时约 `12.2 ms`；单次启动命令外围仍有约 `10.8 s` 的三
句柄冷加载成本。报告新增 route mode/reason、route/tool/total 耗时和
`model_called` 证据。

runtime profile 将 context、模型产物、能力、生成限制和数值策略绑定为一个合同。
最终矩阵为：ctx128 仅 Chat；ctx1024、ctx4096、ctx8192 支持 Chat+Agent。本
Release 只有 ctx1024 已验收；其它 profile 在对应 OM 通过 descriptor、公开输出
cosine 严格大于 `0.98`、greedy token 和板端门禁前保持 fail-closed `pending`。
Agent 选择 ctx128 会在模型加载前拒绝。运行时启动时还会检查 mask/RoPE/K/V
descriptor 几何与所选 context 完全一致。
