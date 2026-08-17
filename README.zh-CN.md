# pico-minicpm5

[English](README.md) · [板端 Demo](app/README.zh-CN.md)

<img src="docs/media/board-chat.svg" alt="MiniCPM5-1B 在 Hi3403 板上回答两个问题，9.9 token/s" width="100%">

板上的真实会话，不是演示稿：三个常驻句柄 `7.3 s` 加载完成，随后模型以
`9.91` 和 `9.92 token/s` 作答。右下角是板子自己的墙钟；等待段按 `2.4x` 播放，
输出段按真实速度播放。

`pico-minicpm5` 将固定版本的
[`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B)
转换为可复现的 Hi3403/PICO 三句柄部署：

```text
Hugging Face checkpoint
  → 真实权重的 24 层 ONNX 与词表 head
  → 图级串联并打包 K/V
  → 分别编译 prefill.om、decode.om、head_flat.om
  → 数值门禁、板端验收与 Release
```

生产路线是在编译前组合 ONNX 图，而不是拼接多个 OM 二进制。图编译器统一负责
内存分配、指令调度、TaskInfo 和层间 hidden bridge。

## 当前状态

`v0.2.0` 提供三个 decode 上下文。三档在同一次板端会话中、于 retain-input
执行器 `cef4edb2…` 上测得，且三档生成的 token 都与各自的已验收基线逐个一致。

| Profile | 每 token p50 | token/s | prompt 送入 | 状态 |
|---|---:|---:|---:|---|
| ctx1024 | 100.40 ms | **9.96** | 79.49 ms/token | 已验收 |
| ctx4096 | 127.96 ms | **7.81** | 106.28 ms/token | 已验收 |
| ctx8192 | 165.71 ms | **6.03** | 144.02 ms/token | pending |

三档之间只有 `decode.om` 不同。每一档的 position 0 都在同一个冻结的 `ctx1024`
`prefill.om` 上引导，并共享同一个 `head_flat.om`，逐字节相同——这就是混合
prefill 窗口契约。实测三档的 position-0 transformer 时间相差 `0.39 ms`，正是
这份契约在时间上的体现。

各档通过的门：

- `ctx1024`：prefill 与 decode 的公开输出最低 cosine 为 `0.996646` 和
  `0.998023`；生成 token 与官方 checkpoint 的 FP64 oracle 对比 `48/48`；为已验收
  49 句柄基线（`4.89–4.92 token/s`）的 `2.03x`。
- `ctx4096`：position 4095 最低公开输出 cosine `0.990820`，板端尾部与模拟器
  逐字节一致，`48/48` 贪心 token，边界 fail-closed。门禁记录见
  `release/contexts/ctx4096.qualification.json`。
- `ctx8192`：公开输出以 `0.986076` 过门、EOS 门也通过，但标定是 donor 零扩展
  而非原生，因此保持 `pending`，使用时需要 `--allow-unqualified-profile`。

三档的 EOS 都能干净停止，48 token oracle 也都通过。若以重新推导的 FP64 参考
为准，`ctx8192` 与参考逐 token 一致，而 `ctx1024` 与 `ctx4096` 会早停一个
token、少了一个句号——不构成阻塞，原因见
[严格 EOS 说明](release/contexts/strict-eos-oracle.md)。

长 prompt 是短板：token 仍然逐个送入，512 token 的 prompt 在 ctx1024 上约需
`41 s`。能摊薄这笔开销的 fail-closed native-prefill 调度器已经实现了
`S128 -> S32 -> S16 -> strict S1 tail` 策略并逐请求记录决策，但当前只启用
S1——还没有任何宽块通过数值门。详见
[native prefill 契约](docs/NATIVE_PREFILL_SCHEDULER.zh-CN.md)与
[性能板](release/perf/README.md)。

这些数字只对应已记录的 Hi3403 配置，不代表所有 Hi3403 产品配置。上游
checkpoint 宣称的上下文远长于此；本 Release 固定为 1024 的是 prefill 窗口，
不是上下文。

## 直接在板端运行

最短路径请阅读 [`app/README.zh-CN.md`](app/README.zh-CN.md)。当 Release 文件
已经放到板端后，只需：

```bash
cd /opt/pico-minicpm5
./app/chat.sh       # 纯对话 REPL
./app/agent.sh      # 工具调用 Agent
```

`chat.sh` 使用 MiniCPM5 官方无工具 chat template 进入多轮对话 REPL；
`agent.sh` 进入默认 `ctx1024` 的三句柄常驻
Agent，使用 MiniCPM5 官方
`<tools>/<function>/<tool_response>` 协议，内置文件、搜索、git 和需确认的
写入/shell 工具。启动时显示彩色 MiniCPM ASCII pet；模型加载、规划和工具执行
都有状态提示，最终回答逐 token 流式输出。支持 `/help`、`/tools`、
`/think on|off`、`/permissions`、`/context`、`/clear`、`/max N` 和 `/quit`。
Thinking 默认关闭，可用 `./app/agent.sh --thinking` 启动，或在运行中切换而不
重载模型。两个应用复用同一套
三只 OM 和运行时，但入口和默认行为相互独立。
单次运行可使用 `./app/chat.sh --prompt '请用一句话解释什么是神经网络。' --max-new 32`。
显式 `--prompt` 和 `--interactive` 继续保留旧的裸文本续写兼容模式。

一套部署由两个 Release 装配而成，因为模型文件没有变化、也就没有重新上传：

| 来源 | 内容 | 原因 |
|---|---|---|
| [`v0.2.0`](https://github.com/GitBubble/pico-minicpm5/releases/tag/v0.2.0) | 运行时归档、`SHA256SUMS` | 内含 `app/` 板端应用与执行器 `cef4edb2…` |
| [`v0.1.0`](https://github.com/GitBubble/pico-minicpm5/releases/tag/v0.1.0) | `prefill.om`、`decode.om`、`head_flat.om`、token embedding、tokenizer | 在 `v0.2.0` 中逐字节相同，因此留在原处 |
| [`v0.1.0-ctx-preview`](https://github.com/GitBubble/pico-minicpm5/releases/tag/v0.1.0-ctx-preview) | `decode.ctx4096.om`、`decode.ctx8192.om` | 只有扩展上下文档位才需要 |

运行时要取 `v0.2.0` 的那份。`v0.1.0` 的运行时归档带的是旧执行器，而那个执行器
无法由它自己随附的源码重建；`v0.2.0` 钉的这个可由
[`docs/EXECUTOR_BUILD.zh-CN.md`](docs/EXECUTOR_BUILD.zh-CN.md) 逐字节复现。

```bash
mkdir pico-minicpm5-deployment-v0.2.0
cd pico-minicpm5-deployment-v0.2.0

gh release download v0.2.0 --repo GitBubble/pico-minicpm5 \
  --pattern 'pico-minicpm5-runtime-v0.2.0.tar.gz' --pattern 'SHA256SUMS'
gh release download v0.1.0 --repo GitBubble/pico-minicpm5 \
  --pattern 'prefill.om' --pattern 'decode.om' --pattern 'head_flat.om' \
  --pattern 'token_embedding.f16.bin' --pattern 'tokenizer.json'

tar xzf pico-minicpm5-runtime-v0.2.0.tar.gz --strip-components=1
mkdir -p models assets
mv prefill.om decode.om head_flat.om models/
mv token_embedding.f16.bin tokenizer.json assets/
sha256sum -c SHA256SUMS
```

需要 `ctx4096` 时，再补上它的 decode OM，并在启动时选择该 profile：

```bash
gh release download v0.1.0-ctx-preview --repo GitBubble/pico-minicpm5 \
  --pattern 'decode.ctx4096.om'
mkdir -p models/ctx4096 && mv decode.ctx4096.om models/ctx4096/decode.om
./app/chat.sh --profile ctx4096
```

然后传到板端：

```bash
tar cf - . | ssh root@BOARD_IP \
  'mkdir -p /opt/pico-minicpm5 && tar xf - -C /opt/pico-minicpm5'
```

板端运行库默认位于 `/root/pico_default_smoke/lib`，这些 SDK 动态库不会在开源
项目中重新分发。runtime 包在 `app/bin/` 提供 AArch64 executor 二进制，
并将其 C 源码和 Makefile 统一归档在 `app/native/`。这些文件不再作为
独立 Release Asset 重复发布。

## 直接使用与 OpenClaw 预览

已发布的 `ctx1024` profile **不**满足 OpenClaw 对本地模型 4096 token 上下文
下限的要求，不得宣传为 OpenClaw-ready 发行包。已经另行部署了兼容服务的
用户可以参考详细中文指南。当前唯一成文的 native JSONL 路径是非生产的 C4096
split-runner 开发预览；C8192 native OM 到 OpenClaw 的链路尚未闭环：

- [MiniCPM5 服务接入 OpenClaw：普通用户使用指南（预览；当前无公开 OpenClaw-ready Asset）](docs/OPENCLAW_USAGE.zh-CN.md)

指南从安全的纯文字路径开始，使用隔离的 OpenClaw profile，把未鉴权的模型端点
保持在回环地址，并明确记录剩余的 native-OM 与工具调用发布阻塞项。

## 从源码构建

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[hub,onnx,reference,dev]'

pico-minicpm5 model fetch --local-dir work/model
pico-minicpm5 model verify --model-dir work/model
pico-minicpm5 reference capture --model-dir work/model --out work/reference --context 1024
pico-minicpm5 reference calibrate --reference work/reference --family decode --out work/calibration/decode
pico-minicpm5 reference calibrate --reference work/reference --family prefill --out work/calibration/prefill
pico-minicpm5 onnx export-layers --model-dir work/model --out work/onnx/layers --context 1024
pico-minicpm5 onnx export-head --model-dir work/model --out work/onnx/head/model.onnx
```

随后分别组合 decode/prefill 的 24 层图，并使用本地合法安装的 ATC/DDK 与
`libsvp_custom.so` 编译三只 OM。完整命令、runtime capture、score、head bridge
和 qualification 流程见 [`docs/PIPELINE.zh-CN.md`](docs/PIPELINE.zh-CN.md)。

## 发布边界

源码仓库包含 Python 包、配置、schema、测试和文档。模型权重、ONNX external
data、OM、token embedding、ATC/DDK/libinstsim、SDK 动态库和私有板端信息不会
进入源码归档。预编译的模型派生产物通过独立 Release asset 发布，并记录来源、
大小与 SHA-256。

更多中文文档：

- [Agent 路由与运行时 Context Profile 设计](docs/AGENT_ROUTING_AND_CONTEXT_PROFILES.zh-CN.md)
- [Native 多 Token Prefill 调度契约](docs/NATIVE_PREFILL_SCHEDULER.zh-CN.md)
- [量化契约](docs/QUANTIZATION_CONTRACT.zh-CN.md)
- [端到端流水线](docs/PIPELINE.zh-CN.md)
- [OM 图级组合契约](docs/OM_COMPOSITION.zh-CN.md)
- [验证阶梯](docs/VALIDATION.zh-CN.md)
- [SDK 环境](docs/SDK_SETUP.zh-CN.md)
- [发布策略](docs/RELEASE.zh-CN.md)
- [Hi3403 验收](docs/Hi3403_ACCEPTANCE.zh-CN.md)

开发检查：

```bash
pytest
pico-minicpm5 doctor
pico-minicpm5 release source --check-only
```
