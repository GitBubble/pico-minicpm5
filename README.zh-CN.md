# pico-minicpm5

[English](README.md) · [板端 Demo](app/README.zh-CN.md)

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

冻结的 `ctx1024` 候选已在 Hi3403 上完成验收：

- prefill 公开输出最低 cosine：`0.996646`；
- decode 公开输出最低 cosine：`0.998023`；
- 生成 token 与官方 checkpoint 的 FP64 oracle 对比为 `48/48`；
- EOS 与中文生成路径通过；
- 优化后 resident-K/V runtime 达到 `9.42–9.48 token/s`，
  即 `105.5–106.1 ms/token`，约为已验收 49 句柄基线的 `1.91x`。
- prompt-only head 抑制通过 token-exact 板端 A/B：rebase 后 810-token 首次请求
  由 `86.70 s` 降至 `69.45 s`（降低 `19.89%`），命中 643-token resident 前缀后
  进一步降至 `14.61 s`。
- fail-closed native-prefill 调度器已实现未来的
  `S128 -> S32 -> S16 -> strict S1 tail` 策略，并把决策写入请求报告。当前
  已验收 Release 仍只启用 S1，详见
  [native prefill 合同](docs/NATIVE_PREFILL_SCHEDULER.zh-CN.md)。

这些数字只对应已记录的 Hi3403 配置和三个冻结 OM 哈希，不代表所有 Hi3403 产品
配置。本 Release 的上下文合同固定为 1024。

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

从 Release 下载和整理文件：

```bash
mkdir pico-minicpm5-deployment-v0.1.0
cd pico-minicpm5-deployment-v0.1.0
gh release download v0.1.0 --repo GitBubble/pico-minicpm5 \
  --pattern 'pico-minicpm5-runtime-v0.1.0.tar.gz' \
  --pattern 'prefill.om' --pattern 'decode.om' --pattern 'head_flat.om' \
  --pattern 'token_embedding.f16.bin' --pattern 'tokenizer.json'
tar xzf pico-minicpm5-runtime-v0.1.0.tar.gz --strip-components=1
mkdir -p models assets
mv prefill.om decode.om head_flat.om models/
mv token_embedding.f16.bin tokenizer.json assets/
sha256sum -c SHA256SUMS
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
- [Native 多 Token Prefill 调度合同](docs/NATIVE_PREFILL_SCHEDULER.zh-CN.md)
- [端到端流水线](docs/PIPELINE.zh-CN.md)
- [OM 图级组合合同](docs/OM_COMPOSITION.zh-CN.md)
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
