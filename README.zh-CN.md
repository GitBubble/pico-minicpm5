# pico-minicpm5

[English](README.md) · [板端 Demo](app/README.zh-CN.md)

<img src="docs/media/board-agent.gif" alt="Hi3403 板端的四轮 agent 会话" width="100%">

一次板端会话，按它真实运行的速度播放。没有加速，也没有剪辑，所以屏幕上的每一个
数字都是板子当时给出的。

问候在 `3.2 s` 后得到回答，因为不需要工具的轮次不会被披露任何工具 schema。写文件
需要，那也就是慢的那一轮：`395` 个 prompt token，每个 `79.5 ms`——进度条把它数出来，
而不是藏起来。之后两轮根本没有惊动模型：列目录 `1.8 ms`，`swish(2)` `0.7 ms`，
由 Python 算出，因为这个数模型自己会算错。列目录同时也是对写入的验证：`a.txt`
就在里面。

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

`v0.2.1` 为五档 decode 上下文提供 runtime 合同。下表数字测自 **Euler Pi**
（商业 `SS928V100_SDK_V2.0.2.2`）。同一套 `ctx1024` OM 在 **Orange Pi
AIfly**（社区 Pegasus / Jammy）上用**同一条**
`chat.sh --prompt '只回复 PICO_OK' --max-new 8` 过门：token ids 相同、文本
相同、`CHAT_EXIT=0`。两板对照见
[`app/README.zh-CN.md`](app/README.zh-CN.md)。长上下文 OM 仍是
owner-supplied 产物；源码发行不重新分发权重、授权运行库或本地编译模型。

| Profile | 每 token p50 | token/s | prompt 送入 | 状态 |
|---|---:|---:|---:|---|
| ctx1024 | 100.40 ms | **9.96** | 79.49 ms/token | 已验收 |
| ctx4096 | 127.96 ms | **7.81** | 106.28 ms/token | 已验收 |
| ctx8192 | 165.71 ms | **6.03** | 146.82 ms/token（4097-token 门） | 已验收 |
| ctx10240 | 185.78 ms | **5.38** | 166.26 ms/token（4097-token 门） | pending |
| ctx16384 | 242.51 ms | **4.12** | 222.25 ms/token（4097-token 门） | pending |

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
- `ctx8192`：最低公开输出 `0.986076`、greedy 48/48、修正后的“句号 + EOS”精确
  一致，4097-token head-skip、实时内存与 JSONL 门均 PASS。
- `ctx10240`：长 prompt、EOS 与运行门通过，但 greedy 仅 36/48、尾 hidden
  cosine `0.978842`，因此保持 pending。
- `ctx16384`：短 greedy/EOS 与长 prompt 运行门通过，但最佳重标定尾部 hidden/K/V
  仅 `0.957146/0.985295/0.967172`，因此保持 pending。

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

一套部署由三个 Release 装配而成，因为模型文件没有变化、也就没有重新上传：

| 来源 | 内容 | 原因 |
|---|---|---|
| [`v0.2.1`](https://github.com/GitBubble/pico-minicpm5/releases/tag/v0.2.1) | 源码包、SBOM、长上下文 runtime/profile 代码 | 仅源码，不重新分发 OM |
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

sha256sum -c --ignore-missing SHA256SUMS

tar xzf pico-minicpm5-runtime-v0.2.0.tar.gz --strip-components=1
mkdir -p models assets
mv prefill.om decode.om head_flat.om models/
mv token_embedding.f16.bin tokenizer.json assets/
```

校验要在移动之前做：`SHA256SUMS` 记的是刚下载时的文件名，而且它还列了本版的
Python 分发件与 SPDX 文档，这段配方并不下载那几个 —— 所以要加
`--ignore-missing`。

需要 `ctx4096` 时，再补上它的 decode OM，并在启动时选择该 profile。扩展上下文
由它自己那份校验和文件覆盖，不在上面那份里：

```bash
gh release download v0.1.0-ctx-preview --repo GitBubble/pico-minicpm5 \
  --pattern 'decode.ctx4096.om' --pattern 'SHA256SUMS.ctx-preview'
sha256sum -c --ignore-missing SHA256SUMS.ctx-preview
mkdir -p models/ctx4096 && mv decode.ctx4096.om models/ctx4096/decode.om
./app/chat.sh --profile ctx4096
```

然后传到板端：

```bash
tar cf - . | ssh root@BOARD_IP \
  'mkdir -p /opt/pico-minicpm5 && tar xf - -C /opt/pico-minicpm5'
```

拷完之后用哪块板、哪套 SDK、跑哪条拉起脚本，见
[`app/README.zh-CN.md`](app/README.zh-CN.md)（Euler Pi 商业版 vs Orange Pi
AIfly 社区版）。简表：

| 板 | SDK | 拷完之后 |
|---|---|---|
| Euler Pi 2.0 | SS928V100_SDK_V2.0.2.2，Linux 4.19.90 | `install_board.sh` / `prepare_npu.sh` |
| Orange Pi AIfly | Pegasus / Jammy `6.6.86-hi3403` | `prepare_community.sh` + glibc 2.39 sidecar |

## Euler Pi 出厂镜像：先卸 pqp，再加载 SVP NPU

已验收的板端数字来自 Hi3403 / SS928。易百纳 **Euler Pi 2.0** 出厂 Linux
（`load_ss928v100 -i`，由 `/etc/init.d/S90autorun` 调用）会插入 `ot_pqp.ko`。
该模块与 `ot_svp_npu.ko` **不能同时存在**——厂方脚本里写明了这一点——因此
`/dev/svp_npu` 不会出现，三句柄 OM 也无法执行。

这块板登录后应看到的环境（来自 `/etc/firmware_version`）：

| 字段 | 值 |
|---|---|
| 产品 | Euler Pi |
| Chip | SS928V100 |
| SDK | SS928V100_SDK_V2.0.2.2 |
| Hardware | HiEuerPI_V1.2 |
| Software | V2.0 |
| Kernel | 4.19.90 aarch64 |
| 出厂账号 | `root` / `ebaina`（手册《海鸥派快速体验手册》） |
| USB 直连 | 主机 `192.168.137.1/24`，板端可加 `192.168.137.100/24` |

换一块新出厂板时，主机上一条命令即可（USB 网卡已是 `192.168.137.1`，
部署树已装配好）：

```bash
./app/bringup_euler_pi.sh \
  --stage /tmp/pico-minicpm5-board-stage \
  --iface en8 --board-ip 192.168.137.100 --smoke
```

它会发现链路上的板、用 `root` / `ebaina` 登录、加上 `192.168.137.100`、
拷入部署（含 `app/lib`）、卸 pqp / 加载 NPU、必要时装 Python，再跑
`chat.sh` 冒烟。

拷贝完成后也可以只在板端跑安装脚本：卸 `pqp`、加载 NPU、把该步骤挂到开机
（排在 `S90autorun` 之后），并在交互式 SSH 登录时打印上述环境：

```bash
ssh root@BOARD_IP \
  '/opt/pico-minicpm5/app/install_board.sh --usb-ipv4 192.168.137.100/24'
```

只做当次模块切换、不改开机与登录提示：

```bash
ssh root@BOARD_IP /opt/pico-minicpm5/app/prepare_npu.sh
ssh root@BOARD_IP /opt/pico-minicpm5/app/board_env.sh
```

`chat.sh` / `agent.sh` 在 `/dev/svp_npu` 缺失且厂方 `svp_npu` ko 目录存在时
会再调用一次 `prepare_npu.sh`。重启后仍依赖 `install_board.sh` 写入的
`/etc/init.d/S91pico_npu`，否则 `S90autorun` 会再次装上 `ot_pqp`。

## 社区 SDK（Orange Pi AIfly / Pegasus）

两板对照和已测 chat 冒烟数字见
[`app/README.zh-CN.md`](app/README.zh-CN.md)。AIfly 是 Ubuntu 22.04（glibc
**2.35**）；社区 `libsvp_aicpu.so` 要 `fmod@GLIBC_2.38`。Python 继续用系统
libc。只有执行器进程走 `app/glibc239/`（Ubuntu 24.04 `libc6` 2.39）。
`chat.sh` 拉起 `pico_persistent_acl_executor.community`，用该 loader 跑
`community.bin`（板上链好的 Pegasus `libsvp_acl.a` + `libss_mpi.a`）。

图形与推理不能共存。`prepare_community.sh` 停 LightDM，对 `sample_gfbg` 发
SIGTERM（它自己的处理函数会跑 `sample_comm_sys_exit()`）。`kill -9` 和
`rmmod ot_vo` 会挂死板子。把 `BUILD_DESKTOP` 写成 `no`，避免
`orangepi-hardware-optimization` 再拉桌面；此时 `load_hi3403` 必须跳过
`ot_tde` / `ot_vo` / `gfbg` / HDMI，否则 ACL `malloc_fix_addr` 会撞上 MMZ
基址上的 framebuffer。`libpico_mmz_anyaddr.so` 仍会在请求地址低于 zone 时
改写 `IOC_MMB_ALLOC_V3`。

USB IPv4 出厂不固定。主机网卡 `192.168.138.1`：

```bash
./app/configure_orangepi_usb_ipv4.sh --iface en10 --board-ip 192.168.138.10
```

```bash
tar cf - -C /tmp/pico-minicpm5-board-stage . \
  | ssh orangepi@192.168.138.10 'echo orangepi | sudo -S tar xf - -C /opt/pico-minicpm5'

ssh -t orangepi@192.168.138.10 \
  'echo orangepi | sudo -S /opt/pico-minicpm5/app/prepare_community.sh'
ssh -t orangepi@192.168.138.10 \
  'cd /opt/pico-minicpm5 && echo orangepi | sudo -S env TOKENIZERS=/opt/pico-minicpm5/pylib PYTHON=python3 ./app/chat.sh --prompt "只回复 PICO_OK" --max-new 8'
```

不要把这块板指到商业 `app/lib`（`svp_acl_init ret=100000`）。

## Euler Pi 出厂镜像：在主机上装 Python 3

出厂 Linux **没有** `python3`、没有 `pip`、也没有 `opkg`/`apt`。glibc 是
`2.29`，不能把 Ubuntu 的 3.10 deb 直接拷上去。`chat.sh` 需要 CPython 3.10
和 `tokenizers`（词表）；OpenClaw 预览再加 `jinja2`。

在**主机**上跑，不要在板端跑：

```bash
# 部署包已经在 /opt/pico-minicpm5 之后
./app/install_python.sh --board root@192.168.137.100
```

脚本会下载钉死的
`cpython-3.10.21+20260814` aarch64 `install_only_stripped`（glibc ≥ 2.17）
以及 `tokenizers` / `jinja2` / `MarkupSafe` 的 manylinux aarch64 轮子，校验
SHA-256，解到 `/opt/pico-minicpm5/venv`。`chat.sh` 会优先用
`$ROOT/venv/bin/python`。

GitHub / PyPI 慢时（常见于国内网络）：

```bash
PICO_GITHUB_MIRROR=https://ghfast.top \
PICO_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn \
  ./app/install_python.sh --board root@192.168.137.100
```

只在主机上预下载、稍后自己拷：

```bash
./app/install_python.sh --stage /tmp/pico-board-python --skip-upload
tar cf - -C /tmp/pico-board-python venv \
  | ssh root@192.168.137.100 'tar xf - -C /opt/pico-minicpm5'
```

板端自检：

```bash
ssh root@192.168.137.100 \
  '/opt/pico-minicpm5/venv/bin/python -c "import tokenizers,jinja2; print(tokenizers.__version__)"'
```

不要在板端 `apt install python3`：这套 rootfs 不是 Debian。不要用要求
glibc 2.34+ 的 `manylinux_2_34` 轮子。

## Euler Pi 出厂镜像：随包交付的 SVP ACL 运行库

`chat.sh` 拉起的执行器链接的是 `libsvp_acl.so`，不是厂方 `/opt/lib/npu` 里的
`libascendcl.so`。这四只库已经放在部署包的 `app/lib/`，来源是
`SS928V100_SDK_V2.0.2.2`。把整个目录拷到板上即可，不必再去翻 SDK：

| 文件 | 作用 |
|---|---|
| `app/lib/libsvp_acl.so` | SVP ACL |
| `app/lib/libsvp_aicpu.so` | AICPU |
| `app/lib/libprotobuf-c.so.1` | protobuf-c |
| `app/lib/libsecurec.so` | 边界检查 |

`chat.sh` 优先使用 `$APP/lib`。校验：

```bash
ssh root@192.168.137.100 \
  'cd /opt/pico-minicpm5/app/lib && sha256sum -c SHA256SUMS'
```

只有在你要换成另一棵 SDK 树时才需要：

```bash
./app/install_runtime_lib.sh --sdk-root /path/to/SS928V100_SDK_V2.0.2.2
```

runtime 包在 `app/bin/` 提供 AArch64 executor，源码与 Makefile 在
`app/native/`。`app/lib/` 是板端加载 OM 所需的运行库，不是 ATC/DDK。

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

源码仓库包含 Python 包、配置、schema、测试、文档，以及板端执行器所需的
`app/lib/` SVP ACL 运行库。模型权重、ONNX external data、OM、token embedding、
ATC/DDK/libinstsim、`libsvp_custom.so` 和私有板端信息不会进入源码归档。预编译
的模型派生产物通过独立 Release asset 发布，并记录来源、大小与 SHA-256。

更多中文文档：

- [Agent 路由与运行时 Context Profile 设计](docs/AGENT_ROUTING_AND_CONTEXT_PROFILES.zh-CN.md)
- [Native 多 Token Prefill 调度契约](docs/NATIVE_PREFILL_SCHEDULER.zh-CN.md)
- [一颗 NPU 上的两个模型：视觉 skill](docs/MULTIMODAL_VISION.zh-CN.md)
- [量化契约](docs/QUANTIZATION_CONTRACT.zh-CN.md)
- [端到端流水线](docs/PIPELINE.zh-CN.md)
- [OM 图级组合契约](docs/OM_COMPOSITION.zh-CN.md)
- [验证阶梯](docs/VALIDATION.zh-CN.md)
- [SDK 环境](docs/SDK_SETUP.zh-CN.md)
- [发布策略](docs/RELEASE.zh-CN.md)
- [Hi3403 验收](docs/Hi3403_ACCEPTANCE.zh-CN.md)
- Euler Pi 出厂镜像：见上文「Euler Pi 出厂镜像：先卸 pqp，再加载 SVP NPU」

开发检查：

```bash
pytest
pico-minicpm5 doctor
pico-minicpm5 release source --check-only
```
