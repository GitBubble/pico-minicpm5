# 发布策略

[English](RELEASE.md)

## 源码发布

```bash
pico-minicpm5 release source --check-only
pico-minicpm5 release source --out artifacts
pico-minicpm5 release sbom --out artifacts/pico-minicpm5-0.1.0.spdx.json
```

源码归档规范化 owner、mode 和 timestamp，并拒绝权重、ONNX、OM、tensor、动态库、
image list、异常大文件、开发机绝对路径和板端地址。公开源码产物保留标准 sdist、
wheel、SBOM 与 checksum；GitHub 自动提供 tag 的 source code 归档。

## 模型派生 Release

```bash
pico-minicpm5 release assemble \
  --models work/om --model-dir work/model \
  --qualification qualification.json --out artifacts/model-release
pico-minicpm5 release verify artifacts/model-release
```

除冻结 v0.1.0 哈希外，任何新 OM 都必须提供显式 PASS qualification。证据包含
prefill、decode、head 三份独立评分，并绑定准确 OM、同一个 ATC build manifest
和对应 runtime capture。全流程固定 ctx1024。

模型 bundle 包含三只 OM、FP16 embedding、tokenizer 与 manifest；板端 runtime
补充包包含 `app/chat.sh`、Python server、AArch64 executor 及其 C++/Makefile。厂商
ATC/DDK、动态库和 custom-op `.so` 始终由用户提供。发布前还需确认模型许可证
以及本地工具链对派生产物的分发条款。
