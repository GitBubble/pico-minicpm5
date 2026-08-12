# 发布策略

[English](RELEASE.md)

## 源码发布

```bash
pico-minicpm5 release source --check-only
pico-minicpm5 release source --out artifacts
pico-minicpm5 release sbom --out artifacts/pico-minicpm5-0.1.0.spdx.json
```

源码 scanner 规范化并检查 owner、mode、timestamp，同时拒绝权重、ONNX、OM、
tensor、动态库、image list、异常大文件、开发机绝对路径和板端地址。发布时只以
`--check-only` 运行该门禁；公开源码产物保留一份标准 sdist、wheel、SBOM 与
checksum，避免近似重复的第二份源码 tarball；GitHub 也会自动提供 tag 源码归档。

## 板端 runtime 归档

executor 源码与 Makefile 统一维护在 `app/native/`，板端 Python 源码与
直接运行 demo 分别放在 `app/src/` 和 `app/chat.sh`。使用已验收的
AArch64 executor 生成唯一 runtime asset：

```bash
make -C app/native \
  SDK_ROOT=/path/to/Hi3403/sdk/smp/a55_linux/mpp/out \
  CC=aarch64-mix210-linux-gcc
pico-minicpm5 release runtime \
  --executor app/bin/pico_persistent_acl_executor.aarch64 \
  --out artifacts
```

打包器会校验 AArch64 ELF、文件大小和 v0.1.0 manifest 中的固定 SHA256。
runtime 包只保留一棵规范的 `app/` 目录，并自带 `SHA256SUMS`。
executor C 源码、Makefile、二进制和 `chat.sh` 不再作为独立 Release
Asset 重复上传；预编译 executor 只存在于 runtime 包的 `app/bin/`。

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
补充包包含完整 `app/` 目录和已验收 AArch64 executor。厂商
ATC/DDK、动态库和 custom-op `.so` 始终由用户提供。发布前还需确认模型许可证
以及本地工具链对派生产物的分发条款。
