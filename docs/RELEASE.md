# Release policy

[中文](RELEASE.zh-CN.md)

## Source release

```bash
pico-minicpm5 release source --check-only
pico-minicpm5 release source --out artifacts
pico-minicpm5 release sbom --out artifacts/pico-minicpm5-0.1.0.spdx.json
```

The deterministic archive normalizes owners, modes and timestamps and rejects
weights, ONNX, OM, binary tensors, shared libraries, image lists, large files,
absolute developer paths and board-address markers.

Expected public source artifacts are one canonical Python sdist, one wheel, the
SBOM and checksums. The custom source scanner runs in `--check-only` mode as a
privacy/portability gate, avoiding a second near-duplicate source tarball. The
release workflow fixes `SOURCE_DATE_EPOCH`, runs the tests and emits an SPDX
2.3 file-level SBOM. It regenerates `SHA256SUMS` through a temporary file, atomically
replaces the previous checksum list, and immediately runs `sha256sum -c` from
the repository root. Re-running the step therefore cannot checksum the old
checksum list or create a self-referential entry. No proprietary or
model-derived payload is needed to run CI.

## Board runtime archive

The executor source and Makefile are maintained in `app/native/`; board Python
sources and the direct demo are maintained in `app/src/` and `app/chat.sh`.
Build the qualified AArch64 executor, then assemble the single runtime asset:

```bash
make -C app/native \
  SDK_ROOT=/path/to/SS928/sdk/smp/a55_linux/mpp/out \
  CC=aarch64-mix210-linux-gcc
pico-minicpm5 release runtime \
  --executor app/bin/pico_persistent_acl_executor.aarch64 \
  --out artifacts
```

The packager verifies the AArch64 ELF, size and frozen SHA256 from the v0.1.0
manifest. The archive has one canonical `app/` tree and embeds its own
`SHA256SUMS`. Executor C, Makefile, binary and `chat.sh` are not uploaded as
duplicate standalone Release assets; the compiled binary exists only inside
the runtime archive.

## Local derived-model release

```bash
pico-minicpm5 release assemble \
  --models work/om --model-dir work/model \
  --qualification qualification.json --out artifacts/model-release
pico-minicpm5 release verify artifacts/model-release
```

For hashes other than the frozen accepted v0.1.0 candidate, a passing explicit
qualification is mandatory. That qualification contains independently scored
prefill, decode and vocabulary-head evidence, all bound to the exact OM hashes
and one ATC build-manifest hash. Each score must also carry a matching
`pico.minicpm5.runtime-capture.v1` lineage manifest generated immediately after
its libinstsim or SS928 run. Capture records are local hash-based audit
evidence, not cryptographic execution proofs or remote attestation. The full
build/capture/score/qualification contract is fixed at ctx1024. The bundle
contains three OM files, the derived FP16 embedding, tokenizer and manifests.
It deliberately excludes ATC/DDK, runtime libraries and custom-op `.so` files.
The separately assembled runtime archive carries the open application and its
qualified prebuilt executor, but no proprietary runtime libraries.

Before publishing a derived-model bundle, confirm both the pinned model license
and the redistribution terms for outputs produced by the locally installed
toolchain. Source release and model release are separate decisions.
