# End-to-end pipeline

[中文](PIPELINE.zh-CN.md)

## 1. Fetch and freeze the source model

```bash
pico-minicpm5 model fetch --local-dir work/model
pico-minicpm5 model verify --model-dir work/model --full-hash
```

The downloader calls the supported `hf` CLI with an immutable revision. The
verification step checks model geometry, config/index hashes, all expected
weight symbols and shapes, the contiguous BF16 safetensors payload and,
optionally, the complete 2.16 GB shard hash.

## 2. Build the float reference

```bash
pico-minicpm5 reference capture \
  --model-dir work/model --out work/reference --context 1024 --dtype float64
```

For every prompt position and all 24 layers this writes:

```text
posP/layer_in_LL.f32.bin
posP/layer_out_LL.f32.bin
posP/k_cache_LL.f16.bin
posP/v_cache_LL.f16.bin
posP/logits.f32.bin
```

The default seven-token fixture is intentional: scoring position 5 needs
position 6's cache to observe the row appended at position 5.
`logits.f32.bin` is the dense vocabulary-head reference for the final hidden
at that same position.

## 3. Regenerate ATC calibration inputs

```bash
pico-minicpm5 reference calibrate \
  --reference work/reference --family decode --out work/calibration/decode
pico-minicpm5 reference calibrate \
  --reference work/reference --family prefill --out work/calibration/prefill
```

Both families use the same graph ABI but different clip contracts. Decode is
calibrated for positions `>=1`; prefill is calibrated for position zero. Mask
and RoPE samples cover the broader static ctx1024 operating domain.

Important: the accepted 2026-08-09 OM used a frozen historical donor corpus.
The open generator reconstructs the semantic calibration contract from the
official checkpoint but does not claim byte identity with that donor. A newly
compiled OM is a new candidate and must pass the numeric gate.

## 4. Export actual-weight ONNX

```bash
pico-minicpm5 onnx export-layers \
  --model-dir work/model --family both --context 1024 --out work/onnx/layers
pico-minicpm5 onnx export-head \
  --model-dir work/model --out work/onnx/head/model.onnx
```

Unlike a frontend coverage fixture, this exporter reads every learned tensor
from the pinned safetensors shard. A layer contains the accepted computation:

```text
ExtendRMSNorm → q/k/v projections → matrix RoPE → KV append
→ batched GQA → o projection → residual
→ ExtendRMSNorm → SwiGLU MLP → residual
```

Norm gamma is folded into following projections. Layer zero uses the qualified
pre-scale. Family-specific Clip nodes pin the activation-range contract.

## 5. Compose 24 layers

```bash
pico-minicpm5 onnx compose \
  --layers-dir work/onnx/layers/decode --family decode \
  --out work/onnx/decode/model.onnx
pico-minicpm5 onnx compose \
  --layers-dir work/onnx/layers/prefill --family prefill \
  --out work/onnx/prefill/model.onnx
```

Defaults enable packed cache inputs, packed current-row outputs and one
external file per initializer. The result has five public inputs and three
outputs regardless of layer depth.
Each composed model and the head must use its own initially empty directory;
all external initializers are separate files with offset zero.

## 6. Compile

ATC/DDK and `libsvp_custom.so` are external prerequisites:

```bash
pico-minicpm5 build \
  --decode-onnx work/onnx/decode/model.onnx \
  --prefill-onnx work/onnx/prefill/model.onnx \
  --head-onnx work/onnx/head/model.onnx \
  --calibration work/calibration --out work/om \
  --context 1024 \
  --atc /opt/atc/bin/atc --custom-ops-lib /opt/pico/libsvp_custom.so
```

The expected products are `decode.om`, `prefill.om` and `head_flat.om`.
`--backend fake` exercises the pipeline in public CI without vendor software;
its tiny PICO-marked files are test artifacts and never deployable models.

## 7. Qualify and release

Run the transformer under `libinstsim` or on SS928 and save its three raw
public outputs. Immediately after that same execution, create its capture
manifest and then score the captured files. First capture and score the
position-zero prefill run:

```bash
pico-minicpm5 capture \
  --runner libinstsim --role prefill --position 0 --context 1024 \
  --om work/om/prefill.om \
  --build-manifest work/om/build-manifest.json \
  --output work/prefill/out.0.bin \
  --output work/prefill/out.1.bin \
  --output work/prefill/out.2.bin \
  --report work/prefill/runtime-capture.json

pico-minicpm5 score \
  --output work/prefill/out.0.bin \
  --output work/prefill/out.1.bin \
  --output work/prefill/out.2.bin \
  --reference work/reference --position 0 --context 1024 \
  --om work/om/prefill.om \
  --build-manifest work/om/build-manifest.json \
  --capture-manifest work/prefill/runtime-capture.json \
  --report prefill-score.json
```

Then capture and score a position-1 decode run:

```bash
pico-minicpm5 capture \
  --runner libinstsim --role decode --position 1 --context 1024 \
  --om work/om/decode.om \
  --build-manifest work/om/build-manifest.json \
  --output work/decode/out.0.bin \
  --output work/decode/out.1.bin \
  --output work/decode/out.2.bin \
  --report work/decode/runtime-capture.json

pico-minicpm5 score \
  --output work/decode/out.0.bin \
  --output work/decode/out.1.bin \
  --output work/decode/out.2.bin \
  --reference work/reference --position 1 --context 1024 \
  --om work/om/decode.om \
  --build-manifest work/om/build-manifest.json \
  --capture-manifest work/decode/runtime-capture.json \
  --report decode-score.json
```

The command exits nonzero unless every published tensor cosine is strictly
greater than `0.98`. Score both position-zero prefill and a position `>=1`
decode execution.

Execute `head_flat.om` using the logical final hidden from one of those exact
transformer positions and a residual file containing exactly 1536 FP32 zeros.
The head output is a dense FP32 logits file. Capture the actual two input files
and output file from that run before scoring:

```bash
pico-minicpm5 capture \
  --runner libinstsim --role head_flat --position 1 --context 1024 \
  --om work/om/head_flat.om \
  --build-manifest work/om/build-manifest.json \
  --input hidden=work/head/pos1/next_hidden.f32.bin \
  --input residual=work/head/pos1/residual_zero.f32.bin \
  --output work/head/pos1/logits.f32.bin \
  --report work/head/pos1/runtime-capture.json

pico-minicpm5 score-head \
  --output work/head/pos1/logits.f32.bin \
  --reference work/reference --position 1 --context 1024 \
  --om work/om/head_flat.om \
  --build-manifest work/om/build-manifest.json \
  --capture-manifest work/head/pos1/runtime-capture.json \
  --hidden-input work/head/pos1/next_hidden.f32.bin \
  --residual-input work/head/pos1/residual_zero.f32.bin \
  --report head-score.json
```

Here `--position 1` means both inputs refer to the same model step: the head
consumes position-1 `next_hidden`, and the scorer loads
`work/reference/pos1/logits.f32.bin`. A head output produced from a different
position is not valid evidence. Qualification verifies that the head
`hidden`-input hash exactly equals the logical `next_hidden` hash in the
same-position transformer score. The residual hash must be the canonical 6144
zero bytes (1536 FP32 zeros). The head gate requires cosine strictly greater
than `0.98` and an exact top-1 token match.

Combine all three portable evidence reports:

```bash
pico-minicpm5 qualify \
  --prefill-score prefill-score.json --decode-score decode-score.json \
  --head-score head-score.json \
  --models work/om \
  --out qualification.json
```

Only a passing prefill + decode + head qualification may accompany new OM
hashes. Optional greedy evidence can be attached with `--greedy-report`; local
raw-output paths are never copied into the portable qualification.
Qualification verifies that every score names the expected role, is bound to
the same ATC `build-manifest.json`, and carries the exact SHA256 of its executed
OM. It then binds the exact SHA256 of all three release handles. Build,
capture and score commands use context 1024; qualification fixes this context
internally and rejects any evidence with a different value.

The `pico.minicpm5.runtime-capture.v1` file is local audit lineage, not a
cryptographic proof that a particular runtime executed the model. `capture`
hashes the files supplied by the operator and records the claimed
`libinstsim`/`ss928-board` runner; it does not provide signing, trusted
timestamps or remote attestation. Generate it immediately after the named run
so outputs cannot be accidentally mixed across executions.
