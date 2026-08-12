# pico-minicpm5

[中文说明](README.zh-CN.md) · [Board demo](app/README.md) · [板端 Demo](app/README.zh-CN.md)

`pico-minicpm5` turns the pinned
[`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B)
checkpoint into a reproducible Hi3403/PICO deployment:

```text
Hugging Face checkpoint
        │ verify revision, config, index and safetensors contract
        ▼
24 real-weight decoder-layer ONNX graphs + vocabulary-head ONNX
        │ prefix every layer, chain hidden by SSA, share mask/RoPE
        ▼
one prefill ONNX + one decode ONNX with packed K/V (5 public inputs / 3 outputs)
        │ ATC/DDK supplied by the user
        ▼
prefill.om + decode.om + head_flat.om
        │ manifest, checksums and numeric qualification
        ▼
three-handle Hi3403 release bundle
```

The production merge is **graph composition before compilation**. It is not
byte concatenation of independently compiled OM files. The historical binary
linker is deliberately excluded from the default pipeline and is documented
only as an experimental, fail-closed recovery path.

## Status

The frozen `ctx1024` three-handle candidate was accepted on an Hi3403 board:

- prefill minimum public-output cosine: `0.996646`;
- decode minimum public-output cosine: `0.998023`;
- `48/48` greedy tokens matched the official-checkpoint FP64 oracle;
- EOS and Chinese text paths passed;
- optimized resident-K/V runtime: `9.42–9.48 token/s` at
  `105.5–106.1 ms/token`, approximately `1.91x` the accepted 49-handle
  baseline;
- prompt-only head suppression passed a token-exact board A/B and reduced a
  rebased 810-token cold request from `86.70 s` to `69.45 s` (`19.89%`), while
  a 643-token resident-prefix hit reduced it further to `14.61 s`.
- The fail-closed native-prefill planner now implements the future
  `S128 -> S32 -> S16 -> strict S1 tail` policy and exposes its decision in
  request reports. Only S1 is enabled in the qualified release; see
  [the native prefill contract](docs/NATIVE_PREFILL_SCHEDULER.md).

These are Hi3403 measurements. They are not a claim that every Hi3403 product
configuration has been qualified. The upstream checkpoint advertises a much
longer context; this release contract is intentionally fixed at `1024`.

## Deploy the prebuilt Hi3403 demo

Release [`v0.1.0`](https://github.com/GitBubble/pico-minicpm5/releases/tag/v0.1.0)
contains the accepted three-handle deployment: `prefill.om`, `decode.om`,
`head_flat.om`, the token embedding and tokenizer, plus a small runtime archive
with the complete `app/` board application and resident AArch64 executor. The
executor C source and Makefile are archived under `app/native/` in both this
repository and the runtime archive; they are not duplicate standalone Release
assets. The three OM files correspond to position-0 prefill, recurrent decode
and the vocabulary projection head respectively.

Download and arrange the files on the host:

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

Copy the assembled directory to the board and start the demo:

```bash
tar cf - . | ssh root@BOARD_IP \
  'mkdir -p /opt/pico-minicpm5 && tar xf - -C /opt/pico-minicpm5'

ssh root@BOARD_IP \
  '/opt/pico-minicpm5/app/chat.sh'
```

The accepted board image provides the licensed runtime libraries under
`/root/pico_default_smoke/lib`; they are intentionally not redistributed in
this repository or release. Override `TOKENIZERS` if the board's `tokenizers`
Python package is installed outside the normal Python environment. The runtime
archive keeps the compiled executor in `app/bin/` and its rebuildable source
and Makefile in `app/native/`.

If the release files are already present on the board, skip all host-side
steps and follow [`app/README.md`](app/README.md). The shortest board command
is:

```bash
cd /opt/pico-minicpm5
./app/chat.sh       # plain conversational REPL
./app/agent.sh      # tool-calling agent
```

`chat.sh` starts a colour-aware conversational REPL using the official
MiniCPM5 chat template without tools. `agent.sh` starts
the resident agent at the default `ctx1024`, using the
official MiniCPM5 `<tools>/<function>/<tool_response>` protocol. It includes
workspace read/search/git tools plus approval-gated write/shell tools, a
MiniCPM ASCII pet, timed planning/tool feedback and streaming final answers.
Commands include `/tools`, `/think on|off`, `/permissions`, `/context`,
`/clear`, `/max N` and `/quit`. Thinking is off by default; start with
`./app/agent.sh --thinking` or toggle it without reloading the models. The two
entry points use the same three resident OM handles but remain
separate applications. For a one-shot completion use
`./app/chat.sh --prompt 'The capital of France is' --max-new 16`.
The explicit `--prompt` and `--interactive` options retain the legacy raw-text
completion mode for compatibility.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[hub,onnx,reference,dev]'

# 1. Fetch the exact checkpoint. hf replaces the old huggingface-cli command.
pico-minicpm5 model fetch --local-dir work/model
pico-minicpm5 model verify --model-dir work/model

# 2. Capture float reference/calibration data from the official checkpoint.
pico-minicpm5 reference capture \
  --model-dir work/model --out work/reference --context 1024
pico-minicpm5 reference calibrate \
  --reference work/reference --family decode --out work/calibration/decode
pico-minicpm5 reference calibrate \
  --reference work/reference --family prefill --out work/calibration/prefill

# 3. Export actual-weight layer graphs and the dense vocabulary head.
pico-minicpm5 onnx export-layers \
  --model-dir work/model --out work/onnx/layers --context 1024
pico-minicpm5 onnx export-head \
  --model-dir work/model --out work/onnx/head/model.onnx

# 4. Compose the two independently calibrated 24-layer graph families.
pico-minicpm5 onnx compose \
  --layers-dir work/onnx/layers/decode --family decode \
  --out work/onnx/decode/model.onnx \
  --pack-input-kv --pack-output-kv --external-data
pico-minicpm5 onnx compose \
  --layers-dir work/onnx/layers/prefill --family prefill \
  --out work/onnx/prefill/model.onnx \
  --pack-input-kv --pack-output-kv --external-data

# 5. Compile with a locally installed/licensed ATC/DDK and custom-op library.
pico-minicpm5 build \
  --decode-onnx work/onnx/decode/model.onnx \
  --prefill-onnx work/onnx/prefill/model.onnx \
  --head-onnx work/onnx/head/model.onnx \
  --calibration work/calibration --out work/om \
  --context 1024 \
  --atc /path/to/atc \
  --custom-ops-lib /path/to/libsvp_custom.so

# 6. Run all three OMs with libinstsim or the Hi3403 runtime. Immediately after
# each completed run, capture the exact files from that run before scoring them.
pico-minicpm5 capture \
  --runner libinstsim --role prefill --position 0 --context 1024 \
  --om work/om/prefill.om \
  --build-manifest work/om/build-manifest.json \
  --output work/prefill/out.0.bin --output work/prefill/out.1.bin \
  --output work/prefill/out.2.bin \
  --report work/prefill/runtime-capture.json
pico-minicpm5 score \
  --output work/prefill/out.0.bin --output work/prefill/out.1.bin \
  --output work/prefill/out.2.bin --reference work/reference \
  --position 0 --context 1024 --om work/om/prefill.om \
  --build-manifest work/om/build-manifest.json \
  --capture-manifest work/prefill/runtime-capture.json \
  --report work/prefill-score.json

pico-minicpm5 capture \
  --runner libinstsim --role decode --position 1 --context 1024 \
  --om work/om/decode.om \
  --build-manifest work/om/build-manifest.json \
  --output work/decode/out.0.bin --output work/decode/out.1.bin \
  --output work/decode/out.2.bin \
  --report work/decode/runtime-capture.json
pico-minicpm5 score \
  --output work/decode/out.0.bin --output work/decode/out.1.bin \
  --output work/decode/out.2.bin --reference work/reference \
  --position 1 --context 1024 --om work/om/decode.om \
  --build-manifest work/om/build-manifest.json \
  --capture-manifest work/decode/runtime-capture.json \
  --report work/decode-score.json

# Feed decode's logical FP32 next_hidden at position 1 and an exactly-zero
# 1536-element FP32 residual to head_flat.om, then save its FP32 logits. The
# paths below must be the actual input/output files used in that same head run.
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
  --report work/head-score.json

# 7. Bind all three numeric gates to this exact ATC-built OM set, then release.
pico-minicpm5 qualify \
  --prefill-score work/prefill-score.json \
  --decode-score work/decode-score.json \
  --head-score work/head-score.json \
  --models work/om \
  --out work/qualification.json
pico-minicpm5 release assemble \
  --models work/om --model-dir work/model \
  --qualification work/qualification.json --out artifacts/release
pico-minicpm5 release verify artifacts/release
```

The head run must use the logical final hidden tensor produced for the same
`--position` as its logits reference. For example, the position-1 decode
hidden is scored against `pos1/layer_out_23.f32.bin`, then passed to
`head_flat.om`; the resulting logits are scored against
`pos1/logits.f32.bin`. Mixing a position-0 hidden with position-1 logits is an
invalid qualification even when tensor shapes happen to match. Qualification
also requires the head capture's `hidden` hash to equal the matching
transformer score's logical `next_hidden` hash, and requires `residual` to be
exactly 1536 FP32 zeros.

`capture` creates a local, auditable lineage record: it hashes the OM, ATC
build manifest and the named files from one completed `libinstsim` or
`ss928-board` run. Generate it immediately after that run, before files can be
replaced or mixed with another execution. It is not a cryptographic proof of
hardware execution or remote attestation; the operator remains responsible
for the truth of `--runner`. Every score command verifies this capture and
records the executed OM and build-manifest SHA256. `qualify` rejects reports
from another OM, another build, a fake compiler, or an unbound raw-output
capture.

Build, capture and scoring are explicitly fixed to `--context 1024` in this
release. `qualify` has no context override and rejects evidence from any other
context.

Only the three frozen hashes listed in `release/v0.1.0/release-manifest.json`
may reuse the recorded board verdict. Every newly compiled OM set requires an
explicit passing qualification file; regenerated calibration is intentionally
not treated as byte-identical to the historical accepted donor corpus.

Each external-data ONNX uses a dedicated empty directory. This preserves the
qualified one-file-per-initializer, offset-zero contract and prevents decode,
prefill or head exports from silently appending into one another's tensor data.

`model fetch` pins revision
`4e9de7a0778dc1c362e983e6858f0e77542cbdca`. Authentication, if ever
required, comes from `HF_TOKEN`; tokens are never accepted on the command line
or written to manifests.

## What is and is not in the source release

The source release contains the Python package, configs, schemas, tests and
documentation. It does not contain:

- checkpoint weights or weight-derived ONNX external-data files;
- OM binaries, token embeddings or copied tokenizer assets;
- ATC/DDK/libinstsim, Docker images, custom-op shared objects or board runtime
  libraries;
- owner-controlled golden models, mapper dumps, calibration image lists,
  board addresses, credentials or raw logs.

The model card currently declares Apache-2.0, but generated model artifacts
remain separately identified as derived model assets. Users must also satisfy
the redistribution terms of their ATC/DDK and runtime installation.

See [docs/PIPELINE.md](docs/PIPELINE.md),
[docs/OM_COMPOSITION.md](docs/OM_COMPOSITION.md) and
[docs/RELEASE.md](docs/RELEASE.md) for the exact build/release contracts. The
[Agent routing and runtime-context profile design](docs/AGENT_ROUTING_AND_CONTEXT_PROFILES.md)
defines hybrid routing and the ctx128/1024/4096/8192 capability matrix; the
[native prefill scheduler](docs/NATIVE_PREFILL_SCHEDULER.md) defines the
`S128 -> S32 -> S16 -> S1` TTFT path and its activation gates.

## Development

```bash
pytest
pico-minicpm5 doctor
pico-minicpm5 release source --out artifacts
```

Public CI uses a tiny synthetic Llama fixture and a fake compiler. Checkpoint
download, ATC compilation, libinstsim and Hi3403 execution are opt-in local or
self-hosted jobs and must never upload private SDK material to a public cache.
