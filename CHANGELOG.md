# Changelog

[中文](CHANGELOG.zh-CN.md)

## Unreleased

- Board scripts `app/prepare_npu.sh`, `app/board_env.sh`, `app/install_board.sh`:
  Euler Pi factory Linux inserts `ot_pqp.ko`, which is mutually exclusive with
  `ot_svp_npu`. The installer unloads pqp, loads the NPU, writes `S91pico_npu`,
  and prints Chip / SDK / Hardware on SSH login.
- Project README adds an "Euler Pi factory image" section.
- Host script `app/install_python.sh`: factory Linux has no python3. It
  downloads pinned CPython 3.10.21 aarch64 plus tokenizers/jinja2 wheels into
  `/opt/pico-minicpm5/venv` on the board.
- Host script `app/install_runtime_lib.sh`: copy `libsvp_acl.so` and siblings
  from the licensed SDK into `/root/pico_default_smoke/lib`. Factory
  `/opt/lib/npu` is the Ascend stack and will not load the executor.
- The four board SVP ACL objects now ship in `app/lib/`. `chat.sh` prefers
  them, so users do not have to hunt the SDK.

## 0.2.0 - 2026-08-17

### The executor is reproducible, and every context got faster

- The board executor is now bound to `cef4edb2…`, built from the source in this
  repository with the sanctioned `aarch64-mix210-linux-gcc` 7.3.0 toolchain and
  verified byte-identical before the manifest was written
  (`docs/EXECUTOR_BUILD.md`). `v0.1.0` pinned a binary its own source could not
  rebuild.
- It retains the workspace input across executes instead of rewriting it, so a
  decode step saves one full workspace write. Measured on all three contexts in
  one session: ctx1024 `9.46 → 9.96 tok/s` (+5.2%), ctx4096
  `6.53 → 7.81 tok/s` (+19.7%), ctx8192 `4.59 → 6.03 tok/s` (+31.6%). All 48
  greedy-oracle tokens stay identical to each profile's qualified baseline.
- The saving is proportional to the retained workspace — 5.5 / 27.6 / 54.9 ms
  against 24.6 / 98.3 / 196.6 MiB — three points on one line, which is what
  makes it a mechanism rather than a coincidence.
- Prompt ingestion, the quantity TTFT is made of, is now measured on all three:
  `79.49` / `106.28` / `144.02` ms per prompt token. The ctx1024 figure agrees
  with an independent cold-prefill measurement to `0.10%`.

### ctx4096 ships qualified under the mixed prefill-window contract

- Runtime profiles carry `context.prefill_window`; ctx4096 and ctx8192 bootstrap
  position zero on the frozen ctx1024 `prefill.om` and share `head_flat.om`,
  byte for byte. Measured directly: their position-zero transformer times agree
  to `0.39 ms`.
- ctx4096 is `qualified`; ctx8192 stays `pending` on donor-zero-extend
  calibration, with the Chinese oracle, memory envelope and long-prompt items
  still open.

### A gate that was measuring the wrong thing

- ctx8192 had carried `eos: FAIL_STRICT_SEQUENCE_MISMATCH` against a sequence
  that was never traced to the reference. Re-derived from the pinned checkpoint
  in float64, the reference writes a terminal period and stops — which ctx8192
  reproduces exactly and ctx1024 and ctx4096 do not. The expectation had been
  recorded from the first artifact that ran. `release/contexts/strict-eos-oracle.md`.

### Documentation

- A recorded board agent session on both homepages as a self-contained animated
  SVG: a tool call returning in `1.9 ms`, a context rebase, and generation at
  `9.74 tok/s`. Waits play at `4.5x` with the factor on screen and the board's
  own clock on every frame.
- `docs/QUANTIZATION_CONTRACT.md`: how a `Clip` caps ATC's IFMR range search,
  why position zero needs its own calibration family (the layer-0 MLP branch,
  not attention — the attention explanation is documented as refuted), and one
  repeated rule retracted because its own proof artifact contradicts it.
- `release/perf/` gains TTFT, the per-context phase breakdown, the superseded
  pre-unification numbers and evidence hashes for each.

## 0.1.0 - 2026-08-09

### Runtime refresh - 2026-08-10

- Added the complete bilingual Hi3403 board application under `app/`, including
  `chat.sh`, runtime sources and the executor C/Makefile build path.
- Enabled resident packed K/V scatter and byte-exact fast RoPE/embedding
  preparation without changing the three accepted OM hashes.
- Improved measured ctx1024 throughput to `9.42–9.48 token/s` with 48/48
  greedy tokens exact, EOS and Chinese prompts passing.
- Consolidated the Release layout: executor source, Makefile and demo are no
  longer duplicated as standalone assets.
- Added a resident stdin REPL (`/help`, `/reset`, `/quit`) so repeated prompts
  reuse the three loaded handles; no-argument `app/chat.sh` enters it directly.
- Streamed REPL output token-by-token, raised the initial response limit from
  32 to 128 tokens and added `/max N` with explicit limit diagnostics.
- Enabled Agent fixed system/tool-prefix resident snapshots by default after a
  Hi3403 token-exact A/B. Restoring 137 prefix tokens took `1.76 ms` and cut a
  repeated 32-token request from `26.97 s` to `12.56 s` (`53.4%`).
- Qualified deterministic context rebase on Hi3403: two long-session runs both
  compacted 12 old tool turns from `2808` to `810` tokens and returned the same
  `[18655, 4569, EOS]`; a 643-token prefix hit made the repeat `4.75x` faster.
- Skip the vocabulary head and argmax on known prompt positions before the
  last input token. A token-exact board A/B reduced cold long-prompt latency by
  `19.89%` (`86.70→69.45 s`) and the resident repeat by `19.59%`
  (`18.17→14.61 s`).
- Added a fail-closed native-prefill scheduler for the future
  `S128 -> S32 -> S16 -> strict S1 tail` path, including absolute-range,
  per-width and invocation telemetry. The accepted bundle remains S1-only
  until each wider context-specific artifact passes all numeric and board gates.
- Added `qualify-prefill-block`, which fixes the strict `>0.98` policy and
  binds S16/S32/S128 OM lineage, full K/V-row publication, absolute-position
  captures, prefill-to-decode handoff, token exactness and board evidence.
- Started the closure workflow: implement and qualify S16→S32→S128, then
  dispatch S128→S32→S16→S1. S16 input RMSNorm, RMSNorm→QKV/RoPE, C4096
  attention, and attention→layer-tail have now passed separate same-graph
  execution gates. A bounded synthetic C256 append→attention gate passes, but
  all four real-calibration/held-out executions fail numeric qualification;
  the full-C4096/B16 join also remains blocked by hidden accuracy and physical
  K/V slot order. These are the first remaining full-layer blockers.
- Added fail-closed prefill activation and MMZ admission over the actual OM,
  build manifest, qualification hashes and physical publisher ABI. Invalid or
  over-budget widths retain strict S1 only.
- Generalized resident K/V scatter to contiguous W-row FP32→FP16 RNE and made
  canonical decode cache its only commit target. Other wide handles rebuild
  prefixes through opcode 9. Wide blocks remain disabled until a complete S16
  OM qualifies.
- Added resident input-to-input copy opcode 9. All 96 channel-wise K/V prefix
  records are validated and source-invalidated before any copy, followed by
  destination flush. Runtime exposes canonical decode-cache to wide-handle
  mirroring without activating an unqualified wide artifact.
- Qualified the same-graph resident S16 attention-to-layer-tail splice in
  libinstsim: private-TEMP bridge only, zero public INOUT, 4092/4092 event edges,
  and cosine `0.99969908`.
- Passed the bounded real S16 input-RMSNorm gate with 142/142 event edges and
  cosine `0.9999759688`. The workflow now joins the complete single layer in
  one OM; neither the single-layer nor 24-layer route is claimed complete yet.
- Passed the same-OM S16 `input RMSNorm→M16 QKV/RoPE` resident splice. The
  normalized hidden and QKV activation remain private TEMP with zero public
  INOUT; event SSA is `540/540`, and Q/K/V cosine is
  `0.9999167/0.9999570/0.9999097`.
- Independently qualified the layer-0
  `RMSNorm→QKV/RoPE→dual-K/V dynamic append` component at absolute starts
  `0,1,31,4080`: no deadloop, event SSA `940/940`, and Q/K/V-publisher cosine
  above `0.9999` at every position. Its 21-input/4-output ABI publishes only
  the truthful `[1,2,16,128]` layer slice of the future `[1,48,16,128]`
  publisher; full C4096 caches have no Report. OM/ONNX/qualification SHA256 are
  `6f31c8284ecee4809eda6692fccbc20f7a6208e1b48a6aff7b4220f9ccea5294`,
  `c91308a83df6cfc4f4f5732cd8f04214b0143860ff7e6b9d5e151524a8c65b48`, and
  `a9007f1bc58d383e8a637969fb7c9a5bec30cce134c69ed37a4f86af60c03862`.
  Sentinel FP16 byte-exact status remains diagnostic only. Attention,
  single-layer, all24-pack/all-24-layer, release-runtime, and production
  readiness all remain false; this publisher-only component does not establish
  an attention consumer.
- Composed all sixteen C256 online-attention states (branch 16 is the causal
  tail) and their merge into one C4096 OM. The selected public ABI is only
  Q/full-K/full-V; 32 cache Slice nodes stay in-graph to avoid the mapper's
  32-input limit without host repacking. libinstsim completes at cosine
  `0.9998773`, event SSA `32076/32076`, and zero public intermediates. Six
  nonfatal large-Jump warnings remain recorded. This gate does not yet contain
  the dynamic sixteen-row K/V append and is not a complete S16 layer or a
  24-layer qualification.
- Localized the first full-layer blocker to the dynamic C4096 H2 KV resident
  bridge. The cause was a duplicate Scatter/Gather VA epoch; NOPing only the
  Gather ACTVA/ALLOCBG makes all four positions execute at approximately
  `0.99999995` cosine with non-target rows byte-exact and SSA 102/102. The
  complete full-C4096/B16 same-OM attention join remains fail-closed.
- Independently passed the bounded synthetic
  `dual-K/V dynamic append→C256 causal-tail attention` gate. The OM has 21 user
  inputs, one public Report, zero public intermediates and a private Neg→Neg
  materialization seam before its real QK/AV consumers. libinstsim reaches
  cosine `0.9966713753`, maximum absolute error `0.0073841021`, no deadloop and
  event SSA `2568/2568`; no-update and no-sentinel negative controls are both
  worse than the reference. OM and qualification SHA256 are
  `90f757be0f2771eae1b1f4108279f1337f94e137f48746f05b15c2600c7ca35d` and
  `1c82512d6246e95a147cf550996817722cb9ffaedea76b710fe4029906f783e8`.
  Its qualification explicitly records `synthetic_domain=true`,
  `real_model_activation_calibration=false`, and
  `held_out_real_snapshot=false`; full-C4096/B16, single-layer, all-24-layer
  and production readiness remain false. Real activation calibration plus the
  full-C4096 boundary join is now the first blocker; S32/S128 have not started.
- Compiled the same bounded C256 graph with three content-bound real-model
  calibration rows and executed three calibration samples plus one isolated
  held-out sample. All four executions finish but fail numeric qualification:
  cosine is `0.902962`, `0.935589`, `0.930731`, and `0.912596`. The executable
  OM is `70f15965725b...3327362`; the independently audited execution
  qualification is `4c944cfb2f...c0c7c`. Evidence completeness is true while
  every owner-numeric, full-C4096, single-layer, all-24-layer, release and
  production readiness flag remains false.
- Narrowed the real C256 first bad boundary without over-attribution. Physical
  layout checks reject NCHW/head-query reinterpretation, W4 and NC1HWC0; an
  independent terminal Neg→Neg variant is byte-identical to the original
  score and remains at cosine `-0.070998`. The first proven bad value is the
  materialised scaled panel-0 score before softmax, not yet a specific MatMul
  or K-quantisation cause.
- Extended the serialized post-Gather bisection with one green and one red
  boundary: the full packed-panel terminal reaches cosine `0.9999994531`, but
  the K-half Crop terminal reaches only `-0.0126486001` with maximum error
  `265.21875`. The physical dataflow simultaneously changes from Gather-FP32
  to Gather-S16/Crop-S16, so the first bad interval is recorded as
  `Gather private-S16 materialisation → K-half Crop`, not as a proven Crop
  root cause. The gate stops `key_panel0`, raw QK, S32 and S128.
- Characterized and independently audited the layer-0 full-C4096 boundary
  join. Main OM `4717e207...d048` closes event SSA `39685/39685`; K/V cosine is
  `0.999958`/`0.999910`, but `next_hidden` is only `0.956954`. Diagnostic OM
  `ffe6da00...793f` isolates attention at `0.995993`; the tail on that hardware
  attention reaches `0.999496`, proving attention error amplification rather
  than a broken private tail seam. The main physical ABI is `V0/K1/H2`, not
  `K0/V1/H2`; no qualification was issued and S32/S128 remain not started.
- Materialized and closed-set audited the four-sample fulljoin reference
  dataset: manifest `b93cfbae...f044`, 134 files/113,855,374 bytes, 21
  three-row calibration image lists with no held-out row, and CPU/ORT minimum
  full-layer cosine `0.999999999998609`. Only dataset readiness is true; owner
  ATC/numeric and every upper readiness flag remain false.
- Made opcode 6 publication transactional: drain and validate every scatter
  record, invalidate every source, convert, and flush every destination before
  ACK. A flush failure terminates the executor instead of continuing with
  potentially incoherent resident cache state.
- Upgraded the strict-S1 release anchor to a dual-route v4 contract: the
  position-zero bootstrap OM, steady canonical decode OM, actual head OM,
  embedding, their descriptors, imported protocol runner and executor are
  independently bound. Token-exact evidence hashes exact little-endian uint32
  token-ID sequences and needs no tokenizer identity. With an activation
  manifest, startup rehashes this complete live route and every registered wide
  OM immediately before process spawn, including when no wide handler exists;
  documentation explicitly requires a trusted read-only deployment tree and
  records the residual path-open race instead of claiming inherited-fd safety.
- Made resident snapshot restore validate every range before mutation and flush
  every cached destination before ACK. Any native or host restore failure now
  poisons and tears down the complete resident session.

- Initial open-source pipeline from pinned MiniCPM5-1B checkpoint to real-weight
  layer ONNX, packed 24-layer prefill/decode ONNX, external ATC compilation and
  reproducible three-handle release manifests.
- Captures the accepted Hi3403 ctx1024 qualification without redistributing
  weights, proprietary SDK components or board binaries.
- Documents the `runtime-capture.v1` lineage step required between each
  libinstsim/Hi3403 execution and strict transformer/head scoring, including
  same-position hidden and canonical zero-residual binding for the head.
