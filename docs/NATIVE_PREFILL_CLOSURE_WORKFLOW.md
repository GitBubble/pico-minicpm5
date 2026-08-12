# Native Prefill S128→S32→S16→S1 Closure Workflow

[中文](NATIVE_PREFILL_CLOSURE_WORKFLOW.zh-CN.md)

When a range starts at position zero, runtime first executes one strict-S1
bootstrap. From position one onward dispatch is fixed to
`S128 → S32 → S16 → strict S1`, while
implementation and qualification are deliberately ordered `S16 → S32 → S128`.
S32 must match two qualified S16 calls; S128 must match four S32 and eight S16
calls. Strict S1 is permanent and is the only enabled path when evidence,
artifact identity, physical ABI, or MMZ admission is incomplete.

As of 2026-08-12, the largest-first scheduler, M16 Q/K/V producer, native
C4096 attention stage, bounded S16 tail execution, and the same-graph resident
attention-to-tail splice have component evidence.
The 97-tile tail closes 1880/1880 event edges and reaches libinstsim cosine
`0.9994710106` without deadloop (OM SHA256
`5007dfd19f7ea970d4bfd2913b5df131cc306dfadc82eb828771a202f5a9c839`). It
is the bounded public-boundary gate. The follow-up resident splice removes the
attention tensor from public I/O: all 61 bridge accesses are private TEMP,
public INOUT is zero, and the only Report is `next_hidden`. It closes 4092/4092
event edges, reaches libinstsim cosine `0.99969908`, and has no deadloop (OM
SHA256 `65de2000252cdcd584864f8b0033ba813d0c309dfa0de1ee5d23a22704c8c434`,
qualification SHA256
`4a7147f60df8dc8227e5bc103b4c5af775abeb0a9eb5d631e2667315f6009126`).
The complete layer, 24-layer graph, S32, and S128 remain fail-closed. M1 tail
evidence remains operator-family/topology authority, not S16 numeric evidence.

The input-RMSNorm gate now also passes: 142/142 event edges close, libinstsim
cosine is `0.9999759688`, and there is no deadloop or continue-event warning
(OM SHA256
`ecedc2ddd1247f981e043cbf893b2269f149627e58bbf7ed54644047f1275c8f`,
qualification SHA256
`165b0ba421a89173d91ec726fa5c4d216c713f8ad46d8d3551a3215929b3aeb1`).
That public RMSNorm boundary has now also been removed. A same-OM
`input RMSNorm → M16 QKV/RoPE` splice exposes only hidden and RoPE inputs;
normalized hidden and the QKV activation stay private TEMP with zero public
INOUT. Event SSA is `540/540`, Q/K/V cosine is
`0.9999167/0.9999570/0.9999097`, and the OM SHA256 is
`ab56ded310b7f7aaf2c376b4e64c5a47852e802ab808fc3d20557317bd90d01d`.

The next component gate now passes independently as well: layer 0
`input RMSNorm → M16 QKV/RoPE → dual-K/V dynamic append` executes as one OM at
absolute starts `0,1,31,4080`, with no deadloop and event SSA `940/940`. Its
public ABI is exactly 21 inputs (hidden, RoPE cosine/sine, K/V caches, and 16
absolute-position scalars) and four outputs (Q, compact K, compact V, and one
bounded packed witness). The K/V outputs are truthfully only the layer-0
publisher slice `[1,2,16,128]`, channel range `[0,2)` of the eventual release
publisher `[1,48,16,128]`; the full C4096 caches never cross a Report boundary.
OM SHA256 is
`6f31c8284ecee4809eda6692fccbc20f7a6208e1b48a6aff7b4220f9ccea5294`,
ONNX SHA256 is
`c91308a83df6cfc4f4f5732cd8f04214b0143860ff7e6b9d5e151524a8c65b48`,
and qualification SHA256 is
`a9007f1bc58d383e8a637969fb7c9a5bec30cce134c69ed37a4f86af60c03862`.

| absolute start | Q cosine | K publisher cosine | V publisher cosine |
|---:|---:|---:|---:|
| 0 | `0.9999173161` | `0.9999574123` | `0.9999096573` |
| 1 | `0.9999173215` | `0.9999573036` | `0.9999096573` |
| 31 | `0.9999172380` | `0.9999573519` | `0.9999096573` |
| 4080 | `0.9999169568` | `0.9999568232` | `0.9999096573` |

The fixed sentinel-row resident-FP16 byte-exact check is diagnostic only; the
qualification gate uses the recorded FP32 sentinel cosine/max-error bounds and
does not promote a byte mismatch into a false release claim. This artifact is
only a bounded component witness: `attention_ready`, `single_layer_ready`,
`all24_pack_ready`, `all_24_layers_ready`, `release_runtime_eligible`, and
`production_ready` all remain false.

An independently audited bounded synthetic gate now closes
`dual-K/V dynamic append → one C256 causal-tail attention consumer` in one OM.
Its physical ABI has 21 user inputs, one public Report (`attention_context`),
zero public intermediate INOUT writes and zero public intermediate Reports.
Gathered K/V panels cross a private Neg→Neg canonical-materialization seam
before the real QK/AV consumers; no host cache repack or public bridge is used.
libinstsim completes without deadloop at cosine `0.9966713753` and reference
maximum absolute error `0.0073841021`; event SSA closes `2568/2568`. The
no-update and no-sentinel negative controls remain materially worse than the
reference (`0.9594016617`/`0.0220721886` and
`0.9824994984`/`0.0106568751` cosine/max-error respectively), binding the
result to both the dynamic updates and sentinel rows. OM SHA256 is
`90f757be0f2771eae1b1f4108279f1337f94e137f48746f05b15c2600c7ca35d`;
qualification SHA256 is
`1c82512d6246e95a147cf550996817722cb9ffaedea76b710fe4029906f783e8`.

**Qualification boundary:** `synthetic_domain=true`,
`real_model_activation_calibration=false`, and
`held_out_real_snapshot=false`. Consequently
`dynamic_full_c4096_attention_ready=false`, `b16_dynamic_ready=false`,
`single_layer_ready=false`, `all24_layers_ready=false`, and
`production_ready=false`. This component is not an S16 release artifact.

The same graph has now been compiled with a content-bound three-row real-model
calibration set and executed against three calibration samples plus one
isolated held-out sample. The executable OM SHA256 is
`70f15965725b3dd2ac430af3c49b1c7fc88edbf16ea581afc3f9b269c3327362`.
All four runs finish and produce finite, non-zero outputs, but all four fail the
numeric gate: cosine/max-error are `0.902962/0.135358`,
`0.935589/0.117447`, `0.930731/0.163486`, and `0.912596/0.107355`.
The independently audited execution qualification SHA256 is
`4c944cfb2f413c723304fc1c76ee3823a66914eed7d511e542f58649c83c0c7c`.
It records complete evidence but keeps owner numeric, single-layer,
full-C4096, all-24-layer, release and production readiness false. Simple
input/QK-K/AV-V factor quantisation simulated on the CPU remains above
`0.999645` cosine, so the current evidence does not justify blaming one S8
factor or input absolute range; the earliest unproven boundary remains the
panel online-state calculation and is being narrowed with single-Report
diagnostics.

The first diagnostic stage is now fail-closed and more precise. The score
output descriptor is dense FP32 `[1,16,16,32]`/32768 bytes; descriptor-native
NCHW and the only descriptor-compatible head/query transpose both fail, while
W4 and NC1HWC0 are excluded by the physical byte contract. A separately
compiled terminal Neg→Neg variant produces output that is byte-identical to
the original score (`8982ab50...`), with the same cosine `-0.0709976621` and
maximum error `67.1187148`. Thus descriptor layout, the original Nop/Report
seam, and terminal materialisation are excluded. The first proven bad semantic
boundary is the materialised scaled panel-0 score before softmax; current
evidence still does not attribute it to the MatMul or K quantisation. The
layout and Neg→Neg execution qualification SHA256 values are
`e3f0c99a0456fc3a92c7dc71934fa91c8a3cf6eb6fecc894104a2daad0c9bb27`
and `43821356cd6529b9a9500dcb42bf78a1a0d84d3247ccda2a68e6bce203e5bb49`.

The subsequent serialized post-Gather bisection narrows this by one more
boundary. A variant that reports the complete
`packed_attention_panel0 [1,32,32,128]` is numerically green at cosine
`0.9999994531` and maximum error `0.015625`. The next variant, which adds only
the K-half Crop and reports
`key_panel0_packed_slice [1,16,32,128]`, immediately fails at cosine
`-0.0126486001` and maximum error `265.21875`. Its OM and execution
qualification SHA256 values are
`9b198f7582ffcb2fa7ff52564044d204d1852fd743a23fa808e7b0a877d4de28`
and `f6e7372474ae1e7b8a95d8cdf1b80c1c9c9b022ea8803f7fc375c658e9fe571f`.
This is not yet evidence that the Crop alone is defective: in the green
terminal graph the Gather publishes FP32 directly, whereas adding the Crop
causes the mapper to schedule both the Gather output and Crop output as
private S16 before the final Report converts to FP32. The first bad interval
is therefore `Gather private-S16 materialisation → K-half Crop`, not a proven
Crop root cause. The serial gate stops here; `key_panel0`, raw QK, S32 and
S128 have not run. The next bounded gate must use an independent
S16-materialisation A/B to separate the Gather quantisation domain from the
Crop/Report boundary.

The complete sixteen-state C4096 attention reduction is likewise one static
same-OM graph now: Q/full-K/full-V are its only public inputs, 32 cache Slice
nodes remain graph-local, all 16 C256 states and 15 merges are private, event
SSA is `32076/32076`, and libinstsim cosine is `0.9998773019` (OM SHA256
`d8cad715b344334b923026b9c26bccd20e696e01b170ddd1f0e77bce2fc86dd6`).
This is a full-cache attention gate, not a dynamic-append gate.

A full-C4096 layer-0 boundary join has also been executed and independently
audited, but it is a blocked characterization rather than a qualification.
The main OM SHA256 is
`4717e207dc536dbe758bcd3da30ba219b88eb9296a90e6e1ec389139a016d048`;
its global event SSA is `39685/39685`. K/V publishers remain numerically green
(`0.999958`/`0.999910` cosine), while `next_hidden` is only `0.956954`.
An independent four-output diagnostic OM
`ffe6da001272c6f21fdee4aea028da1cb203b3fa898bffacb0b3c77bc5ca793f`
shows attention-context cosine `0.995993`; the same tail evaluated on that
hardware attention reaches `0.999496`. Therefore the current error is the
attention error amplified by o_proj/MLP, not a broken private tail seam. The
main physical output order is also `V0/K1/H2`, not the release contract
`K0/V1/H2`. No release qualification exists and every boundary, single-layer,
all-24-layer, runtime and production readiness flag remains false.

The four-sample fulljoin reference dataset is now physically materialised and
closed-set audited. Its manifest SHA256 is
`b93cfbae41dad32ce35499b4a3ef826bbc5bbe8b83b59cc9b91449ff234df044`;
134/134 files and 113,855,374 bytes are registered, 84 raw inputs and 20
reference arrays are rehashed, and all 21 calibration image lists contain
exactly three rows with zero held-out rows. CPU/ORT full-layer references retain
minimum cosine `0.999999999998609`. The content-bound recipe SHA256 is
`9c20840249757c44e1f2ae043cf1c48a32c6d09bbe78a29ef72d5435ba52f8c5`.
The audit only sets `dataset_materialized_and_audited=true`; owner ATC, owner
numeric, single-layer, all-24-layer, release and production remain false. It
also records that three calibration rows do not guarantee a numeric fix.

The first full-layer blocker is now precise: first close the real-C256
`Gather-S16 → Crop` boundary above, then correct and qualify real-domain
attention accuracy, fix the physical K/V output-slot order, and rerun the
full-C4096/B16 same-OM boundary with the bound real calibration and held-out
samples. Only then may the workflow connect the already-green RMSNorm/QKV and
resident tail and aggregate 24 truthful layer slices into
`[1,48,16,128]`. The bounded H2 resident bridge also
proves the underlying Scatter/Gather execute contract at
positions `0,1,31,4095`: selected/witness cosine is approximately
`0.99999995`, non-target rows are byte-exact and event SSA is `102/102` (OM
SHA256 `76eeedd497b5e3cbfad74b5c10392cf3e031e50972eea7c2ae8a12f7fedb990d`).
Neither bounded component proves real-model calibration or the dynamic full
C4096/B16 attention boundary.

The workflow next closes real-domain attention accuracy and the full-C4096/B16
boundary join, then joins the already-green RMSNorm/QKV, attention and tail
components into one full layer, followed by 24 layers without reference-hidden
injection, S16→S1/decode handoff, and board token/boundary/performance gates. A
failure stops work at S16; S32 and S128 have not started.

Wide blocks publish contiguous channel-major K/V tensors as
`[1,48,W,128] FP32`. Executor opcode 6 converts complete per-channel blocks to
resident FP16 using RNE. The runtime validates exact source bytes, cache
descriptors, absolute offsets, and the context-1 boundary. Logical cache ABI
and publisher ABI are recorded separately.

Before a wide invocation, opcode 9 mirrors the valid canonical decode K/V
prefix into the target handle with 96 fixed records. The executor drains and
validates every record and invalidates every source before performing any copy,
then flushes all destinations. Model indices remain valid only for the current
executor model-table lifetime and must be rebuilt after a process/phase change.

`app/src/minicpm_prefill_activation.py` accepts release qualification v4 only.
It independently re-reads and verifies the OM, build manifest, runner,
executor, board-ready descriptor, every capture/workload/timing artifact, and
the complete baseline qualification/OM ladder. MMZ admission is derived from a
bound clean-board before/after observation. It rejects symlinks, path escape,
size/SHA drift and development-only qualification v2. An individual failure
disables that width and retains S1. Repeated residency groups are disabled and
each admitted width is charged independently.

The activation manifest must explicitly bind the live content-qualified S1
OM/build/runner/executor/descriptor identity; a boolean flag is insufficient.
All wide baseline chains must resolve to that same anchor, and the caller's
`base_resident_bytes` must be at least its measured S1 residency. An invalid
top-level S1 anchor rejects activation; an anchor mismatch or base underreport
disables wide blocks while retaining the verified S1 route.

The board application accepts the same live inputs through
`--prefill-activation-manifest`, `--available-bytes`,
`--base-resident-bytes`, and `--reserve-bytes`. At startup,
`minicpm_prefill_runtime.py` calls `load_activation` and then intersects its
result with typed wide-handle handlers registered by this process. The typed
dispatcher and fake transport closure now cover exact descriptor/publisher
binding, opcode-9 prefix restore, one wide execute, opcode-6 canonical row
publication, no-head intermediate blocks and final-hidden handoff. Any failure
poisons and terminates the resident session; it cannot continue as S1 without
a rebuild. The production registry remains empty and has no CLI injection
path, so qualified widths appear only in `qualified_widths` and do not enter
scheduler `enabled_widths`; production telemetry remains strict S1.

Mask row `j` exposes absolute K/V prefix `[0,start+j)` plus the final-column
current-token sentinel. A real qualified wide graph must dynamically append
earlier rows from the same block at their absolute slots before later rows
consume them. The runtime preflights the complete plan and forces the terminal
context position to S1 because the v1 opcode-6 publisher cannot write beyond
the `context-1` cache. These are software contracts verified with fakes, not a
claim that a wide OM exists.

Independent 24-layer S16/S32/S128 OMs are not assumed to fit simultaneously.
Release qualification v4 admits them independently only. A shared carrier
requires a future schema that binds one physical S128 descriptor and its valid-width
branches; otherwise release needs lazy-wide admission or a clean-board
measurement covering base models, all enabled blocks and the reserve. Serial
per-width build and qualification may proceed before that runtime gate closes.

See [native prefill residency and MMZ closure](NATIVE_PREFILL_RESIDENCY.md) for
the measured lower bound, lazy-wide bring-up, canonical KV/input-copy design,
and the static-S128 `valid_len` release candidate.

See [native-prefill release qualification v4](NATIVE_PREFILL_RELEASE_QUALIFICATION.md)
for the evidence index and explicit release CLI.

Run the release-side gates with:

```bash
cd release_work/pico-minicpm5
PYTHONPATH=src ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_prefill_schedule.py tests/test_prefill_runtime.py \
  tests/test_prefill_wide_dispatch.py \
  tests/test_prefill_blocks.py \
  tests/test_prefill_activation.py tests/test_board_repl.py
make -C app/native contract-check
```

The route is complete only after all three widths pass 24-layer numeric,
token, boundary, board, and performance gates; runtime dispatch and fallback
are observed; canonical scatter/input-copy/snapshot/handoffs are exact; MMZ failure is
fail-closed; and Agent/OpenClaw TTFT improves against a same-run strict-S1
control without generated-token regression.
