# Native prefill residency and MMZ closure

[中文](NATIVE_PREFILL_RESIDENCY.zh-CN.md)

This document owns only the residency, KV ownership and model-switching
contract for publishing S16/S32/S128. Numeric qualification remains defined by
[`NATIVE_PREFILL_CLOSURE_WORKFLOW.md`](NATIVE_PREFILL_CLOSURE_WORKFLOW.md).

Status labels are strict: **PASS** means source-contract or recorded board
evidence exists, **CANDIDATE** means the route is implementable but has no
qualified wide-block board result, and **BLOCKED** means it must not activate.

## Decision

Do not keep three independent approximately 687 MB, 24-layer block OMs beside
the existing three base handles.

Use two stages:

1. **Bring-up candidate:** keep decode and head resident. After the position-0
   bootstrap, unload `prefill.om` and expose one manifest-pinned lazy-wide slot.
   Load only the requested S128, S32 or S16 OM. Decode owns the canonical KV;
   copy its used prefix into the active block and publish only new rows back
   through executor opcode 6.
2. **Recommended Agent release candidate:** one static-S128 carrier with a
   `valid_len` dispatcher for exactly 16, 32 or 128. It has a fixed maximum
   descriptor and shares weights inside one graph. Native branches must skip
   unused M16 groups. This is not ACL dynamic shape and remains blocked until
   runtime-scalar ingress, numeric, token and PMU gates pass.

Putting several GraphDefs into one OM and selecting an entry at runtime is not
available in the current stack. `svp_acl_mdl_execute` selects only a `model_id`,
the SDK exposes no graph selector, and `PackPicoOm.build_om()` assembles one
parameter/instruction/item region. Existing merged containers are one composed
execution graph, not independently callable entries.

## Evidence and present limits

- `app/native/pico_persistent_acl_executor.c` allocates separate OM, input and
  output buffers for every model in `load_om_source()` and `create_dataset()`.
- Models load before the ready frame and unload only during process cleanup.
  The protocol has no load/unload operation, although the ACL API provides
  `svp_acl_mdl_load_from_mem()` and `svp_acl_mdl_unload()`.
- `src/pico_minicpm5/compiler/atc.py` always supplies a static `--input_shape`.
  This SDK exposes dynamic batch, HW and total-T APIs, not an arbitrary sequence
  dimension. Its ATC configuration disables `dynamic_batch_size` for ND input.
- Fixed-trip native branch/loop component evidence exists in the integration
  worktree, but runtime `valid_len` ingress into that control path does not.
- Executor output-to-input chaining, FP32-to-FP16 scatter, same-model input
  snapshots and input-to-input copy exist as local protocol contracts. Shared
  input-buffer binding does not.
- Current snapshot limits are 64 MiB per slot and 128 MiB in total, and a
  snapshot is bound to its original model index.

## MMZ lower bound

The current executor allocates the exact OM file size with
`svp_acl_rt_malloc_cached()` before model load. OM source bytes are therefore a
safe allocation lower bound; descriptor datasets and runtime-private memory are
additional. A lower bound above available MMZ is conclusive, while one below it
still requires clean-board admission.

| Item | Bytes | GiB |
|---|---:|---:|
| decode OM | 686,997,372 | 0.640 |
| position-0 prefill OM | 686,999,901 | 0.640 |
| head OM | 202,651,666 | 0.189 |
| base three handles | 1,576,648,939 | 1.468 |
| representative 24L wide OM | 687,076,012 | 0.640 |
| recorded clean/post-exit MMZ | 2,896,191,488 | 2.697 |

Consequences:

- base plus three wide OMs is 3.388 GiB, already **707.3 MiB over MMZ** before
  any IO allocation: **BLOCKED**;
- base plus one wide OM is 2.108 GiB and leaves 603.2 MiB: a board-admission
  candidate, not a pass;
- adding a 256 MiB reserve leaves 347.2 MiB before IO/runtime-private memory;
- decode + head + one wide OM is 1.468 GiB, making replacement of the bootstrap
  prefill handle the preferred lazy residency set.

At C4096 one packed FP16 cache kind is 50,319,360 bytes and K+V is
100,638,720 bytes (95.98 MiB). Until shared binding passes, admission must price
one cache allocation per model.

## Static S128 plus `valid_len`

Arbitrary dynamic sequence shape is blocked. The candidate keeps maximum S128
physical inputs/outputs and dispatches before any width-dependent load or
compute:

```text
valid_len=16  -> 1 M16 group
valid_len=32  -> 2 M16 groups
valid_len=128 -> 8 M16 groups
```

The carrier must reject every other value. Instruction trace must show no DLD,
compute, DSTR or KV publication for skipped groups. Poisoning invalid rows must
not change hidden state or the next strict-S1 token. Board PMU must report
executed group counts 1/2/8 and `T16 < T32 < T128`; a linear latency ratio is
not required because weight bandwidth creates a fixed floor.

The current per-width qualification ABI (v2 development and v3 release) cannot
represent this shared carrier because its exact publisher ABI is W-specific.
A carrier needs a later schema with
`physical_width=128`, explicit `valid_width=16/32/128`, and either opcode-6
source-stride metadata or a proven compact publisher before MMZ is counted once.

## Lazy load/unload

Add one lazy slot whose artifact paths and SHA256 values are registered by a
trusted activation manifest. Requests select only a width, never a path. A
switch waits for the synchronous executor to become idle, unloads the old slot,
invalidates slot-generation state, loads the qualified artifact and returns its
exact descriptor plus load latency.

Recorded evidence is not an isolated wide-load benchmark: 904.5 MB across 52
handles reached ready in 3.91/4.36 seconds, while the 1.576 GB three-handle demo
shows about 6.4 seconds. Measure 30 `S128 -> S32 -> S16 -> empty` cycles and
record p50/p95 load and unload, model-id reuse and MMZ before/after. A width is
eligible only when the complete cost wins:

```text
load + prefix copy + execute + unload < qualified lower-width route
```

Any model-id leak, MMZ growth, descriptor drift, error 500004 or next-token
change disables lazy-wide and falls back to strict S1.

## Canonical KV, input copy and aliasing

Decode input slots 3/4 are the sole canonical K/V. Before a block executes,
copy `[0,start)` from decode to the active block. Opcode 6 scatters only the
new `[start,start+W)` FP32 publisher rows back to decode. A later block copies
from decode again, never from the previous block.

The implemented opcode-9 input-copy record names source/destination model and
input indices, offsets and length. The executor drains and validates every
record and invalidates all cached sources before copying any byte, uses
`memmove` within one buffer and `memcpy` across buffers, then flushes all
destinations. It intentionally has no model-table generation guard; callers
must rebuild records after a process or phase transition.
Packed cache requires 48 K plus 48 V channel records:

```text
row_bytes      = 128 * 2
channel_stride = (context - 1) * row_bytes
offset(c)      = c * channel_stride
length(c)      = start * row_bytes
```

Shared buffer binding is a later optimization. The ACL dataset API accepts an
address, but the current executor unconditionally frees every dataset buffer;
naive aliasing would double-free. Add ownership/reference counting first, then
prove exact size, default stride, cache flush semantics and unload ordering.
Until that gate passes, price and allocate separate caches.

## Snapshot policy

Snapshots protect only canonical decode KV. Current host-heap snapshots are not
cross-model copies. A 643-token Agent prefix uses about 15.1 MiB and fits the
existing single-slot path. One slot can hold at most 2730 C4096 tokens of K+V.
For a full C4096 cache, use a transactional pair: one approximately 48 MiB K
snapshot and one approximately 48 MiB V snapshot. Validate both before restore;
on partial failure invalidate resident-token metadata and execute nothing.

Full C8192 K+V is approximately 192 MiB and exceeds the present 128 MiB total
snapshot budget, so it is blocked under this contract.

## Release gates

1. Build and qualify S16, then S32, then S128 serially; simultaneous residency
   is not required for compiler qualification.
2. Use the implemented input-to-input copy and compare block outputs
   byte-exactly with a full host-payload baseline.
3. Add the manifest-pinned lazy slot and pass 30-cycle MMZ/model-id/latency tests.
4. Run the complete `S128 -> S32 -> S16 -> S1` route with canonical KV,
   block-to-block, block-to-S1 and snapshot restore exactness.
5. Build the static-S128 dispatcher and pass runtime-scalar, trace, PMU,
   numeric and board-token gates for all widths.
6. Replace lazy switching only after the universal carrier passes. Shared cache
   aliasing is optional; input-copy remains the correctness fallback.

The local protocol check is reproducible with:

```bash
cd release_work/pico-minicpm5
make -C app/native contract-check
```

Its current report has `resident_scatter_f32_to_f16=true`,
`resident_scatter_record_bytes=48`, `resident_scatter_validate_all=true`,
`resident_scatter_flush_failure_fatal=true`,
`resident_input_snapshots=true`, `resident_input_copy=true`,
`resident_input_copy_opcode=9`, `resident_input_copy_record_bytes=44`,
`resident_input_copy_generation_guard=false`, `self_test=true` and
`model_execution=false`. It proves protocol encoding only, requires records to
be rebuilt after a process/phase change, and is not a libinstsim or Hi3403
wide-block execution gate.
