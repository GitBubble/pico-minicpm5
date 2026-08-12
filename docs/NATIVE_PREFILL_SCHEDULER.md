# Native multi-token prefill scheduler

[中文](NATIVE_PREFILL_SCHEDULER.zh-CN.md)

The staged implementation, qualification, MMZ-admission, and release gates are
defined in the [S128→S32→S16→S1 closure workflow](NATIVE_PREFILL_CLOSURE_WORKFLOW.md).

## Objective

MiniCPM currently ingests a new prompt one token at a time. Resident K/V,
fixed-prefix snapshots and prompt-only head suppression avoid unnecessary
replay and vocabulary projection, but every remaining new prompt token still
executes a transformer handle. True TTFT reduction therefore requires native
multi-token transformer artifacts.

The target policy is:

```text
S128 -> S32 -> S16 -> strict S1 tail
```

`S<N>` means that one qualified invocation consumes exactly `N` consecutive
prompt tokens and publishes all K/V rows needed by the following block or by
decode. It is not a different model and does not change the context capacity.

## Scheduling contract

The runtime covers `[resident_prefix_tokens, prompt_tokens)` greedily in the
canonical order above. Caller ordering cannot change the policy. S1 is
mandatory and guarantees exact coverage. If the range begins at position zero,
one S1 call is the mandatory startup bootstrap. For example, 433 tokens execute
as `1xS1(startup) + 3xS128 + 1xS32 + 1xS16`; a resident prefix at 643 followed by a prompt
ending at 810 becomes `1xS128 + 1xS32 + 7xS1`.

`app/src/minicpm_prefill_schedule.py` implements this fail-closed planner. Each
request report records its schema, absolute range, enabled widths, per-width
counts, invocation count and compact segments. The current qualified release
enables only S1, so adding the planner does **not** claim a TTFT improvement by
itself.

`app/src/minicpm_prefill_runtime.py` is the runtime registry between release
activation and this planner. The board CLI may load a v3 manifest at startup:

```bash
./app/chat.sh \
  --prefill-activation-manifest work/prefill/activation.json \
  --available-bytes "$MMZ_AVAILABLE" \
  --base-resident-bytes "$BASE_RESIDENT" \
  --reserve-bytes 268435456
```

The manifest and all three live MMZ values are an all-or-nothing tuple. With
no manifest, the exact default remains strict S1. An invalid top-level S1
anchor aborts before any model handle executes. The report distinguishes
`qualified_widths` from executable `enabled_widths`; the latter is what feeds
`prefill_schedule.enabled_widths`. The merged runtime now has a typed,
injectable wide-handler boundary, including exact width/context/model-index,
ready descriptor and publisher ABI checks. However, this release registers no
production handler and exposes no CLI injection switch because no complete
wide OM has passed the release gates. Therefore even a release-qualified
S16/S32/S128 is reported as handler-unavailable and the executable set stays
`[1]`; it is never run through S1 under a wide label.

The boundary is exercised only by fake-transport tests. A wide transaction
prepares `W` embedding rows, `W×context` mask rows and `W` absolute-position
RoPE rows, mirrors canonical decode K/V `[0,start)` into the wide handle with
opcode 9, performs exactly one wide execute, and publishes only rows
`[start,start+W)` back to canonical decode with opcode 6. The returned final
hidden is chained to the head with an exact byte-size match. Any prepare,
copy, execute or publish error discards the complete resident process; S1 may
be retried only after constructing a new session. The fake tests prove call
order, byte ranges and fail-closed state transitions, not OM correctness or
Hi3403 performance.

Because the canonical cache has only `context-1` writable rows and opcode 6
cannot partially publish a wide tensor, the complete request plan is
preflighted before any model execute. A terminal position at `context-1` is
replanned as strict S1, whose existing contract safely omits that unused final
K/V scatter.

Fixed-prefix snapshot positions are also hard scheduling boundaries. The
planner may restart largest-first ordering after a named boundary, but no wide
segment may cross it; the snapshot is therefore created at the exact fixed
token count on the first Agent request and restored safely on later requests.

## Artifact eligibility

A block width may be enabled only when its exact context-specific artifact has
passed all of the following:

1. OM and build-manifest hash binding;
2. input/output descriptor and physical-stride validation;
3. every public output cosine strictly greater than `0.98`;
4. byte-exact absolute-position mask and RoPE construction;
5. byte-exact publication of all `N` K/V rows for all 24 layers;
6. block-to-block and prefill-to-S1/decode handoff validation;
7. greedy token, EOS, Chinese/English and context-boundary gates on Hi3403;
8. measured TTFT benefit over the next smaller qualified family.

The physical role map is fixed, not inferred from equal tensor sizes: inputs
are embedding/mask/RoPE/K/V at slots `0/1/2/3/4`, and outputs are K/V/hidden at
slots `0/1/2`. Qualification independently regenerates every capture's FP32
mask hash from `(context,width,start)`, including prefix visibility, masked
future range and the `context-1` current-token sentinel.

Eligibility is context-specific: an S16 artifact qualified for ctx1024 does
not authorize the ctx4096 or ctx8192 profile. Missing, stale or mismatched
evidence disables that width and falls through to the next qualified family.

`pico_minicpm5.prefill_blocks` makes this policy machine-checkable. Steady S16
requires starts `1,15,16,31,32,255,256,643,context-16`; S32 requires
`1,31,32,127,128,643,context-32`; and S128 requires
`1,127,128,511,512,643,context-128`. Position zero remains on the strict-S1
startup quantization domain. Each K/V
publisher is contiguous `[1,48,16,128]` FP32, while opcode 6 converts it into
the logical FP16 resident-cache ABI. Final hidden is `[1,1536,1,1]` FP32. Each
capture binds physical descriptor, mask, RoPE and raw-output hashes and must
prove all 768 K/V rows per role, handoff, token exactness and a board pass.
Generate a release-eligible v3 record with:

```bash
pico-minicpm5 qualify-prefill-block-release \
  --evidence work/s16/release-evidence.json \
  --out work/s16/qualification.json
```

The threshold is fixed by policy and cannot be weakened from the CLI or the
evidence JSON. The shorter `qualify-prefill-block` command remains a
development-only v2 compatibility path and cannot activate a release. See
[native-prefill release qualification v4](NATIVE_PREFILL_RELEASE_QUALIFICATION.md).

## Native data path

A block consumes embeddings for `N` consecutive absolute positions, the
resident packed K/V prefix, causal-mask rows and the matching RoPE rows. The
24-layer graph computes the sequence together and publishes:

- the final-layer hidden row required at the last prompt position;
- `N` K rows and `N` V rows for each transformer layer;
- a resident-cache extent advanced by exactly `N` positions.

For query row `j`, the mask exposes absolute prefix `[0,start+j)` and the last
context column as the current-token sentinel. The wide graph—not the host—must
dynamically append earlier rows from the same block to absolute cache slots so
later query rows consume them under that mask. Publisher row `j` maps back to
canonical absolute row `start+j`.

Prompt-only head suppression remains valid: vocabulary projection is skipped
inside the block and runs only for the final known prompt position. A block
must never publish only its final K/V row, inject reference hidden states, or
use host-visible output staging as an implicit layer-to-layer bridge.

## Activation order

1. **S16 first:** close one byte-exact native block, then validate its handoff
   to the existing S1/decode path over the complete steady-position matrix
   above; position zero remains strict S1.
2. **S32:** compose or compile a dedicated family only after S16 is qualified;
   compare S32 against two consecutive S16 executions using the same inputs.
3. **S128:** compare against four S32 and eight S16 executions, including
   resident-prefix and context-rebase starts that are not width-aligned.
4. Enable all accepted widths and run end-to-end Agent TTFT, token-exact and
   long-session gates. Keep strict S1 independently testable.

Local native bring-up has validated S16 input RMSNorm, the same-graph
RMSNorm→M16 QKV/RoPE edge, the dynamic C4096 H2 KV resident witness bridge,
and the attention-to-layer-tail private-TEMP splice. The former fifteen full
states, one causal-tail state, and one merge have also been composed into one
three-input C4096 attention OM: all 32 cache Slice nodes stay in-graph, event
SSA is `32076/32076`, and libinstsim cosine is `0.9998773`. That OM still
starts from full Q/K/V cache inputs; the current S16 dual-K/V dynamic append,
24-layer replication, and board handoff remain incomplete. These bounded
results justify starting with S16 but are not release artifacts.

## Performance accounting

Report both transformer invocations and wall time. Replacing 433 S1 calls with
six scheduled block calls is an invocation-count result, not a latency claim:
S16/S32/S128 execution costs must be measured. Required metrics are cold and
resident-prefix TTFT, per-family execution time, cache-publication time,
fallback count, generated-token latency and end-to-end total time.
