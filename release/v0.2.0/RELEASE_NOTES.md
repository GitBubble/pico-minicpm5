# pico-minicpm5 v0.2.0

[中文](RELEASE_NOTES.zh-CN.md)

The three `ctx1024` OM files are byte-identical to `v0.1.0`. Every throughput
number below moved because the executor changed, not because a model was
recompiled.

## The executor is now reproducible from the source in this repository

`v0.1.0` pinned an executor that the shipped source could not rebuild. This
release pins `cef4edb2…`, built from `app/native/pico_persistent_acl_executor.c`
with the sanctioned `aarch64-mix210-linux-gcc` 7.3.0 toolchain. The recipe is
[`docs/EXECUTOR_BUILD.md`](../../docs/EXECUTOR_BUILD.md); it was re-run against
the committed source before this manifest was written, and reproduces the hash
byte for byte.

The new executor retains the workspace input across executes instead of
rewriting it, so each decode step saves one full workspace write. The saving is
proportional to the context, and three independent contexts agree on the
mechanism:

| context | workspace retained | transformer saving | implied bandwidth |
|---|---:|---:|---:|
| ctx1024 | 24.6 MiB | 5.5 ms | 4.66 GB/s |
| ctx4096 | 98.3 MiB | 27.6 ms | 3.74 GB/s |
| ctx8192 | 196.6 MiB | 54.9 ms | 3.76 GB/s |

## Measured, one board session, all three profiles

| profile | p50 / token | token/s | vs v0.1.0 path | prompt ingestion |
|---|---:|---:|---:|---:|
| ctx1024 | 100.40 ms | **9.96** | +5.2% | 79.49 ms/token |
| ctx4096 | 127.96 ms | **7.81** | +19.7% | 106.28 ms/token |
| ctx8192 | 165.71 ms | **6.03** | +31.6% | 144.02 ms/token |

All 48 greedy-oracle tokens are identical to each profile's qualified baseline
on all three. The `ctx1024` ingestion figure agrees with an independent
cold-prefill measurement to `0.10%`.

## ctx4096 is qualified; ctx8192 remains a candidate

`ctx4096` passes its numeric gate (minimum public-output cosine `0.9908` at
position 4095, board tail byte-exact with the simulator, 48/48 greedy tokens,
boundary fail-closed) and ships as a **qualified** profile. Its decode OM is a
release asset; `prefill.om` and `head_flat.om` are the frozen `v0.1.0` files,
shared byte for byte across every context — that is the mixed prefill-window
contract, now proven on the board at all three widths in one session.

`ctx8192` stays **pending**. Its public outputs clear the gate and its EOS
verdict is now `PASS`, but its calibration is donor-zero-extended rather than
native, and the Chinese oracle, memory envelope and long-prompt items are open.

## A gate that was measuring the wrong thing

`ctx8192` had carried `eos: FAIL_STRICT_SEQUENCE_MISMATCH` against a sequence
that was never traced to the reference model. Re-derived from the pinned
checkpoint in float64, the reference writes a terminal period and then stops —
which is exactly what `ctx8192` produces and what `ctx1024` and `ctx4096` do
not. The expectation had been recorded from the first artifact that ran.
Details in [`release/contexts/strict-eos-oracle.md`](../contexts/strict-eos-oracle.md).

This does not make the other two profiles defective: their 48-token oracle
passes, and the reference is near a tie at that step.

## Also in this release

- A recorded board agent session on the project and app homepages, rendered to
  a self-contained animated SVG. Waits play at `4.5x` with the compression
  stated on screen and the board's own clock on every frame.
- [`docs/QUANTIZATION_CONTRACT.md`](../../docs/QUANTIZATION_CONTRACT.md):
  what a `Clip` does to ATC's IFMR range search, why position zero needs its
  own calibration family (the layer-0 MLP branch, not attention), and one
  widely repeated rule retracted because its own proof artifact contradicts it.
- [`release/perf/`](../perf/README.md): the performance board now carries TTFT,
  the per-context phase breakdown, the superseded pre-unification numbers and
  the evidence hashes behind each.

## Known limits

Long-prompt TTFT is unattractive and honest about it: prompt tokens are still
ingested one at a time, so a 512-token prompt costs `40.7 s` on ctx1024 and
`54.4 s` on ctx4096. The wide-block prefill path that would amortise this is
not in this release; no wide block has passed a numeric gate.
