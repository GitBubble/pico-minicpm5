# Performance board

[中文](README.zh-CN.md)

Human-written summary of [`perf-board.json`](perf-board.json); where the two disagree the JSON is authoritative. The headline table is a
2026-08-17 board session that ran all three profiles back to back on the
retain-input unified executor (`cef4edb2…`); the numbers it superseded are kept
alongside so the delta is always visible. Numbers that live outside this
repository are bound by the sha256 of their evidence file, the way the numeric
gates are.

Target: Hi3403 / V101, SS928-class development board.

## Decode phases

One decode step is five host-observed phases: **prepare** (embedding row,
attention mask, RoPE matrix), **transformer** (the resident 24-layer handle),
**kv** (packed K/V publication into the canonical resident cache), **head**
(vocabulary head handle) and **argmax**. Steady state is the median over
positions `>= 1`; position 0 runs on the prefill handle and is reported apart.
Throughput is `1000 / total p50` — it is derived here, not stored in evidence.

Each profile's gate record publishes the headline (`gate_reported` in the
JSON); the phase breakdown is pooled over every `position >= 1` record, which
in a greedy-oracle run also includes teacher-forced prompt steps. In the
2026-08-17 session both scopes are the same 67 pooled steps across the same
three prompts, so the two agree exactly.

## Steady-state decode, p50 ms

| Profile | prepare | transformer | kv | head | argmax | **total** | tok/s | vs superseded | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctx1024 | 1.32 | 77.12 | 0.95 | 19.90 | 1.01 | **100.40** | 9.96 | +5.2% | qualified |
| ctx4096 | 1.38 | 102.13 | 2.66 | 20.69 | 1.00 | **127.96** | 7.81 | +19.7% | qualified |
| ctx8192 | 1.36 | 139.88 | 2.67 | 20.70 | 0.99 | **165.71** | 6.03 | +31.6% | pending |

The head and argmax phases are flat across all three contexts — they do not see
the KV window. Context cost is almost entirely the transformer phase.

### Where the gain comes from

The unified executor retains the workspace input across executes instead of
rewriting it, so a decode step saves one full workspace write. That predicts a
saving proportional to the context, and the three contexts agree:

| Profile | workspace retained | transformer saving | implied bandwidth |
|---|---:|---:|---:|
| ctx1024 | 24.6 MiB | 82.66 → 77.12 = 5.54 ms | 4.66 GB/s |
| ctx4096 | 98.3 MiB | 129.70 → 102.13 = 27.57 ms | 3.74 GB/s |
| ctx8192 | 196.6 MiB | 194.75 → 139.88 = 54.87 ms | 3.76 GB/s |

A proportional fit through these three points sits at `0.28 ms/MiB`. ctx4096
and ctx8192 land within `0.6%` of it. ctx1024 saves `24%` less than the fit
predicts, which is what a fixed per-execute cost looks like when the smallest
context has the least to amortise it over. The trend is why the longer contexts
gain the most; it is not three points on one line.

### Position zero is the same handle everywhere

| Profile | pos-0 transformer | pos-0 total | with prompt head skip |
|---|---:|---:|---:|
| ctx1024 | 80.59 | 103.67 | 82.77 |
| ctx4096 | 86.40 | 109.52 | 88.11 |
| ctx8192 | 86.01 | 109.15 | 87.74 |

ctx4096 and ctx8192 agree to `0.39 ms` because they bootstrap position zero on
the *same* frozen ctx1024 `prefill.om`, byte for byte. That is the mixed
prefill-window contract measured directly.

ctx1024 additionally reports `1.91x` throughput over the accepted 49-handle
baseline (`4.89–4.92 tok/s`).

## The tail

| Profile | tail position | tail total |
|---|---:|---:|
| ctx4096 | 4095 | 327.91 |
| ctx8192 | 8191 | 479.95 (legacy 574.97) |

Tail steps are excluded from steady state. Each was run as a single step
immediately after model load, so first-step warm-up cannot be separated from
the cost of attending over a full window; head and argmax are unchanged there,
so the inflation is attention plus warm-up.

## Executor-mode A/B (ctx8192, 16 generated tokens)

All five modes produced byte-identical position-1 raw outputs and the same 16
token ids, so this compares cost only.

| Mode | transformer | argmax | **total p50** | tok/s | p50 change | Verdict |
|---|---:|---:|---:|---:|---:|---|
| cached | 194.54 | 1.00 | **219.30** | 4.56 | — | baseline |
| nocache | 171.25 | 25.21 | **222.34** | 4.50 | +1.4% slower | rejected |
| per-model-mixed | 171.30 | 1.00 | **198.57** | 5.04 | −9.5% faster | superseded |
| zero-once | 139.85 | 0.99 | **167.11** | 5.98 | −23.8% faster | **adopted** |
| promptskip + zero-once | 139.87 | 0.99 | **167.28** | 5.98 | −23.7% faster | **adopted** |

The nocache arm is the instructive one: uncaching makes the transformer `23 ms`
faster and the argmax `24 ms` slower, so it loses overall. Per-model-mixed
isolates that — uncached transformer, cached argmax — and banks the gain. The
zero-once executor then takes the transformer down another `31 ms`. Promptskip
is deliberately flat in steady state; its gain is at prompt positions only.

`zero-once` is the mode that became the shipped retain-input executor
(`cef4edb2…`). Its `167.11 ms` here and the `165.71 ms` in the headline table
are the same configuration measured in two sessions; the difference is
run-to-run variance, not a change.

## Time to first token

TTFT is the wall time from request start until the first generated token: every
prompt-ingestion step summed. Position 0 runs on the prefill handle, positions
`1..N-1` run on the decode handle **one token at a time**, and the head plus
argmax run once at the last prompt position. Model load is excluded.

So all three OMs contribute, but the counts differ: prefill and head execute
once each, the decode handle executes `N-1` times. Past a handful of prompt
tokens TTFT is almost entirely decode-handle time — which is why the wide-block
planner (S16/S32/S128) targets the decode side rather than a per-context prefill
binary.

```
ttft ≈ position_zero_total + (N − 1) × decode_step_total
```

This model reproduces every measured point to within **0.12%**:

| Case | model | measured | error |
|---|---:|---:|---:|
| ctx128 bucket, 121 tokens | 11509.6 ms | 11495.3 ms | +0.12% |
| ctx4096, 12 tokens | 1795.8 ms | 1796.2 ms | −0.02% |
| ctx8192 legacy, 12 tokens | 2511.7 ms | 2511.3 ms | +0.02% |
| ctx8192 zero-once, 12 tokens | 1931.3 ms | 1931.9 ms | −0.03% |

### Prompt ingestion, measured on the shipped executor

The per-token cost is what TTFT is made of. It is the steady step minus head
and argmax, because the runtime skips both on non-terminal prompt positions:

| Profile | ingestion per prompt token | 47 tok | 128 tok | 512 tok |
|---|---:|---:|---:|---:|
| ctx1024 | **79.49 ms** | 3.77 s | 10.20 s | 40.73 s |
| ctx4096 | **106.28 ms** | 5.00 s | 13.61 s | 54.42 s |
| ctx8192 | **144.02 ms** | 6.73 s | 18.40 s | 73.70 s |

The per-token figures are measured; the totals are the model above. ctx1024's
`79.49 ms` agrees with an independent cold-prefill measurement of `79.570 ms`
to `0.10%`, which is the cross-check that makes the model usable.

### Earlier measured TTFT, pre-unification

| Profile | 5 tok | 6 tok | 7 tok | 12 tok | 121 tok |
|---|---:|---:|---:|---:|---:|
| ctx1024 (2026-08-09 runtime) | 831 ms | 965 ms | 1064 ms | 1639 ms | — |
| ctx4096 | 723 ms | 881 ms | 1044 ms | 1796 ms | — |
| ctx8192 (legacy cached) | 984 ms | 1206 ms | 1425 ms | 2511 ms | — |
| ctx8192 (zero-once) | — | 946 ms | 1111 ms | 1932 ms | — |
| ctx128 bucket | 491 ms | 585 ms | — | — | **11495 ms** |

The 121-token ctx128-bucket run remains the longest single prompt measured
end to end anywhere in this project, at a mean `95.00 ms` per prompt position.

## Gate binding

A throughput number is publishable only next to the numeric gate the same
artifact passed. Throughput never substitutes for a cosine or token gate.

| Profile | Gate record | Verdict | Minimum public-output cosine |
|---|---|---|---:|
| ctx1024 | `release/v0.1.0/qualification.json` | PASS | 0.996646 |
| ctx4096 | `release/contexts/ctx4096.qualification.json` | PASS | 0.990820 |
| ctx8192 | `release/contexts/ctx8192.qualification.json` | CANDIDATE_CALIBRATION_NOT_NATIVE | 0.986076 |

ctx8192's throughput is real and reproducible, and the profile is still
`pending` — because its calibration is donor-zero-extended rather than native,
and the Chinese oracle, memory envelope and long-prompt items are open. Its EOS
gate passes: measured against the re-derived FP64 reference it is the only one
of the three profiles that reproduces the reference exactly
([why](../contexts/strict-eos-oracle.md)). Use it with
`--allow-unqualified-profile` for development only.

## What is not measured

**TTFT.** No published profile has an end-to-end long-prompt run: the per-token
ingestion cost is measured on all three, but the longest prompt actually taken
to a first token on ctx1024, ctx4096 or ctx8192 is **47 tokens**. Nothing on
disk records a resident-prefix hit, a fixed-prefix snapshot, or a wide prefill
block.

**Three prose-only figures.** The 810-token cold request falling
`86.70 s → 69.45 s`, the resident repeat falling `18.17 s → 14.61 s`, and the
643-token prefix hit reaching `14.61 s` have **no retrievable per-step record**.
They were produced live on the board and only summarised into prose. They sit in
`perf-board.json` under `ttft.unbound_prose_claims`; no gate depends on them,
and the fix is to re-run all three with the report written to disk.

**Everything else:**

- Per-layer or per-kernel NPU breakdown — only the five host-observed phases exist.
- KV snapshot hydration bytes and time at tail positions — no such field is recorded.
- Separation of first-step warm-up from tail-position cost — tail runs execute one step.
- ctx128 — no board performance run exists.
- Executor-mode A/B at ctx1024 or ctx4096 — the five-mode comparison ran at ctx8192 only.
- Prompts longer than the prefill window under the mixed contract — every recorded run used 2–13 token prompts.
