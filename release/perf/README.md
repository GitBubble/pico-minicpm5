# Performance board

Rendered view of [`perf-board.json`](perf-board.json). Every number here is
distilled from board runs that already existed; building this board took no new
measurement. Numbers that live outside this repository are bound by the sha256
of their evidence file, the way the numeric gates are.

Target: Hi3403 / V101, SS928-class development board.

## Decode phases

One decode step is five host-observed phases: **prepare** (embedding row,
attention mask, RoPE matrix), **transformer** (the resident 24-layer handle),
**kv** (packed K/V publication into the canonical resident cache), **head**
(vocabulary head handle) and **argmax**. Steady state is the median over
positions `>= 1`; position 0 runs on the prefill handle and is reported apart.
Throughput is `1000 / total p50` — it is derived here, not stored in evidence.

Two median scopes exist and the board keeps both. Each profile's gate record
publishes the authoritative headline (`gate_reported` in the JSON — for ctx4096
the 48 scored generated tokens, `153.117 ms`); the phase breakdown is pooled
over every `position >= 1` record, which in a greedy-oracle run also includes
teacher-forced prompt steps (`153.202 ms`, `+0.06%`). The tables below show the
phase-breakdown scope; the gate-binding table at the end shows the headline.

## Steady-state decode, p50 ms

| Profile | prepare | transformer | kv | head | argmax | **total** | tok/s | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ctx1024 | 1.31 | 82.66 | 0.37 | 20.37 | 0.94 | **105.66** | 9.42–9.48 | qualified |
| ctx4096 | 1.54 | 129.70 | 0.45 | 20.40 | 1.00 | **153.20** | 6.53 | qualified |
| ctx8192 | 2.93 | 139.85 | 2.69 | 20.67 | 0.99 | **167.11** | 5.98 | pending |

The head and argmax phases are flat across all three contexts — they do not see
the KV window. Context cost is almost entirely the transformer phase, and the
step from 4096 to 8192 (`+10.2 ms`) is far smaller than the step from 1024 to
4096 (`+47.0 ms`): the ctx8192 numbers use the zero-once executor candidate,
whose gain (see the A/B below) offsets most of the window growth.

ctx1024 additionally reports `1.91x` throughput over the accepted 49-handle
baseline (`4.89–4.92 tok/s`).

## Position zero and the tail

| Profile | pos-0 transformer | pos-0 total | pos-0 with head skip | tail position | tail total |
|---|---:|---:|---:|---:|---:|
| ctx4096 | 86.31 | 109.52 | — | 4095 | 327.91 |
| ctx8192 | 85.93 | 109.09 | 87.85 | 8191 | 479.95 (legacy 574.97) |

Position zero costs the *same* `~86 ms` in both extended contexts. That is the
mixed prefill-window contract visible in the timing: both bootstrap on the same
frozen ctx1024 `prefill.om`, which attends over a 1024-entry window regardless
of the profile's capacity — cheaper than a steady decode step at 4096 or 8192.

Prompt-position head skip removes the head and argmax phases on non-terminal
teacher-forced positions: `109.09 -> 87.85 ms`.

Tail steps are excluded from steady state. Each was run as a single step
immediately after model load, so first-step warm-up cannot be separated from
the cost of attending over a full window; head and argmax are unchanged there,
so the inflation is attention plus warm-up.

## Executor-mode A/B (ctx8192, 16 generated tokens)

All five modes produced byte-identical position-1 raw outputs and the same 16
token ids, so this compares cost only.

| Mode | transformer | argmax | **total p50** | tok/s | vs baseline | Verdict |
|---|---:|---:|---:|---:|---:|---|
| cached | 194.54 | 1.00 | **219.30** | 4.56 | — | baseline |
| nocache | 171.25 | 25.21 | **222.34** | 4.50 | +1.4% | rejected |
| per-model-mixed | 171.30 | 1.00 | **198.57** | 5.04 | −9.5% | superseded |
| zero-once | 139.85 | 0.99 | **167.11** | 5.98 | −23.8% | **adopted** |
| promptskip + zero-once | 139.87 | 0.99 | **167.28** | 5.98 | −23.7% | **adopted** |

The nocache arm is the instructive one: uncaching makes the transformer `23 ms`
faster and the argmax `24 ms` slower, so it loses overall. Per-model-mixed
isolates that — uncached transformer, cached argmax — and banks the gain. The
zero-once executor then takes the transformer down another `31 ms`. Promptskip
is deliberately flat in steady state; its gain is at prompt positions only.

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

### Measured TTFT

| Profile | 5 tok | 6 tok | 7 tok | 12 tok | 121 tok |
|---|---:|---:|---:|---:|---:|
| ctx1024 (2026-08-09 runtime) | 831 ms | 965 ms | 1064 ms | 1639 ms | — |
| ctx4096 | 723 ms | 881 ms | 1044 ms | 1796 ms | — |
| ctx8192 (legacy cached) | 984 ms | 1206 ms | 1425 ms | 2511 ms | — |
| ctx8192 (zero-once) | — | 946 ms | 1111 ms | 1932 ms | — |
| ctx128 bucket | 491 ms | 585 ms | — | — | **11495 ms** |

The ctx1024 row predates the resident-K/V and head-skip refresh (its decode step
was 116–123 ms against today's 105.66 ms), so those are upper bounds for the
shipped runtime. The 121-token ctx128-bucket run is the **only long-prompt TTFT
measured anywhere in this project**, at a mean `95.00 ms` per prompt position.

### What is not measured

No published profile has a long-prompt TTFT: the largest measured prompt on
ctx1024, ctx4096 or ctx8192 is **12 tokens**. There is also no on-disk run with
a resident-prefix hit, a fixed-prefix snapshot, or wide prefill blocks.

Two figures quoted in the README, CHANGELOG and release notes — the 810-token
cold request falling `86.70 s → 69.45 s`, and the 643-token prefix hit reaching
`14.61 s` — have **no retrievable per-step record**. They were produced live on
the board and only summarized into prose. They are carried in `perf-board.json`
under `ttft.unbound_prose_claims` for provenance; no gate depends on them, and
the fix is to re-run both with the report written to disk.

## Gate binding

A throughput number is publishable only next to the numeric gate the same
artifact passed. Throughput never substitutes for a cosine or token gate.

| Profile | Gate record | Verdict | Minimum public-output cosine |
|---|---|---|---:|
| ctx1024 | `release/v0.1.0/qualification.json` | PASS | 0.996646 |
| ctx4096 | `release/contexts/ctx4096.qualification.json` | PASS | 0.990820 |
| ctx8192 | `release/contexts/ctx8192.qualification.json` | CANDIDATE_STRICT_EOS_FAIL | 0.986076 |

ctx8192's throughput is real and reproducible, and it is still `pending`: its
strict-EOS sequence gate fails. Use it with `--allow-unqualified-profile` for
development only.

## Not measured

- Per-layer or per-kernel NPU breakdown — only the five host-observed phases exist.
- KV snapshot hydration bytes and time at tail positions — no such field is recorded.
- Separation of first-step warm-up from tail-position cost — tail runs execute one step.
- ctx128 — no board performance run exists.
- Executor-mode A/B at ctx1024 or ctx4096 — the five-mode comparison ran at ctx8192 only.
- Prompts longer than the prefill window under the mixed contract — every recorded run used 2–13 token prompts.
