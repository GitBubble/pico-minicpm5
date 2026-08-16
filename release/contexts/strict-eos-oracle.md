# The strict-EOS expectation, re-derived

The ctx8192 record carried `eos: FAIL_STRICT_SEQUENCE_MISMATCH` against this
sequence:

```text
[242, 39, 220, 608, 4219, 357, 242, 39, 130073]     " 2\nThe answer is 2"
```

That sequence was never traced to the reference model. It was re-derived on
2026-08-17 by running the pinned checkpoint
(`openbmb/MiniCPM5-1B`, revision `4e9de7a0778…`) in float64, greedy, stopping
on `eos_token_id = [1, 130073]`, from the same prompt ids the board uses
(`[0] + encode("1+1 equals")`):

```text
[242, 39, 220, 608, 4219, 357, 242, 39, 35, 1]      " 2\nThe answer is 2."
```

The reference writes a terminal period and then stops. At the divergent step
its choice is unambiguous in both precisions:

| token | logit (fp64) | logit (fp32) | p (fp32) |
|---|---:|---:|---:|
| `35` `"."` | 20.183120 | 20.18317 | 39.14% |
| `130073` EOS | 19.877989 | 19.87802 | 28.85% |

So the expectation was inverted. Measured against the reference:

| profile | generated | matches reference |
|---|---|---|
| ctx1024 | `[…242, 39, 130073]` | no — stops one token early |
| ctx4096 | `[…242, 39, 130073]` | no — stops one token early |
| **ctx8192** | `[…242, 39, 35, 1]` | **yes, all ten tokens** |

All three runs shared `prefill.om` and `head_flat.om` byte for byte and ran on
the same executor, so the only variable is each profile's own `decode.om`.

## What this does and does not change

It removes `eos` as a blocker on the ctx8192 line: judged against the
reference rather than against ctx1024's output, ctx8192 passes and the other
two carry a one-token deviation.

It does not qualify ctx8192. Its calibration is still donor-zero-extended
rather than native, and the Chinese oracle, memory envelope and long-prompt
items are still open. It also does not make ctx1024 or ctx4096 defective: the
48-token three-prompt greedy oracle passes on all three, and the deviation
here is a terminal period at a step where the reference itself is close to a
tie.

The lesson worth keeping is the one about provenance. An expected sequence
that is recorded from the first thing that ran, rather than derived from the
reference, will always confirm whatever ran first — and will report the one
artifact that disagrees with it as the broken one.

Evidence: `scripts/pico_aot/minicpm5_strict_eos_oracle_recheck_qualification.json`
in the integration monorepo.
