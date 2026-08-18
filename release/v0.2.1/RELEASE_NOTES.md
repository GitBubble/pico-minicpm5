# pico-minicpm5 v0.2.1

This source release adds the long-context runtime contracts needed to operate
owner-built MiniCPM5 decode artifacts at 8192, 10240 and 16384 tokens. It does
not redistribute model weights, licensed SDK libraries or locally compiled OM
files.

## Highlights

- `ctx8192` is now qualified. Its frozen 48-token greedy oracle is exact, all
  reported public outputs are strictly above cosine `0.98`, and the corrected
  official EOS sequence ends with a period followed by EOS.
- `ctx10240` and `ctx16384` profiles are available but remain fail-closed
  `pending`; selecting either one for controlled testing still requires the
  explicit unqualified-profile override.
- The 4097-token head-skip workload passes on all three profiles (8192/10240/
  16384 ingest: 602.48/681.89/910.42 seconds). ctx10240 remains pending because
  its frozen greedy suite is 36/48 and tail hidden cosine is `0.978842`;
  ctx16384 remains pending because its best recalibrated tail reaches only
  `0.957146/0.985295/0.967172` for hidden/K/V.
- The 8192/10240/16384 decode contracts automatically retain their exact
  context-specific workspace input and validate the seven-input descriptor
  before loading.
- Teacher-forced prompt ingestion skips the vocabulary head at every known
  prompt position except the last. The ctx8192 4097-token gate skipped 4096
  head calls and emitted the exact expected next token.
- `app/agent.sh` accepts `CONTEXT_PROFILE=ctx8192|ctx10240|ctx16384`; conflicting
  environment and command-line profile selections fail closed.
- Optional eager tool-output prefill is included but remains disabled by
  default and guarded by resident-K/V and prefix-reuse requirements.

The release workflow publishes the Python sdist/wheel, an SPDX SBOM and
checksums. Board OM artifacts continue to be supplied and qualified by their
owner separately.
