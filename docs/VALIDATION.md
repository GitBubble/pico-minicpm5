# Validation ladder

[中文](VALIDATION.zh-CN.md)

1. Checkpoint: pinned revision, hashes, geometry, symbols and BF16 spans.
2. Graph: ONNX checker with known custom ops stubbed, unique prefixes, no
   dangling values, five public inputs and three outputs at depth 24.
3. Incremental depth: N=1 identity, N=2 output differs from layer zero and
   agrees with the two-layer reference, then 4/8/12/24.
4. Local execution: raw public K, V and hidden tensors, strict cosine `>0.98`.
   Immediately after each libinstsim or SS928 run, create a
   `pico.minicpm5.runtime-capture.v1` manifest bound to the executed
   transformer OM, ATC build manifest, position, ctx1024 and raw files.
5. Board: load exactly three handles, compare board raw output to the same OM
   under libinstsim, then compare to FP64.
6. Head: pass a transformer's logical final hidden to `head_flat.om` at the
   same reference position; require logits cosine `>0.98`, exact top-1 and
   execution evidence bound to the head OM.
7. Generation: stable-margin greedy exact, EOS, code-sensitive prompt and
   multilingual text.
8. Performance: record load time, per-token latency distribution and token/s.

`pico-minicpm5 score` identifies K/V order structurally against the reference,
decodes the physical C4 hidden carrier when necessary, records raw hashes and
returns exit code 1 on a failed strict gate.

Aggregate packed-output cosine is the release gate. Per-layer K/V slice cosine
is always emitted as a diagnostic so a low-energy slice cannot disappear from
the report.

`pico-minicpm5 score-head --position P` reads
`reference/posP/logits.f32.bin`. The scored runtime logits must come from
`head_flat.om` fed by the logical final hidden produced at position `P`; shape
compatibility alone does not make outputs from different positions comparable.
Pass that exact file as `--hidden-input` and pass an exactly-zero,
1536-element FP32 file as `--residual-input`. Qualification requires the hidden
hash to equal the same-position transformer score's logical `next_hidden` hash
and the residual hash to equal the canonical zero tensor.

Both scorers require `--om`, `--build-manifest` and `--capture-manifest`.
Build, capture and score evidence must use context 1024; qualification fixes
that context and offers no override. The resulting lineage checks prevent raw
tensors from one build being accidentally used to qualify binaries from
another build.

Runtime capture is deliberately modest in scope: it is an auditable local
record of hashes supplied after an execution, not a cryptographic proof of
execution, signed board report or remote attestation. The `--runner` value is
an operator assertion. Create the manifest immediately after the corresponding
libinstsim/SS928 run and preserve it with the raw files.
