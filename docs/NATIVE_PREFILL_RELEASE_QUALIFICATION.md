# Native-prefill release qualification v4

Native S16/S32/S128 blocks have two deliberately separate qualification
contracts:

- `pico.minicpm5.prefill-block-qualification.v2` is retained for development
  captures and compatibility. It is not release activatable.
- `pico.minicpm5.prefill-block-qualification.v4` is the only schema accepted by
  `pico.minicpm5.prefill-activation.v4`. Release v3 is legacy and is rejected.

The old command therefore remains development-only:

```bash
pico-minicpm5 qualify-prefill-block \
  --evidence work/prefill/dev-evidence.json \
  --out work/prefill/dev-qualification.json
```

Use the explicit release command for a v4 evidence index:

```bash
pico-minicpm5 qualify-prefill-block-release \
  --evidence work/prefill/release-evidence.json \
  --out work/prefill/release-qualification.json
```

Before qualifying S16, create its content-bound strict-S1 baseline:

```bash
pico-minicpm5 qualify-prefill-s1-release \
  --evidence work/prefill/s1-release-evidence.json \
  --out work/prefill/s1-release-qualification.json
```

## Evidence which v4 binds

The release evidence index names real files using exact `path`, `bytes` and
lowercase `sha256` fields. Paths must be relative, remain below the evidence
directory and must not traverse symlinks. Qualification reads every file and
fails closed on a missing file, byte-count drift, hash drift, duplicate JSON
key or unexpected field.

A v4 record binds:

- candidate OM, the actual head OM, the embedding artifact and build manifest;
- board runner, native executor and ready descriptor;
- one JSON capture artifact at every required absolute position;
- EOS, English, Chinese and context-boundary workload artifacts;
- real timing measurement artifacts, including warm-up and measured sample
  arrays (means and speedup are recomputed, not accepted as claims);
- each required baseline qualification and baseline OM: S16 against S1, S32
  against S16, and S128 against both S32 and S16;
- a clean-board MMZ before/after observation.

Each workload artifact binds the head OM and embedding SHA-256 values used for
that run. `prompt_sha256` and `output_tokens_sha256` hash the exact ordered token
ID sequence encoded as packed little-endian unsigned 32-bit integers; the
contract is named `sha256-le32-u32-token-id-sequence`. These are token-ID
comparisons, so no tokenizer artifact is needed to interpret or reproduce the
hash. Precisely, hash `b"".join(struct.pack("<I", token_id) for token_id in
token_ids)` with no count, delimiter or other prefix; every ID must be in
`0..2^32-1`. `token_id_sequence_sha256()` in
`pico_minicpm5.prefill_blocks` is the canonical producer implementation.

Wide baseline qualifications are verified transitively as complete v4 records.
The S1 trust anchor must be a complete
`pico.minicpm5.strict-s1-baseline-qualification.v4` record. It binds the real
position-zero bootstrap OM separately from the position>=1 canonical decode
OM, the actual head OM and embedding, their two ready descriptors, build
manifest, runner, executor, the required absolute-position capture matrix, an
exact 48-token run, EOS, English, Chinese and context-boundary workloads, and
its clean-board MMZ observation. The old
six-field `strict-s1-baseline-qualification.v1` form is development-only; the
single-OM v2 and dual-route v3 forms are legacy-only. All are explicitly
rejected by the wide-block builder and release activation.
The builder and verifier also derive the complete S1 identity transitively for
every baseline. An S128 ladder whose independently valid S32 and S16 records
terminate at different S1 deployments is rejected before a release PASS can
be emitted.

## MMZ admission

`admission_bytes` is never supplied to the v4 qualification builder. It is
derived as:

```text
before_available_bytes - after_available_bytes
```

The observation must be `PASS`, identify Hi3403, set `clean_board=true`, and
bind the candidate OM, runner, executor and ready descriptor hashes. Release
activation requires the manifest's `admission_bytes` to equal this derived
delta. Every active width is charged independently; repeated residency groups
are disabled and never used to discount a second model.

Until real v4 evidence artifacts exist for a width, that width remains release
blocked and activation retains strict S1 fallback.

The strict-S1 observation uses `role=base-resident` and
`accounting=included-in-base_resident_bytes`; its measured `resident_bytes` is
therefore evidence for the base accounting rather than another wide-block
admission charge.

## Live strict-S1 activation identity

Activation v4 does not accept the former `"strict_s1": true` flag. The
manifest must name the actual resident trust anchor:

```json
{
  "schema": "pico.minicpm5.prefill-activation.v4",
  "context": 4096,
  "deployment_mode": "trusted-read-only-process-lifetime",
  "strict_s1": {
    "bootstrap_model": "models/prefill-position0.om",
    "canonical_decode_model": "models/decode.om",
    "head_model": "models/head.om",
    "embedding": "assets/token_embedding.f16.bin",
    "qualification": "evidence/s1-qualification.json",
    "qualification_sha256": "<64 lowercase hex>",
    "build_manifest": "evidence/s1-build.json",
    "runner": "app/src/pico_minicpm5_split_board_runner.py",
    "executor": "app/pico_persistent_acl_executor",
    "bootstrap_ready_descriptor": "evidence/prefill-ready-descriptor.bin",
    "canonical_ready_descriptor": "evidence/decode-ready-descriptor.bin"
  },
  "blocks": []
}
```

Every direct or transitive wide-block baseline must resolve to exactly this S1
qualification, both route OMs, head OM, embedding, build, runner, executor and
both descriptor identities. Each wide qualification must also use exactly the
live S1 runner/executor and head/embedding identities. A different anchor
disables that width. Whenever an activation manifest is used, even when no wide
handler is registered, immediately before `probe._start` the runtime resolves
and rehashes the live executor, imported protocol runner, bootstrap OM,
canonical decode OM, head OM, embedding, both descriptor artifacts and every
registered wide OM.

The exact `deployment_mode` value is a required operator trust assertion;
missing or different values reject the manifest. It does not let the runtime
prove filesystem immutability. This preflight detects stale or passively
replaced files and minimizes the
qualification-to-load interval. It is not an inherited-file-descriptor handoff:
the executor reopens model paths, so an actively malicious writer can still race
between the final hash and those opens. Release activation is therefore
fail-closed on an operational prerequisite: the deployment tree must be trusted
and kept read-only/immutable for the entire process lifetime. If the
supplied `base_resident_bytes` is below the
anchor's measured `resident_bytes`, every wide width is disabled while the
verified strict-S1 fallback remains reported. If the top-level S1 anchor itself
is missing, unreadable, non-UTF-8, tampered or incomplete, activation rejects
the manifest rather than claiming a fallback it did not verify.

Memory admission uses
`effective_base_resident_bytes = max(base_resident_bytes, s1.resident_bytes)`.
If this effective lower bound plus the reserve exceeds available MMZ,
activation rejects the manifest because even the claimed S1 fallback cannot be
admitted.
