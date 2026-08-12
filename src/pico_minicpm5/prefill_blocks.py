"""Fail-closed qualification contract for native multi-token prefill OMs.

The qualification record is deliberately stricter than a numeric score.  A
PASS binds the files which may be activated, the physical FP32 publisher seen
by the native executor, the logical FP16 resident-cache ABI, byte-exact input
construction, complete K/V publication, end-to-end workloads and a measured
board-speed win.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Mapping, Sequence


# v2 remains intentionally available for development captures and for reading
# older qualification bundles.  Release activation must require
# ``RELEASE_SCHEMA``; a v2 PASS never carries enough independently verifiable
# evidence to be called release eligible.
SCHEMA = "pico.minicpm5.prefill-block-qualification.v2"
LEGACY_RELEASE_SCHEMA = "pico.minicpm5.prefill-block-qualification.v3"
RELEASE_SCHEMA = "pico.minicpm5.prefill-block-qualification.v4"
CAPTURE_SCHEMA = "pico.minicpm5.prefill-capture.v1"
WORKLOAD_SCHEMA = "pico.minicpm5.prefill-workload.v2"
PERFORMANCE_SCHEMA = "pico.minicpm5.prefill-performance-measurement.v1"
MMZ_SCHEMA = "pico.minicpm5.prefill-mmz-observation.v1"
STRICT_S1_DEVELOPMENT_SCHEMA = (
    "pico.minicpm5.strict-s1-baseline-qualification.v1")
STRICT_S1_LEGACY_RELEASE_SCHEMA = (
    "pico.minicpm5.strict-s1-baseline-qualification.v2")
STRICT_S1_PREVIOUS_RELEASE_SCHEMA = (
    "pico.minicpm5.strict-s1-baseline-qualification.v3")
STRICT_S1_BASELINE_SCHEMA = (
    "pico.minicpm5.strict-s1-baseline-qualification.v4")
STRICT_S1_CAPTURE_SCHEMA = "pico.minicpm5.strict-s1-capture.v1"
STRICT_S1_WORKLOAD_SCHEMA = "pico.minicpm5.strict-s1-workload.v2"
STRICT_S1_MMZ_SCHEMA = "pico.minicpm5.strict-s1-mmz-observation.v1"
TOKEN_ID_HASH_CONTRACT = "sha256-le32-u32-token-id-sequence"
POLICY_THRESHOLD_EXCLUSIVE = 0.98
SUPPORTED_WIDTHS = (16, 32, 128)
LAYERS = 24
KV_HEADS = 2
HEAD_DIM = 128
HIDDEN = 1536
PUBLISHER_DTYPE = "FP32"
PUBLISHER_LAYOUT = "contiguous-channel-major"
MASK_NEGATIVE = -64.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COSINE_GROUPS = ("public_cosines", "handoff_cosines")
_COSINE_NAMES = ("hidden", "k", "v")
_WORKLOAD_KINDS = ("eos", "english", "chinese", "context_boundary")
_PERFORMANCE_BASELINES = {16: (1,), 32: (16,), 128: (32, 16)}
_POSITION_PROBES = {
    16: (1, 15, 16, 31, 32, 255, 256, 643),
    32: (1, 31, 32, 127, 128, 643),
    128: (1, 127, 128, 511, 512, 643),
}
_STRICT_S1_POSITION_PROBES = (
    0, 1, 15, 16, 31, 32, 127, 128, 255, 256, 511, 512, 643)
_STRICT_S1_WORKLOAD_KINDS = (
    "tokens_48", "eos", "english", "chinese", "context_boundary")


class PrefillBlockQualificationError(ValueError):
    """A block artifact or its evidence is not eligible for activation."""


def required_starts(context: int, width: int) -> tuple[int, ...]:
    """Return minimum absolute-position coverage for a steady block family.

    Position zero intentionally is not part of this contract.  The current
    quantization contract uses strict S1 for startup and enables a wide block
    only after at least one resident token has been published.
    """
    _geometry(context, width)
    last = context - width
    return tuple(sorted({
        *[position for position in _POSITION_PROBES[width]
          if 1 <= position <= last],
        last,
    }))


def required_strict_s1_positions(context: int) -> tuple[int, ...]:
    """Return the release position matrix for the strict-S1 baseline route."""
    if type(context) is not int or context < 128:
        raise PrefillBlockQualificationError(
            "strict-S1 context must be an integer >=128")
    return tuple(sorted({
        *[position for position in _STRICT_S1_POSITION_PROBES
          if position < context],
        context - 1,
    }))


def canonical_mask_sha256(context: int, width: int, start: int) -> str:
    """Hash the exact absolute-position mask consumed by a wide OM.

    Row ``j`` represents absolute position ``start + j``.  It may attend to
    the resident prefix ``[0, position)`` and to the current-token sentinel at
    column ``context - 1``; every intervening column is exactly FP32 ``-64``.
    Keeping this construction next to the release ABI makes a capture's mask
    hash independently reproducible instead of trusting a self-reported
    ``mask_bytes_exact`` flag.
    """
    _geometry(context, width)
    if type(start) is not int or not 1 <= start <= context - width:
        raise PrefillBlockQualificationError(
            "steady mask start must be within [1, context-width]")
    zero = struct.pack("<f", 0.0)
    negative = struct.pack("<f", MASK_NEGATIVE)
    digest = hashlib.sha256()
    for position in range(start, start + width):
        digest.update(zero * position)
        digest.update(negative * (context - 1 - position))
        digest.update(zero)
    return digest.hexdigest()


def token_id_sequence_sha256(token_ids: Sequence[int]) -> str:
    """Hash an exact ordered token-ID sequence under the release v4 contract.

    Token IDs are packed consecutively as unsigned little-endian 32-bit values,
    with no count, delimiter or tokenizer-dependent representation.
    """
    if not isinstance(token_ids, Sequence) or isinstance(
            token_ids, (str, bytes)):
        raise PrefillBlockQualificationError(
            "token_ids must be a sequence of uint32 values")
    digest = hashlib.sha256()
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int or not 0 <= token_id <= 0xFFFFFFFF:
            raise PrefillBlockQualificationError(
                f"token_ids[{index}] must be a uint32 value")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def logical_abi(context: int, width: int) -> dict[str, object]:
    """Describe the physical publisher and logical resident-cache contracts."""
    _geometry(context, width)
    packed_channels = LAYERS * KV_HEADS
    rows_per_role = packed_channels * width
    values_per_role = rows_per_role * HEAD_DIM
    return {
        "context": context,
        "width": width,
        "input_slots": {
            "embedding": 0,
            "mask": 1,
            "rope": 2,
            "k_cache": 3,
            "v_cache": 4,
        },
        "output_slots": {"k": 0, "v": 1, "hidden": 2},
        "hidden": {
            "shape": [1, HIDDEN, 1, 1],
            "dtype": "FP32",
            "logical_bytes": HIDDEN * 4,
            "meaning": "final layer hidden for the last token in the block",
        },
        # K and V are published by the OM as separate, contiguous FP32
        # channel-major tensors.  Executor opcode 6 performs RNE conversion
        # while scattering them into the logical FP16 resident cache.
        "publisher": {
            "layout": PUBLISHER_LAYOUT,
            "dtype": PUBLISHER_DTYPE,
            "shape": [1, packed_channels, width, HEAD_DIM],
            "logical_rows": rows_per_role,
            "logical_bytes": values_per_role * 4,
            "roles": ["k", "v"],
            "resident_conversion": {
                "opcode": 6,
                "dst_dtype": "FP16",
                "rounding": "RNE",
            },
        },
        "mask": {
            "shape": [width, context],
            "dtype": "FP32",
            "layout": "row-major",
            "logical_bytes": width * context * 4,
            "negative_value": MASK_NEGATIVE,
            "position_mapping": "row_j_is_absolute_start_plus_j",
            "visible_prefix": "[0,position)",
            "masked_future": "[position,context-1)",
            "current_token_column": context - 1,
        },
        "rope": {
            "shape": [width, HEAD_DIM, HEAD_DIM],
            "dtype": "FP32",
            "layout": "row-major",
            "logical_bytes": width * HEAD_DIM * HEAD_DIM * 4,
        },
        "k": {
            "shape": [1, packed_channels, width, HEAD_DIM],
            "dtype": "FP16",
            "logical_rows": rows_per_role,
            "logical_bytes": values_per_role * 2,
            "meaning": "resident cache after opcode-6 RNE scatter",
        },
        "v": {
            "shape": [1, packed_channels, width, HEAD_DIM],
            "dtype": "FP16",
            "logical_rows": rows_per_role,
            "logical_bytes": values_per_role * 2,
            "meaning": "resident cache after opcode-6 RNE scatter",
        },
        "cache_advance": width,
    }


def build_qualification(
    *,
    context: int,
    width: int,
    om: Mapping[str, object],
    build_manifest: Mapping[str, object],
    captures: Sequence[Mapping[str, object]],
    workloads: Sequence[Mapping[str, object]],
    performance: Sequence[Mapping[str, object]],
    artifact_root: Path,
    threshold_exclusive: float = POLICY_THRESHOLD_EXCLUSIVE,
) -> dict[str, object]:
    """Verify all evidence and return the only runtime-eligible PASS record.

    ``artifact_root`` makes file verification unavoidable even for direct
    callers.  ``qualify_file`` supplies the evidence document's directory, so
    portable evidence normally uses relative artifact paths.
    """
    _geometry(context, width)
    threshold = _real(threshold_exclusive, "threshold_exclusive")
    if threshold != POLICY_THRESHOLD_EXCLUSIVE:
        raise PrefillBlockQualificationError(
            f"threshold must remain exactly {POLICY_THRESHOLD_EXCLUSIVE}")
    root = Path(artifact_root)
    artifact = _artifact(om, "om", root, require_json=False)
    build = _artifact(build_manifest, "build_manifest", root, require_json=True)
    normalized, minimum = _captures(
        captures, context=context, width=width, threshold=threshold)
    normalized_workloads = _workloads(workloads, context=context)
    normalized_performance = _performance(
        performance, width=width, om_sha256=str(artifact["sha256"]))
    abi = logical_abi(context, width)
    activation = {
        "key": f"ctx{context}.s{width}.steady",
        "phase": "steady",
        "minimum_start": 1,
        "startup_requires_strict_s1": True,
        "runtime_eligible": True,
        "context": context,
        "width": width,
        "om_sha256": artifact["sha256"],
        "build_manifest_sha256": build["sha256"],
        "qualification_schema": SCHEMA,
    }
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "context": context,
        "width": width,
        "threshold_exclusive": threshold,
        "minimum_cosine": minimum,
        "om": artifact,
        "build_manifest": build,
        "required_starts": list(required_starts(context, width)),
        "abi": abi,
        "captures": normalized,
        "workloads": normalized_workloads,
        "performance": normalized_performance,
        "activation": activation,
    }


def qualify_file(*, evidence: Path, output: Path) -> dict[str, object]:
    """Verify one portable evidence document and write its PASS record."""
    value = json.loads(evidence.read_text(encoding="utf-8"))
    required = {
        "context", "width", "om", "build_manifest", "captures",
        "workloads", "performance",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise PrefillBlockQualificationError(
            f"evidence fields must be exactly {sorted(required)}")
    report = build_qualification(**value, artifact_root=evidence.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _captures(
    captures: Sequence[Mapping[str, object]], *, context: int, width: int,
    threshold: float,
) -> tuple[list[dict[str, object]], float]:
    if not isinstance(captures, Sequence) or isinstance(captures, (str, bytes)):
        raise PrefillBlockQualificationError("captures must be a sequence")
    normalized: list[dict[str, object]] = []
    seen: set[int] = set()
    descriptor_hashes: set[str] = set()
    minimum = 1.0
    rows = LAYERS * KV_HEADS * width
    capture_fields = {
        "start", "capture_sha256", "physical_descriptor_sha256",
        "publisher_source_sha256", "publisher_source_dtype",
        "publisher_source_layout", "descriptor_exact", "mask_sha256",
        "mask_bytes_exact", "rope_sha256", "rope_bytes_exact", "kv_rows",
        "kv_rows_exact", "prefill_decode_handoff", "token_exact",
        "board_pass", *_COSINE_GROUPS,
    }
    for capture in captures:
        if not isinstance(capture, Mapping) or set(capture) != capture_fields:
            raise PrefillBlockQualificationError(
                f"capture fields must be exactly {sorted(capture_fields)}")
        start = capture.get("start")
        if type(start) is not int or not 1 <= start <= context - width:
            raise PrefillBlockQualificationError(
                "steady capture start must be within [1, context-width]")
        if start in seen:
            raise PrefillBlockQualificationError("capture starts must not repeat")
        seen.add(start)
        source_hashes = capture.get("publisher_source_sha256")
        if not isinstance(source_hashes, Mapping) or set(source_hashes) != {"k", "v"}:
            raise PrefillBlockQualificationError(
                "publisher_source_sha256 must contain k and v")
        if capture.get("publisher_source_dtype") != PUBLISHER_DTYPE:
            raise PrefillBlockQualificationError("publisher source dtype must be FP32")
        if capture.get("publisher_source_layout") != PUBLISHER_LAYOUT:
            raise PrefillBlockQualificationError(
                f"publisher source layout must be {PUBLISHER_LAYOUT}")
        kv_rows = capture.get("kv_rows")
        if not isinstance(kv_rows, Mapping) or set(kv_rows) != {"k", "v"}:
            raise PrefillBlockQualificationError("kv_rows must contain k and v")
        if any(type(kv_rows[role]) is not int or kv_rows[role] != rows
               for role in ("k", "v")):
            raise PrefillBlockQualificationError(
                f"each K/V role must publish exactly {rows} rows (24*2*width)")
        descriptor_hash = _hash(
            capture.get("physical_descriptor_sha256"),
            "physical_descriptor_sha256")
        descriptor_hashes.add(descriptor_hash)
        record: dict[str, object] = {
            "start": start,
            "stop": start + width,
            "capture_sha256": _hash(capture.get("capture_sha256"), "capture_sha256"),
            "physical_descriptor_sha256": descriptor_hash,
            "publisher_source_sha256": {
                role: _hash(source_hashes[role], f"publisher_source_sha256.{role}")
                for role in ("k", "v")
            },
            "publisher_source_dtype": PUBLISHER_DTYPE,
            "publisher_source_layout": PUBLISHER_LAYOUT,
            "mask_sha256": _hash(capture.get("mask_sha256"), "mask_sha256"),
            "rope_sha256": _hash(capture.get("rope_sha256"), "rope_sha256"),
            "kv_rows": {"k": rows, "v": rows},
        }
        if record["mask_sha256"] != canonical_mask_sha256(
                context, width, start):
            raise PrefillBlockQualificationError(
                "mask_sha256 does not match the canonical absolute mask")
        for field in (
            "descriptor_exact", "mask_bytes_exact", "rope_bytes_exact",
            "kv_rows_exact", "prefill_decode_handoff", "token_exact",
            "board_pass",
        ):
            if capture.get(field) is not True:
                raise PrefillBlockQualificationError(f"{field} must be true")
            record[field] = True
        for group in _COSINE_GROUPS:
            values = capture.get(group)
            if not isinstance(values, Mapping) or set(values) != set(_COSINE_NAMES):
                raise PrefillBlockQualificationError(
                    f"{group} must contain hidden, k and v")
            scored = {name: _real(values[name], f"{group}.{name}")
                      for name in _COSINE_NAMES}
            if min(scored.values()) <= threshold:
                raise PrefillBlockQualificationError(
                    f"{group} must be strictly greater than {threshold}")
            minimum = min(minimum, *scored.values())
            record[group] = scored
        normalized.append(record)

    required = set(required_starts(context, width))
    missing = sorted(required - seen)
    if missing:
        raise PrefillBlockQualificationError(
            f"missing required block starts: {missing}")
    extra = sorted(seen - required)
    if extra:
        raise PrefillBlockQualificationError(
            f"unexpected block starts outside the qualification matrix: {extra}")
    if len(descriptor_hashes) != 1:
        raise PrefillBlockQualificationError(
            "all captures must bind one physical descriptor SHA-256")
    normalized.sort(key=lambda value: int(value["start"]))
    return normalized, minimum


def _workloads(
    workloads: Sequence[Mapping[str, object]], *, context: int,
) -> list[dict[str, object]]:
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)):
        raise PrefillBlockQualificationError("workloads must be a sequence")
    base_fields = {
        "kind", "capture_sha256", "prompt_sha256", "output_tokens_sha256",
        "board_pass", "token_exact",
    }
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for workload in workloads:
        if not isinstance(workload, Mapping):
            raise PrefillBlockQualificationError("workload must be an object")
        kind = workload.get("kind")
        if kind not in _WORKLOAD_KINDS or kind in seen:
            raise PrefillBlockQualificationError(
                "workload kinds must be unique eos, english, chinese and context_boundary")
        expected = set(base_fields)
        if kind == "eos":
            expected.add("eos_exact")
        if kind == "context_boundary":
            expected.update(("boundary_exact", "terminal_position"))
        if set(workload) != expected:
            raise PrefillBlockQualificationError(
                f"{kind} workload fields must be exactly {sorted(expected)}")
        record: dict[str, object] = {
            "kind": kind,
            "capture_sha256": _hash(
                workload.get("capture_sha256"), f"workloads.{kind}.capture_sha256"),
            "prompt_sha256": _hash(
                workload.get("prompt_sha256"), f"workloads.{kind}.prompt_sha256"),
            "output_tokens_sha256": _hash(
                workload.get("output_tokens_sha256"),
                f"workloads.{kind}.output_tokens_sha256"),
        }
        for field in ("board_pass", "token_exact"):
            if workload.get(field) is not True:
                raise PrefillBlockQualificationError(
                    f"{kind}.{field} must be true")
            record[field] = True
        if kind == "eos":
            if workload.get("eos_exact") is not True:
                raise PrefillBlockQualificationError("eos.eos_exact must be true")
            record["eos_exact"] = True
        if kind == "context_boundary":
            if workload.get("boundary_exact") is not True:
                raise PrefillBlockQualificationError(
                    "context_boundary.boundary_exact must be true")
            if workload.get("terminal_position") != context - 1:
                raise PrefillBlockQualificationError(
                    "context boundary must terminate at context-1")
            record["boundary_exact"] = True
            record["terminal_position"] = context - 1
        normalized.append(record)
        seen.add(str(kind))
    missing = sorted(set(_WORKLOAD_KINDS) - seen)
    if missing:
        raise PrefillBlockQualificationError(
            f"missing required end-to-end workloads: {missing}")
    normalized.sort(key=lambda value: _WORKLOAD_KINDS.index(str(value["kind"])))
    return normalized


def _performance(
    value: Sequence[Mapping[str, object]], *, width: int, om_sha256: str,
) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PrefillBlockQualificationError(
            "performance must be a sequence of ladder comparisons")
    expected_baselines = _PERFORMANCE_BASELINES[width]
    if len(value) != len(expected_baselines):
        raise PrefillBlockQualificationError(
            f"S{width} performance must compare baselines "
            f"{list(expected_baselines)}")
    by_baseline: dict[int, Mapping[str, object]] = {}
    for comparison in value:
        if not isinstance(comparison, Mapping):
            raise PrefillBlockQualificationError(
                "performance comparison must be an object")
        baseline = comparison.get("baseline_width")
        if type(baseline) is not int or baseline in by_baseline:
            raise PrefillBlockQualificationError(
                "performance baseline widths must be unique integers")
        by_baseline[baseline] = comparison
    if set(by_baseline) != set(expected_baselines):
        raise PrefillBlockQualificationError(
            f"S{width} performance must compare baselines "
            f"{list(expected_baselines)}")
    return [_performance_comparison(
        by_baseline[baseline], width=width, baseline_width=baseline,
        om_sha256=om_sha256) for baseline in expected_baselines]


def _performance_comparison(
    value: Mapping[str, object], *, width: int, baseline_width: int,
    om_sha256: str,
) -> dict[str, object]:
    fields = {
        "metric", "board", "tokens", "candidate_width", "baseline_width",
        "candidate_invocations", "baseline_invocations", "warmup_runs",
        "measured_runs", "candidate_ms", "baseline_ms",
        "candidate_om_sha256", "baseline_artifact_sha256",
        "measurement_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PrefillBlockQualificationError(
            f"performance fields must be exactly {sorted(fields)}")
    expected_baseline_invocations = width // baseline_width
    exact = {
        "metric": "board-wall-time-ms",
        "board": "Hi3403",
        "tokens": width,
        "candidate_width": width,
        "baseline_width": baseline_width,
        "candidate_invocations": 1,
        "baseline_invocations": expected_baseline_invocations,
    }
    for field, expected in exact.items():
        if value.get(field) != expected:
            raise PrefillBlockQualificationError(
                f"performance.{field} must be {expected!r}")
    warmup = value.get("warmup_runs")
    measured = value.get("measured_runs")
    if type(warmup) is not int or warmup < 1:
        raise PrefillBlockQualificationError("performance.warmup_runs must be >=1")
    if type(measured) is not int or measured < 3:
        raise PrefillBlockQualificationError("performance.measured_runs must be >=3")
    candidate_ms = _real(value.get("candidate_ms"), "performance.candidate_ms")
    baseline_ms = _real(value.get("baseline_ms"), "performance.baseline_ms")
    if candidate_ms <= 0 or baseline_ms <= 0:
        raise PrefillBlockQualificationError("performance times must be positive")
    if candidate_ms >= baseline_ms:
        raise PrefillBlockQualificationError(
            "wide block must be strictly faster than its qualified baseline")
    candidate_hash = _hash(
        value.get("candidate_om_sha256"), "performance.candidate_om_sha256")
    if candidate_hash != om_sha256:
        raise PrefillBlockQualificationError(
            "performance candidate OM hash does not match qualified OM")
    return {
        **exact,
        "warmup_runs": warmup,
        "measured_runs": measured,
        "candidate_ms": candidate_ms,
        "baseline_ms": baseline_ms,
        "speedup": baseline_ms / candidate_ms,
        "candidate_om_sha256": candidate_hash,
        "baseline_artifact_sha256": _hash(
            value.get("baseline_artifact_sha256"),
            "performance.baseline_artifact_sha256"),
        "measurement_sha256": _hash(
            value.get("measurement_sha256"), "performance.measurement_sha256"),
    }


def _geometry(context: int, width: int) -> None:
    if type(context) is not int or context < 128:
        raise PrefillBlockQualificationError("context must be an integer >=128")
    if type(width) is not int or width not in SUPPORTED_WIDTHS:
        raise PrefillBlockQualificationError(
            f"width must be one of {SUPPORTED_WIDTHS}")
    if width >= context or context % 16:
        raise PrefillBlockQualificationError(
            "steady width must fit after strict-S1 startup in a context divisible by 16")


def _artifact(
    value: Mapping[str, object], field: str, root: Path, *, require_json: bool,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise PrefillBlockQualificationError(
            f"{field} must contain path, bytes and sha256")
    declared = value.get("path")
    if not isinstance(declared, str) or not declared or "\x00" in declared:
        raise PrefillBlockQualificationError(f"{field}.path must name a file")
    size = value.get("bytes")
    if type(size) is not int or size <= 0:
        raise PrefillBlockQualificationError(
            f"{field}.bytes must be a positive integer")
    expected_hash = _hash(value.get("sha256"), f"{field}.sha256")
    path = Path(declared)
    actual = path if path.is_absolute() else root / path
    if not actual.is_file():
        raise PrefillBlockQualificationError(
            f"{field}.path is not a real file: {declared}")
    if actual.stat().st_size != size:
        raise PrefillBlockQualificationError(
            f"{field}.bytes does not match the real file")
    digest_builder = hashlib.sha256()
    with actual.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != expected_hash:
        raise PrefillBlockQualificationError(
            f"{field}.sha256 does not match the real file")
    if require_json:
        try:
            manifest = json.loads(actual.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PrefillBlockQualificationError(
                "build_manifest.path must contain valid JSON") from error
        if not isinstance(manifest, dict):
            raise PrefillBlockQualificationError(
                "build_manifest.path must contain a JSON object")
    return {
        "path": declared,
        "bytes": size,
        "sha256": expected_hash,
        "file_verified": True,
    }


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PrefillBlockQualificationError(
            f"{field} must be a lowercase SHA-256")
    return value


def _real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillBlockQualificationError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise PrefillBlockQualificationError(f"{field} must be finite")
    return result


def build_release_qualification(
    *,
    context: int,
    width: int,
    om: Mapping[str, object],
    head_om: Mapping[str, object],
    embedding: Mapping[str, object],
    build_manifest: Mapping[str, object],
    runner: Mapping[str, object],
    executor: Mapping[str, object],
    ready_descriptor: Mapping[str, object],
    capture_artifacts: Sequence[Mapping[str, object]],
    workload_artifacts: Sequence[Mapping[str, object]],
    performance_artifacts: Sequence[Mapping[str, object]],
    baselines: Sequence[Mapping[str, object]],
    mmz_observation: Mapping[str, object],
    artifact_root: Path,
    threshold_exclusive: float = POLICY_THRESHOLD_EXCLUSIVE,
) -> dict[str, object]:
    """Build a release-eligible v4 record from real, content-bound evidence.

    Unlike :func:`build_qualification` (the retained v2 development contract),
    every capture, workload and timing comparison is loaded from a real JSON
    artifact.  Runtime binaries, baseline qualifications/OMs and a clean-board
    MMZ observation are also byte/size/SHA bound.  ``admission_bytes`` is never
    accepted as input: it is derived as ``before_available - after_available``.
    """
    _geometry(context, width)
    threshold = _real(threshold_exclusive, "threshold_exclusive")
    if threshold != POLICY_THRESHOLD_EXCLUSIVE:
        raise PrefillBlockQualificationError(
            f"threshold must remain exactly {POLICY_THRESHOLD_EXCLUSIVE}")
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise PrefillBlockQualificationError("artifact_root must be a directory")

    model = _release_artifact(om, "om", root, require_json=False)
    head = _release_artifact(head_om, "head_om", root, require_json=False)
    embedding_artifact = _release_artifact(
        embedding, "embedding", root, require_json=False)
    build = _release_artifact(
        build_manifest, "build_manifest", root, require_json=True)
    runtime = {
        "runner": _release_artifact(
            runner, "runner", root, require_json=False),
        "executor": _release_artifact(
            executor, "executor", root, require_json=False),
        "ready_descriptor": _release_artifact(
            ready_descriptor, "ready_descriptor", root, require_json=False),
    }
    runtime_hashes = {
        name: str(value["sha256"]) for name, value in runtime.items()}

    normalized_captures, minimum = _release_captures(
        capture_artifacts, root=root, context=context, width=width,
        threshold=threshold, model_sha256=str(model["sha256"]),
        runner_sha256=runtime_hashes["runner"],
        executor_sha256=runtime_hashes["executor"],
        descriptor_sha256=runtime_hashes["ready_descriptor"])
    normalized_workloads = _release_workloads(
        workload_artifacts, root=root, context=context, width=width,
        model_sha256=str(model["sha256"]),
        head_om_sha256=str(head["sha256"]),
        embedding_sha256=str(embedding_artifact["sha256"]),
        runner_sha256=runtime_hashes["runner"],
        executor_sha256=runtime_hashes["executor"])
    normalized_baselines = _release_baselines(
        baselines, root=root, context=context, width=width,
        head_om_sha256=str(head["sha256"]),
        embedding_sha256=str(embedding_artifact["sha256"]))
    normalized_performance = _release_performance(
        performance_artifacts, root=root, context=context, width=width,
        om_sha256=str(model["sha256"]),
        runner_sha256=runtime_hashes["runner"],
        executor_sha256=runtime_hashes["executor"],
        baselines=normalized_baselines)
    admission = _release_mmz_observation(
        mmz_observation, root=root, context=context, width=width,
        model_sha256=str(model["sha256"]),
        runner_sha256=runtime_hashes["runner"],
        executor_sha256=runtime_hashes["executor"],
        descriptor_sha256=runtime_hashes["ready_descriptor"])
    abi = logical_abi(context, width)
    activation = {
        "key": f"ctx{context}.s{width}.steady",
        "phase": "steady",
        "minimum_start": 1,
        "startup_requires_strict_s1": True,
        "runtime_eligible": True,
        "release_eligible": True,
        "context": context,
        "width": width,
        "om_sha256": model["sha256"],
        "head_om_sha256": head["sha256"],
        "embedding_sha256": embedding_artifact["sha256"],
        "build_manifest_sha256": build["sha256"],
        "runner_sha256": runtime_hashes["runner"],
        "executor_sha256": runtime_hashes["executor"],
        "ready_descriptor_sha256": runtime_hashes["ready_descriptor"],
        "admission_bytes": admission["admission_bytes"],
        "qualification_schema": RELEASE_SCHEMA,
    }
    return {
        "schema": RELEASE_SCHEMA,
        "status": "PASS",
        "release_eligible": True,
        "context": context,
        "width": width,
        "threshold_exclusive": threshold,
        "minimum_cosine": minimum,
        "om": model,
        "head_om": head,
        "embedding": embedding_artifact,
        "build_manifest": build,
        "runtime": runtime,
        "required_starts": list(required_starts(context, width)),
        "abi": abi,
        "captures": normalized_captures,
        "workloads": normalized_workloads,
        "baselines": normalized_baselines,
        "performance": normalized_performance,
        "mmz_admission": admission,
        "activation": activation,
    }


def qualify_release_file(*, evidence: Path, output: Path) -> dict[str, object]:
    """Verify a portable v4 evidence index and write its release PASS record."""
    value = _read_json_path(Path(evidence), "release evidence")
    required = {
        "context", "width", "om", "head_om", "embedding",
        "build_manifest", "runner", "executor",
        "ready_descriptor", "capture_artifacts", "workload_artifacts",
        "performance_artifacts", "baselines", "mmz_observation",
    }
    if set(value) != required:
        raise PrefillBlockQualificationError(
            f"release evidence fields must be exactly {sorted(required)}")
    report = build_release_qualification(
        **value, artifact_root=Path(evidence).parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def verify_release_qualification(
    report: Mapping[str, object], *, artifact_root: Path,
) -> dict[str, object]:
    """Rebuild a v4 qualification from its bound files and require equality.

    This is also used transitively for S32/S128 baselines, so a file which only
    claims the v3 schema/status cannot stand in for a complete qualification.
    """
    fields = {
        "schema", "status", "release_eligible", "context", "width",
        "threshold_exclusive", "minimum_cosine", "om", "head_om",
        "embedding", "build_manifest",
        "runtime", "required_starts", "abi", "captures", "workloads",
        "baselines", "performance", "mmz_admission", "activation",
    }
    if not isinstance(report, Mapping) or set(report) != fields \
            or report.get("schema") != RELEASE_SCHEMA \
            or report.get("status") != "PASS" \
            or report.get("release_eligible") is not True:
        raise PrefillBlockQualificationError(
            "qualification is not a complete release v4 PASS record")
    runtime = report.get("runtime")
    captures = report.get("captures")
    workloads = report.get("workloads")
    performance = report.get("performance")
    baselines = report.get("baselines")
    mmz = report.get("mmz_admission")
    if not isinstance(runtime, Mapping) or set(runtime) != {
            "runner", "executor", "ready_descriptor"} \
            or not isinstance(captures, Sequence) \
            or isinstance(captures, (str, bytes)) \
            or not isinstance(workloads, Sequence) \
            or isinstance(workloads, (str, bytes)) \
            or not isinstance(performance, Sequence) \
            or isinstance(performance, (str, bytes)) \
            or not isinstance(baselines, Sequence) \
            or isinstance(baselines, (str, bytes)) \
            or not isinstance(mmz, Mapping):
        raise PrefillBlockQualificationError(
            "qualification release evidence index is malformed")
    try:
        rebuilt = build_release_qualification(
            context=report["context"],
            width=report["width"],
            om=_artifact_declaration(report["om"], "om"),
            head_om=_artifact_declaration(report["head_om"], "head_om"),
            embedding=_artifact_declaration(
                report["embedding"], "embedding"),
            build_manifest=_artifact_declaration(
                report["build_manifest"], "build_manifest"),
            runner=_artifact_declaration(runtime["runner"], "runner"),
            executor=_artifact_declaration(runtime["executor"], "executor"),
            ready_descriptor=_artifact_declaration(
                runtime["ready_descriptor"], "ready_descriptor"),
            capture_artifacts=[_artifact_declaration(
                item["artifact"], "capture.artifact")
                for item in captures if isinstance(item, Mapping)],
            workload_artifacts=[_artifact_declaration(
                item["artifact"], "workload.artifact")
                for item in workloads if isinstance(item, Mapping)],
            performance_artifacts=[_artifact_declaration(
                item["measurement_artifact"],
                "performance.measurement_artifact")
                for item in performance if isinstance(item, Mapping)],
            baselines=[{
                "width": item["width"],
                "qualification": _artifact_declaration(
                    item["qualification"], "baseline.qualification"),
                "om": _artifact_declaration(item["om"], "baseline.om"),
            } for item in baselines if isinstance(item, Mapping)],
            mmz_observation=_artifact_declaration(
                mmz["observation_artifact"], "mmz.observation_artifact"),
            artifact_root=artifact_root,
        )
    except (KeyError, TypeError) as error:
        raise PrefillBlockQualificationError(
            "qualification release evidence index is malformed") from error
    if rebuilt != report:
        raise PrefillBlockQualificationError(
            "qualification fields do not match its real release evidence")
    return rebuilt


def _artifact_declaration(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
            "path", "bytes", "sha256", "file_verified"} \
            or value.get("file_verified") is not True:
        raise PrefillBlockQualificationError(
            f"{field} is not a verified artifact descriptor")
    return {key: value[key] for key in ("path", "bytes", "sha256")}


def _release_artifact(
    value: Mapping[str, object], field: str, root: Path, *, require_json: bool,
) -> dict[str, object]:
    """Verify one portable artifact without allowing path or symlink escape."""
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise PrefillBlockQualificationError(
            f"{field} must contain path, bytes and sha256")
    declared = value.get("path")
    if not isinstance(declared, str) or not declared or "\x00" in declared:
        raise PrefillBlockQualificationError(f"{field}.path must name a file")
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        raise PrefillBlockQualificationError(
            f"{field}.path must stay under artifact_root")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PrefillBlockQualificationError(
                f"{field}.path must not traverse a symlink")
    try:
        actual = cursor.resolve(strict=True)
    except OSError as error:
        raise PrefillBlockQualificationError(
            f"{field}.path is unavailable: {error}") from error
    if not actual.is_file() or root not in actual.parents:
        raise PrefillBlockQualificationError(
            f"{field}.path must be a file under artifact_root")
    size = value.get("bytes")
    if type(size) is not int or size <= 0 or actual.stat().st_size != size:
        raise PrefillBlockQualificationError(
            f"{field}.bytes does not match the real file")
    expected_hash = _hash(value.get("sha256"), f"{field}.sha256")
    digest = _file_sha256(actual)
    if digest != expected_hash:
        raise PrefillBlockQualificationError(
            f"{field}.sha256 does not match the real file")
    if require_json:
        _read_json_path(actual, field)
    return {
        "path": declared,
        "bytes": size,
        "sha256": expected_hash,
        "file_verified": True,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_path(path: Path, field: str) -> dict[str, object]:
    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise PrefillBlockQualificationError(
                    f"{field} contains duplicate key {key!r}")
            result[key] = item
        return result

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrefillBlockQualificationError(
            f"{field} must contain valid JSON: {error}") from error
    if not isinstance(result, dict):
        raise PrefillBlockQualificationError(
            f"{field} must contain one JSON object")
    return result


def _evidence_json(
    descriptor: Mapping[str, object], field: str, root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = _release_artifact(
        descriptor, field, root, require_json=True)
    payload = _read_json_path(root / str(artifact["path"]), field)
    return artifact, payload


def _release_captures(
    descriptors: Sequence[Mapping[str, object]], *, root: Path, context: int,
    width: int, threshold: float, model_sha256: str, runner_sha256: str,
    executor_sha256: str, descriptor_sha256: str,
) -> tuple[list[dict[str, object]], float]:
    if not isinstance(descriptors, Sequence) or isinstance(
            descriptors, (str, bytes)):
        raise PrefillBlockQualificationError(
            "capture_artifacts must be a sequence")
    records: list[dict[str, object]] = []
    artifacts: dict[int, dict[str, object]] = {}
    metadata = {
        "schema": CAPTURE_SCHEMA,
        "board": "Hi3403",
        "context": context,
        "width": width,
        "model_sha256": model_sha256,
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
        "ready_descriptor_sha256": descriptor_sha256,
    }
    evidence_fields = {
        *metadata, "start", "physical_descriptor_sha256",
        "publisher_source_sha256", "publisher_source_dtype",
        "publisher_source_layout", "descriptor_exact", "mask_sha256",
        "mask_bytes_exact", "rope_sha256", "rope_bytes_exact", "kv_rows",
        "kv_rows_exact", "prefill_decode_handoff", "token_exact",
        "board_pass", *_COSINE_GROUPS,
    }
    for index, item in enumerate(descriptors):
        artifact, payload = _evidence_json(
            item, f"capture_artifacts[{index}]", root)
        if set(payload) != evidence_fields:
            raise PrefillBlockQualificationError(
                "capture evidence fields do not match the release v4 schema")
        for field, expected in metadata.items():
            if payload.get(field) != expected:
                raise PrefillBlockQualificationError(
                    f"capture evidence {field} is not bound to this candidate")
        if payload.get("physical_descriptor_sha256") != descriptor_sha256:
            raise PrefillBlockQualificationError(
                "capture physical descriptor does not match the ready descriptor")
        start = payload.get("start")
        if type(start) is not int or start in artifacts:
            raise PrefillBlockQualificationError(
                "capture evidence starts must be unique integers")
        artifacts[start] = artifact
        legacy = {field: payload[field] for field in evidence_fields - set(metadata)}
        legacy["capture_sha256"] = artifact["sha256"]
        records.append(legacy)
    normalized, minimum = _captures(
        records, context=context, width=width, threshold=threshold)
    for record in normalized:
        record["artifact"] = artifacts[int(record["start"])]
    return normalized, minimum


def _release_workloads(
    descriptors: Sequence[Mapping[str, object]], *, root: Path, context: int,
    width: int, model_sha256: str, runner_sha256: str,
    executor_sha256: str, head_om_sha256: str, embedding_sha256: str,
) -> list[dict[str, object]]:
    if not isinstance(descriptors, Sequence) or isinstance(
            descriptors, (str, bytes)):
        raise PrefillBlockQualificationError(
            "workload_artifacts must be a sequence")
    metadata = {
        "schema": WORKLOAD_SCHEMA,
        "board": "Hi3403",
        "context": context,
        "width": width,
        "model_sha256": model_sha256,
        "head_om_sha256": head_om_sha256,
        "embedding_sha256": embedding_sha256,
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
        "token_id_hash_contract": TOKEN_ID_HASH_CONTRACT,
    }
    payloads: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, object]] = {}
    for index, item in enumerate(descriptors):
        artifact, payload = _evidence_json(
            item, f"workload_artifacts[{index}]", root)
        expected = {
            *metadata, "kind", "capture_sha256", "prompt_sha256",
            "output_tokens_sha256", "board_pass", "token_exact",
        }
        if payload.get("kind") == "eos":
            expected.add("eos_exact")
        if payload.get("kind") == "context_boundary":
            expected.update(("boundary_exact", "terminal_position"))
        if set(payload) != expected:
            raise PrefillBlockQualificationError(
                "workload evidence fields do not match the v4 schema")
        for field, expected_value in metadata.items():
            if payload.get(field) != expected_value:
                raise PrefillBlockQualificationError(
                    f"workload evidence {field} is not bound to this candidate")
        kind = payload.get("kind")
        if not isinstance(kind, str) or kind in artifacts:
            raise PrefillBlockQualificationError(
                "workload evidence kinds must be unique strings")
        artifacts[kind] = artifact
        payloads.append({
            field: payload[field] for field in set(payload) - set(metadata)})
    normalized = _workloads(payloads, context=context)
    for record in normalized:
        record["artifact"] = artifacts[str(record["kind"])]
    return normalized


def _verified_strict_s1_identity(
    report: Mapping[str, object], *, qualification_sha256: str, root: Path,
) -> tuple[str, str, str, str, str, str, str, str, str, str, int]:
    """Return the one release S1 trust anchor under a verified baseline.

    The caller must first run the corresponding release verifier.  This helper
    then walks the already verified ladder and makes the transitive S1 identity
    explicit, so an S128 record cannot combine independently valid S32 and S16
    baselines which terminate at different resident S1 deployments.
    """
    schema = report.get("schema")
    if schema == STRICT_S1_BASELINE_SCHEMA:
        runtime = report.get("runtime")
        bootstrap_om = report.get("bootstrap_om")
        canonical_om = report.get("canonical_decode_om")
        head_om = report.get("head_om")
        embedding = report.get("embedding")
        build = report.get("build_manifest")
        mmz = report.get("mmz_residency")
        if not isinstance(runtime, Mapping) \
                or not isinstance(bootstrap_om, Mapping) \
                or not isinstance(canonical_om, Mapping) \
                or not isinstance(head_om, Mapping) \
                or not isinstance(embedding, Mapping) \
                or not isinstance(build, Mapping) or not isinstance(mmz, Mapping):
            raise PrefillBlockQualificationError(
                "strict-S1 identity lineage is incomplete")
        runner = runtime.get("runner")
        executor = runtime.get("executor")
        bootstrap_descriptor = runtime.get("bootstrap_ready_descriptor")
        canonical_descriptor = runtime.get("canonical_ready_descriptor")
        if not all(isinstance(item, Mapping) for item in (
                runner, executor, bootstrap_descriptor,
                canonical_descriptor)):
            raise PrefillBlockQualificationError(
                "strict-S1 runtime identity is incomplete")
        resident = mmz.get("resident_bytes")
        if type(resident) is not int or resident <= 0:
            raise PrefillBlockQualificationError(
                "strict-S1 resident identity is invalid")
        return (
            _hash(qualification_sha256, "strict-S1 qualification identity"),
            _hash(bootstrap_om.get("sha256"),
                  "strict-S1 bootstrap OM identity"),
            _hash(canonical_om.get("sha256"),
                  "strict-S1 canonical decode OM identity"),
            _hash(head_om.get("sha256"), "strict-S1 head OM identity"),
            _hash(embedding.get("sha256"), "strict-S1 embedding identity"),
            _hash(build.get("sha256"), "strict-S1 build identity"),
            _hash(runner.get("sha256"), "strict-S1 runner identity"),
            _hash(executor.get("sha256"), "strict-S1 executor identity"),
            _hash(bootstrap_descriptor.get("sha256"),
                  "strict-S1 bootstrap descriptor identity"),
            _hash(canonical_descriptor.get("sha256"),
                  "strict-S1 canonical descriptor identity"),
            resident,
        )
    if schema != RELEASE_SCHEMA:
        raise PrefillBlockQualificationError(
            "baseline does not terminate in a release S1 identity")
    baselines = report.get("baselines")
    if not isinstance(baselines, Sequence) or isinstance(
            baselines, (str, bytes)) or not baselines:
        raise PrefillBlockQualificationError(
            "wide baseline has no release ladder")
    identities: list[
        tuple[str, str, str, str, str, str, str, str, str, str, int]] = []
    for index, item in enumerate(baselines):
        if not isinstance(item, Mapping):
            raise PrefillBlockQualificationError(
                "wide baseline ladder entry is malformed")
        declaration = _artifact_declaration(
            item.get("qualification"),
            f"wide baseline[{index}].qualification")
        qualification = _release_artifact(
            declaration, f"wide baseline[{index}].qualification", root,
            require_json=True)
        nested = _read_json_path(
            root / str(qualification["path"]),
            f"wide baseline[{index}].qualification")
        identities.append(_verified_strict_s1_identity(
            nested, qualification_sha256=str(qualification["sha256"]),
            root=root))
    if any(identity != identities[0] for identity in identities[1:]):
        raise PrefillBlockQualificationError(
            "wide baseline ladder does not share one strict-S1 identity")
    return identities[0]


def _release_baselines(
    values: Sequence[Mapping[str, object]], *, root: Path, context: int,
    width: int, head_om_sha256: str, embedding_sha256: str,
) -> list[dict[str, object]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PrefillBlockQualificationError("baselines must be a sequence")
    expected_widths = _PERFORMANCE_BASELINES[width]
    if len(values) != len(expected_widths):
        raise PrefillBlockQualificationError(
            f"S{width} baselines must be {list(expected_widths)}")
    by_width: dict[int, dict[str, object]] = {}
    strict_s1_identities: list[
        tuple[str, str, str, str, str, str, str, str, str, str, int]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {
                "width", "qualification", "om"}:
            raise PrefillBlockQualificationError(
                "baseline must contain width, qualification and om")
        baseline_width = value.get("width")
        if type(baseline_width) is not int or baseline_width in by_width:
            raise PrefillBlockQualificationError(
                "baseline widths must be unique integers")
        qualification = _release_artifact(
            value["qualification"], f"baselines[{index}].qualification",
            root, require_json=True)
        om = _release_artifact(
            value["om"], f"baselines[{index}].om", root,
            require_json=False)
        report = _read_json_path(
            root / str(qualification["path"]),
            f"baselines[{index}].qualification")
        if report.get("status") != "PASS" \
                or report.get("release_eligible") is not True \
                or report.get("context") != context \
                or report.get("width") != baseline_width:
            raise PrefillBlockQualificationError(
                "baseline qualification is not a release PASS for this context")
        if baseline_width == 1:
            if report.get("schema") == STRICT_S1_DEVELOPMENT_SCHEMA:
                raise PrefillBlockQualificationError(
                    "development strict-S1 v1 baseline is not release eligible")
            if report.get("schema") in {
                    STRICT_S1_LEGACY_RELEASE_SCHEMA,
                    STRICT_S1_PREVIOUS_RELEASE_SCHEMA}:
                raise PrefillBlockQualificationError(
                    "legacy strict-S1 v2/v3 baseline is not release activatable")
            if report.get("schema") != STRICT_S1_BASELINE_SCHEMA:
                raise PrefillBlockQualificationError(
                    "S1 baseline must use the strict-S1 release schema")
            verify_strict_s1_release_qualification(
                report, artifact_root=root)
            declared = report.get("canonical_decode_om")
            declared_om = declared.get("sha256") if isinstance(
                declared, Mapping) else None
        else:
            if report.get("schema") == LEGACY_RELEASE_SCHEMA:
                raise PrefillBlockQualificationError(
                    "legacy wide v3 baseline is not release activatable")
            if report.get("schema") != RELEASE_SCHEMA:
                raise PrefillBlockQualificationError(
                    "wide baseline must use a release v4 qualification")
            verify_release_qualification(report, artifact_root=root)
            declared = report.get("om")
            declared_om = declared.get("sha256") if isinstance(
                declared, Mapping) else None
        if declared_om != om["sha256"]:
            raise PrefillBlockQualificationError(
                "baseline qualification does not bind the baseline OM")
        identity = _verified_strict_s1_identity(
            report, qualification_sha256=str(qualification["sha256"]),
            root=root)
        if identity[3] != head_om_sha256 or identity[4] != embedding_sha256:
            raise PrefillBlockQualificationError(
                "release candidate head/embedding identity does not match its "
                "strict-S1 baseline")
        strict_s1_identities.append(identity)
        by_width[baseline_width] = {
            "width": baseline_width,
            "qualification": qualification,
            "om": om,
        }
    if set(by_width) != set(expected_widths):
        raise PrefillBlockQualificationError(
            f"S{width} baselines must be {list(expected_widths)}")
    if not strict_s1_identities or any(
            identity != strict_s1_identities[0]
            for identity in strict_s1_identities[1:]):
        raise PrefillBlockQualificationError(
            "release baseline ladder does not share one strict-S1 identity")
    return [by_width[item] for item in expected_widths]


def _timing_series(value: object, field: str, count: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) \
            or len(value) != count:
        raise PrefillBlockQualificationError(
            f"{field} must contain exactly {count} samples")
    result = [_real(item, field) for item in value]
    if any(item <= 0 for item in result):
        raise PrefillBlockQualificationError(
            f"{field} samples must be positive")
    return result


def _release_performance(
    descriptors: Sequence[Mapping[str, object]], *, root: Path, context: int,
    width: int, om_sha256: str, runner_sha256: str, executor_sha256: str,
    baselines: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected_widths = _PERFORMANCE_BASELINES[width]
    if not isinstance(descriptors, Sequence) or isinstance(
            descriptors, (str, bytes)) or len(descriptors) != len(expected_widths):
        raise PrefillBlockQualificationError(
            f"S{width} performance artifacts must compare "
            f"{list(expected_widths)}")
    baseline_by_width = {int(item["width"]): item for item in baselines}
    by_width: dict[int, dict[str, object]] = {}
    fields = {
        "schema", "board", "context", "width", "metric", "tokens",
        "candidate_width", "baseline_width", "candidate_invocations",
        "baseline_invocations", "warmup_runs", "measured_runs",
        "candidate_warmup_ms", "baseline_warmup_ms",
        "candidate_samples_ms", "baseline_samples_ms",
        "candidate_om_sha256", "baseline_qualification_sha256",
        "baseline_om_sha256", "runner_sha256", "executor_sha256",
    }
    for index, item in enumerate(descriptors):
        artifact, payload = _evidence_json(
            item, f"performance_artifacts[{index}]", root)
        if set(payload) != fields:
            raise PrefillBlockQualificationError(
                "performance evidence fields do not match the release v4 schema")
        baseline_width = payload.get("baseline_width")
        if type(baseline_width) is not int or baseline_width in by_width \
                or baseline_width not in baseline_by_width:
            raise PrefillBlockQualificationError(
                "performance evidence baseline is not in the release ladder")
        baseline = baseline_by_width[baseline_width]
        exact = {
            "schema": PERFORMANCE_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": width,
            "metric": "board-wall-time-ms",
            "tokens": width,
            "candidate_width": width,
            "baseline_width": baseline_width,
            "candidate_invocations": 1,
            "baseline_invocations": width // baseline_width,
            "candidate_om_sha256": om_sha256,
            "baseline_qualification_sha256": baseline["qualification"]["sha256"],
            "baseline_om_sha256": baseline["om"]["sha256"],
            "runner_sha256": runner_sha256,
            "executor_sha256": executor_sha256,
        }
        for field, expected in exact.items():
            if payload.get(field) != expected:
                raise PrefillBlockQualificationError(
                    f"performance evidence {field} is not bound to the release ladder")
        warmup = payload.get("warmup_runs")
        measured = payload.get("measured_runs")
        if type(warmup) is not int or warmup < 1:
            raise PrefillBlockQualificationError(
                "performance warmup_runs must be >=1")
        if type(measured) is not int or measured < 3:
            raise PrefillBlockQualificationError(
                "performance measured_runs must be >=3")
        candidate_warmup = _timing_series(
            payload.get("candidate_warmup_ms"), "candidate_warmup_ms", warmup)
        baseline_warmup = _timing_series(
            payload.get("baseline_warmup_ms"), "baseline_warmup_ms", warmup)
        candidate_samples = _timing_series(
            payload.get("candidate_samples_ms"), "candidate_samples_ms", measured)
        baseline_samples = _timing_series(
            payload.get("baseline_samples_ms"), "baseline_samples_ms", measured)
        candidate_ms = sum(candidate_samples) / measured
        baseline_ms = sum(baseline_samples) / measured
        if candidate_ms >= baseline_ms:
            raise PrefillBlockQualificationError(
                "wide block must be strictly faster than its qualified baseline")
        by_width[baseline_width] = {
            "metric": "board-wall-time-ms",
            "board": "Hi3403",
            "tokens": width,
            "candidate_width": width,
            "baseline_width": baseline_width,
            "candidate_invocations": 1,
            "baseline_invocations": width // baseline_width,
            "warmup_runs": warmup,
            "measured_runs": measured,
            "candidate_warmup_ms": candidate_warmup,
            "baseline_warmup_ms": baseline_warmup,
            "candidate_samples_ms": candidate_samples,
            "baseline_samples_ms": baseline_samples,
            "candidate_ms": candidate_ms,
            "baseline_ms": baseline_ms,
            "speedup": baseline_ms / candidate_ms,
            "candidate_om_sha256": om_sha256,
            "baseline_qualification_sha256": baseline["qualification"]["sha256"],
            "baseline_om_sha256": baseline["om"]["sha256"],
            "runner_sha256": runner_sha256,
            "executor_sha256": executor_sha256,
            "measurement_artifact": artifact,
        }
    if set(by_width) != set(expected_widths):
        raise PrefillBlockQualificationError(
            f"S{width} performance artifacts must compare "
            f"{list(expected_widths)}")
    return [by_width[item] for item in expected_widths]


def _release_mmz_observation(
    descriptor: Mapping[str, object], *, root: Path, context: int, width: int,
    model_sha256: str, runner_sha256: str, executor_sha256: str,
    descriptor_sha256: str,
) -> dict[str, object]:
    artifact, payload = _evidence_json(descriptor, "mmz_observation", root)
    fields = {
        "schema", "status", "board", "clean_board", "context", "width",
        "model_sha256", "runner_sha256", "executor_sha256",
        "ready_descriptor_sha256", "before_available_bytes",
        "after_available_bytes",
    }
    exact = {
        "schema": MMZ_SCHEMA,
        "status": "PASS",
        "board": "Hi3403",
        "clean_board": True,
        "context": context,
        "width": width,
        "model_sha256": model_sha256,
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
        "ready_descriptor_sha256": descriptor_sha256,
    }
    if set(payload) != fields:
        raise PrefillBlockQualificationError(
            "MMZ observation fields do not match the release v4 schema")
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise PrefillBlockQualificationError(
                f"MMZ observation {field} is not bound to this candidate")
    before = payload.get("before_available_bytes")
    after = payload.get("after_available_bytes")
    if type(before) is not int or type(after) is not int \
            or before <= 0 or after < 0 or after >= before:
        raise PrefillBlockQualificationError(
            "MMZ before/after bytes must show a positive clean-board delta")
    return {
        "observation_artifact": artifact,
        "clean_board": True,
        "before_available_bytes": before,
        "after_available_bytes": after,
        "admission_bytes": before - after,
    }


def build_strict_s1_release_qualification(
    *,
    context: int,
    bootstrap_om: Mapping[str, object],
    canonical_decode_om: Mapping[str, object],
    head_om: Mapping[str, object],
    embedding: Mapping[str, object],
    build_manifest: Mapping[str, object],
    runner: Mapping[str, object],
    executor: Mapping[str, object],
    bootstrap_ready_descriptor: Mapping[str, object],
    canonical_ready_descriptor: Mapping[str, object],
    capture_artifacts: Sequence[Mapping[str, object]],
    workload_artifacts: Sequence[Mapping[str, object]],
    mmz_observation: Mapping[str, object],
    artifact_root: Path,
    threshold_exclusive: float = POLICY_THRESHOLD_EXCLUSIVE,
) -> dict[str, object]:
    """Build the only strict-S1 route eligible for release performance.

    The former v1/v2/v3 records are intentionally not accepted here.  A v4
    baseline binds the position-zero bootstrap OM separately from the
    position>=1 canonical decode OM, plus enough token/position evidence to
    make a wide-block speed comparison meaningful.
    """
    positions = required_strict_s1_positions(context)
    threshold = _real(threshold_exclusive, "threshold_exclusive")
    if threshold != POLICY_THRESHOLD_EXCLUSIVE:
        raise PrefillBlockQualificationError(
            f"threshold must remain exactly {POLICY_THRESHOLD_EXCLUSIVE}")
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise PrefillBlockQualificationError("artifact_root must be a directory")
    bootstrap = _release_artifact(
        bootstrap_om, "strict_s1.bootstrap_om", root, require_json=False)
    canonical = _release_artifact(
        canonical_decode_om, "strict_s1.canonical_decode_om", root,
        require_json=False)
    head = _release_artifact(
        head_om, "strict_s1.head_om", root, require_json=False)
    embedding_artifact = _release_artifact(
        embedding, "strict_s1.embedding", root, require_json=False)
    build = _release_artifact(
        build_manifest, "strict_s1.build_manifest", root, require_json=True)
    runtime = {
        "runner": _release_artifact(
            runner, "strict_s1.runner", root, require_json=False),
        "executor": _release_artifact(
            executor, "strict_s1.executor", root, require_json=False),
        "bootstrap_ready_descriptor": _release_artifact(
            bootstrap_ready_descriptor,
            "strict_s1.bootstrap_ready_descriptor", root,
            require_json=False),
        "canonical_ready_descriptor": _release_artifact(
            canonical_ready_descriptor,
            "strict_s1.canonical_ready_descriptor", root,
            require_json=False),
    }
    hashes = {name: str(value["sha256"]) for name, value in runtime.items()}
    captures, minimum = _strict_s1_captures(
        capture_artifacts, root=root, context=context, positions=positions,
        threshold=threshold,
        bootstrap_om_sha256=str(bootstrap["sha256"]),
        canonical_om_sha256=str(canonical["sha256"]),
        runner_sha256=hashes["runner"], executor_sha256=hashes["executor"],
        bootstrap_descriptor_sha256=hashes["bootstrap_ready_descriptor"],
        canonical_descriptor_sha256=hashes["canonical_ready_descriptor"])
    workloads = _strict_s1_workloads(
        workload_artifacts, root=root, context=context,
        bootstrap_om_sha256=str(bootstrap["sha256"]),
        canonical_om_sha256=str(canonical["sha256"]),
        head_om_sha256=str(head["sha256"]),
        embedding_sha256=str(embedding_artifact["sha256"]),
        runner_sha256=hashes["runner"], executor_sha256=hashes["executor"])
    residency = _strict_s1_mmz_observation(
        mmz_observation, root=root, context=context,
        bootstrap_om_sha256=str(bootstrap["sha256"]),
        canonical_om_sha256=str(canonical["sha256"]),
        runner_sha256=hashes["runner"], executor_sha256=hashes["executor"],
        bootstrap_descriptor_sha256=hashes["bootstrap_ready_descriptor"],
        canonical_descriptor_sha256=hashes["canonical_ready_descriptor"])
    baseline = {
        "role": "strict-s1",
        "performance_baseline_eligible": True,
        "accounting": "included-in-base_resident_bytes",
        "bootstrap_om_sha256": bootstrap["sha256"],
        "canonical_decode_om_sha256": canonical["sha256"],
        "head_om_sha256": head["sha256"],
        "embedding_sha256": embedding_artifact["sha256"],
        "build_manifest_sha256": build["sha256"],
        "runner_sha256": hashes["runner"],
        "executor_sha256": hashes["executor"],
        "bootstrap_ready_descriptor_sha256": hashes[
            "bootstrap_ready_descriptor"],
        "canonical_ready_descriptor_sha256": hashes[
            "canonical_ready_descriptor"],
        "resident_bytes": residency["resident_bytes"],
        "qualification_schema": STRICT_S1_BASELINE_SCHEMA,
    }
    return {
        "schema": STRICT_S1_BASELINE_SCHEMA,
        "status": "PASS",
        "release_eligible": True,
        "context": context,
        "width": 1,
        "threshold_exclusive": threshold,
        "minimum_cosine": minimum,
        "bootstrap_om": bootstrap,
        "canonical_decode_om": canonical,
        "head_om": head,
        "embedding": embedding_artifact,
        "build_manifest": build,
        "runtime": runtime,
        "required_positions": list(positions),
        "captures": captures,
        "workloads": workloads,
        "mmz_residency": residency,
        "baseline": baseline,
    }


def qualify_strict_s1_release_file(
    *, evidence: Path, output: Path,
) -> dict[str, object]:
    """Verify a portable strict-S1 release evidence index."""
    value = _read_json_path(Path(evidence), "strict-S1 release evidence")
    required = {
        "context", "bootstrap_om", "canonical_decode_om",
        "head_om", "embedding",
        "build_manifest", "runner", "executor",
        "bootstrap_ready_descriptor", "canonical_ready_descriptor",
        "capture_artifacts", "workload_artifacts", "mmz_observation",
    }
    if set(value) != required:
        raise PrefillBlockQualificationError(
            f"strict-S1 release evidence fields must be exactly {sorted(required)}")
    report = build_strict_s1_release_qualification(
        **value, artifact_root=Path(evidence).parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def verify_strict_s1_release_qualification(
    report: Mapping[str, object], *, artifact_root: Path,
) -> dict[str, object]:
    """Rebuild a strict-S1 v4 route from real evidence and require equality."""
    fields = {
        "schema", "status", "release_eligible", "context", "width",
        "threshold_exclusive", "minimum_cosine", "bootstrap_om",
        "canonical_decode_om", "head_om", "embedding", "build_manifest",
        "runtime",
        "required_positions", "captures", "workloads", "mmz_residency",
        "baseline",
    }
    if not isinstance(report, Mapping) or set(report) != fields \
            or report.get("schema") != STRICT_S1_BASELINE_SCHEMA \
            or report.get("status") != "PASS" \
            or report.get("release_eligible") is not True \
            or report.get("width") != 1:
        raise PrefillBlockQualificationError(
            "strict-S1 baseline is not a complete release v4 PASS record")
    runtime = report.get("runtime")
    captures = report.get("captures")
    workloads = report.get("workloads")
    mmz = report.get("mmz_residency")
    if not isinstance(runtime, Mapping) or set(runtime) != {
            "runner", "executor", "bootstrap_ready_descriptor",
            "canonical_ready_descriptor"} \
            or not isinstance(captures, Sequence) \
            or isinstance(captures, (str, bytes)) \
            or not isinstance(workloads, Sequence) \
            or isinstance(workloads, (str, bytes)) \
            or not isinstance(mmz, Mapping):
        raise PrefillBlockQualificationError(
            "strict-S1 release evidence index is malformed")
    try:
        rebuilt = build_strict_s1_release_qualification(
            context=report["context"],
            bootstrap_om=_artifact_declaration(
                report["bootstrap_om"], "strict_s1.bootstrap_om"),
            canonical_decode_om=_artifact_declaration(
                report["canonical_decode_om"],
                "strict_s1.canonical_decode_om"),
            head_om=_artifact_declaration(
                report["head_om"], "strict_s1.head_om"),
            embedding=_artifact_declaration(
                report["embedding"], "strict_s1.embedding"),
            build_manifest=_artifact_declaration(
                report["build_manifest"], "strict_s1.build_manifest"),
            runner=_artifact_declaration(runtime["runner"], "strict_s1.runner"),
            executor=_artifact_declaration(
                runtime["executor"], "strict_s1.executor"),
            bootstrap_ready_descriptor=_artifact_declaration(
                runtime["bootstrap_ready_descriptor"],
                "strict_s1.bootstrap_ready_descriptor"),
            canonical_ready_descriptor=_artifact_declaration(
                runtime["canonical_ready_descriptor"],
                "strict_s1.canonical_ready_descriptor"),
            capture_artifacts=[_artifact_declaration(
                item["artifact"], "strict_s1.capture.artifact")
                for item in captures if isinstance(item, Mapping)],
            workload_artifacts=[_artifact_declaration(
                item["artifact"], "strict_s1.workload.artifact")
                for item in workloads if isinstance(item, Mapping)],
            mmz_observation=_artifact_declaration(
                mmz["observation_artifact"],
                "strict_s1.mmz.observation_artifact"),
            artifact_root=artifact_root,
        )
    except (KeyError, TypeError) as error:
        raise PrefillBlockQualificationError(
            "strict-S1 release evidence index is malformed") from error
    if rebuilt != report:
        raise PrefillBlockQualificationError(
            "strict-S1 fields do not match real release evidence")
    return rebuilt


def _strict_s1_captures(
    descriptors: Sequence[Mapping[str, object]], *, root: Path, context: int,
    positions: Sequence[int], threshold: float,
    bootstrap_om_sha256: str, canonical_om_sha256: str,
    runner_sha256: str, executor_sha256: str,
    bootstrap_descriptor_sha256: str,
    canonical_descriptor_sha256: str,
) -> tuple[list[dict[str, object]], float]:
    if not isinstance(descriptors, Sequence) or isinstance(
            descriptors, (str, bytes)):
        raise PrefillBlockQualificationError(
            "strict-S1 capture_artifacts must be a sequence")
    metadata = {
        "schema": STRICT_S1_CAPTURE_SCHEMA,
        "board": "Hi3403",
        "context": context,
        "width": 1,
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
    }
    fields = {
        *metadata, "model_sha256", "ready_descriptor_sha256", "position",
        "physical_descriptor_sha256",
        "publisher_source_sha256", "publisher_source_dtype",
        "publisher_source_layout", "descriptor_exact", "mask_sha256",
        "mask_bytes_exact", "rope_sha256", "rope_bytes_exact", "kv_rows",
        "kv_rows_exact", "route_handoff_exact", "token_exact", "board_pass",
        "public_cosines",
    }
    by_position: dict[int, dict[str, object]] = {}
    minimum = 1.0
    for index, descriptor in enumerate(descriptors):
        artifact, payload = _evidence_json(
            descriptor, f"strict_s1.capture_artifacts[{index}]", root)
        if set(payload) != fields:
            raise PrefillBlockQualificationError(
                "strict-S1 capture fields do not match the release schema")
        for field, expected in metadata.items():
            if payload.get(field) != expected:
                raise PrefillBlockQualificationError(
                    f"strict-S1 capture {field} is not bound to the baseline")
        position = payload.get("position")
        if type(position) is not int or position in by_position:
            raise PrefillBlockQualificationError(
                "strict-S1 capture positions must be unique integers")
        expected_model = (
            bootstrap_om_sha256 if position == 0 else canonical_om_sha256)
        expected_descriptor = (
            bootstrap_descriptor_sha256
            if position == 0 else canonical_descriptor_sha256)
        if payload.get("model_sha256") != expected_model \
                or payload.get("ready_descriptor_sha256") != \
                expected_descriptor \
                or payload.get("physical_descriptor_sha256") != \
                expected_descriptor \
                or payload.get("publisher_source_dtype") != "FP32" \
                or payload.get("publisher_source_layout") != \
                PUBLISHER_LAYOUT \
                or payload.get("kv_rows") != {"k": 48, "v": 48}:
            raise PrefillBlockQualificationError(
                "strict-S1 capture physical ABI/descriptor mismatch")
        for field in ("mask_sha256", "rope_sha256"):
            _hash(payload.get(field), f"strict_s1.capture.{field}")
        sources = payload.get("publisher_source_sha256")
        if not isinstance(sources, Mapping) or set(sources) != {"k", "v"}:
            raise PrefillBlockQualificationError(
                "strict-S1 publisher source hashes are incomplete")
        for role in ("k", "v"):
            _hash(sources[role], f"strict_s1.publisher_source_sha256.{role}")
        for gate in (
            "descriptor_exact", "mask_bytes_exact", "rope_bytes_exact",
            "kv_rows_exact", "route_handoff_exact", "token_exact",
            "board_pass",
        ):
            if payload.get(gate) is not True:
                raise PrefillBlockQualificationError(
                    f"strict-S1 capture {gate} must be true")
        scores = payload.get("public_cosines")
        if not isinstance(scores, Mapping) or set(scores) != set(_COSINE_NAMES):
            raise PrefillBlockQualificationError(
                "strict-S1 public_cosines must contain hidden, k and v")
        normalized_scores = {
            role: _real(scores[role], f"strict_s1.public_cosines.{role}")
            for role in _COSINE_NAMES}
        if min(normalized_scores.values()) <= threshold:
            raise PrefillBlockQualificationError(
                f"strict-S1 public cosines must be greater than {threshold}")
        minimum = min(minimum, *normalized_scores.values())
        by_position[position] = {
            "position": position,
            "capture_sha256": artifact["sha256"],
            "physical_descriptor_sha256": expected_descriptor,
            "publisher_source_sha256": dict(sources),
            "publisher_source_dtype": "FP32",
            "publisher_source_layout": PUBLISHER_LAYOUT,
            "descriptor_exact": True,
            "mask_sha256": payload["mask_sha256"],
            "mask_bytes_exact": True,
            "rope_sha256": payload["rope_sha256"],
            "rope_bytes_exact": True,
            "kv_rows": {"k": 48, "v": 48},
            "kv_rows_exact": True,
            "route_handoff_exact": True,
            "token_exact": True,
            "board_pass": True,
            "public_cosines": normalized_scores,
            "artifact": artifact,
        }
    if set(by_position) != set(positions):
        raise PrefillBlockQualificationError(
            "strict-S1 capture position matrix is incomplete or has extras")
    return [by_position[position] for position in positions], minimum


def _strict_s1_workloads(
    descriptors: Sequence[Mapping[str, object]], *, root: Path, context: int,
    bootstrap_om_sha256: str, canonical_om_sha256: str,
    head_om_sha256: str, embedding_sha256: str,
    runner_sha256: str, executor_sha256: str,
) -> list[dict[str, object]]:
    if not isinstance(descriptors, Sequence) or isinstance(
            descriptors, (str, bytes)):
        raise PrefillBlockQualificationError(
            "strict-S1 workload_artifacts must be a sequence")
    metadata = {
        "schema": STRICT_S1_WORKLOAD_SCHEMA,
        "board": "Hi3403",
        "context": context,
        "width": 1,
        "bootstrap_om_sha256": bootstrap_om_sha256,
        "canonical_decode_om_sha256": canonical_om_sha256,
        "head_om_sha256": head_om_sha256,
        "embedding_sha256": embedding_sha256,
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
        "token_id_hash_contract": TOKEN_ID_HASH_CONTRACT,
    }
    by_kind: dict[str, dict[str, object]] = {}
    for index, descriptor in enumerate(descriptors):
        artifact, payload = _evidence_json(
            descriptor, f"strict_s1.workload_artifacts[{index}]", root)
        expected_fields = {
            *metadata, "kind", "capture_sha256", "prompt_sha256",
            "output_tokens_sha256", "generated_tokens", "board_pass",
            "token_exact",
        }
        kind = payload.get("kind")
        if kind == "tokens_48":
            expected_fields.add("sequence_exact")
        if kind == "eos":
            expected_fields.add("eos_exact")
        if kind == "context_boundary":
            expected_fields.update(("boundary_exact", "terminal_position"))
        if set(payload) != expected_fields:
            raise PrefillBlockQualificationError(
                "strict-S1 workload fields do not match the release schema")
        for field, expected in metadata.items():
            if payload.get(field) != expected:
                raise PrefillBlockQualificationError(
                    f"strict-S1 workload {field} is not bound to the baseline")
        if kind not in _STRICT_S1_WORKLOAD_KINDS or kind in by_kind:
            raise PrefillBlockQualificationError(
                "strict-S1 workload kinds are missing, extra or duplicated")
        for field in ("capture_sha256", "prompt_sha256", "output_tokens_sha256"):
            _hash(payload.get(field), f"strict_s1.workload.{field}")
        generated = payload.get("generated_tokens")
        if type(generated) is not int or generated <= 0:
            raise PrefillBlockQualificationError(
                "strict-S1 generated_tokens must be a positive integer")
        if payload.get("board_pass") is not True \
                or payload.get("token_exact") is not True:
            raise PrefillBlockQualificationError(
                "strict-S1 workload board/token gate is not PASS")
        record = {
            key: payload[key] for key in set(payload) - set(metadata)}
        if kind == "tokens_48" and (
                generated != 48 or payload.get("sequence_exact") is not True):
            raise PrefillBlockQualificationError(
                "strict-S1 tokens_48 workload must prove exactly 48 tokens")
        if kind == "eos" and payload.get("eos_exact") is not True:
            raise PrefillBlockQualificationError(
                "strict-S1 EOS workload is not exact")
        if kind == "context_boundary" and (
                payload.get("boundary_exact") is not True
                or payload.get("terminal_position") != context - 1):
            raise PrefillBlockQualificationError(
                "strict-S1 context-boundary workload is not exact")
        record["artifact"] = artifact
        by_kind[str(kind)] = record
    if set(by_kind) != set(_STRICT_S1_WORKLOAD_KINDS):
        raise PrefillBlockQualificationError(
            "strict-S1 workloads must include 48-token, EOS, English, Chinese "
            "and context-boundary evidence")
    return [by_kind[kind] for kind in _STRICT_S1_WORKLOAD_KINDS]


def _strict_s1_mmz_observation(
    descriptor: Mapping[str, object], *, root: Path, context: int,
    bootstrap_om_sha256: str, canonical_om_sha256: str,
    runner_sha256: str, executor_sha256: str,
    bootstrap_descriptor_sha256: str,
    canonical_descriptor_sha256: str,
) -> dict[str, object]:
    artifact, payload = _evidence_json(
        descriptor, "strict_s1.mmz_observation", root)
    fields = {
        "schema", "status", "board", "clean_board", "context", "width",
        "role", "accounting", "bootstrap_om_sha256",
        "canonical_decode_om_sha256", "runner_sha256",
        "executor_sha256", "bootstrap_ready_descriptor_sha256",
        "canonical_ready_descriptor_sha256",
        "before_available_bytes", "after_available_bytes",
    }
    exact = {
        "schema": STRICT_S1_MMZ_SCHEMA,
        "status": "PASS",
        "board": "Hi3403",
        "clean_board": True,
        "context": context,
        "width": 1,
        "role": "base-resident",
        "accounting": "included-in-base_resident_bytes",
        "bootstrap_om_sha256": bootstrap_om_sha256,
        "canonical_decode_om_sha256": canonical_om_sha256,
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
        "bootstrap_ready_descriptor_sha256": bootstrap_descriptor_sha256,
        "canonical_ready_descriptor_sha256": canonical_descriptor_sha256,
    }
    if set(payload) != fields:
        raise PrefillBlockQualificationError(
            "strict-S1 MMZ fields do not match the release schema")
    for field, expected in exact.items():
        if payload.get(field) != expected:
            raise PrefillBlockQualificationError(
                f"strict-S1 MMZ {field} is not bound to the baseline")
    before = payload.get("before_available_bytes")
    after = payload.get("after_available_bytes")
    if type(before) is not int or type(after) is not int \
            or before <= 0 or after < 0 or after >= before:
        raise PrefillBlockQualificationError(
            "strict-S1 MMZ before/after must show a positive clean-board delta")
    return {
        "observation_artifact": artifact,
        "role": "base-resident",
        "accounting": "included-in-base_resident_bytes",
        "clean_board": True,
        "before_available_bytes": before,
        "after_available_bytes": after,
        "resident_bytes": before - after,
    }
