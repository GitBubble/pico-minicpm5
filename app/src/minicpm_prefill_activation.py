#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed activation and residency admission for native prefill blocks.

Qualification and runtime activation are deliberately separate.  S16/S32/S128
may be built and scored serially on a host, while the board may only load the
subset whose exact artifacts, physical publisher ABI and measured MMZ cost fit
the live deployment.  Strict S1 is never represented by this registry and is
always retained by the caller.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct


SCHEMA = "pico.minicpm5.prefill-activation.v4"
QUALIFICATION_SCHEMA = "pico.minicpm5.prefill-block-qualification.v4"
LEGACY_QUALIFICATION_SCHEMA = "pico.minicpm5.prefill-block-qualification.v3"
DEVELOPMENT_QUALIFICATION_SCHEMA = (
    "pico.minicpm5.prefill-block-qualification.v2")
ADMISSION_SCHEMA = "pico.minicpm5.prefill-mmz-observation.v1"
CAPTURE_SCHEMA = "pico.minicpm5.prefill-capture.v1"
WORKLOAD_SCHEMA = "pico.minicpm5.prefill-workload.v2"
PERFORMANCE_SCHEMA = "pico.minicpm5.prefill-performance-measurement.v1"
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
TOKEN_ID_HASH_CONTRACT = "sha256-le32-u32-token-id-sequence"
TRUSTED_DEPLOYMENT_MODE = "trusted-read-only-process-lifetime"
STRICT_S1_MMZ_SCHEMA = "pico.minicpm5.strict-s1-mmz-observation.v1"
POLICY_THRESHOLD_EXCLUSIVE = 0.98
MASK_NEGATIVE = -64.0
WIDTHS = (128, 32, 16)
_POSITION_PROBES = {
    16: (1, 15, 16, 31, 32, 255, 256, 643),
    32: (1, 31, 32, 127, 128, 643),
    128: (1, 127, 128, 511, 512, 643),
}
_PERFORMANCE_BASELINES = {16: (1,), 32: (16,), 128: (32, 16)}
_STRICT_S1_POSITION_PROBES = (
    0, 1, 15, 16, 31, 32, 127, 128, 255, 256, 511, 512, 643)
_STRICT_S1_WORKLOAD_KINDS = (
    "tokens_48", "eos", "english", "chinese", "context_boundary")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GROUP = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")


class PrefillActivationError(ValueError):
    """The activation manifest itself is unsafe or internally ambiguous."""


@dataclass(frozen=True)
class StrictS1Identity:
    qualification_sha256: str
    bootstrap_om_sha256: str
    canonical_decode_om_sha256: str
    head_om_sha256: str
    embedding_sha256: str
    build_manifest_sha256: str
    runner_sha256: str
    executor_sha256: str
    bootstrap_ready_descriptor_sha256: str
    canonical_ready_descriptor_sha256: str
    resident_bytes: int


@dataclass(frozen=True)
class StrictS1Anchor:
    qualification: Path
    bootstrap_model: Path
    canonical_decode_model: Path
    head_model: Path
    embedding: Path
    build_manifest: Path
    runner: Path
    executor: Path
    bootstrap_ready_descriptor: Path
    canonical_ready_descriptor: Path
    identity: StrictS1Identity
    declaration: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.declaration)


@dataclass(frozen=True)
class ActivatedBlock:
    width: int
    model: Path
    qualification: Path
    build_manifest: Path
    ready_descriptor: Path
    runner: Path
    executor: Path
    admission_report: Path
    admission_report_sha256: str
    model_sha256: str
    head_om_sha256: str
    embedding_sha256: str
    ready_descriptor_sha256: str
    runner_sha256: str
    executor_sha256: str
    residency_group: str
    admission_bytes: int
    strict_s1_identity: StrictS1Identity


@dataclass(frozen=True)
class PrefillActivation:
    context: int
    enabled_widths: tuple[int, ...]
    blocks: tuple[ActivatedBlock, ...]
    disabled: dict[str, str]
    base_resident_bytes: int
    block_resident_bytes: int
    reserve_bytes: int
    available_bytes: int
    strict_s1: StrictS1Anchor
    base_underreported: bool
    effective_base_resident_bytes: int
    deployment_mode: str = TRUSTED_DEPLOYMENT_MODE

    def to_dict(self) -> dict[str, object]:
        groups: dict[str, dict[str, object]] = {}
        for block in self.blocks:
            group = groups.setdefault(block.residency_group, {
                "name": block.residency_group,
                "model": str(block.model),
                "ready_descriptor_sha256": block.ready_descriptor_sha256,
                "admission_bytes": block.admission_bytes,
                "widths": [],
            })
            group["widths"].append(block.width)
        return {
            "schema": SCHEMA,
            "context": self.context,
            "deployment_mode": self.deployment_mode,
            "enabled_widths": list(self.enabled_widths),
            "strict_s1": self.strict_s1.to_dict(),
            "blocks": [
                {
                    "width": block.width,
                    "model": str(block.model),
                    "qualification": str(block.qualification),
                    "build_manifest": str(block.build_manifest),
                    "ready_descriptor": str(block.ready_descriptor),
                    "runner": str(block.runner),
                    "executor": str(block.executor),
                    "admission_report": str(block.admission_report),
                    "admission_report_sha256": block.admission_report_sha256,
                    "model_sha256": block.model_sha256,
                    "head_om_sha256": block.head_om_sha256,
                    "embedding_sha256": block.embedding_sha256,
                    "ready_descriptor_sha256": block.ready_descriptor_sha256,
                    "runner_sha256": block.runner_sha256,
                    "executor_sha256": block.executor_sha256,
                    "residency_group": block.residency_group,
                    "admission_bytes": block.admission_bytes,
                }
                for block in self.blocks
            ],
            # v4 permits exactly one independently measured block per group.
            # This view is only a canonical runtime index; it never discounts
            # admission bytes across widths.
            "residency_groups": list(groups.values()),
            "disabled": dict(self.disabled),
            "memory": {
                "base_resident_bytes": self.base_resident_bytes,
                "block_resident_bytes": self.block_resident_bytes,
                "reserve_bytes": self.reserve_bytes,
                "available_bytes": self.available_bytes,
                "strict_s1_resident_bytes": (
                    self.strict_s1.identity.resident_bytes),
                "base_underreported": self.base_underreported,
                "effective_base_resident_bytes": (
                    self.effective_base_resident_bytes),
            },
        }


def _unsigned(value: object, label: str, *, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise PrefillActivationError(
            f"{label} must be an integer >= {minimum}")
    return value


def _relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PrefillActivationError(f"{label} must be a relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PrefillActivationError(f"{label} must stay under deployment root")
    return path


def _safe_file(root: Path, value: object, label: str) -> Path:
    relative = _relative(value, label)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PrefillActivationError(f"{label} must not traverse a symlink")
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as error:
        raise PrefillActivationError(f"{label} is unavailable: {error}") from error
    if not resolved.is_file() or root not in resolved.parents:
        raise PrefillActivationError(f"{label} must be a file under deployment root")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
    except OSError as error:
        raise PrefillActivationError(
            f"cannot hash {path.name}: {error}") from error
    return digest.hexdigest()


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PrefillActivationError(f"{label} must be a lowercase SHA-256")
    return value


def _load_json(path: Path, label: str) -> dict[str, object]:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise PrefillActivationError(
                    f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrefillActivationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PrefillActivationError(f"{label} must contain one JSON object")
    return value


def _verified_artifact(
    root: Path, value: object, label: str, *, require_json: bool = False,
) -> tuple[Path, dict[str, object] | None]:
    """Re-read one qualification-bound artifact from the deployment root."""
    if not isinstance(value, dict) or set(value) != {
            "path", "bytes", "sha256", "file_verified"} \
            or value.get("file_verified") is not True:
        raise PrefillActivationError(
            f"{label} is not a complete verified artifact descriptor")
    path = _safe_file(root, value.get("path"), label)
    size = value.get("bytes")
    if type(size) is not int or size <= 0 or path.stat().st_size != size:
        raise PrefillActivationError(f"{label} byte size mismatch")
    expected = _hash(value.get("sha256"), f"{label}.sha256")
    if _sha256(path) != expected:
        raise PrefillActivationError(f"{label} SHA-256 mismatch")
    payload = _load_json(path, label) if require_json else None
    return path, payload


def _positive_series(value: object, field: str, count: int) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise PrefillActivationError(
            f"{field} must contain exactly {count} samples")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) \
                or not math.isfinite(float(item)) or float(item) <= 0:
            raise PrefillActivationError(
                f"{field} samples must be finite and positive")
        result.append(float(item))
    return result


def _canonical_mask_sha256(context: int, width: int, start: int) -> str:
    if type(context) is not int or type(width) is not int \
            or type(start) is not int \
            or not 1 <= start <= context - width:
        raise PrefillActivationError("canonical mask range is invalid")
    zero = struct.pack("<f", 0.0)
    negative = struct.pack("<f", MASK_NEGATIVE)
    digest = hashlib.sha256()
    for position in range(start, start + width):
        digest.update(zero * position)
        digest.update(negative * (context - 1 - position))
        digest.update(zero)
    return digest.hexdigest()


def _publisher_abi(qualification: dict[str, object], width: int) -> None:
    abi = qualification.get("abi")
    if not isinstance(abi, dict):
        raise PrefillActivationError("qualification.abi is missing")
    context = qualification.get("context")
    rows = 48 * width
    values = rows * 128
    expected = {
        "context": context,
        "width": width,
        "input_slots": {
            "embedding": 0, "mask": 1, "rope": 2,
            "k_cache": 3, "v_cache": 4,
        },
        "output_slots": {"k": 0, "v": 1, "hidden": 2},
        "hidden": {
            "shape": [1, 1536, 1, 1], "dtype": "FP32",
            "logical_bytes": 1536 * 4,
            "meaning": "final layer hidden for the last token in the block",
        },
        "publisher": {
            "layout": "contiguous-channel-major", "dtype": "FP32",
            "shape": [1, 48, width, 128], "logical_rows": rows,
            "logical_bytes": values * 4, "roles": ["k", "v"],
            "resident_conversion": {
                "opcode": 6, "dst_dtype": "FP16", "rounding": "RNE"},
        },
        "mask": {
            "shape": [width, context], "dtype": "FP32",
            "layout": "row-major", "logical_bytes": width * context * 4,
            "negative_value": MASK_NEGATIVE,
            "position_mapping": "row_j_is_absolute_start_plus_j",
            "visible_prefix": "[0,position)",
            "masked_future": "[position,context-1)",
            "current_token_column": context - 1,
        },
        "rope": {
            "shape": [width, 128, 128], "dtype": "FP32",
            "layout": "row-major", "logical_bytes": width * 128 * 128 * 4,
        },
        "k": {
            "shape": [1, 48, width, 128], "dtype": "FP16",
            "logical_rows": rows, "logical_bytes": values * 2,
            "meaning": "resident cache after opcode-6 RNE scatter",
        },
        "v": {
            "shape": [1, 48, width, 128], "dtype": "FP16",
            "logical_rows": rows, "logical_bytes": values * 2,
            "meaning": "resident cache after opcode-6 RNE scatter",
        },
        "cache_advance": width,
    }
    if abi != expected:
        raise PrefillActivationError(
            "qualification physical/resident ABI is not exact")
    activation = qualification.get("activation")
    artifact = qualification.get("om")
    head = qualification.get("head_om")
    embedding = qualification.get("embedding")
    build = qualification.get("build_manifest")
    runtime = qualification.get("runtime")
    admission = qualification.get("mmz_admission")
    if not isinstance(artifact, dict) or not isinstance(head, dict) \
            or not isinstance(embedding, dict) or not isinstance(build, dict) \
            or not isinstance(runtime, dict) or not isinstance(admission, dict):
        raise PrefillActivationError("qualification artifact lineage is missing")
    runner = runtime.get("runner")
    executor = runtime.get("executor")
    descriptor = runtime.get("ready_descriptor")
    if not all(isinstance(item, dict) for item in (
            runner, executor, descriptor)):
        raise PrefillActivationError("qualification runtime lineage is missing")
    expected_activation = {
        "key": f"ctx{qualification.get('context')}.s{width}.steady",
        "phase": "steady",
        "minimum_start": 1,
        "startup_requires_strict_s1": True,
        "runtime_eligible": True,
        "release_eligible": True,
        "context": qualification.get("context"),
        "width": width,
        "om_sha256": artifact.get("sha256"),
        "head_om_sha256": head.get("sha256"),
        "embedding_sha256": embedding.get("sha256"),
        "build_manifest_sha256": build.get("sha256"),
        "runner_sha256": runner.get("sha256"),
        "executor_sha256": executor.get("sha256"),
        "ready_descriptor_sha256": descriptor.get("sha256"),
        "admission_bytes": admission.get("admission_bytes"),
        "qualification_schema": QUALIFICATION_SCHEMA,
    }
    if activation != expected_activation:
        raise PrefillActivationError(
            "qualification is not a steady runtime-eligible artifact")


def _required_starts(context: int, width: int) -> list[int]:
    last = context - width
    return sorted({
        *[position for position in _POSITION_PROBES[width]
          if 1 <= position <= last],
        last,
    })


def _strict_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillActivationError(f"{label} must be numeric")
    score = float(value)
    if not POLICY_THRESHOLD_EXCLUSIVE < score <= 1.0:
        raise PrefillActivationError(
            f"{label} must be in ({POLICY_THRESHOLD_EXCLUSIVE}, 1]")
    return score


def _qualification_gates(
    qualification: dict[str, object], *, context: int, width: int,
    descriptor_sha256: str, runner_sha256: str, executor_sha256: str,
    root: Path,
) -> StrictS1Identity:
    fields = {
        "schema", "status", "release_eligible", "context", "width",
        "threshold_exclusive", "minimum_cosine", "om", "head_om",
        "embedding", "build_manifest",
        "runtime", "required_starts", "abi", "captures", "workloads",
        "baselines", "performance", "mmz_admission", "activation",
    }
    if set(qualification) != fields:
        raise PrefillActivationError(
            "qualification is not a complete release v4 PASS record")
    if qualification.get("release_eligible") is not True:
        raise PrefillActivationError("qualification is not release eligible")
    _strict_score(qualification.get("minimum_cosine"), "minimum_cosine")
    required = _required_starts(context, width)
    if qualification.get("required_starts") != required:
        raise PrefillActivationError("qualification position matrix is incomplete")
    captures = qualification.get("captures")
    if not isinstance(captures, list) or len(captures) != len(required) or {
            capture.get("start") for capture in captures
            if isinstance(capture, dict)} != set(required):
        raise PrefillActivationError("qualification captures are incomplete")
    capture_fields = {
        "start", "stop", "capture_sha256", "physical_descriptor_sha256",
        "publisher_source_sha256", "publisher_source_dtype",
        "publisher_source_layout", "descriptor_exact", "mask_sha256",
        "mask_bytes_exact", "rope_sha256", "rope_bytes_exact", "kv_rows",
        "kv_rows_exact", "prefill_decode_handoff", "token_exact",
        "board_pass", "public_cosines", "handoff_cosines", "artifact",
    }
    observed_minimum = 1.0
    for capture in captures:
        if not isinstance(capture, dict) or set(capture) != capture_fields \
                or type(capture.get("start")) is not int \
                or capture.get("physical_descriptor_sha256") != descriptor_sha256 \
                or capture.get("publisher_source_dtype") != "FP32" \
                or capture.get("publisher_source_layout") != \
                "contiguous-channel-major" \
                or capture.get("kv_rows") != {
                    "k": 48 * width, "v": 48 * width}:
            raise PrefillActivationError(
                "qualification capture physical ABI/descriptor mismatch")
        if capture.get("stop") != int(capture["start"]) + width:
            raise PrefillActivationError("qualification capture range mismatch")
        for field in ("capture_sha256", "mask_sha256", "rope_sha256"):
            _hash(capture.get(field), f"capture.{field}")
        if capture.get("mask_sha256") != _canonical_mask_sha256(
                context, width, int(capture["start"])):
            raise PrefillActivationError(
                "qualification capture mask is not the canonical absolute mask")
        source_hashes = capture.get("publisher_source_sha256")
        if not isinstance(source_hashes, dict) or set(source_hashes) != {"k", "v"}:
            raise PrefillActivationError(
                "qualification publisher source hashes are incomplete")
        for role in ("k", "v"):
            _hash(source_hashes[role], f"publisher_source_sha256.{role}")
        for gate in (
            "descriptor_exact", "mask_bytes_exact", "rope_bytes_exact",
            "kv_rows_exact", "prefill_decode_handoff", "token_exact",
            "board_pass",
        ):
            if capture.get(gate) is not True:
                raise PrefillActivationError(
                    f"qualification capture {gate} is not PASS")
        for group in ("public_cosines", "handoff_cosines"):
            scores = capture.get(group)
            if not isinstance(scores, dict) or set(scores) != {"hidden", "k", "v"}:
                raise PrefillActivationError(
                    f"qualification capture {group} is incomplete")
            for role, score in scores.items():
                observed_minimum = min(
                    observed_minimum, _strict_score(score, f"{group}.{role}"))
        _, evidence = _verified_artifact(
            root, capture.get("artifact"), "qualification capture artifact",
            require_json=True)
        assert evidence is not None
        artifact = capture["artifact"]
        assert isinstance(artifact, dict)
        if capture.get("capture_sha256") != artifact.get("sha256"):
            raise PrefillActivationError(
                "capture hash is not bound to its real evidence artifact")
        expected_evidence = {
            "schema": CAPTURE_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": width,
            "model_sha256": qualification["om"]["sha256"],
            "runner_sha256": runner_sha256,
            "executor_sha256": executor_sha256,
            "ready_descriptor_sha256": descriptor_sha256,
            **{key: value for key, value in capture.items()
               if key not in {"artifact", "capture_sha256", "stop"}},
        }
        if evidence != expected_evidence:
            raise PrefillActivationError(
                "capture evidence content does not match qualification")
    if float(qualification["minimum_cosine"]) != observed_minimum:
        raise PrefillActivationError(
            "qualification minimum_cosine does not match captures")
    workloads = qualification.get("workloads")
    if not isinstance(workloads, list) or len(workloads) != 4 or {
            item.get("kind") for item in workloads if isinstance(item, dict)} != {
                "eos", "english", "chinese", "context_boundary"}:
        raise PrefillActivationError("qualification workloads are incomplete")
    workload_base = {
        "kind", "capture_sha256", "prompt_sha256", "output_tokens_sha256",
        "board_pass", "token_exact", "artifact",
    }
    for item in workloads:
        expected_workload = set(workload_base)
        if isinstance(item, dict) and item.get("kind") == "eos":
            expected_workload.add("eos_exact")
        if isinstance(item, dict) and item.get("kind") == "context_boundary":
            expected_workload.update(("boundary_exact", "terminal_position"))
        if not isinstance(item, dict) or set(item) != expected_workload \
                or item.get("board_pass") is not True \
                or item.get("token_exact") is not True:
            raise PrefillActivationError("qualification workload is not PASS")
        for field in ("capture_sha256", "prompt_sha256", "output_tokens_sha256"):
            _hash(item.get(field), f"workload.{field}")
        if item.get("kind") == "eos" and item.get("eos_exact") is not True:
            raise PrefillActivationError("qualification EOS gate is not PASS")
        if item.get("kind") == "context_boundary" and (
                item.get("boundary_exact") is not True
                or item.get("terminal_position") != context - 1):
            raise PrefillActivationError(
                "qualification context boundary is not PASS")
        _, evidence = _verified_artifact(
            root, item.get("artifact"), "qualification workload artifact",
            require_json=True)
        assert evidence is not None
        expected_evidence = {
            "schema": WORKLOAD_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": width,
            "model_sha256": qualification["om"]["sha256"],
            "head_om_sha256": qualification["head_om"]["sha256"],
            "embedding_sha256": qualification["embedding"]["sha256"],
            "runner_sha256": runner_sha256,
            "executor_sha256": executor_sha256,
            "token_id_hash_contract": TOKEN_ID_HASH_CONTRACT,
            **{key: value for key, value in item.items() if key != "artifact"},
        }
        if evidence != expected_evidence:
            raise PrefillActivationError(
                "workload evidence content does not match qualification")

    baselines = qualification.get("baselines")
    expected_baselines = _PERFORMANCE_BASELINES[width]
    if not isinstance(baselines, list) or len(baselines) != len(
            expected_baselines):
        raise PrefillActivationError(
            "qualification baseline ladder is incomplete")
    baseline_by_width: dict[int, dict[str, object]] = {}
    strict_s1_identities: list[StrictS1Identity] = []
    for item in baselines:
        if not isinstance(item, dict) or set(item) != {
                "width", "qualification", "om"}:
            raise PrefillActivationError(
                "qualification baseline descriptor is malformed")
        baseline_width = item.get("width")
        if type(baseline_width) is not int or baseline_width in baseline_by_width:
            raise PrefillActivationError(
                "qualification baseline widths must be unique integers")
        _, baseline_report = _verified_artifact(
            root, item.get("qualification"),
            f"S{baseline_width} baseline qualification", require_json=True)
        _, _ = _verified_artifact(
            root, item.get("om"), f"S{baseline_width} baseline OM")
        assert baseline_report is not None
        baseline_om = item.get("om")
        assert isinstance(baseline_om, dict)
        if baseline_report.get("status") != "PASS" \
                or baseline_report.get("release_eligible") is not True \
                or baseline_report.get("context") != context \
                or baseline_report.get("width") != baseline_width:
            raise PrefillActivationError(
                "baseline qualification is not a release PASS")
        if baseline_width == 1:
            if baseline_report.get("schema") == STRICT_S1_DEVELOPMENT_SCHEMA:
                raise PrefillActivationError(
                    "development strict-S1 v1 baseline is not release activatable")
            if baseline_report.get("schema") == \
                    STRICT_S1_LEGACY_RELEASE_SCHEMA:
                raise PrefillActivationError(
                    "legacy single-OM strict-S1 v2 baseline is not release "
                    "activatable")
            if baseline_report.get("schema") == \
                    STRICT_S1_PREVIOUS_RELEASE_SCHEMA:
                raise PrefillActivationError(
                    "legacy strict-S1 v3 baseline is not release activatable")
            if baseline_report.get("schema") != STRICT_S1_BASELINE_SCHEMA:
                raise PrefillActivationError(
                    "S1 baseline qualification schema mismatch")
            strict_s1_identities.append(
                _verify_nested_strict_s1_qualification(
                    baseline_report,
                    qualification_sha256=str(
                        item["qualification"]["sha256"]),
                    root=root, context=context))
            baseline_artifact = baseline_report.get("canonical_decode_om")
            baseline_om_hash = baseline_artifact.get("sha256") \
                if isinstance(baseline_artifact, dict) else None
        else:
            if baseline_report.get("schema") == LEGACY_QUALIFICATION_SCHEMA:
                raise PrefillActivationError(
                    "legacy wide v3 baseline is not release activatable")
            if baseline_report.get("schema") != QUALIFICATION_SCHEMA:
                raise PrefillActivationError(
                    "wide baseline qualification is not release v4")
            strict_s1_identities.append(
                _verify_nested_release_qualification(
                    baseline_report, root=root, context=context,
                    width=baseline_width))
            baseline_artifact = baseline_report.get("om")
            baseline_om_hash = baseline_artifact.get("sha256") \
                if isinstance(baseline_artifact, dict) else None
        if baseline_om_hash != baseline_om.get("sha256"):
            raise PrefillActivationError(
                "baseline qualification does not bind the baseline OM")
        baseline_by_width[baseline_width] = item
    if tuple(baseline_by_width) != expected_baselines:
        # The builder emits canonical ladder order; activation rejects reordered
        # or substituted baselines to keep evidence review deterministic.
        raise PrefillActivationError(
            "qualification baseline ladder order is not canonical")
    if not strict_s1_identities or any(
            identity != strict_s1_identities[0]
            for identity in strict_s1_identities[1:]):
        raise PrefillActivationError(
            "wide baseline ladder does not share one strict-S1 identity")
    head = qualification.get("head_om")
    embedding = qualification.get("embedding")
    assert isinstance(head, dict) and isinstance(embedding, dict)
    if strict_s1_identities[0].head_om_sha256 != head.get("sha256") \
            or strict_s1_identities[0].embedding_sha256 != \
            embedding.get("sha256"):
        raise PrefillActivationError(
            "release candidate head/embedding identity does not match its "
            "strict-S1 baseline")

    performance = qualification.get("performance")
    if not isinstance(performance, list) \
            or len(performance) != len(_PERFORMANCE_BASELINES[width]) or {
            item.get("baseline_width") for item in performance
            if isinstance(item, dict)} != set(_PERFORMANCE_BASELINES[width]):
        raise PrefillActivationError(
            "qualification performance ladder is incomplete")
    artifact = qualification.get("om")
    assert isinstance(artifact, dict)
    for item in performance:
        performance_fields = {
            "metric", "board", "tokens", "candidate_width", "baseline_width",
            "candidate_invocations", "baseline_invocations", "warmup_runs",
            "measured_runs", "candidate_ms", "baseline_ms", "speedup",
            "candidate_warmup_ms", "baseline_warmup_ms",
            "candidate_samples_ms", "baseline_samples_ms",
            "candidate_om_sha256", "baseline_qualification_sha256",
            "baseline_om_sha256", "runner_sha256", "executor_sha256",
            "measurement_artifact",
        }
        if not isinstance(item, dict) or set(item) != performance_fields \
                or item.get("metric") != "board-wall-time-ms" \
                or item.get("board") != "Hi3403" \
                or item.get("tokens") != width \
                or item.get("candidate_om_sha256") != artifact.get("sha256") \
                or item.get("runner_sha256") != runner_sha256 \
                or item.get("executor_sha256") != executor_sha256 \
                or item.get("candidate_width") != width \
                or item.get("candidate_invocations") != 1 \
                or type(item.get("baseline_width")) is not int \
                or item.get("baseline_invocations") != \
                width // item["baseline_width"] \
                or type(item.get("warmup_runs")) is not int \
                or item.get("warmup_runs") < 1 \
                or type(item.get("measured_runs")) is not int \
                or item.get("measured_runs") < 3 \
                or isinstance(item.get("candidate_ms"), bool) \
                or isinstance(item.get("baseline_ms"), bool) \
                or not isinstance(item.get("candidate_ms"), (int, float)) \
                or not isinstance(item.get("baseline_ms"), (int, float)) \
                or not 0 < float(item["candidate_ms"]) < float(item["baseline_ms"]):
            raise PrefillActivationError(
                "qualification performance comparison is not PASS")
        baseline_width = int(item["baseline_width"])
        baseline = baseline_by_width.get(baseline_width)
        if baseline is None:
            raise PrefillActivationError(
                "performance baseline is absent from the release ladder")
        baseline_q = baseline.get("qualification")
        baseline_om = baseline.get("om")
        assert isinstance(baseline_q, dict) and isinstance(baseline_om, dict)
        if item.get("baseline_qualification_sha256") != baseline_q.get("sha256") \
                or item.get("baseline_om_sha256") != baseline_om.get("sha256"):
            raise PrefillActivationError(
                "performance is not bound to baseline qualification and OM")
        warmup = int(item["warmup_runs"])
        measured = int(item["measured_runs"])
        candidate_warmup = _positive_series(
            item.get("candidate_warmup_ms"), "candidate_warmup_ms", warmup)
        baseline_warmup = _positive_series(
            item.get("baseline_warmup_ms"), "baseline_warmup_ms", warmup)
        candidate_samples = _positive_series(
            item.get("candidate_samples_ms"), "candidate_samples_ms", measured)
        baseline_samples = _positive_series(
            item.get("baseline_samples_ms"), "baseline_samples_ms", measured)
        candidate_mean = sum(candidate_samples) / measured
        baseline_mean = sum(baseline_samples) / measured
        if float(item["candidate_ms"]) != candidate_mean \
                or float(item["baseline_ms"]) != baseline_mean:
            raise PrefillActivationError(
                "performance means do not match the bound samples")
        expected_speedup = baseline_mean / candidate_mean
        if isinstance(item.get("speedup"), bool) \
                or not isinstance(item.get("speedup"), (int, float)) \
                or not math.isfinite(float(item["speedup"])) \
                or not math.isclose(float(item["speedup"]), expected_speedup,
                                    rel_tol=0.0, abs_tol=1e-12):
            raise PrefillActivationError(
                "qualification performance speedup is inconsistent")
        _, evidence = _verified_artifact(
            root, item.get("measurement_artifact"),
            "qualification performance measurement", require_json=True)
        assert evidence is not None
        expected_evidence = {
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
            "warmup_runs": warmup,
            "measured_runs": measured,
            "candidate_warmup_ms": candidate_warmup,
            "baseline_warmup_ms": baseline_warmup,
            "candidate_samples_ms": candidate_samples,
            "baseline_samples_ms": baseline_samples,
            "candidate_om_sha256": artifact.get("sha256"),
            "baseline_qualification_sha256": baseline_q.get("sha256"),
            "baseline_om_sha256": baseline_om.get("sha256"),
            "runner_sha256": runner_sha256,
            "executor_sha256": executor_sha256,
        }
        if evidence != expected_evidence:
            raise PrefillActivationError(
                "performance measurement content does not match qualification")

    mmz = qualification.get("mmz_admission")
    if not isinstance(mmz, dict) or set(mmz) != {
            "observation_artifact", "clean_board", "before_available_bytes",
            "after_available_bytes", "admission_bytes"} \
            or mmz.get("clean_board") is not True:
        raise PrefillActivationError(
            "qualification MMZ admission evidence is incomplete")
    before = mmz.get("before_available_bytes")
    after = mmz.get("after_available_bytes")
    admission = mmz.get("admission_bytes")
    if type(before) is not int or type(after) is not int \
            or type(admission) is not int or before <= 0 or after < 0 \
            or after >= before or admission != before - after:
        raise PrefillActivationError(
            "MMZ admission is not the clean-board before/after delta")
    _, observation = _verified_artifact(
        root, mmz.get("observation_artifact"),
        "qualification MMZ observation", require_json=True)
    assert observation is not None
    expected_observation = {
        "schema": ADMISSION_SCHEMA,
        "status": "PASS",
        "board": "Hi3403",
        "clean_board": True,
        "context": context,
        "width": width,
        "model_sha256": artifact.get("sha256"),
        "runner_sha256": runner_sha256,
        "executor_sha256": executor_sha256,
        "ready_descriptor_sha256": descriptor_sha256,
        "before_available_bytes": before,
        "after_available_bytes": after,
    }
    if observation != expected_observation:
        raise PrefillActivationError(
            "MMZ observation content does not match qualification")
    return strict_s1_identities[0]


def _verify_nested_release_qualification(
    report: dict[str, object], *, root: Path, context: int, width: int,
) -> StrictS1Identity:
    """Transitively verify a wide baseline and every file it binds."""
    if report.get("schema") != QUALIFICATION_SCHEMA \
            or report.get("status") != "PASS" \
            or report.get("release_eligible") is not True \
            or report.get("context") != context \
            or report.get("width") != width \
            or report.get("threshold_exclusive") != POLICY_THRESHOLD_EXCLUSIVE:
        raise PrefillActivationError(
            "wide baseline is not a release v4 PASS for this context")
    _verified_artifact(root, report.get("om"), "baseline model")
    _verified_artifact(root, report.get("head_om"), "baseline head OM")
    _verified_artifact(root, report.get("embedding"), "baseline embedding")
    _verified_artifact(
        root, report.get("build_manifest"), "baseline build manifest",
        require_json=True)
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
            "runner", "executor", "ready_descriptor"}:
        raise PrefillActivationError("baseline runtime lineage is missing")
    _, _ = _verified_artifact(root, runtime["runner"], "baseline runner")
    _, _ = _verified_artifact(root, runtime["executor"], "baseline executor")
    _, _ = _verified_artifact(
        root, runtime["ready_descriptor"], "baseline ready descriptor")
    _publisher_abi(report, width)
    return _qualification_gates(
        report, context=context, width=width,
        descriptor_sha256=str(runtime["ready_descriptor"]["sha256"]),
        runner_sha256=str(runtime["runner"]["sha256"]),
        executor_sha256=str(runtime["executor"]["sha256"]), root=root)


def _strict_s1_required_positions(context: int) -> list[int]:
    return sorted({
        *[position for position in _STRICT_S1_POSITION_PROBES
          if position < context],
        context - 1,
    })


def _verify_nested_strict_s1_qualification(
    report: dict[str, object], *, qualification_sha256: str, root: Path,
    context: int,
) -> StrictS1Identity:
    """Verify the content-bound strict-S1 release trust anchor."""
    fields = {
        "schema", "status", "release_eligible", "context", "width",
        "threshold_exclusive", "minimum_cosine", "bootstrap_om",
        "canonical_decode_om", "head_om", "embedding", "build_manifest",
        "runtime",
        "required_positions", "captures", "workloads", "mmz_residency",
        "baseline",
    }
    if set(report) != fields \
            or report.get("schema") != STRICT_S1_BASELINE_SCHEMA \
            or report.get("status") != "PASS" \
            or report.get("release_eligible") is not True \
            or report.get("context") != context \
            or report.get("width") != 1 \
            or report.get("threshold_exclusive") != POLICY_THRESHOLD_EXCLUSIVE:
        raise PrefillActivationError(
            "strict-S1 baseline is not a complete release v4 PASS")
    bootstrap_model_path, _ = _verified_artifact(
        root, report.get("bootstrap_om"), "strict-S1 bootstrap OM")
    canonical_model_path, _ = _verified_artifact(
        root, report.get("canonical_decode_om"),
        "strict-S1 canonical decode OM")
    head_model_path, _ = _verified_artifact(
        root, report.get("head_om"), "strict-S1 head OM")
    embedding_path, _ = _verified_artifact(
        root, report.get("embedding"), "strict-S1 embedding")
    _verified_artifact(
        root, report.get("build_manifest"), "strict-S1 build manifest",
        require_json=True)
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
            "runner", "executor", "bootstrap_ready_descriptor",
            "canonical_ready_descriptor"}:
        raise PrefillActivationError("strict-S1 runtime lineage is missing")
    _, _ = _verified_artifact(root, runtime["runner"], "strict-S1 runner")
    _, _ = _verified_artifact(root, runtime["executor"], "strict-S1 executor")
    _, _ = _verified_artifact(
        root, runtime["bootstrap_ready_descriptor"],
        "strict-S1 bootstrap ready descriptor")
    _, _ = _verified_artifact(
        root, runtime["canonical_ready_descriptor"],
        "strict-S1 canonical ready descriptor")
    bootstrap_model_hash = _sha256(bootstrap_model_path)
    canonical_model_hash = _sha256(canonical_model_path)
    head_model_hash = _sha256(head_model_path)
    embedding_hash = _sha256(embedding_path)
    runner_hash = str(runtime["runner"]["sha256"])
    executor_hash = str(runtime["executor"]["sha256"])
    bootstrap_descriptor_hash = str(
        runtime["bootstrap_ready_descriptor"]["sha256"])
    canonical_descriptor_hash = str(
        runtime["canonical_ready_descriptor"]["sha256"])

    positions = _strict_s1_required_positions(context)
    if report.get("required_positions") != positions:
        raise PrefillActivationError(
            "strict-S1 position matrix is incomplete")
    captures = report.get("captures")
    capture_fields = {
        "position", "capture_sha256", "physical_descriptor_sha256",
        "publisher_source_sha256", "publisher_source_dtype",
        "publisher_source_layout", "descriptor_exact", "mask_sha256",
        "mask_bytes_exact", "rope_sha256", "rope_bytes_exact", "kv_rows",
        "kv_rows_exact", "route_handoff_exact", "token_exact", "board_pass",
        "public_cosines", "artifact",
    }
    if not isinstance(captures, list) or len(captures) != len(positions) \
            or [item.get("position") for item in captures
                if isinstance(item, dict)] != positions:
        raise PrefillActivationError(
            "strict-S1 captures are incomplete or not canonical")
    observed_minimum = 1.0
    for item in captures:
        if not isinstance(item, dict) or set(item) != capture_fields \
                or item.get("physical_descriptor_sha256") != (
                    bootstrap_descriptor_hash
                    if item.get("position") == 0
                    else canonical_descriptor_hash) \
                or item.get("publisher_source_dtype") != "FP32" \
                or item.get("publisher_source_layout") != \
                "contiguous-channel-major" \
                or item.get("kv_rows") != {"k": 48, "v": 48}:
            raise PrefillActivationError(
                "strict-S1 capture physical ABI/descriptor mismatch")
        for field in ("capture_sha256", "mask_sha256", "rope_sha256"):
            _hash(item.get(field), f"strict_s1.capture.{field}")
        sources = item.get("publisher_source_sha256")
        if not isinstance(sources, dict) or set(sources) != {"k", "v"}:
            raise PrefillActivationError(
                "strict-S1 publisher hashes are incomplete")
        for role in ("k", "v"):
            _hash(sources[role], f"strict_s1.publisher.{role}")
        for gate in (
            "descriptor_exact", "mask_bytes_exact", "rope_bytes_exact",
            "kv_rows_exact", "route_handoff_exact", "token_exact",
            "board_pass",
        ):
            if item.get(gate) is not True:
                raise PrefillActivationError(
                    f"strict-S1 capture {gate} is not PASS")
        scores = item.get("public_cosines")
        if not isinstance(scores, dict) or set(scores) != {"hidden", "k", "v"}:
            raise PrefillActivationError(
                "strict-S1 capture public cosines are incomplete")
        for role, score in scores.items():
            observed_minimum = min(
                observed_minimum,
                _strict_score(score, f"strict_s1.public_cosines.{role}"))
        _, evidence = _verified_artifact(
            root, item.get("artifact"), "strict-S1 capture artifact",
            require_json=True)
        assert evidence is not None
        artifact = item["artifact"]
        assert isinstance(artifact, dict)
        if item.get("capture_sha256") != artifact.get("sha256"):
            raise PrefillActivationError(
                "strict-S1 capture hash is not artifact-bound")
        expected_evidence = {
            "schema": STRICT_S1_CAPTURE_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": 1,
            "model_sha256": (
                bootstrap_model_hash
                if item["position"] == 0 else canonical_model_hash),
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
            "ready_descriptor_sha256": (
                bootstrap_descriptor_hash
                if item["position"] == 0
                else canonical_descriptor_hash),
            **{key: value for key, value in item.items()
               if key not in {"artifact", "capture_sha256"}},
        }
        if evidence != expected_evidence:
            raise PrefillActivationError(
                "strict-S1 capture evidence content mismatch")
    if float(report["minimum_cosine"]) != observed_minimum:
        raise PrefillActivationError(
            "strict-S1 minimum cosine does not match captures")

    workloads = report.get("workloads")
    if not isinstance(workloads, list) or [
            item.get("kind") for item in workloads if isinstance(item, dict)
            ] != list(_STRICT_S1_WORKLOAD_KINDS):
        raise PrefillActivationError(
            "strict-S1 workload matrix is incomplete")
    for item in workloads:
        if not isinstance(item, dict):
            raise PrefillActivationError("strict-S1 workload is malformed")
        expected_fields = {
            "kind", "capture_sha256", "prompt_sha256",
            "output_tokens_sha256", "generated_tokens", "board_pass",
            "token_exact", "artifact",
        }
        kind = item.get("kind")
        if kind == "tokens_48":
            expected_fields.add("sequence_exact")
        if kind == "eos":
            expected_fields.add("eos_exact")
        if kind == "context_boundary":
            expected_fields.update(("boundary_exact", "terminal_position"))
        if set(item) != expected_fields \
                or item.get("board_pass") is not True \
                or item.get("token_exact") is not True:
            raise PrefillActivationError(
                "strict-S1 workload is not a complete PASS")
        for field in ("capture_sha256", "prompt_sha256", "output_tokens_sha256"):
            _hash(item.get(field), f"strict_s1.workload.{field}")
        generated = item.get("generated_tokens")
        if type(generated) is not int or generated <= 0:
            raise PrefillActivationError(
                "strict-S1 generated token count is invalid")
        if kind == "tokens_48" and (
                generated != 48 or item.get("sequence_exact") is not True):
            raise PrefillActivationError(
                "strict-S1 48-token gate is not exact")
        if kind == "eos" and item.get("eos_exact") is not True:
            raise PrefillActivationError("strict-S1 EOS gate is not exact")
        if kind == "context_boundary" and (
                item.get("boundary_exact") is not True
                or item.get("terminal_position") != context - 1):
            raise PrefillActivationError(
                "strict-S1 context-boundary gate is not exact")
        _, evidence = _verified_artifact(
            root, item.get("artifact"), "strict-S1 workload artifact",
            require_json=True)
        assert evidence is not None
        expected_evidence = {
            "schema": STRICT_S1_WORKLOAD_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": 1,
            "bootstrap_om_sha256": bootstrap_model_hash,
            "canonical_decode_om_sha256": canonical_model_hash,
            "head_om_sha256": head_model_hash,
            "embedding_sha256": embedding_hash,
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
            "token_id_hash_contract": TOKEN_ID_HASH_CONTRACT,
            **{key: value for key, value in item.items() if key != "artifact"},
        }
        if evidence != expected_evidence:
            raise PrefillActivationError(
                "strict-S1 workload evidence content mismatch")

    mmz = report.get("mmz_residency")
    if not isinstance(mmz, dict) or set(mmz) != {
            "observation_artifact", "role", "accounting", "clean_board",
            "before_available_bytes", "after_available_bytes",
            "resident_bytes"} \
            or mmz.get("role") != "base-resident" \
            or mmz.get("accounting") != "included-in-base_resident_bytes" \
            or mmz.get("clean_board") is not True:
        raise PrefillActivationError(
            "strict-S1 MMZ accounting role is incomplete")
    before = mmz.get("before_available_bytes")
    after = mmz.get("after_available_bytes")
    resident = mmz.get("resident_bytes")
    if type(before) is not int or type(after) is not int \
            or type(resident) is not int or before <= 0 or after < 0 \
            or after >= before or resident != before - after:
        raise PrefillActivationError(
            "strict-S1 MMZ residency is not a clean-board delta")
    _, observation = _verified_artifact(
        root, mmz.get("observation_artifact"),
        "strict-S1 MMZ observation", require_json=True)
    assert observation is not None
    if observation != {
            "schema": STRICT_S1_MMZ_SCHEMA,
            "status": "PASS",
            "board": "Hi3403",
            "clean_board": True,
            "context": context,
            "width": 1,
            "role": "base-resident",
            "accounting": "included-in-base_resident_bytes",
            "bootstrap_om_sha256": bootstrap_model_hash,
            "canonical_decode_om_sha256": canonical_model_hash,
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
            "bootstrap_ready_descriptor_sha256": (
                bootstrap_descriptor_hash),
            "canonical_ready_descriptor_sha256": (
                canonical_descriptor_hash),
            "before_available_bytes": before,
            "after_available_bytes": after,
            }:
        raise PrefillActivationError(
            "strict-S1 MMZ evidence content mismatch")
    build = report.get("build_manifest")
    assert isinstance(build, dict)
    expected_baseline = {
        "role": "strict-s1",
        "performance_baseline_eligible": True,
        "accounting": "included-in-base_resident_bytes",
        "bootstrap_om_sha256": bootstrap_model_hash,
        "canonical_decode_om_sha256": canonical_model_hash,
        "head_om_sha256": head_model_hash,
        "embedding_sha256": embedding_hash,
        "build_manifest_sha256": build.get("sha256"),
        "runner_sha256": runner_hash,
        "executor_sha256": executor_hash,
        "bootstrap_ready_descriptor_sha256": bootstrap_descriptor_hash,
        "canonical_ready_descriptor_sha256": canonical_descriptor_hash,
        "resident_bytes": resident,
        "qualification_schema": STRICT_S1_BASELINE_SCHEMA,
    }
    if report.get("baseline") != expected_baseline:
        raise PrefillActivationError(
            "strict-S1 performance baseline lineage is incomplete")
    return StrictS1Identity(
        qualification_sha256=_hash(
            qualification_sha256, "strict-S1 qualification SHA-256"),
        bootstrap_om_sha256=bootstrap_model_hash,
        canonical_decode_om_sha256=canonical_model_hash,
        head_om_sha256=head_model_hash,
        embedding_sha256=embedding_hash,
        build_manifest_sha256=str(build["sha256"]),
        runner_sha256=runner_hash,
        executor_sha256=executor_hash,
        bootstrap_ready_descriptor_sha256=bootstrap_descriptor_hash,
        canonical_ready_descriptor_sha256=canonical_descriptor_hash,
        resident_bytes=int(resident),
    )


def _block(
    raw: object,
    *,
    root: Path,
    context: int,
) -> ActivatedBlock:
    if not isinstance(raw, dict):
        raise PrefillActivationError("block entry must be an object")
    expected_fields = {
        "width", "phase", "model", "qualification", "qualification_sha256",
        "build_manifest", "residency_group", "admission_bytes",
        "admission_report", "admission_report_sha256", "ready_descriptor",
        "runner", "executor",
    }
    if set(raw) != expected_fields:
        raise PrefillActivationError("block entry fields mismatch")
    width = raw["width"]
    if type(width) is not int or width not in WIDTHS:
        raise PrefillActivationError(f"block width must be one of {WIDTHS}")
    if raw["phase"] != "steady":
        raise PrefillActivationError("wide prefill blocks must be steady-only")
    group = raw["residency_group"]
    if not isinstance(group, str) or _GROUP.fullmatch(group) is None:
        raise PrefillActivationError("residency_group is invalid")
    admission = _unsigned(
        raw["admission_bytes"], "admission_bytes", allow_zero=False)
    model = _safe_file(root, raw["model"], "block.model")
    qualification_path = _safe_file(
        root, raw["qualification"], "block.qualification")
    build_manifest = _safe_file(
        root, raw["build_manifest"], "block.build_manifest")
    ready_descriptor = _safe_file(
        root, raw["ready_descriptor"], "block.ready_descriptor")
    runner = _safe_file(root, raw["runner"], "block.runner")
    executor = _safe_file(root, raw["executor"], "block.executor")
    admission_report_path = _safe_file(
        root, raw["admission_report"], "block.admission_report")
    expected_qualification_hash = _hash(
        raw["qualification_sha256"], "qualification_sha256")
    if _sha256(qualification_path) != expected_qualification_hash:
        raise PrefillActivationError("qualification SHA-256 mismatch")
    expected_admission_hash = _hash(
        raw["admission_report_sha256"], "admission_report_sha256")
    if _sha256(admission_report_path) != expected_admission_hash:
        raise PrefillActivationError("admission report SHA-256 mismatch")

    qualification = _load_json(qualification_path, "block qualification")
    if qualification.get("schema") == DEVELOPMENT_QUALIFICATION_SCHEMA:
        raise PrefillActivationError(
            "development v2 qualification is not release activatable")
    if qualification.get("schema") == LEGACY_QUALIFICATION_SCHEMA:
        raise PrefillActivationError(
            "legacy wide v3 qualification is not release activatable")
    if qualification.get("schema") != QUALIFICATION_SCHEMA \
            or qualification.get("status") != "PASS" \
            or qualification.get("release_eligible") is not True:
        raise PrefillActivationError(
            "block qualification is not a release v4 PASS")
    if qualification.get("context") != context \
            or qualification.get("width") != width:
        raise PrefillActivationError("block qualification geometry mismatch")
    if qualification.get("threshold_exclusive") != POLICY_THRESHOLD_EXCLUSIVE:
        raise PrefillActivationError("block qualification weakened the numeric gate")
    artifact = qualification.get("om")
    if not isinstance(artifact, dict) \
            or artifact.get("file_verified") is not True \
            or artifact.get("bytes") != model.stat().st_size \
            or artifact.get("sha256") != _sha256(model):
        raise PrefillActivationError("block model bytes/SHA do not match qualification")
    head_path, _ = _verified_artifact(
        root, qualification.get("head_om"), "block head OM")
    embedding_path, _ = _verified_artifact(
        root, qualification.get("embedding"), "block embedding")
    head_hash = _sha256(head_path)
    embedding_hash = _sha256(embedding_path)
    build_artifact = qualification.get("build_manifest")
    if not isinstance(build_artifact, dict) \
            or build_artifact.get("file_verified") is not True \
            or build_artifact.get("bytes") != build_manifest.stat().st_size \
            or build_artifact.get("sha256") != _sha256(build_manifest):
        raise PrefillActivationError("build manifest SHA-256 mismatch")
    runtime = qualification.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
            "runner", "executor", "ready_descriptor"}:
        raise PrefillActivationError("qualification runtime lineage is missing")
    runtime_files = {
        "runner": runner,
        "executor": executor,
        "ready_descriptor": ready_descriptor,
    }
    for name, path in runtime_files.items():
        descriptor = runtime.get(name)
        if not isinstance(descriptor, dict) \
                or descriptor.get("file_verified") is not True \
                or descriptor.get("bytes") != path.stat().st_size \
                or descriptor.get("sha256") != _sha256(path):
            raise PrefillActivationError(
                f"{name} bytes/SHA do not match qualification")
    _publisher_abi(qualification, width)
    admission_report = _load_json(
        admission_report_path, "block admission report")
    admission_fields = {
        "schema", "status", "board", "clean_board", "context", "width",
        "model_sha256", "runner_sha256", "executor_sha256",
        "ready_descriptor_sha256", "before_available_bytes",
        "after_available_bytes",
    }
    descriptor_hash = _sha256(ready_descriptor)
    runner_hash = _sha256(runner)
    executor_hash = _sha256(executor)
    before = admission_report.get("before_available_bytes")
    after = admission_report.get("after_available_bytes")
    if set(admission_report) != admission_fields \
            or admission_report.get("schema") != ADMISSION_SCHEMA \
            or admission_report.get("status") != "PASS" \
            or admission_report.get("board") != "Hi3403" \
            or admission_report.get("clean_board") is not True \
            or admission_report.get("context") != context \
            or admission_report.get("width") != width \
            or admission_report.get("model_sha256") != artifact.get("sha256") \
            or admission_report.get("runner_sha256") != runner_hash \
            or admission_report.get("executor_sha256") != executor_hash \
            or admission_report.get("ready_descriptor_sha256") != descriptor_hash \
            or type(before) is not int or type(after) is not int \
            or before <= 0 or after < 0 or after >= before \
            or before - after != admission:
        raise PrefillActivationError(
            "admission is not the bound clean-board MMZ before/after delta")
    mmz = qualification.get("mmz_admission")
    if not isinstance(mmz, dict) \
            or mmz.get("admission_bytes") != admission \
            or not isinstance(mmz.get("observation_artifact"), dict) \
            or mmz["observation_artifact"].get("sha256") != expected_admission_hash:
        raise PrefillActivationError(
            "activation admission does not match qualification MMZ evidence")
    strict_s1_identity = _qualification_gates(
        qualification, context=context, width=width,
        descriptor_sha256=descriptor_hash, runner_sha256=runner_hash,
        executor_sha256=executor_hash, root=root)
    return ActivatedBlock(
        width=width, model=model, qualification=qualification_path,
        build_manifest=build_manifest, ready_descriptor=ready_descriptor,
        runner=runner, executor=executor,
        admission_report=admission_report_path,
        admission_report_sha256=expected_admission_hash,
        model_sha256=str(artifact["sha256"]),
        head_om_sha256=head_hash,
        embedding_sha256=embedding_hash,
        ready_descriptor_sha256=str(
            admission_report["ready_descriptor_sha256"]),
        runner_sha256=runner_hash, executor_sha256=executor_hash,
        residency_group=group,
        admission_bytes=admission,
        strict_s1_identity=strict_s1_identity)


def _strict_s1_anchor(
    raw: object, *, root: Path, context: int,
) -> StrictS1Anchor:
    if not isinstance(raw, dict) or set(raw) != {
            "bootstrap_model", "canonical_decode_model", "qualification",
            "head_model", "embedding", "qualification_sha256",
            "build_manifest", "runner", "executor",
            "bootstrap_ready_descriptor", "canonical_ready_descriptor"}:
        raise PrefillActivationError(
            "strict_s1 must explicitly bind the live release baseline")
    bootstrap_model = _safe_file(
        root, raw["bootstrap_model"], "strict_s1.bootstrap_model")
    canonical_model = _safe_file(
        root, raw["canonical_decode_model"],
        "strict_s1.canonical_decode_model")
    head_model = _safe_file(root, raw["head_model"], "strict_s1.head_model")
    embedding = _safe_file(root, raw["embedding"], "strict_s1.embedding")
    qualification_path = _safe_file(
        root, raw["qualification"], "strict_s1.qualification")
    build_manifest = _safe_file(
        root, raw["build_manifest"], "strict_s1.build_manifest")
    runner = _safe_file(root, raw["runner"], "strict_s1.runner")
    executor = _safe_file(root, raw["executor"], "strict_s1.executor")
    bootstrap_descriptor = _safe_file(
        root, raw["bootstrap_ready_descriptor"],
        "strict_s1.bootstrap_ready_descriptor")
    canonical_descriptor = _safe_file(
        root, raw["canonical_ready_descriptor"],
        "strict_s1.canonical_ready_descriptor")
    qualification_sha256 = _hash(
        raw["qualification_sha256"], "strict_s1.qualification_sha256")
    if _sha256(qualification_path) != qualification_sha256:
        raise PrefillActivationError(
            "strict-S1 qualification SHA-256 mismatch")
    report = _load_json(qualification_path, "strict-S1 qualification")
    identity = _verify_nested_strict_s1_qualification(
        report, qualification_sha256=qualification_sha256,
        root=root, context=context)
    artifacts = {
        "bootstrap_om": (report.get("bootstrap_om"), bootstrap_model),
        "canonical_decode_om": (
            report.get("canonical_decode_om"), canonical_model),
        "head_om": (report.get("head_om"), head_model),
        "embedding": (report.get("embedding"), embedding),
        "build_manifest": (report.get("build_manifest"), build_manifest),
    }
    runtime = report.get("runtime")
    if not isinstance(runtime, dict):
        raise PrefillActivationError("strict-S1 runtime lineage is missing")
    artifacts.update({
        "runner": (runtime.get("runner"), runner),
        "executor": (runtime.get("executor"), executor),
        "bootstrap_ready_descriptor": (
            runtime.get("bootstrap_ready_descriptor"), bootstrap_descriptor),
        "canonical_ready_descriptor": (
            runtime.get("canonical_ready_descriptor"), canonical_descriptor),
    })
    for name, pair in artifacts.items():
        descriptor, path = pair
        if not isinstance(descriptor, dict) \
                or descriptor.get("file_verified") is not True \
                or descriptor.get("bytes") != path.stat().st_size \
                or descriptor.get("sha256") != _sha256(path):
            raise PrefillActivationError(
                f"live strict-S1 {name} does not match its qualification")
    return StrictS1Anchor(
        qualification=qualification_path,
        bootstrap_model=bootstrap_model,
        canonical_decode_model=canonical_model,
        head_model=head_model,
        embedding=embedding,
        build_manifest=build_manifest,
        runner=runner,
        executor=executor,
        bootstrap_ready_descriptor=bootstrap_descriptor,
        canonical_ready_descriptor=canonical_descriptor,
        identity=identity,
        declaration={
            "bootstrap_model": str(raw["bootstrap_model"]),
            "canonical_decode_model": str(raw["canonical_decode_model"]),
            "head_model": str(raw["head_model"]),
            "embedding": str(raw["embedding"]),
            "qualification": str(raw["qualification"]),
            "qualification_sha256": qualification_sha256,
            "build_manifest": str(raw["build_manifest"]),
            "runner": str(raw["runner"]),
            "executor": str(raw["executor"]),
            "bootstrap_ready_descriptor": str(
                raw["bootstrap_ready_descriptor"]),
            "canonical_ready_descriptor": str(
                raw["canonical_ready_descriptor"]),
        },
    )


def load_activation(
    manifest: Path,
    *,
    deployment_root: Path,
    context: int,
    available_bytes: int,
    base_resident_bytes: int,
    reserve_bytes: int,
) -> PrefillActivation:
    """Verify eligible blocks and admit the subset that fits live MMZ.

    Individual block failures disable only that width and are reported.  A
    malformed top-level manifest is an operator error and raises.  Every active
    width is charged its independently observed clean-board MMZ delta.  A
    repeated residency group is disabled instead of being used to discount a
    second 24-layer OM.
    """
    if type(context) is not int or context < 128:
        raise PrefillActivationError("context must be an integer >= 128")
    available = _unsigned(available_bytes, "available_bytes", allow_zero=False)
    base = _unsigned(base_resident_bytes, "base_resident_bytes")
    reserve = _unsigned(reserve_bytes, "reserve_bytes")
    if base + reserve > available:
        raise PrefillActivationError(
            "base resident models plus reserve exceed available MMZ")
    root = deployment_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PrefillActivationError("deployment_root must be a directory")
    raw = _load_json(manifest, "prefill activation manifest")
    if set(raw) != {
            "schema", "context", "deployment_mode", "strict_s1", "blocks"} \
            or raw.get("schema") != SCHEMA:
        raise PrefillActivationError("activation manifest schema/fields mismatch")
    if raw.get("deployment_mode") != TRUSTED_DEPLOYMENT_MODE:
        raise PrefillActivationError(
            "activation requires a trusted read-only deployment for the "
            "complete process lifetime")
    if raw.get("context") != context:
        raise PrefillActivationError("activation manifest context mismatch")
    try:
        strict_s1 = _strict_s1_anchor(
            raw.get("strict_s1"), root=root, context=context)
    except PrefillActivationError:
        raise
    except (OSError, UnicodeError) as error:
        raise PrefillActivationError(
            f"cannot verify live strict-S1 anchor: {error}") from error
    base_underreported = base < strict_s1.identity.resident_bytes
    effective_base = max(base, strict_s1.identity.resident_bytes)
    if effective_base + reserve > available:
        raise PrefillActivationError(
            "measured live strict-S1 residency plus reserve exceeds "
            "available MMZ")
    entries = raw.get("blocks")
    if not isinstance(entries, list):
        raise PrefillActivationError("activation blocks must be a list")

    verified: dict[int, ActivatedBlock] = {}
    supplied_widths: set[int] = set()
    disabled: dict[str, str] = {}
    for entry in entries:
        width = entry.get("width") if isinstance(entry, dict) else None
        label = f"S{width}" if type(width) is int else "unknown"
        if type(width) is int and width in supplied_widths:
            raise PrefillActivationError(f"duplicate activation block S{width}")
        if type(width) is int:
            supplied_widths.add(width)
        try:
            block = _block(entry, root=root, context=context)
        except (PrefillActivationError, OSError, UnicodeError) as error:
            disabled[label] = str(error)
            continue
        if block.strict_s1_identity != strict_s1.identity:
            disabled[label] = (
                "wide qualification strict-S1 identity does not match the "
                "live activation anchor")
            continue
        if block.runner_sha256 != strict_s1.identity.runner_sha256 \
                or block.executor_sha256 != strict_s1.identity.executor_sha256:
            disabled[label] = (
                "wide qualification runtime does not match the live "
                "strict-S1 runner/executor")
            continue
        verified[block.width] = block

    used_group_names: set[str] = set()
    active: list[ActivatedBlock] = []
    block_bytes = 0
    for width in WIDTHS:
        block = verified.get(width)
        if block is None:
            disabled.setdefault(f"S{width}", "no qualified artifact")
            continue
        if base_underreported:
            disabled[f"S{width}"] = (
                "base_resident_bytes is below the measured live strict-S1 "
                "residency lower bound")
            continue
        if block.residency_group in used_group_names:
            disabled[f"S{width}"] = (
                "release v4 requires one independently charged width per "
                "residency group")
            continue
        incremental = block.admission_bytes
        if effective_base + block_bytes + incremental + reserve > available:
            disabled[f"S{width}"] = "measured residency exceeds available MMZ"
            continue
        active.append(block)
        block_bytes += incremental
        used_group_names.add(block.residency_group)

    return PrefillActivation(
        context=context,
        enabled_widths=tuple(block.width for block in active) + (1,),
        blocks=tuple(active), disabled=disabled,
        base_resident_bytes=base, block_resident_bytes=block_bytes,
        reserve_bytes=reserve, available_bytes=available,
        strict_s1=strict_s1, base_underreported=base_underreported,
        effective_base_resident_bytes=effective_base,
        deployment_mode=TRUSTED_DEPLOYMENT_MODE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify qualified native-prefill blocks and MMZ admission")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--available-bytes", type=int, required=True)
    parser.add_argument("--base-resident-bytes", type=int, required=True)
    parser.add_argument("--reserve-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = load_activation(
        args.manifest, deployment_root=args.deployment_root,
        context=args.context, available_bytes=args.available_bytes,
        base_resident_bytes=args.base_resident_bytes,
        reserve_bytes=args.reserve_bytes).to_dict()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
