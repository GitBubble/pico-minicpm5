# SPDX-License-Identifier: Apache-2.0
"""Pure-JSON numeric gate for extended-context runtime profiles.

A context-profile qualification record binds the identity (bytes + sha256)
of a per-context decode OM, the inherited frozen v0.1.0 prefill/head
artifacts of the mixed prefill-window contract, and the externally
produced cosine/token/boundary evidence. The heavy evidence (OMs, score
captures, board reports) lives outside this repository; only hashes and
scores are committed, so this validator never touches the filesystem —
mirroring how ``release.bundle._qualified`` gates the frozen ctx1024
release.

A profile in ``app/profiles`` may declare ``status: qualified`` for an
extended context only when the matching record in ``release/contexts``
passes this gate with ``verdict.overall == "PASS"``.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

SCHEMA = "pico.minicpm5.context-profile-qualification.v1"
COSINE_FLOOR_EXCLUSIVE = 0.98
KNOWN_CONTEXTS = (128, 1024, 4096, 8192)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOCALHOST_LEAK = re.compile(r"/(?:Users|home|root)/|(?:\d{1,3}\.){3}\d{1,3}")

_GATE_VERDICTS = (
    "public_output_numeric",
    "greedy_exact",
    "eos",
    "context_boundary",
    "board_load",
)


class ContextQualificationError(ValueError):
    """A context-profile qualification record fails the fail-closed gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContextQualificationError(message)


def _exact_keys(record: dict, expected: set[str], label: str) -> None:
    _require(isinstance(record, dict) and set(record) == expected,
             f"{label} fields mismatch: expected {sorted(expected)}")


def _artifact(record: dict, label: str, *, prefix: str) -> None:
    _exact_keys(record, {"deployment_path", "bytes", "sha256"}, label)
    path = record["deployment_path"]
    _require(isinstance(path, str) and path.startswith(prefix)
             and ".." not in path,
             f"{label}.deployment_path must stay under {prefix}")
    _require(type(record["bytes"]) is int and record["bytes"] > 0,
             f"{label}.bytes must be a positive integer")
    sha = record["sha256"]
    _require(isinstance(sha, str) and _SHA256.fullmatch(sha) is not None,
             f"{label}.sha256 must be 64 lowercase hex characters")


def _cosine(value, label: str, threshold: float) -> float:
    _require(not isinstance(value, bool)
             and isinstance(value, (int, float))
             and threshold < float(value) <= 1.0,
             f"{label} must be a cosine strictly above {threshold}")
    return float(value)


def _scan_leaks(value, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _scan_leaks(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_leaks(item, f"{label}[{index}]")
    elif isinstance(value, str):
        _require(_LOCALHOST_LEAK.search(value) is None,
                 f"{label} leaks a local path or network address")


def validate_record(record: dict) -> dict:
    """Validate one record; returns a compact summary, raises on any hole."""
    _exact_keys(record, {
        "schema", "profile", "target", "contract", "numeric_gate",
        "gates", "diagnostics_nonblocking", "evidence", "verdict",
    }, "context qualification")
    _require(record["schema"] == SCHEMA, f"schema must be {SCHEMA}")
    _scan_leaks(record, "record")

    target = record["target"]
    _exact_keys(target, {"soc", "npu_arch", "context", "prefill_window"},
                "target")
    _require(target["soc"] == "Hi3403" and target["npu_arch"] == "V101",
             "target must be Hi3403/V101")
    context = target["context"]
    window = target["prefill_window"]
    _require(context in KNOWN_CONTEXTS and window in KNOWN_CONTEXTS,
             f"context and prefill_window must be one of {KNOWN_CONTEXTS}")
    _require(window <= context, "prefill_window cannot exceed context")
    _require(record["profile"] == f"ctx{context}",
             "profile name must match the target context")

    contract = record["contract"]
    _exact_keys(contract, {"kind", "decode", "prefill", "head"}, "contract")
    _require(contract["kind"] == "mixed-prefill-window",
             "contract.kind must be mixed-prefill-window")
    _artifact(contract["decode"], "contract.decode",
              prefix=f"models/ctx{context}/")
    _artifact(contract["prefill"], "contract.prefill", prefix="models/")
    _artifact(contract["head"], "contract.head", prefix="models/")

    gate = record["numeric_gate"]
    _exact_keys(gate, {
        "cosine_threshold_exclusive", "public_outputs",
        "minimum_public_output", "board_tail_byte_exact_position",
    }, "numeric_gate")
    threshold = gate["cosine_threshold_exclusive"]
    _require(not isinstance(threshold, bool)
             and isinstance(threshold, (int, float))
             and float(threshold) >= COSINE_FLOOR_EXCLUSIVE,
             f"cosine threshold floor is {COSINE_FLOOR_EXCLUSIVE}")
    threshold = float(threshold)
    outputs = gate["public_outputs"]
    _require(isinstance(outputs, list) and outputs,
             "numeric_gate.public_outputs must be a non-empty list")
    seen_positions = set()
    observed = []
    for index, row in enumerate(outputs):
        label = f"public_outputs[{index}]"
        _exact_keys(row, {"position", "runner", "next_hidden",
                          "packed_k", "packed_v"}, label)
        _require(type(row["position"]) is int
                 and 0 <= row["position"] < context,
                 f"{label}.position must be in [0, context)")
        _require(row["runner"] in ("libinstsim", "board"),
                 f"{label}.runner must be libinstsim or board")
        seen_positions.add(row["position"])
        for metric in ("next_hidden", "packed_k", "packed_v"):
            observed.append(
                _cosine(row[metric], f"{label}.{metric}", threshold))
    _require({1, context - 1} <= seen_positions,
             "public outputs must cover position 1 and the last valid "
             "position (context - 1)")
    minimum = _cosine(
        gate["minimum_public_output"], "minimum_public_output", threshold)
    _require(minimum == min(observed),
             "minimum_public_output must equal the smallest reported cosine")
    tail = gate["board_tail_byte_exact_position"]
    _require(tail == context - 1,
             "board tail byte-exact evidence must sit at context - 1")

    verdict = record["verdict"]
    _exact_keys(verdict, set(_GATE_VERDICTS) | {"overall"}, "verdict")
    gates = record["gates"]
    _require(isinstance(gates, dict) and gates,
             "gates must be a non-empty object")
    for name in _GATE_VERDICTS:
        value = verdict[name]
        _require(isinstance(value, str) and value,
                 f"verdict.{name} must be a non-empty string")
    overall = verdict["overall"]
    _require(isinstance(overall, str) and overall,
             "verdict.overall must be a non-empty string")
    if overall == "PASS":
        failing = [name for name in _GATE_VERDICTS
                   if verdict[name] != "PASS"]
        _require(not failing,
                 f"overall PASS requires every gate PASS; failing: {failing}")

    evidence = record["evidence"]
    _exact_keys(evidence, {"source_record", "source_sha256", "location"},
                "evidence")
    _require(_SHA256.fullmatch(str(evidence["source_sha256"])) is not None,
             "evidence.source_sha256 must be 64 lowercase hex characters")

    return {
        "profile": record["profile"],
        "context": context,
        "prefill_window": window,
        "minimum_public_output": minimum,
        "threshold_exclusive": threshold,
        "overall": overall,
        "passes": overall == "PASS",
    }


def record_passes(record: dict) -> bool:
    return validate_record(record)["passes"]


def verify_file(path: Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_record(record)
