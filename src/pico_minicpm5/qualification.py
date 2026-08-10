"""Build a portable two-family qualification consumed by release assembly."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any

from .contract import (
    HF_REPO_ID,
    HF_REVISION,
    PUBLIC_COSINE_THRESHOLD_EXCLUSIVE,
    sha256_file,
)

MODEL_FILES = {"decode": "decode.om", "prefill": "prefill.om", "head_flat": "head_flat.om"}
CONTEXT = 1024
ZERO_RESIDUAL = {
    "bytes": 1536 * 4,
    "sha256": hashlib.sha256(bytes(1536 * 4)).hexdigest(),
}


def _portable_hash_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} hash record is malformed")
    size, digest = value.get("bytes"), value.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} hash record is malformed")
    return {"bytes": size, "sha256": digest}


def _portable_execution(
    value: Any,
    *,
    role: str,
    position: int,
    context: int,
    outputs: list[dict[str, Any]],
    inputs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("role") != role:
        raise ValueError(f"{role} score has no bound execution evidence")
    compiler = value.get("compiler")
    if not isinstance(compiler, dict) or compiler.get("backend") != "atc":
        raise ValueError(f"{role} score was not produced from an ATC build")
    build_digest = compiler.get("build_manifest_sha256")
    if (
        not isinstance(build_digest, str)
        or len(build_digest) != 64
        or any(character not in "0123456789abcdef" for character in build_digest)
    ):
        raise ValueError(f"{role} score has no build-manifest hash")
    capture = value.get("capture")
    if not isinstance(capture, dict):
        raise ValueError(f"{role} score has no runtime capture evidence")
    capture_digest = capture.get("manifest_sha256")
    if (
        capture.get("runner") not in {"libinstsim", "ss928-board"}
        or capture.get("status") != "PASS"
        or capture.get("position") != position
        or capture.get("context") != context
        or not isinstance(capture_digest, str)
        or len(capture_digest) != 64
        or any(character not in "0123456789abcdef" for character in capture_digest)
    ):
        raise ValueError(f"{role} runtime capture is malformed")
    capture_outputs = [
        _portable_hash_record(record, label=f"{role} captured output {index}")
        for index, record in enumerate(capture.get("outputs", []))
    ]
    if capture_outputs != outputs:
        raise ValueError(f"{role} runtime capture output hashes disagree")
    capture_inputs_raw = capture.get("inputs")
    if not isinstance(capture_inputs_raw, dict):
        raise ValueError(f"{role} runtime capture inputs are malformed")
    capture_inputs = {
        str(name): _portable_hash_record(record, label=f"{role} captured input {name}")
        for name, record in capture_inputs_raw.items()
    }
    if inputs is not None and capture_inputs != inputs:
        raise ValueError(f"{role} runtime capture input hashes disagree")
    return {
        "role": role,
        "om": _portable_hash_record(value.get("om"), label=f"{role} OM"),
        "compiler": {
            "backend": "atc",
            "build_manifest_sha256": build_digest,
        },
        "capture": {
            "manifest_sha256": capture_digest,
            "runner": capture["runner"],
            "status": "PASS",
            "position": position,
            "context": context,
            "inputs": capture_inputs,
            "outputs": capture_outputs,
        },
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _portable_score(path: Path, *, role: str, threshold: float) -> dict[str, Any]:
    score = _load(path)
    if score.get("schema") != "pico.minicpm5.public-output-score.v1":
        raise ValueError(f"{role} score has an unsupported schema")
    if score.get("role") != role:
        raise ValueError(f"{role} score role does not match its position")
    position = score.get("position")
    context = score.get("context")
    if context != CONTEXT:
        raise ValueError(f"{role} score must use context {CONTEXT}")
    if role == "prefill" and position != 0:
        raise ValueError("prefill score must be captured at position 0")
    if role == "decode" and (not isinstance(position, int) or position < 1):
        raise ValueError("decode score must be captured at position >=1")
    minimum = score.get("minimum_public_output")
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or float(minimum) <= threshold
    ):
        raise ValueError(f"{role} minimum public-output cosine does not exceed {threshold}")
    cosine = score.get("cosine")
    if not isinstance(cosine, dict) or set(cosine) != {"next_hidden", "k_cur_all", "v_cur_all"}:
        raise ValueError(f"{role} score does not contain the three public tensors")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in cosine.values()
    ):
        raise ValueError(f"{role} public tensor cosine is not numeric")
    values = [float(value) for value in cosine.values()]
    if any(not math.isfinite(value) or value <= threshold or value > 1.0 for value in values):
        raise ValueError(f"{role} public tensor does not exceed {threshold}")
    if not math.isclose(float(minimum), min(values), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{role} minimum does not match its public tensors")
    outputs = []
    for index, record in enumerate(score.get("outputs", [])):
        outputs.append(
            _portable_hash_record(record, label=f"{role} output {index}")
        )
    if len(outputs) != 3:
        raise ValueError(f"{role} score must identify exactly three raw outputs")
    output_order = score.get("output_order")
    if output_order not in {"direct", "swapped"}:
        raise ValueError(f"{role} score has an invalid output order")
    logical_raw = score.get("logical_outputs")
    if not isinstance(logical_raw, dict) or set(logical_raw) != {"next_hidden"}:
        raise ValueError(f"{role} score has no logical next_hidden hash")
    logical_outputs = {
        "next_hidden": _portable_hash_record(
            logical_raw["next_hidden"], label=f"{role} logical next_hidden"
        )
    }
    execution = _portable_execution(
        score.get("execution"),
        role=role,
        position=position,
        context=context,
        outputs=outputs,
    )
    return {
        "position": position,
        "context": context,
        "cosine": {key: float(value) for key, value in cosine.items()},
        "minimum_public_output": float(minimum),
        "output_order": output_order,
        "outputs": outputs,
        "logical_outputs": logical_outputs,
        "execution": execution,
    }


def _portable_head_score(path: Path, *, threshold: float) -> dict[str, Any]:
    score = _load(path)
    if score.get("schema") != "pico.minicpm5.head-output-score.v1":
        raise ValueError("head score has an unsupported schema")
    if score.get("role") != "head_flat":
        raise ValueError("head score has an invalid role")
    context = score.get("context")
    if context != CONTEXT:
        raise ValueError(f"head score must use context {CONTEXT}")
    minimum = score.get("minimum_public_output")
    if (
        not isinstance(minimum, (int, float))
        or isinstance(minimum, bool)
        or not math.isfinite(float(minimum))
        or float(minimum) <= threshold
        or float(minimum) > 1.0
    ):
        raise ValueError(f"head minimum public-output cosine does not exceed {threshold}")
    cosine = score.get("cosine")
    if (
        not isinstance(cosine, (int, float))
        or isinstance(cosine, bool)
        or not math.isclose(float(cosine), float(minimum), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("head minimum does not match its cosine")
    if score.get("top1_exact") is not True:
        raise ValueError("head top1 token does not match the float reference")
    position = score.get("position")
    top1_output, top1_reference = score.get("top1_output"), score.get("top1_reference")
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or position < 0
        or not isinstance(top1_output, int)
        or isinstance(top1_output, bool)
        or not isinstance(top1_reference, int)
        or isinstance(top1_reference, bool)
        or top1_output != top1_reference
    ):
        raise ValueError("head score position or top1 evidence is malformed")
    output = _portable_hash_record(score.get("output"), label="head output")
    inputs_raw = score.get("inputs")
    if not isinstance(inputs_raw, dict) or set(inputs_raw) != {"hidden", "residual"}:
        raise ValueError("head score must identify hidden and residual inputs")
    inputs = {
        name: _portable_hash_record(record, label=f"head {name} input")
        for name, record in inputs_raw.items()
    }
    if inputs["residual"] != ZERO_RESIDUAL:
        raise ValueError("head residual input is not the canonical zero tensor")
    execution = _portable_execution(
        score.get("execution"),
        role="head_flat",
        position=position,
        context=context,
        outputs=[output],
        inputs=inputs,
    )
    return {
        "position": position,
        "context": context,
        "cosine": float(cosine),
        "minimum_public_output": float(minimum),
        "top1_output": top1_output,
        "top1_reference": top1_reference,
        "top1_exact": True,
        "output": output,
        "inputs": inputs,
        "execution": execution,
    }


def _model_evidence(models: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = models / "build-manifest.json"
    build = _load(manifest_path)
    reports = build.get("builds")
    if (
        build.get("schema") != "pico.minicpm5.three-handle-build.v1"
        or build.get("handles") != 3
        or build.get("context") != CONTEXT
        or not isinstance(reports, list)
    ):
        raise ValueError("build manifest schema, handle count or context is invalid")
    by_role = {
        record.get("role"): record for record in reports if isinstance(record, dict)
    }
    if set(by_role) != set(MODEL_FILES):
        raise ValueError("build manifest must contain decode, prefill and head_flat")
    artifacts = {}
    for role, name in MODEL_FILES.items():
        record = by_role[role]
        if record.get("backend") != "atc":
            raise ValueError(f"{role} was not built by the ATC backend")
        path = models / name
        magic = b""
        if path.is_file():
            with path.open("rb") as stream:
                magic = stream.read(4)
        if magic != b"PICO":
            raise ValueError(f"missing or invalid {path}")
        digest = sha256_file(path)
        expected = record.get("output_sha256")
        if expected != digest:
            raise ValueError(f"build manifest hash mismatch for {name}")
        artifacts[role] = {"bytes": path.stat().st_size, "sha256": digest}
    compiler = {
        "backend": "atc",
        "build_manifest_sha256": sha256_file(manifest_path),
    }
    return artifacts, compiler


def build_qualification(
    *,
    prefill_score: Path,
    decode_score: Path,
    head_score: Path,
    models: Path,
    output: Path,
    greedy_report: Path | None = None,
) -> dict[str, Any]:
    threshold_exclusive = PUBLIC_COSINE_THRESHOLD_EXCLUSIVE
    artifacts, compiler = _model_evidence(models)
    families = {
        "prefill": _portable_score(
            prefill_score, role="prefill", threshold=threshold_exclusive
        ),
        "decode": _portable_score(
            decode_score, role="decode", threshold=threshold_exclusive
        ),
    }
    head = _portable_head_score(head_score, threshold=threshold_exclusive)

    def require_bound(record: dict[str, Any], role: str) -> None:
        execution = record["execution"]
        om = execution.get("om", {})
        compiler_record = execution.get("compiler", {})
        if om != artifacts[role]:
            raise ValueError(f"{role} score is bound to a different OM")
        if (
            compiler_record.get("backend") != "atc"
            or compiler_record.get("build_manifest_sha256")
            != compiler["build_manifest_sha256"]
        ):
            raise ValueError(f"{role} score is bound to a different build manifest")

    for role, record in families.items():
        require_bound(record, role)
    require_bound(head, "head_flat")
    source_role = "prefill" if head["position"] == 0 else "decode"
    source = families[source_role]
    if (
        head["position"] != source["position"]
        or head["inputs"]["hidden"] != source["logical_outputs"]["next_hidden"]
    ):
        raise ValueError("head input is not the matching transformer logical next_hidden")
    head["source_transformer_role"] = source_role
    greedy = None
    if greedy_report is not None:
        greedy = _load(greedy_report)
        verdict = greedy.get("verdict", greedy.get("status"))
        if verdict != "PASS" and not (
            isinstance(verdict, dict) and verdict.get("overall") == "PASS"
        ):
            raise ValueError("greedy report is not passing")
        # Do not copy arbitrary local paths from a runtime report into a release.
        greedy = {
            "sha256": hashlib.sha256(greedy_report.read_bytes()).hexdigest(),
            "verdict": "PASS",
        }
    minimum = min(
        [item["minimum_public_output"] for item in families.values()]
        + [head["minimum_public_output"]]
    )
    report: dict[str, Any] = {
        "schema": "pico.minicpm5.qualification.v1",
        "model": {"repository": HF_REPO_ID, "revision": HF_REVISION},
        "target": {"soc": "SS928", "npu_arch": "V101", "context": CONTEXT},
        "threshold_exclusive": threshold_exclusive,
        "families": families,
        "head": head,
        "minimum_public_output": minimum,
        "artifacts": artifacts,
        "compiler": compiler,
        "greedy": greedy,
        "verdict": {
            "public_output_numeric": "PASS",
            "head_numeric": "PASS",
            "overall": "PASS",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
