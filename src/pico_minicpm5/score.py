"""Strict public-output scorer for simulator or board raw output files."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .contract import OFFICIAL_CONTRACT, sha256_file


def _file_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _execution_evidence(
    *, om: Path, build_manifest: Path, role: str, context: int
) -> dict:
    manifest = json.loads(build_manifest.read_text(encoding="utf-8"))
    reports = manifest.get("builds") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "pico.minicpm5.three-handle-build.v1"
        or manifest.get("handles") != 3
        or manifest.get("context") != context
        or not isinstance(reports, list)
    ):
        raise ValueError("build manifest schema, handle count or context is invalid")
    matches = [
        record for record in reports
        if isinstance(record, dict) and record.get("role") == role
    ]
    if len(matches) != 1 or matches[0].get("backend") != "atc":
        raise ValueError(f"build manifest has no unique ATC record for {role}")
    if not om.is_file():
        raise FileNotFoundError(om)
    with om.open("rb") as stream:
        if stream.read(4) != b"PICO":
            raise ValueError(f"invalid PICO model: {om}")
    digest = sha256_file(om)
    if matches[0].get("output_sha256") != digest:
        raise ValueError(f"build manifest hash does not match {om.name}")
    return {
        "role": role,
        "om": _file_record(om),
        "compiler": {
            "backend": "atc",
            "build_manifest_sha256": sha256_file(build_manifest),
        },
    }


def _portable_records(value, *, label: str) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(f"capture {label} must be a list")
    records = []
    for record in value:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
        ):
            raise ValueError(f"capture {label} contains a malformed hash record")
        records.append({"bytes": record["bytes"], "sha256": record["sha256"]})
    return records


def _capture_evidence(
    *,
    capture_manifest: Path,
    execution: dict,
    role: str,
    position: int,
    context: int,
    outputs: list[dict],
    inputs: dict[str, dict] | None = None,
) -> dict:
    capture = json.loads(capture_manifest.read_text(encoding="utf-8"))
    if not isinstance(capture, dict):
        raise ValueError("runtime capture manifest must be a JSON object")
    runner = capture.get("runner")
    if (
        capture.get("schema") != "pico.minicpm5.runtime-capture.v1"
        or runner not in {"libinstsim", "ss928-board"}
        or capture.get("status") != "PASS"
        or capture.get("role") != role
        or capture.get("position") != position
        or capture.get("context") != context
        or capture.get("om") != execution["om"]
        or capture.get("build_manifest_sha256")
        != execution["compiler"]["build_manifest_sha256"]
    ):
        raise ValueError("runtime capture does not match this execution")
    captured_outputs = _portable_records(capture.get("outputs"), label="outputs")
    if captured_outputs != outputs:
        raise ValueError("runtime capture output hashes do not match the scored files")
    captured_inputs = capture.get("inputs", {})
    if not isinstance(captured_inputs, dict):
        raise ValueError("runtime capture inputs must be an object")
    portable_inputs = {}
    for name, record in captured_inputs.items():
        values = _portable_records([record], label=f"input {name}")
        portable_inputs[str(name)] = values[0]
    if inputs is not None and portable_inputs != inputs:
        raise ValueError("runtime capture input hashes do not match the scored files")
    return {
        "manifest_sha256": sha256_file(capture_manifest),
        "runner": runner,
        "status": "PASS",
        "position": position,
        "context": context,
        "inputs": portable_inputs,
        "outputs": captured_outputs,
    }


def build_runtime_capture(
    *,
    runner: str,
    role: str,
    position: int,
    context: int,
    om: Path,
    build_manifest: Path,
    inputs: dict[str, Path],
    outputs: tuple[Path, ...],
    report: Path,
) -> dict:
    if runner not in {"libinstsim", "ss928-board"}:
        raise ValueError("runner must be libinstsim or ss928-board")
    if role not in {"prefill", "decode", "head_flat"}:
        raise ValueError("invalid runtime capture role")
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or position < 0
        or (role == "prefill" and position != 0)
        or (role == "decode" and position < 1)
    ):
        raise ValueError("runtime capture position does not match its role")
    expected_outputs = 1 if role == "head_flat" else 3
    if len(outputs) != expected_outputs:
        raise ValueError(f"{role} capture requires {expected_outputs} output file(s)")
    if role == "head_flat" and set(inputs) != {"hidden", "residual"}:
        raise ValueError("head capture requires hidden and residual input files")
    execution = _execution_evidence(
        om=om, build_manifest=build_manifest, role=role, context=context
    )
    value = {
        "schema": "pico.minicpm5.runtime-capture.v1",
        "runner": runner,
        "status": "PASS",
        "role": role,
        "position": position,
        "context": context,
        "om": execution["om"],
        "build_manifest_sha256": execution["compiler"]["build_manifest_sha256"],
        "inputs": {name: _file_record(path) for name, path in sorted(inputs.items())},
        "outputs": [_file_record(path) for path in outputs],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _cosine(left, right) -> float:
    import numpy as np

    a = np.asarray(left, np.float64).reshape(-1)
    b = np.asarray(right, np.float64).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        raise ValueError("cosine is undefined for a zero-norm tensor")
    return float(a @ b / denominator)


def _from_c4(payload: bytes, element: int = 4, stride: int = 16) -> bytes:
    return b"".join(payload[offset : offset + element] for offset in range(0, len(payload), stride))


def score_public_outputs(
    *,
    output_files: tuple[Path, Path, Path],
    reference: Path,
    position: int,
    om: Path,
    build_manifest: Path,
    capture_manifest: Path,
    context: int = 1024,
    threshold_exclusive: float = 0.98,
) -> dict:
    import numpy as np

    c = OFFICIAL_CONTRACT
    role = "prefill" if position == 0 else "decode"
    execution = _execution_evidence(
        om=om, build_manifest=build_manifest, role=role, context=context
    )
    if not 0.0 < threshold_exclusive < 1.0:
        raise ValueError("threshold must be in (0,1)")
    raw = [path.read_bytes() for path in output_files]
    output_records = [_file_record(path) for path in output_files]
    execution["capture"] = _capture_evidence(
        capture_manifest=capture_manifest,
        execution=execution,
        role=role,
        position=position,
        context=context,
        outputs=output_records,
    )
    expected_kv_values = c.num_hidden_layers * c.num_key_value_heads * c.head_dim
    candidates = [np.frombuffer(blob, np.float32) for blob in raw]
    kv_slots = [index for index, value in enumerate(candidates) if value.size == expected_kv_values]
    if len(kv_slots) < 2:
        # All three accepted transformer outputs have the same physical byte
        # size. Preserve the observed output convention as a fallback.
        kv_slots = [0, 1]
    hidden_slot = next(index for index in range(3) if index not in kv_slots[:2])
    hidden_raw = raw[hidden_slot]
    if len(hidden_raw) == c.hidden_size * 16:
        hidden_raw = _from_c4(hidden_raw)
    hidden = np.frombuffer(hidden_raw, np.float32)
    if hidden.size != c.hidden_size:
        raise ValueError(f"hidden output has {hidden.size} values")

    next_position = reference / f"pos{position + 1}"
    if not next_position.is_dir():
        raise FileNotFoundError(
            f"reference requires pos{position + 1} to identify current K/V rows"
        )

    def reference_rows(role: str):
        rows = []
        for layer in range(c.num_hidden_layers):
            cache = np.fromfile(
                next_position / f"{role}_cache_{layer:02d}.f16.bin", np.float16
            ).reshape(c.num_key_value_heads, context - 1, c.head_dim)
            rows.append(cache[:, position].astype(np.float32))
        return np.concatenate(rows, axis=0).reshape(-1)

    ref_k, ref_v = reference_rows("k"), reference_rows("v")
    first, second = candidates[kv_slots[0]], candidates[kv_slots[1]]
    direct = _cosine(first, ref_k) + _cosine(second, ref_v)
    swapped = _cosine(second, ref_k) + _cosine(first, ref_v)
    k_out, v_out = (first, second) if direct >= swapped else (second, first)
    ref_hidden = np.fromfile(
        reference / f"pos{position}" / "layer_out_23.f32.bin", np.float32
    )
    cosine = {
        "next_hidden": _cosine(hidden, ref_hidden),
        "k_cur_all": _cosine(k_out, ref_k),
        "v_cur_all": _cosine(v_out, ref_v),
    }
    minimum = min(cosine.values())
    per_layer = []
    width = c.num_key_value_heads * c.head_dim
    for layer in range(c.num_hidden_layers):
        begin, end = layer * width, (layer + 1) * width
        per_layer.append(
            {
                "layer": layer,
                "k_cur": _cosine(k_out[begin:end], ref_k[begin:end]),
                "v_cur": _cosine(v_out[begin:end], ref_v[begin:end]),
            }
        )
    return {
        "schema": "pico.minicpm5.public-output-score.v1",
        "position": position,
        "context": context,
        "role": role,
        "execution": execution,
        "threshold_exclusive": threshold_exclusive,
        "cosine": cosine,
        "minimum_public_output": minimum,
        "output_order": "direct" if direct >= swapped else "swapped",
        "per_layer_diagnostic": per_layer,
        "logical_outputs": {
            "next_hidden": {
                "bytes": len(hidden_raw),
                "sha256": hashlib.sha256(hidden_raw).hexdigest(),
            }
        },
        "outputs": [
            {"path": str(path.resolve()), **record}
            for path, record in zip(output_files, output_records)
        ],
        "status": "PASS" if minimum > threshold_exclusive else "FAIL",
    }


def score_head_output(
    *,
    output_file: Path,
    reference: Path,
    position: int,
    om: Path,
    build_manifest: Path,
    capture_manifest: Path,
    hidden_input: Path,
    residual_input: Path,
    context: int = 1024,
    threshold_exclusive: float = 0.98,
) -> dict:
    import numpy as np

    if not 0.0 < threshold_exclusive < 1.0:
        raise ValueError("threshold must be in (0,1)")
    execution = _execution_evidence(
        om=om, build_manifest=build_manifest, role="head_flat", context=context
    )
    inputs = {
        "hidden": _file_record(hidden_input),
        "residual": _file_record(residual_input),
    }
    output_record = _file_record(output_file)
    execution["capture"] = _capture_evidence(
        capture_manifest=capture_manifest,
        execution=execution,
        role="head_flat",
        position=position,
        context=context,
        outputs=[output_record],
        inputs=inputs,
    )
    hidden = np.fromfile(hidden_input, np.float32)
    residual = np.fromfile(residual_input, np.float32)
    if hidden.size != OFFICIAL_CONTRACT.hidden_size:
        raise ValueError(f"head hidden input has {hidden.size} values")
    if residual.size != OFFICIAL_CONTRACT.hidden_size or not np.all(residual == 0.0):
        raise ValueError("head residual input must be exactly 1536 FP32 zeros")
    output = np.fromfile(output_file, np.float32)
    expected = np.fromfile(reference / f"pos{position}" / "logits.f32.bin", np.float32)
    if output.size != OFFICIAL_CONTRACT.vocab_size or expected.size != output.size:
        raise ValueError(
            f"head logits size drift: output={output.size} reference={expected.size}"
        )
    cosine = _cosine(output, expected)
    top1_output, top1_reference = int(np.argmax(output)), int(np.argmax(expected))
    top1_exact = top1_output == top1_reference
    return {
        "schema": "pico.minicpm5.head-output-score.v1",
        "position": position,
        "context": context,
        "role": "head_flat",
        "threshold_exclusive": threshold_exclusive,
        "cosine": cosine,
        "minimum_public_output": cosine,
        "top1_output": top1_output,
        "top1_reference": top1_reference,
        "top1_exact": top1_exact,
        "execution": execution,
        "inputs": inputs,
        "output": output_record,
        "status": "PASS" if cosine > threshold_exclusive and top1_exact else "FAIL",
    }


def write_score(report: dict, output: Path) -> None:
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
