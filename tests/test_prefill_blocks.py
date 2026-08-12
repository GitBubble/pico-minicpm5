from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct

import pytest

from pico_minicpm5.prefill_blocks import (
    CAPTURE_SCHEMA,
    MMZ_SCHEMA,
    POLICY_THRESHOLD_EXCLUSIVE,
    PERFORMANCE_SCHEMA,
    RELEASE_SCHEMA,
    SCHEMA,
    STRICT_S1_BASELINE_SCHEMA,
    STRICT_S1_CAPTURE_SCHEMA,
    STRICT_S1_DEVELOPMENT_SCHEMA,
    STRICT_S1_MMZ_SCHEMA,
    STRICT_S1_WORKLOAD_SCHEMA,
    TOKEN_ID_HASH_CONTRACT,
    WORKLOAD_SCHEMA,
    PrefillBlockQualificationError,
    build_qualification,
    build_release_qualification,
    build_strict_s1_release_qualification,
    canonical_mask_sha256,
    logical_abi,
    qualify_file,
    qualify_release_file,
    qualify_strict_s1_release_file,
    required_starts,
    required_strict_s1_positions,
    token_id_sequence_sha256,
    verify_release_qualification,
    verify_strict_s1_release_qualification,
)


def _capture(
    start: int, width: int = 16, cosine: float = 0.999,
    context: int = 1024,
) -> dict[str, object]:
    rows = 24 * 2 * width
    return {
        "start": start,
        "capture_sha256": f"{start + 1:064x}",
        "physical_descriptor_sha256": "a" * 64,
        "publisher_source_sha256": {
            "k": f"{start + 3:064x}", "v": f"{start + 4:064x}"},
        "publisher_source_dtype": "FP32",
        "publisher_source_layout": "contiguous-channel-major",
        "descriptor_exact": True,
        "mask_sha256": (
            canonical_mask_sha256(context, width, start)
            if 1 <= start <= context - width else "0" * 64),
        "mask_bytes_exact": True,
        "rope_sha256": f"{start + 6:064x}",
        "rope_bytes_exact": True,
        "kv_rows": {"k": rows, "v": rows},
        "kv_rows_exact": True,
        "prefill_decode_handoff": True,
        "token_exact": True,
        "board_pass": True,
        "public_cosines": {"hidden": cosine, "k": cosine, "v": cosine},
        "handoff_cosines": {"hidden": cosine, "k": cosine, "v": cosine},
    }


def _workloads(context: int = 1024) -> list[dict[str, object]]:
    records = []
    for index, kind in enumerate(("eos", "english", "chinese", "context_boundary")):
        record = {
            "kind": kind,
            "capture_sha256": f"{100 + index:064x}",
            "prompt_sha256": f"{110 + index:064x}",
            "output_tokens_sha256": f"{120 + index:064x}",
            "board_pass": True,
            "token_exact": True,
        }
        if kind == "eos":
            record["eos_exact"] = True
        if kind == "context_boundary":
            record["boundary_exact"] = True
            record["terminal_position"] = context - 1
        records.append(record)
    return records


def _files(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    om_path = root / "s16.om"
    manifest_path = root / "build-manifest.json"
    om_path.write_bytes(b"real-om-bytes")
    manifest_path.write_text(json.dumps({"build": "s16"}), encoding="utf-8")

    def artifact(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    return artifact(om_path), artifact(manifest_path)


def _values(tmp_path: Path, *, context: int = 1024, width: int = 16) -> dict[str, object]:
    om, build_manifest = _files(tmp_path)
    baseline_width = {16: 1, 32: 16, 128: 32}[width]
    return {
        "context": context,
        "width": width,
        "om": om,
        "build_manifest": build_manifest,
        "captures": [
            _capture(start, width, context=context)
            for start in required_starts(context, width)],
        "workloads": _workloads(context),
        "performance": [{
            "metric": "board-wall-time-ms",
            "board": "Hi3403",
            "tokens": width,
            "candidate_width": width,
            "baseline_width": baseline_width,
            "candidate_invocations": 1,
            "baseline_invocations": width // baseline_width,
            "warmup_runs": 2,
            "measured_runs": 10,
            "candidate_ms": 8.0,
            "baseline_ms": 10.0,
            "candidate_om_sha256": om["sha256"],
            "baseline_artifact_sha256": "c" * 64,
            "measurement_sha256": "d" * 64,
        }] + ([{
            "metric": "board-wall-time-ms",
            "board": "Hi3403",
            "tokens": 128,
            "candidate_width": 128,
            "baseline_width": 16,
            "candidate_invocations": 1,
            "baseline_invocations": 8,
            "warmup_runs": 2,
            "measured_runs": 10,
            "candidate_ms": 8.0,
            "baseline_ms": 10.0,
            "candidate_om_sha256": om["sha256"],
            "baseline_artifact_sha256": "e" * 64,
            "measurement_sha256": "f" * 64,
        }] if width == 128 else []),
    }


def _qualified(tmp_path: Path, **overrides) -> dict[str, object]:
    values = _values(tmp_path)
    values.update(overrides)
    return build_qualification(**values, artifact_root=tmp_path)


def test_s16_qualification_binds_publisher_resident_and_activation(tmp_path: Path) -> None:
    report = _qualified(tmp_path)

    assert report["schema"] == SCHEMA
    assert report["status"] == "PASS"
    assert report["threshold_exclusive"] == POLICY_THRESHOLD_EXCLUSIVE
    assert report["required_starts"] == [
        1, 15, 16, 31, 32, 255, 256, 643, 1008]
    assert report["minimum_cosine"] == 0.999
    assert report["om"]["file_verified"] is True
    assert report["build_manifest"]["file_verified"] is True
    assert report["abi"] == logical_abi(1024, 16)
    assert report["abi"]["publisher"] == {
        "layout": "contiguous-channel-major",
        "dtype": "FP32",
        "shape": [1, 48, 16, 128],
        "logical_rows": 768,
        "logical_bytes": 393_216,
        "roles": ["k", "v"],
        "resident_conversion": {
            "opcode": 6, "dst_dtype": "FP16", "rounding": "RNE"},
    }
    assert report["abi"]["k"]["dtype"] == "FP16"
    assert report["abi"]["k"]["logical_rows"] == 24 * 2 * 16
    assert report["activation"] == {
        "key": "ctx1024.s16.steady",
        "phase": "steady",
        "minimum_start": 1,
        "startup_requires_strict_s1": True,
        "runtime_eligible": True,
        "context": 1024,
        "width": 16,
        "om_sha256": report["om"]["sha256"],
        "build_manifest_sha256": report["build_manifest"]["sha256"],
        "qualification_schema": SCHEMA,
    }


def test_steady_contract_rejects_position_zero(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"].append(_capture(0))
    with pytest.raises(PrefillBlockQualificationError, match="steady capture"):
        build_qualification(**values, artifact_root=tmp_path)


def test_captures_bind_descriptor_mask_rope_and_are_sorted(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"] = list(reversed(values["captures"]))
    report = build_qualification(**values, artifact_root=tmp_path)
    assert [entry["start"] for entry in report["captures"]] == [
        1, 15, 16, 31, 32, 255, 256, 643, 1008]
    assert all(entry["descriptor_exact"] and entry["mask_bytes_exact"]
               and entry["rope_bytes_exact"] for entry in report["captures"])
    assert all(len(entry["physical_descriptor_sha256"]) == 64
               for entry in report["captures"])


def test_capture_mask_hash_must_match_absolute_prefix_and_sentinel(
        tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"][0]["mask_sha256"] = "f" * 64

    with pytest.raises(
            PrefillBlockQualificationError,
            match="canonical absolute mask"):
        build_qualification(**values, artifact_root=tmp_path)


@pytest.mark.parametrize("field", [
    "descriptor_exact", "mask_bytes_exact", "rope_bytes_exact", "kv_rows_exact",
    "prefill_decode_handoff", "token_exact", "board_pass",
])
def test_every_capture_boolean_gate_is_required(tmp_path: Path, field: str) -> None:
    values = _values(tmp_path)
    values["captures"][0][field] = False
    with pytest.raises(PrefillBlockQualificationError, match=field):
        build_qualification(**values, artifact_root=tmp_path)


@pytest.mark.parametrize("field,value,match", [
    ("publisher_source_dtype", "FP16", "source dtype"),
    ("publisher_source_layout", "row-major", "source layout"),
])
def test_physical_publisher_is_independent_of_resident_abi(
    tmp_path: Path, field: str, value: str, match: str,
) -> None:
    values = _values(tmp_path)
    values["captures"][0][field] = value
    with pytest.raises(PrefillBlockQualificationError, match=match):
        build_qualification(**values, artifact_root=tmp_path)


def test_all_24_by_2_by_width_rows_are_required_per_role(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"][0]["kv_rows"]["v"] -= 1
    with pytest.raises(PrefillBlockQualificationError, match=r"24\*2\*width"):
        build_qualification(**values, artifact_root=tmp_path)


def test_cosine_gate_is_strict_and_rejects_json_booleans(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"][0]["public_cosines"]["v"] = 0.98
    with pytest.raises(PrefillBlockQualificationError, match="strictly greater"):
        build_qualification(**values, artifact_root=tmp_path)

    values = _values(tmp_path)
    values["captures"][0]["handoff_cosines"]["hidden"] = True
    with pytest.raises(PrefillBlockQualificationError, match="real number"):
        build_qualification(**values, artifact_root=tmp_path)


def test_missing_boundary_or_duplicate_capture_fails_closed(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"] = values["captures"][:-1]
    with pytest.raises(PrefillBlockQualificationError, match="missing required"):
        build_qualification(**values, artifact_root=tmp_path)

    values = _values(tmp_path)
    values["captures"].append(copy.deepcopy(values["captures"][0]))
    with pytest.raises(PrefillBlockQualificationError, match="must not repeat"):
        build_qualification(**values, artifact_root=tmp_path)


def test_all_positions_must_bind_one_ready_descriptor(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"][-1]["physical_descriptor_sha256"] = "b" * 64
    with pytest.raises(PrefillBlockQualificationError, match="one physical"):
        build_qualification(**values, artifact_root=tmp_path)


def test_extra_position_cannot_create_a_nonactivatable_pass(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["captures"].append(_capture(777))
    with pytest.raises(PrefillBlockQualificationError, match="unexpected block"):
        build_qualification(**values, artifact_root=tmp_path)


@pytest.mark.parametrize("context,width", [
    (1024, 1), (1024, 64), (120, 16), (1025, 16), (128, 128),
])
def test_unqualified_steady_geometry_is_rejected(context: int, width: int) -> None:
    with pytest.raises(PrefillBlockQualificationError):
        required_starts(context, width)


def test_policy_threshold_cannot_be_weakened(tmp_path: Path) -> None:
    values = _values(tmp_path)
    with pytest.raises(PrefillBlockQualificationError, match="exactly 0.98"):
        build_qualification(
            **values, artifact_root=tmp_path, threshold_exclusive=0.5)


@pytest.mark.parametrize("kind,field", [
    ("eos", "eos_exact"),
    ("english", "token_exact"),
    ("chinese", "board_pass"),
    ("context_boundary", "boundary_exact"),
])
def test_eos_language_and_context_workloads_are_hard_gates(
    tmp_path: Path, kind: str, field: str,
) -> None:
    values = _values(tmp_path)
    workload = next(item for item in values["workloads"] if item["kind"] == kind)
    workload[field] = False
    with pytest.raises(PrefillBlockQualificationError, match=field):
        build_qualification(**values, artifact_root=tmp_path)


def test_context_workload_must_reach_final_position(tmp_path: Path) -> None:
    values = _values(tmp_path)
    workload = next(
        item for item in values["workloads"] if item["kind"] == "context_boundary")
    workload["terminal_position"] = 1022
    with pytest.raises(PrefillBlockQualificationError, match="context-1"):
        build_qualification(**values, artifact_root=tmp_path)


def test_wide_block_must_win_same_token_board_measurement(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["performance"][0]["candidate_ms"] = 10.0
    with pytest.raises(PrefillBlockQualificationError, match="strictly faster"):
        build_qualification(**values, artifact_root=tmp_path)

    values = _values(tmp_path)
    values["performance"][0]["baseline_invocations"] = 15
    with pytest.raises(PrefillBlockQualificationError, match="baseline_invocations"):
        build_qualification(**values, artifact_root=tmp_path)


@pytest.mark.parametrize("width,baseline,invocations", [
    (16, 1, 16), (32, 16, 2), (128, 32, 4),
])
def test_performance_baseline_follows_activation_ladder(
    tmp_path: Path, width: int, baseline: int, invocations: int,
) -> None:
    context = 1024
    values = _values(tmp_path, context=context, width=width)
    report = build_qualification(**values, artifact_root=tmp_path)
    assert report["performance"][0]["baseline_width"] == baseline
    assert report["performance"][0]["baseline_invocations"] == invocations
    assert report["performance"][0]["speedup"] == 1.25
    if width == 128:
        assert [item["baseline_width"] for item in report["performance"]] == [
            32, 16]


def test_s128_requires_both_s32_and_eight_s16_comparisons(
        tmp_path: Path) -> None:
    values = _values(tmp_path, width=128)
    values["performance"] = values["performance"][:1]
    with pytest.raises(PrefillBlockQualificationError, match=r"\[32, 16\]"):
        build_qualification(**values, artifact_root=tmp_path)


def test_real_om_and_build_manifest_bytes_and_hash_are_verified(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["om"]["sha256"] = "a" * 64
    with pytest.raises(PrefillBlockQualificationError, match="real file"):
        build_qualification(**values, artifact_root=tmp_path)

    values = _values(tmp_path)
    (tmp_path / "build-manifest.json").write_text("not-json", encoding="utf-8")
    payload = (tmp_path / "build-manifest.json").read_bytes()
    values["build_manifest"]["bytes"] = len(payload)
    values["build_manifest"]["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(PrefillBlockQualificationError, match="valid JSON"):
        build_qualification(**values, artifact_root=tmp_path)


def test_performance_must_bind_the_same_real_om(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["performance"][0]["candidate_om_sha256"] = "e" * 64
    with pytest.raises(PrefillBlockQualificationError, match="does not match"):
        build_qualification(**values, artifact_root=tmp_path)


def test_portable_evidence_verifies_files_and_writes_pass(tmp_path: Path) -> None:
    values = _values(tmp_path)
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "nested" / "qualification.json"
    evidence.write_text(json.dumps(values), encoding="utf-8")

    report = qualify_file(evidence=evidence, output=output)

    assert report["status"] == "PASS"
    assert report["activation"]["phase"] == "steady"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_portable_evidence_rejects_unbound_extra_fields(tmp_path: Path) -> None:
    values = _values(tmp_path)
    values["threshold_exclusive"] = 0.1
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(PrefillBlockQualificationError, match="fields must be exactly"):
        qualify_file(evidence=evidence, output=tmp_path / "qualification.json")


def _real_artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json_artifact(path: Path, value: object) -> dict[str, object]:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _real_artifact(path)


def test_token_id_hash_contract_is_exact_uint32_little_endian() -> None:
    token_ids = (0, 1, 0x12345678, 0xFFFFFFFF)
    expected = hashlib.sha256(b"".join(
        struct.pack("<I", token_id) for token_id in token_ids)).hexdigest()
    assert token_id_sequence_sha256(token_ids) == expected
    for bad in ("1,2", [True], [-1], [0x1_0000_0000]):
        with pytest.raises(PrefillBlockQualificationError, match="uint32"):
            token_id_sequence_sha256(bad)


def _strict_s1_release_values(
    root: Path, *, context: int = 1024, stem: str = "strict-s1",
) -> dict[str, object]:
    bootstrap_model = root / f"{stem}-bootstrap.om"
    canonical_model = root / f"{stem}-decode.om"
    head_model = root / "release-head.om"
    embedding = root / "release-embedding.bin"
    build = root / f"{stem}-build.json"
    runner = root / f"{stem}-runner.py"
    executor = root / f"{stem}-executor"
    bootstrap_descriptor = root / f"{stem}-bootstrap-descriptor.bin"
    canonical_descriptor = root / f"{stem}-canonical-descriptor.bin"
    bootstrap_model.write_bytes(f"{stem}-bootstrap-model".encode())
    canonical_model.write_bytes(f"{stem}-decode-model".encode())
    head_model.write_bytes(b"release-head-model")
    embedding.write_bytes(b"release-embedding")
    build.write_text(json.dumps({"family": "strict-s1"}), encoding="utf-8")
    runner.write_text("# strict-S1 release runner\n", encoding="utf-8")
    executor.write_bytes(b"strict-S1 release executor")
    bootstrap_descriptor.write_bytes(
        f"{stem}-bootstrap-descriptor".encode())
    canonical_descriptor.write_bytes(
        f"{stem}-canonical-descriptor".encode())
    bootstrap_model_hash = _real_artifact(bootstrap_model)["sha256"]
    canonical_model_hash = _real_artifact(canonical_model)["sha256"]
    head_model_hash = _real_artifact(head_model)["sha256"]
    embedding_hash = _real_artifact(embedding)["sha256"]
    runner_hash = _real_artifact(runner)["sha256"]
    executor_hash = _real_artifact(executor)["sha256"]
    bootstrap_descriptor_hash = _real_artifact(
        bootstrap_descriptor)["sha256"]
    canonical_descriptor_hash = _real_artifact(
        canonical_descriptor)["sha256"]

    captures = []
    for position in required_strict_s1_positions(context):
        model_hash = (
            bootstrap_model_hash if position == 0 else canonical_model_hash)
        descriptor_hash = (
            bootstrap_descriptor_hash
            if position == 0 else canonical_descriptor_hash)
        captures.append(_json_artifact(
            root / f"{stem}-capture-{position}.json", {
                "schema": STRICT_S1_CAPTURE_SCHEMA,
                "board": "Hi3403",
                "context": context,
                "width": 1,
                "model_sha256": model_hash,
                "runner_sha256": runner_hash,
                "executor_sha256": executor_hash,
                "ready_descriptor_sha256": descriptor_hash,
                "position": position,
                "physical_descriptor_sha256": descriptor_hash,
                "publisher_source_sha256": {
                    "k": f"{700 + position:064x}",
                    "v": f"{800 + position:064x}"},
                "publisher_source_dtype": "FP32",
                "publisher_source_layout": "contiguous-channel-major",
                "descriptor_exact": True,
                "mask_sha256": f"{900 + position:064x}",
                "mask_bytes_exact": True,
                "rope_sha256": f"{1000 + position:064x}",
                "rope_bytes_exact": True,
                "kv_rows": {"k": 48, "v": 48},
                "kv_rows_exact": True,
                "route_handoff_exact": True,
                "token_exact": True,
                "board_pass": True,
                "public_cosines": {"hidden": .999, "k": .999, "v": .999},
            }))
    workloads = []
    for index, kind in enumerate((
            "tokens_48", "eos", "english", "chinese", "context_boundary")):
        value = {
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
            "kind": kind,
            "capture_sha256": f"{1100 + index:064x}",
            "prompt_sha256": f"{1200 + index:064x}",
            "output_tokens_sha256": f"{1300 + index:064x}",
            "generated_tokens": 48 if kind == "tokens_48" else 8,
            "board_pass": True,
            "token_exact": True,
        }
        if kind == "tokens_48":
            value["sequence_exact"] = True
        if kind == "eos":
            value["eos_exact"] = True
        if kind == "context_boundary":
            value["boundary_exact"] = True
            value["terminal_position"] = context - 1
        workloads.append(_json_artifact(
            root / f"{stem}-workload-{kind}.json", value))
    mmz = _json_artifact(root / f"{stem}-mmz.json", {
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
        "bootstrap_ready_descriptor_sha256": bootstrap_descriptor_hash,
        "canonical_ready_descriptor_sha256": canonical_descriptor_hash,
        "before_available_bytes": 2_000_000,
        "after_available_bytes": 1_900_000,
    })
    return {
        "context": context,
        "bootstrap_om": _real_artifact(bootstrap_model),
        "canonical_decode_om": _real_artifact(canonical_model),
        "head_om": _real_artifact(head_model),
        "embedding": _real_artifact(embedding),
        "build_manifest": _real_artifact(build),
        "runner": _real_artifact(runner),
        "executor": _real_artifact(executor),
        "bootstrap_ready_descriptor": _real_artifact(bootstrap_descriptor),
        "canonical_ready_descriptor": _real_artifact(canonical_descriptor),
        "capture_artifacts": captures,
        "workload_artifacts": workloads,
        "mmz_observation": mmz,
    }


def _release_values(
    root: Path, *, context: int = 1024, width: int = 16,
    strict_s1_baseline: tuple[
        dict[str, object], dict[str, object]] | None = None,
) -> dict[str, object]:
    model = root / f"release-s{width}.om"
    head_model = root / "release-head.om"
    embedding = root / "release-embedding.bin"
    build = root / f"release-s{width}-build.json"
    runner = root / "release-runner.py"
    executor = root / "release-executor"
    descriptor = root / f"release-s{width}-descriptor.bin"
    model.write_bytes(f"release-model-{width}".encode())
    head_model.write_bytes(b"release-head-model")
    embedding.write_bytes(b"release-embedding")
    build.write_text(json.dumps({"width": width}), encoding="utf-8")
    runner.write_text("# release runner\n", encoding="utf-8")
    executor.write_bytes(b"release executor")
    descriptor.write_bytes(f"descriptor-{width}".encode())
    model_hash = _real_artifact(model)["sha256"]
    head_model_hash = _real_artifact(head_model)["sha256"]
    embedding_hash = _real_artifact(embedding)["sha256"]
    runner_hash = _real_artifact(runner)["sha256"]
    executor_hash = _real_artifact(executor)["sha256"]
    descriptor_hash = _real_artifact(descriptor)["sha256"]

    captures = []
    for start in required_starts(context, width):
        record = {
            "schema": CAPTURE_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": width,
            "model_sha256": model_hash,
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
            "ready_descriptor_sha256": descriptor_hash,
            "start": start,
            "physical_descriptor_sha256": descriptor_hash,
            "publisher_source_sha256": {
                "k": f"{start + 2:064x}", "v": f"{start + 3:064x}"},
            "publisher_source_dtype": "FP32",
            "publisher_source_layout": "contiguous-channel-major",
            "descriptor_exact": True,
            "mask_sha256": canonical_mask_sha256(
                context, width, start),
            "mask_bytes_exact": True,
            "rope_sha256": f"{start + 5:064x}",
            "rope_bytes_exact": True,
            "kv_rows": {"k": 48 * width, "v": 48 * width},
            "kv_rows_exact": True,
            "prefill_decode_handoff": True,
            "token_exact": True,
            "board_pass": True,
            "public_cosines": {"hidden": .999, "k": .999, "v": .999},
            "handoff_cosines": {"hidden": .999, "k": .999, "v": .999},
        }
        captures.append(_json_artifact(
            root / f"release-s{width}-capture-{start}.json", record))

    workloads = []
    for index, kind in enumerate((
            "eos", "english", "chinese", "context_boundary")):
        record = {
            "schema": WORKLOAD_SCHEMA,
            "board": "Hi3403",
            "context": context,
            "width": width,
            "model_sha256": model_hash,
            "head_om_sha256": head_model_hash,
            "embedding_sha256": embedding_hash,
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
            "token_id_hash_contract": TOKEN_ID_HASH_CONTRACT,
            "kind": kind,
            "capture_sha256": f"{500 + index:064x}",
            "prompt_sha256": f"{510 + index:064x}",
            "output_tokens_sha256": f"{520 + index:064x}",
            "board_pass": True,
            "token_exact": True,
        }
        if kind == "eos":
            record["eos_exact"] = True
        if kind == "context_boundary":
            record["boundary_exact"] = True
            record["terminal_position"] = context - 1
        workloads.append(_json_artifact(
            root / f"release-s{width}-workload-{kind}.json", record))

    if strict_s1_baseline is None:
        s1_values = _strict_s1_release_values(
            root, context=context, stem=f"release-s{width}-strict-s1")
        s1_report = build_strict_s1_release_qualification(
            **s1_values, artifact_root=root)
        s1_report_path = root / (
            f"release-s{width}-strict-s1-baseline-q.json")
        strict_s1_baseline = (
            _json_artifact(s1_report_path, s1_report),
            s1_values["canonical_decode_om"])

    baseline_widths = {16: (1,), 32: (16,), 128: (32, 16)}[width]
    baselines = []
    baseline_hashes: dict[int, tuple[str, str]] = {}
    for baseline_width in baseline_widths:
        if baseline_width == 1:
            baseline_q_artifact, baseline_om_artifact = strict_s1_baseline
        else:
            nested_values = _release_values(
                root, context=context, width=baseline_width,
                strict_s1_baseline=strict_s1_baseline)
            nested_report = build_release_qualification(
                **nested_values, artifact_root=root)
            baseline_report = root / (
                f"release-s{width}-baseline-s{baseline_width}-q.json")
            baseline_q_artifact = _json_artifact(
                baseline_report, nested_report)
            baseline_om_artifact = nested_values["om"]
        baselines.append({
            "width": baseline_width,
            "qualification": baseline_q_artifact,
            "om": baseline_om_artifact,
        })
        baseline_hashes[baseline_width] = (
            str(baseline_q_artifact["sha256"]),
            str(baseline_om_artifact["sha256"]))

    performance = []
    for baseline_width in baseline_widths:
        baseline_q_hash, baseline_om_hash = baseline_hashes[baseline_width]
        measurement = {
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
            "warmup_runs": 2,
            "measured_runs": 3,
            "candidate_warmup_ms": [8.2, 8.1],
            "baseline_warmup_ms": [10.2, 10.1],
            "candidate_samples_ms": [8.0, 8.0, 8.0],
            "baseline_samples_ms": [10.0, 10.0, 10.0],
            "candidate_om_sha256": model_hash,
            "baseline_qualification_sha256": baseline_q_hash,
            "baseline_om_sha256": baseline_om_hash,
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
        }
        performance.append(_json_artifact(
            root / f"release-s{width}-perf-vs-s{baseline_width}.json",
            measurement))

    mmz = {
        "schema": MMZ_SCHEMA,
        "status": "PASS",
        "board": "Hi3403",
        "clean_board": True,
        "context": context,
        "width": width,
        "model_sha256": model_hash,
        "runner_sha256": runner_hash,
        "executor_sha256": executor_hash,
        "ready_descriptor_sha256": descriptor_hash,
        "before_available_bytes": 1_000_000,
        "after_available_bytes": 900_000,
    }
    return {
        "context": context,
        "width": width,
        "om": _real_artifact(model),
        "head_om": _real_artifact(head_model),
        "embedding": _real_artifact(embedding),
        "build_manifest": _real_artifact(build),
        "runner": _real_artifact(runner),
        "executor": _real_artifact(executor),
        "ready_descriptor": _real_artifact(descriptor),
        "capture_artifacts": captures,
        "workload_artifacts": workloads,
        "performance_artifacts": performance,
        "baselines": baselines,
        "mmz_observation": _json_artifact(
            root / f"release-s{width}-mmz.json", mmz),
    }


def test_release_v4_binds_real_evidence_runtime_baseline_and_mmz(
        tmp_path: Path) -> None:
    values = _release_values(tmp_path)
    report = build_release_qualification(**values, artifact_root=tmp_path)

    assert report["schema"] == RELEASE_SCHEMA
    assert report["release_eligible"] is True
    assert report["runtime"]["runner"]["file_verified"] is True
    assert report["captures"][0]["capture_sha256"] == \
        report["captures"][0]["artifact"]["sha256"]
    assert report["performance"][0]["candidate_ms"] == 8.0
    assert report["performance"][0]["baseline_ms"] == 10.0
    assert report["performance"][0]["speedup"] == 1.25
    assert report["mmz_admission"]["admission_bytes"] == 100_000
    assert report["activation"]["admission_bytes"] == 100_000


def test_strict_s1_release_baseline_is_content_bound(tmp_path: Path) -> None:
    values = _strict_s1_release_values(tmp_path)
    report = build_strict_s1_release_qualification(
        **values, artifact_root=tmp_path)

    assert report["schema"] == STRICT_S1_BASELINE_SCHEMA
    assert report["required_positions"] == list(
        required_strict_s1_positions(1024))
    assert [item["kind"] for item in report["workloads"]] == [
        "tokens_48", "eos", "english", "chinese", "context_boundary"]
    assert report["mmz_residency"]["role"] == "base-resident"
    assert report["mmz_residency"]["accounting"] == \
        "included-in-base_resident_bytes"
    assert report["mmz_residency"]["resident_bytes"] == 100_000
    assert report["captures"][0]["physical_descriptor_sha256"] == \
        report["runtime"]["bootstrap_ready_descriptor"]["sha256"]
    assert report["captures"][1]["physical_descriptor_sha256"] == \
        report["runtime"]["canonical_ready_descriptor"]["sha256"]
    assert report["baseline"]["bootstrap_om_sha256"] == \
        report["bootstrap_om"]["sha256"]
    assert report["baseline"]["canonical_decode_om_sha256"] == \
        report["canonical_decode_om"]["sha256"]
    assert verify_strict_s1_release_qualification(
        report, artifact_root=tmp_path) == report


def test_strict_s1_release_rejects_swapped_route_artifacts(
        tmp_path: Path) -> None:
    values = _strict_s1_release_values(tmp_path)
    values["bootstrap_om"], values["canonical_decode_om"] = (
        values["canonical_decode_om"], values["bootstrap_om"])

    with pytest.raises(
            PrefillBlockQualificationError,
            match="physical ABI/descriptor mismatch"):
        build_strict_s1_release_qualification(
            **values, artifact_root=tmp_path)


def test_forged_six_field_strict_s1_pass_is_not_a_release_baseline(
        tmp_path: Path) -> None:
    values = _release_values(tmp_path)
    baseline = values["baselines"][0]
    baseline_q = tmp_path / str(baseline["qualification"]["path"])
    forged = {
        "schema": STRICT_S1_DEVELOPMENT_SCHEMA,
        "status": "PASS",
        "release_eligible": True,
        "context": 1024,
        "width": 1,
        "om_sha256": baseline["om"]["sha256"],
    }
    baseline["qualification"] = _json_artifact(baseline_q, forged)
    measurement = tmp_path / str(values["performance_artifacts"][0]["path"])
    payload = json.loads(measurement.read_text())
    payload["baseline_qualification_sha256"] = \
        baseline["qualification"]["sha256"]
    values["performance_artifacts"][0] = _json_artifact(measurement, payload)

    with pytest.raises(
            PrefillBlockQualificationError, match="development strict-S1"):
        build_release_qualification(**values, artifact_root=tmp_path)


def test_strict_s1_release_rejects_evidence_and_mmz_tamper(
        tmp_path: Path) -> None:
    values = _strict_s1_release_values(tmp_path)
    capture = tmp_path / str(values["capture_artifacts"][0]["path"])
    capture.write_text("{}", encoding="utf-8")
    with pytest.raises(PrefillBlockQualificationError, match="bytes|sha256"):
        build_strict_s1_release_qualification(
            **values, artifact_root=tmp_path)

    values = _strict_s1_release_values(tmp_path)
    workload = tmp_path / str(values["workload_artifacts"][0]["path"])
    payload = json.loads(workload.read_text())
    payload["generated_tokens"] = 47
    values["workload_artifacts"][0] = _json_artifact(workload, payload)
    with pytest.raises(PrefillBlockQualificationError, match="exactly 48"):
        build_strict_s1_release_qualification(
            **values, artifact_root=tmp_path)

    values = _strict_s1_release_values(tmp_path)
    observation = tmp_path / str(values["mmz_observation"]["path"])
    payload = json.loads(observation.read_text())
    payload["accounting"] = "not-counted"
    values["mmz_observation"] = _json_artifact(observation, payload)
    with pytest.raises(PrefillBlockQualificationError, match="accounting"):
        build_strict_s1_release_qualification(
            **values, artifact_root=tmp_path)


@pytest.mark.parametrize("field", [
    "head_om_sha256", "embedding_sha256", "token_id_hash_contract",
])
def test_strict_s1_token_evidence_binds_inference_artifacts_and_id_contract(
        tmp_path: Path, field: str) -> None:
    values = _strict_s1_release_values(tmp_path)
    workload = tmp_path / str(values["workload_artifacts"][0]["path"])
    payload = json.loads(workload.read_text())
    payload[field] = "0" * 64 if field.endswith("sha256") else "tokenizer-text"
    values["workload_artifacts"][0] = _json_artifact(workload, payload)

    with pytest.raises(PrefillBlockQualificationError, match=field):
        build_strict_s1_release_qualification(
            **values, artifact_root=tmp_path)


@pytest.mark.parametrize("field", [
    "head_om_sha256", "embedding_sha256", "token_id_hash_contract",
])
def test_wide_token_evidence_binds_inference_artifacts_and_id_contract(
        tmp_path: Path, field: str) -> None:
    values = _release_values(tmp_path)
    workload = tmp_path / str(values["workload_artifacts"][0]["path"])
    payload = json.loads(workload.read_text())
    payload[field] = "0" * 64 if field.endswith("sha256") else "tokenizer-text"
    values["workload_artifacts"][0] = _json_artifact(workload, payload)

    with pytest.raises(PrefillBlockQualificationError, match=field):
        build_release_qualification(**values, artifact_root=tmp_path)


@pytest.mark.parametrize("artifact_field,workload_field", [
    ("head_om", "head_om_sha256"),
    ("embedding", "embedding_sha256"),
])
def test_wide_candidate_cannot_change_strict_s1_token_artifact_identity(
        tmp_path: Path, artifact_field: str, workload_field: str) -> None:
    values = _release_values(tmp_path)
    alternate = tmp_path / f"alternate-{artifact_field}.bin"
    alternate.write_bytes(f"alternate-{artifact_field}".encode())
    values[artifact_field] = _real_artifact(alternate)
    for index, descriptor in enumerate(values["workload_artifacts"]):
        workload = tmp_path / str(descriptor["path"])
        payload = json.loads(workload.read_text())
        payload[workload_field] = values[artifact_field]["sha256"]
        values["workload_artifacts"][index] = _json_artifact(workload, payload)

    with pytest.raises(
            PrefillBlockQualificationError,
            match="head/embedding identity"):
        build_release_qualification(**values, artifact_root=tmp_path)


def test_portable_strict_s1_release_evidence_writes_pass(
        tmp_path: Path) -> None:
    values = _strict_s1_release_values(tmp_path)
    evidence = tmp_path / "strict-s1-release-evidence.json"
    output = tmp_path / "strict-s1-release-qualification.json"
    evidence.write_text(json.dumps(values), encoding="utf-8")

    report = qualify_strict_s1_release_file(evidence=evidence, output=output)

    assert report["schema"] == STRICT_S1_BASELINE_SCHEMA
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("width,expected", [
    (32, [16]),
    (128, [32, 16]),
])
def test_release_v4_transitively_binds_the_full_baseline_ladder(
        tmp_path: Path, width: int, expected: list[int]) -> None:
    values = _release_values(tmp_path, width=width)
    report = build_release_qualification(**values, artifact_root=tmp_path)

    assert [item["width"] for item in report["baselines"]] == expected
    assert [item["baseline_width"] for item in report["performance"]] == expected


def test_s128_builder_and_verifier_reject_mixed_strict_s1_anchors(
        tmp_path: Path) -> None:
    values = _release_values(tmp_path, width=128)
    report = build_release_qualification(**values, artifact_root=tmp_path)
    alternate_s1_values = _strict_s1_release_values(
        tmp_path, stem="alternate-strict-s1")
    alternate_s1_report = build_strict_s1_release_qualification(
        **alternate_s1_values, artifact_root=tmp_path)
    alternate_s1 = (
        _json_artifact(
            tmp_path / "alternate-strict-s1-q.json", alternate_s1_report),
        alternate_s1_values["canonical_decode_om"],
    )
    original_s16 = next(
        item for item in values["baselines"] if item["width"] == 16)
    original_s16_path = tmp_path / str(
        original_s16["qualification"]["path"])
    alternate_s16_report = json.loads(original_s16_path.read_text())
    alternate_s16_report["baselines"][0]["qualification"] = {
        **alternate_s1[0], "file_verified": True}
    alternate_s16_report["baselines"][0]["om"] = {
        **alternate_s1[1], "file_verified": True}
    alternate_s16_performance = alternate_s16_report["performance"][0]
    alternate_s16_performance["baseline_qualification_sha256"] = \
        alternate_s1[0]["sha256"]
    alternate_s16_performance["baseline_om_sha256"] = \
        alternate_s1[1]["sha256"]
    alternate_s16_measurement_payload = {
        key: value for key, value in alternate_s16_performance.items()
        if key not in {
            "candidate_ms", "baseline_ms", "speedup",
            "measurement_artifact"}
    }
    alternate_s16_measurement_payload.update({
        "schema": PERFORMANCE_SCHEMA, "context": 1024, "width": 16})
    alternate_s16_measurement = _json_artifact(
        tmp_path / "alternate-s16-vs-s1-performance.json",
        alternate_s16_measurement_payload)
    alternate_s16_performance["measurement_artifact"] = {
        **alternate_s16_measurement, "file_verified": True}
    alternate_s16_q = _json_artifact(
        tmp_path / "alternate-s16-q.json", alternate_s16_report)

    mixed_values = copy.deepcopy(values)
    s16_baseline = next(
        item for item in mixed_values["baselines"] if item["width"] == 16)
    s16_baseline["qualification"] = alternate_s16_q
    for index, descriptor in enumerate(mixed_values["performance_artifacts"]):
        path = tmp_path / str(descriptor["path"])
        payload = json.loads(path.read_text())
        if payload["baseline_width"] == 16:
            payload["baseline_qualification_sha256"] = \
                alternate_s16_q["sha256"]
            mixed_values["performance_artifacts"][index] = _json_artifact(
                tmp_path / "mixed-s128-vs-s16-performance-builder.json",
                payload)
    with pytest.raises(
            PrefillBlockQualificationError, match="one strict-S1 identity"):
        build_release_qualification(**mixed_values, artifact_root=tmp_path)

    report_s16 = next(
        item for item in report["baselines"] if item["width"] == 16)
    report_s16["qualification"] = {
        **alternate_s16_q, "file_verified": True}
    performance = next(
        item for item in report["performance"]
        if item["baseline_width"] == 16)
    performance["baseline_qualification_sha256"] = \
        alternate_s16_q["sha256"]
    measurement_path = tmp_path / "mixed-s128-vs-s16-performance.json"
    measurement_payload = {
        key: value for key, value in performance.items()
        if key not in {
            "candidate_ms", "baseline_ms", "speedup",
            "measurement_artifact"}
    }
    measurement_payload.update({
        "schema": PERFORMANCE_SCHEMA, "context": 1024, "width": 128})
    measurement = _json_artifact(measurement_path, measurement_payload)
    performance["measurement_artifact"] = {
        **measurement, "file_verified": True}

    with pytest.raises(
            PrefillBlockQualificationError, match="one strict-S1 identity"):
        verify_release_qualification(report, artifact_root=tmp_path)


def test_release_v4_rejects_tampered_capture_and_baseline_lineage(
        tmp_path: Path) -> None:
    values = _release_values(tmp_path)
    capture = tmp_path / str(values["capture_artifacts"][0]["path"])
    capture.write_text("{}", encoding="utf-8")
    with pytest.raises(PrefillBlockQualificationError, match="bytes|sha256"):
        build_release_qualification(**values, artifact_root=tmp_path)

    values = _release_values(tmp_path)
    capture = tmp_path / str(values["capture_artifacts"][0]["path"])
    capture_value = json.loads(capture.read_text())
    capture_value["physical_descriptor_sha256"] = "0" * 64
    values["capture_artifacts"][0] = _json_artifact(capture, capture_value)
    with pytest.raises(PrefillBlockQualificationError, match="ready descriptor"):
        build_release_qualification(**values, artifact_root=tmp_path)

    values = _release_values(tmp_path)
    baseline = tmp_path / str(values["baselines"][0]["qualification"]["path"])
    baseline_value = json.loads(baseline.read_text())
    baseline_value["canonical_decode_om"]["sha256"] = "0" * 64
    values["baselines"][0]["qualification"] = _json_artifact(
        baseline, baseline_value)
    with pytest.raises(
            PrefillBlockQualificationError,
            match="strict_s1.canonical_decode_om.sha256"):
        build_release_qualification(**values, artifact_root=tmp_path)


def test_release_v4_derives_admission_and_rejects_non_clean_delta(
        tmp_path: Path) -> None:
    values = _release_values(tmp_path)
    observation = tmp_path / str(values["mmz_observation"]["path"])
    payload = json.loads(observation.read_text())
    payload["clean_board"] = False
    values["mmz_observation"] = _json_artifact(observation, payload)
    with pytest.raises(PrefillBlockQualificationError, match="clean_board"):
        build_release_qualification(**values, artifact_root=tmp_path)

    values = _release_values(tmp_path)
    observation = tmp_path / str(values["mmz_observation"]["path"])
    payload = json.loads(observation.read_text())
    payload["after_available_bytes"] = payload["before_available_bytes"]
    values["mmz_observation"] = _json_artifact(observation, payload)
    with pytest.raises(PrefillBlockQualificationError, match="positive"):
        build_release_qualification(**values, artifact_root=tmp_path)


def test_release_v4_performance_is_recomputed_from_real_samples(
        tmp_path: Path) -> None:
    values = _release_values(tmp_path)
    measurement = tmp_path / str(values["performance_artifacts"][0]["path"])
    payload = json.loads(measurement.read_text())
    payload["candidate_samples_ms"] = [12.0, 12.0, 12.0]
    values["performance_artifacts"][0] = _json_artifact(measurement, payload)
    with pytest.raises(PrefillBlockQualificationError, match="strictly faster"):
        build_release_qualification(**values, artifact_root=tmp_path)


def test_v2_pass_remains_development_only(tmp_path: Path) -> None:
    report = _qualified(tmp_path)
    assert report["schema"] == SCHEMA
    assert "release_eligible" not in report


def test_portable_release_evidence_writes_v3_pass(tmp_path: Path) -> None:
    values = _release_values(tmp_path)
    evidence = tmp_path / "release-evidence.json"
    output = tmp_path / "release-qualification.json"
    evidence.write_text(json.dumps(values), encoding="utf-8")

    report = qualify_release_file(evidence=evidence, output=output)

    assert report["schema"] == RELEASE_SCHEMA
    assert json.loads(output.read_text()) == report
