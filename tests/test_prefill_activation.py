from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pico_minicpm5.prefill_blocks import (
    CAPTURE_SCHEMA,
    MMZ_SCHEMA,
    PERFORMANCE_SCHEMA,
    RELEASE_SCHEMA,
    STRICT_S1_BASELINE_SCHEMA,
    STRICT_S1_CAPTURE_SCHEMA,
    STRICT_S1_DEVELOPMENT_SCHEMA,
    STRICT_S1_MMZ_SCHEMA,
    STRICT_S1_WORKLOAD_SCHEMA,
    TOKEN_ID_HASH_CONTRACT,
    WORKLOAD_SCHEMA,
    build_release_qualification,
    build_strict_s1_release_qualification,
    canonical_mask_sha256,
    required_starts,
    required_strict_s1_positions,
)


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"


def _module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_prefill_activation_test",
        APP_SRC / "minicpm_prefill_activation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _json_artifact(path: Path, value: object) -> dict[str, object]:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _artifact(path)


def _strict_s1_baseline(
    root: Path, *, stem: str, context: int = 1024, resident: int = 100,
) -> tuple[dict[str, object], dict[str, object]]:
    bootstrap_model = root / f"{stem}-bootstrap.om"
    canonical_model = root / f"{stem}-decode.om"
    head_model = root / "live-head.om"
    embedding = root / "live-embedding.bin"
    build = root / f"{stem}-build.json"
    runner = root / f"{stem}-runner.py"
    executor = root / f"{stem}-executor"
    bootstrap_descriptor = root / f"{stem}-bootstrap-descriptor.bin"
    canonical_descriptor = root / f"{stem}-canonical-descriptor.bin"
    bootstrap_model.write_bytes(f"{stem}-bootstrap-model".encode())
    canonical_model.write_bytes(f"{stem}-decode-model".encode())
    head_model.write_bytes(b"live-head-model")
    embedding.write_bytes(b"live-embedding")
    build.write_text(json.dumps({"family": "strict-s1"}), encoding="utf-8")
    runner.write_text("# strict-S1 runner\n", encoding="utf-8")
    executor.write_bytes(b"strict-S1 executor")
    bootstrap_descriptor.write_bytes(
        f"{stem}-bootstrap-descriptor".encode())
    canonical_descriptor.write_bytes(
        f"{stem}-canonical-descriptor".encode())
    bootstrap_model_hash = _sha(bootstrap_model)
    canonical_model_hash = _sha(canonical_model)
    head_model_hash = _sha(head_model)
    embedding_hash = _sha(embedding)
    runner_hash = _sha(runner)
    executor_hash = _sha(executor)
    bootstrap_descriptor_hash = _sha(bootstrap_descriptor)
    canonical_descriptor_hash = _sha(canonical_descriptor)
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
        "after_available_bytes": 2_000_000 - resident,
    })
    report = build_strict_s1_release_qualification(
        context=context, bootstrap_om=_artifact(bootstrap_model),
        canonical_decode_om=_artifact(canonical_model),
        head_om=_artifact(head_model), embedding=_artifact(embedding),
        build_manifest=_artifact(build),
        runner=_artifact(runner), executor=_artifact(executor),
        bootstrap_ready_descriptor=_artifact(bootstrap_descriptor),
        canonical_ready_descriptor=_artifact(canonical_descriptor),
        capture_artifacts=captures,
        workload_artifacts=workloads, mmz_observation=mmz,
        artifact_root=root)
    qualification = root / f"{stem}-qualification.json"
    return _json_artifact(qualification, report), _artifact(canonical_model)


def _block(root: Path, width: int, *, group: str | None = None,
           admission: int = 100, namespace: str | None = None,
           s1_stem: str = "live-strict-s1",
           s1_resident: int = 100,
           runtime_drift: bool = False) -> dict[str, object]:
    stem = namespace or f"s{width}"
    model = root / f"{stem}.om"
    head_model = root / "live-head.om"
    embedding = root / "live-embedding.bin"
    model.write_bytes(f"model-{width}".encode())
    head_model.write_bytes(b"live-head-model")
    embedding.write_bytes(b"live-embedding")
    build = root / f"{stem}-build.json"
    build.write_text(json.dumps({"width": width}), encoding="utf-8")
    model_hash = _sha(model)
    head_model_hash = _sha(head_model)
    embedding_hash = _sha(embedding)
    descriptor = root / f"{stem}-ready-descriptor.txt"
    descriptor.write_text(f"descriptor-{width}\n", encoding="utf-8")
    descriptor_hash = _sha(descriptor)
    runner = root / f"{stem}-runner.py"
    executor = root / f"{stem}-executor"
    runner.write_text(
        "# candidate-only runner\n" if runtime_drift
        else "# strict-S1 runner\n",
        encoding="utf-8")
    executor.write_bytes(
        b"candidate-only executor" if runtime_drift
        else b"strict-S1 executor")
    runner_hash = _sha(runner)
    executor_hash = _sha(executor)
    captures = []
    for start in required_starts(1024, width):
        captures.append(_json_artifact(
            root / f"{stem}-capture-{start}.json", {
                "schema": CAPTURE_SCHEMA,
                "board": "Hi3403",
                "context": 1024,
                "width": width,
                "model_sha256": model_hash,
                "runner_sha256": runner_hash,
                "executor_sha256": executor_hash,
                "ready_descriptor_sha256": descriptor_hash,
                "start": start,
                "physical_descriptor_sha256": descriptor_hash,
                "publisher_source_sha256": {
                    "k": f"{start + 2:064x}",
                    "v": f"{start + 3:064x}"},
                "publisher_source_dtype": "FP32",
                "publisher_source_layout": "contiguous-channel-major",
                "descriptor_exact": True,
                "mask_sha256": canonical_mask_sha256(
                    1024, width, start),
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
            }))
    workloads = []
    for index, kind in enumerate((
            "eos", "english", "chinese", "context_boundary")):
        workload = {
            "schema": WORKLOAD_SCHEMA,
            "board": "Hi3403",
            "context": 1024,
            "width": width,
            "model_sha256": model_hash,
            "head_om_sha256": head_model_hash,
            "embedding_sha256": embedding_hash,
            "runner_sha256": runner_hash,
            "executor_sha256": executor_hash,
            "token_id_hash_contract": TOKEN_ID_HASH_CONTRACT,
            "kind": kind,
            "capture_sha256": f"{100 + index:064x}",
            "prompt_sha256": f"{110 + index:064x}",
            "output_tokens_sha256": f"{120 + index:064x}",
            "board_pass": True,
            "token_exact": True,
        }
        if kind == "eos":
            workload["eos_exact"] = True
        if kind == "context_boundary":
            workload["boundary_exact"] = True
            workload["terminal_position"] = 1023
        workloads.append(_json_artifact(
            root / f"{stem}-workload-{kind}.json", workload))
    baselines = {16: (1,), 32: (16,), 128: (32, 16)}[width]
    baseline_records = []
    baseline_hashes = {}
    for baseline in baselines:
        if baseline == 1:
            baseline_q, baseline_om_artifact = _strict_s1_baseline(
                root, stem=s1_stem, resident=s1_resident)
        else:
            nested = _block(
                root, baseline, namespace=f"{stem}-baseline-s{baseline}",
                s1_stem=s1_stem, s1_resident=s1_resident)
            baseline_om_artifact = _artifact(
                root / str(nested["model"]))
            baseline_q = _artifact(root / str(nested["qualification"]))
        baseline_records.append({
            "width": baseline,
            "qualification": baseline_q,
            "om": baseline_om_artifact,
        })
        baseline_hashes[baseline] = (
            baseline_q["sha256"], baseline_om_artifact["sha256"])
    performance = []
    for baseline in baselines:
        baseline_q_hash, baseline_om_hash = baseline_hashes[baseline]
        performance.append(_json_artifact(
            root / f"{stem}-performance-vs-s{baseline}.json", {
                "schema": PERFORMANCE_SCHEMA,
                "board": "Hi3403",
                "context": 1024,
                "width": width,
                "metric": "board-wall-time-ms",
                "tokens": width,
                "candidate_width": width,
                "baseline_width": baseline,
                "candidate_invocations": 1,
                "baseline_invocations": width // baseline,
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
            }))
    before = 1_000_000
    after = before - admission
    admission_report = root / f"{stem}-admission.json"
    admission_artifact = _json_artifact(admission_report, {
        "schema": MMZ_SCHEMA,
        "status": "PASS",
        "board": "Hi3403",
        "clean_board": True,
        "context": 1024,
        "width": width,
        "model_sha256": model_hash,
        "runner_sha256": runner_hash,
        "executor_sha256": executor_hash,
        "ready_descriptor_sha256": descriptor_hash,
        "before_available_bytes": before,
        "after_available_bytes": after,
    })
    report = build_release_qualification(
        context=1024, width=width,
        om=_artifact(model), head_om=_artifact(head_model),
        embedding=_artifact(embedding), build_manifest=_artifact(build),
        runner=_artifact(runner), executor=_artifact(executor),
        ready_descriptor=_artifact(descriptor),
        capture_artifacts=captures, workload_artifacts=workloads,
        performance_artifacts=performance, baselines=baseline_records,
        mmz_observation=admission_artifact,
        artifact_root=root)
    qualification = root / f"{stem}-qualification.json"
    qualification.write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8")
    return {
        "width": width,
        "phase": "steady",
        "model": model.name,
        "qualification": qualification.name,
        "qualification_sha256": _sha(qualification),
        "build_manifest": build.name,
        "ready_descriptor": descriptor.name,
        "runner": runner.name,
        "executor": executor.name,
        "residency_group": group or f"s{width}",
        "admission_bytes": admission,
        "admission_report": admission_report.name,
        "admission_report_sha256": admission_artifact["sha256"],
    }


def _live_strict_s1_descriptor(root: Path) -> dict[str, object]:
    qualification = root / "live-strict-s1-qualification.json"
    if not qualification.is_file():
        _strict_s1_baseline(root, stem="live-strict-s1")
    report = json.loads(qualification.read_text())
    return {
        "bootstrap_model": report["bootstrap_om"]["path"],
        "canonical_decode_model": report["canonical_decode_om"]["path"],
        "head_model": report["head_om"]["path"],
        "embedding": report["embedding"]["path"],
        "qualification": qualification.name,
        "qualification_sha256": _sha(qualification),
        "build_manifest": report["build_manifest"]["path"],
        "runner": report["runtime"]["runner"]["path"],
        "executor": report["runtime"]["executor"]["path"],
        "bootstrap_ready_descriptor": report["runtime"][
            "bootstrap_ready_descriptor"]["path"],
        "canonical_ready_descriptor": report["runtime"][
            "canonical_ready_descriptor"]["path"],
    }


def _manifest(root: Path, blocks: list[dict[str, object]]) -> Path:
    path = root / "activation.json"
    path.write_text(json.dumps({
        "schema": "pico.minicpm5.prefill-activation.v4",
        "context": 1024,
        "deployment_mode": "trusted-read-only-process-lifetime",
        "strict_s1": _live_strict_s1_descriptor(root),
        "blocks": blocks,
    }), encoding="utf-8")
    return path


def test_activation_verifies_files_and_uses_canonical_order(tmp_path: Path) -> None:
    module = _module()
    manifest = _manifest(tmp_path, [
        _block(tmp_path, 16), _block(tmp_path, 128), _block(tmp_path, 32),
    ])

    activation = module.load_activation(
        manifest, deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=100)

    assert activation.enabled_widths == (128, 32, 16, 1)
    assert [block.width for block in activation.blocks] == [128, 32, 16]
    assert activation.block_resident_bytes == 300
    assert activation.disabled == {}
    rendered = activation.to_dict()
    assert set(rendered["strict_s1"]) == {
        "bootstrap_model", "canonical_decode_model", "head_model",
        "embedding", "qualification",
        "qualification_sha256", "build_manifest", "runner", "executor",
        "bootstrap_ready_descriptor", "canonical_ready_descriptor"}
    assert rendered["strict_s1"] == json.loads(
        manifest.read_text())["strict_s1"]
    assert rendered["strict_s1"]["qualification_sha256"] == _sha(
        tmp_path / "live-strict-s1-qualification.json")
    assert rendered["strict_s1"]["bootstrap_model"].endswith(
        "live-strict-s1-bootstrap.om")
    assert rendered["strict_s1"]["canonical_decode_model"].endswith(
        "live-strict-s1-decode.om")
    assert rendered["memory"]["strict_s1_resident_bytes"] == 100
    assert rendered["memory"]["base_underreported"] is False
    assert rendered["memory"]["effective_base_resident_bytes"] == 100
    assert all(block.strict_s1_identity == activation.strict_s1.identity
               for block in activation.blocks)
    assert activation.to_dict()["blocks"][0][
        "ready_descriptor_sha256"] == _sha(
            tmp_path / "s128-ready-descriptor.txt")


def test_bad_hash_context_or_physical_abi_disables_only_that_width(
        tmp_path: Path) -> None:
    module = _module()
    s16 = _block(tmp_path, 16)
    s32 = _block(tmp_path, 32)
    s128 = _block(tmp_path, 128)
    s16["qualification_sha256"] = "0" * 64
    q32 = tmp_path / str(s32["qualification"])
    value = json.loads(q32.read_text())
    value["context"] = 4096
    q32.write_text(json.dumps(value), encoding="utf-8")
    s32["qualification_sha256"] = _sha(q32)
    q128 = tmp_path / str(s128["qualification"])
    value = json.loads(q128.read_text())
    value["abi"]["publisher"]["dtype"] = "FP16"
    q128.write_text(json.dumps(value), encoding="utf-8")
    s128["qualification_sha256"] = _sha(q128)

    activation = module.load_activation(
        _manifest(tmp_path, [s128, s32, s16]), deployment_root=tmp_path,
        context=1024, available_bytes=1000, base_resident_bytes=100,
        reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "physical/resident ABI" in activation.disabled["S128"]
    assert "geometry mismatch" in activation.disabled["S32"]
    assert "qualification SHA-256 mismatch" in activation.disabled["S16"]


def test_activation_rejects_noncanonical_absolute_mask_hash(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification_path = tmp_path / str(block["qualification"])
    qualification = json.loads(qualification_path.read_text())
    qualification["captures"][0]["mask_sha256"] = "f" * 64
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
    block["qualification_sha256"] = _sha(qualification_path)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path,
        context=1024, available_bytes=1000, base_resident_bytes=100,
        reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "canonical absolute mask" in activation.disabled["S16"]


def test_memory_admission_counts_shared_model_once_and_falls_back(
        tmp_path: Path) -> None:
    module = _module()
    # Separate files cannot claim one shared residency group.
    blocks = [
        _block(tmp_path, 128, admission=400),
        _block(tmp_path, 32, admission=200),
        _block(tmp_path, 16, admission=100),
    ]
    activation = module.load_activation(
        _manifest(tmp_path, blocks), deployment_root=tmp_path, context=1024,
        available_bytes=750, base_resident_bytes=200, reserve_bytes=100)
    assert activation.enabled_widths == (128, 1)
    assert activation.disabled["S32"] == \
        "measured residency exceeds available MMZ"
    assert activation.disabled["S16"] == \
        "measured residency exceeds available MMZ"
    assert activation.block_resident_bytes == 400


def test_strict_s1_base_underreport_disables_all_wide_blocks(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16, admission=100, s1_resident=100_000)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=200_000, base_resident_bytes=0, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert activation.strict_s1.identity.resident_bytes == 100_000
    assert activation.base_underreported is True
    assert "strict-S1 residency lower bound" in activation.disabled["S16"]


def test_strict_s1_and_reserve_must_physically_fit_available_mmz(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16, admission=100, s1_resident=100_000)

    with pytest.raises(module.PrefillActivationError, match="strict-S1 residency"):
        module.load_activation(
            _manifest(tmp_path, [block]), deployment_root=tmp_path,
            context=1024, available_bytes=100, base_resident_bytes=0,
            reserve_bytes=0)


def test_wide_block_with_different_strict_s1_anchor_is_disabled(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16, s1_stem="alternate-strict-s1")

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "does not match the live activation anchor" in \
        activation.disabled["S16"]


def test_wide_block_with_candidate_runtime_drift_is_disabled(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16, runtime_drift=True)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "runner/executor" in activation.disabled["S16"]


def test_v4_never_co_admits_a_shared_width_group(tmp_path: Path) -> None:
    module = _module()
    blocks = [
        _block(tmp_path, 128, group="universal", admission=400),
        _block(tmp_path, 32, group="universal", admission=400),
    ]
    activation = module.load_activation(
        _manifest(tmp_path, blocks), deployment_root=tmp_path, context=1024,
        available_bytes=2000, base_resident_bytes=100, reserve_bytes=0)
    assert activation.enabled_widths == (128, 1)
    assert activation.disabled["S32"] == \
        "release v4 requires one independently charged width per residency group"


def test_shared_group_rejects_descriptor_drift(tmp_path: Path) -> None:
    module = _module()
    s32 = _block(tmp_path, 32, admission=400)
    report = tmp_path / str(s32["admission_report"])
    value = json.loads(report.read_text())
    value["ready_descriptor_sha256"] = "d" * 64
    report.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    s32["admission_report_sha256"] = _sha(report)

    activation = module.load_activation(
        _manifest(tmp_path, [s32]), deployment_root=tmp_path,
        context=1024, available_bytes=2000, base_resident_bytes=100,
        reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "clean-board MMZ" in activation.disabled["S32"]


def test_symlink_phase_and_missing_artifact_fail_closed(tmp_path: Path) -> None:
    module = _module()
    outside = tmp_path / "outside.om"
    outside.write_bytes(b"outside")
    block = _block(tmp_path, 16)
    model = tmp_path / str(block["model"])
    model.unlink()
    model.symlink_to(outside)
    block["phase"] = "startup"
    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)
    assert activation.enabled_widths == (1,)
    assert "steady-only" in activation.disabled["S16"]

    block["phase"] = "steady"
    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)
    assert activation.enabled_widths == (1,)
    assert "must not traverse a symlink" in activation.disabled["S16"]


def test_top_level_contract_never_allows_removing_s1(tmp_path: Path) -> None:
    module = _module()
    manifest = _manifest(tmp_path, [])
    value = json.loads(manifest.read_text())
    value["strict_s1"] = False
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(module.PrefillActivationError, match="strict_s1"):
        module.load_activation(
            manifest, deployment_root=tmp_path, context=1024,
            available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)


@pytest.mark.parametrize("mode", [None, "writable-development-tree"])
def test_activation_requires_explicit_trusted_read_only_deployment(
        tmp_path: Path, mode: str | None) -> None:
    module = _module()
    manifest = _manifest(tmp_path, [])
    value = json.loads(manifest.read_text())
    if mode is None:
        del value["deployment_mode"]
        expected = "schema/fields"
    else:
        value["deployment_mode"] = mode
        expected = "trusted read-only deployment"
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(module.PrefillActivationError, match=expected):
        module.load_activation(
            manifest, deployment_root=tmp_path, context=1024,
            available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)


def test_duplicate_width_or_json_key_is_rejected(tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    with pytest.raises(module.PrefillActivationError, match="duplicate.*S16"):
        module.load_activation(
            _manifest(tmp_path, [block, dict(block)]),
            deployment_root=tmp_path, context=1024,
            available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    manifest = tmp_path / "duplicate-key.json"
    strict_s1 = json.dumps(_live_strict_s1_descriptor(tmp_path))
    manifest.write_text(
        '{"schema":"pico.minicpm5.prefill-activation.v4",'
        '"deployment_mode":"trusted-read-only-process-lifetime",'
        f'"context":1024,"context":4096,"strict_s1":{strict_s1},'
        '"blocks":[]}',
        encoding="utf-8")
    with pytest.raises(module.PrefillActivationError, match="duplicate key"):
        module.load_activation(
            manifest, deployment_root=tmp_path, context=1024,
            available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)


def test_admission_report_must_bind_model_descriptor_and_bytes(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16, admission=100)
    report = tmp_path / str(block["admission_report"])
    value = json.loads(report.read_text())
    value["model_sha256"] = "0" * 64
    value["after_available_bytes"] += 1
    report.write_text(json.dumps(value), encoding="utf-8")
    block["admission_report_sha256"] = _sha(report)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)
    assert activation.enabled_widths == (1,)
    assert "clean-board MMZ" in activation.disabled["S16"]


def test_incomplete_pass_qualification_cannot_activate(tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification = tmp_path / str(block["qualification"])
    value = json.loads(qualification.read_text())
    del value["captures"]
    qualification.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    block["qualification_sha256"] = _sha(qualification)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "complete release v4 PASS" in activation.disabled["S16"]


def test_base_models_and_reserve_must_fit_before_any_block_admission(
        tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module.PrefillActivationError, match="base resident"):
        module.load_activation(
            _manifest(tmp_path, []), deployment_root=tmp_path, context=1024,
            available_bytes=100, base_resident_bytes=91, reserve_bytes=10)


def test_development_v2_pass_is_never_release_activatable(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification = tmp_path / str(block["qualification"])
    value = json.loads(qualification.read_text())
    value["schema"] = "pico.minicpm5.prefill-block-qualification.v2"
    value.pop("release_eligible")
    qualification.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    block["qualification_sha256"] = _sha(qualification)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert activation.disabled["S16"] == \
        "development v2 qualification is not release activatable"


def test_forged_six_field_strict_s1_baseline_cannot_enable_s16(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification_path = tmp_path / str(block["qualification"])
    qualification = json.loads(qualification_path.read_text())
    baseline = qualification["baselines"][0]
    baseline_path = tmp_path / "forged-strict-s1-v1.json"
    forged = {
        "schema": STRICT_S1_DEVELOPMENT_SCHEMA,
        "status": "PASS",
        "release_eligible": True,
        "context": 1024,
        "width": 1,
        "om_sha256": baseline["om"]["sha256"],
    }
    forged_artifact = _json_artifact(baseline_path, forged)
    baseline["qualification"] = {
        **forged_artifact, "file_verified": True}
    qualification_path.write_text(
        json.dumps(qualification, sort_keys=True), encoding="utf-8")
    block["qualification_sha256"] = _sha(qualification_path)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "development strict-S1 v1" in activation.disabled["S16"]


def test_non_utf8_optional_qualification_disables_width_and_keeps_s1(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification_path = tmp_path / str(block["qualification"])
    qualification_path.write_bytes(b"\xff\xfe\x00")
    block["qualification_sha256"] = _sha(qualification_path)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "cannot read block qualification" in activation.disabled["S16"]


def test_unreadable_optional_artifact_disables_width_and_keeps_s1(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification_path = (
        tmp_path / str(block["qualification"])).resolve()
    real_sha256 = module._sha256

    def denied(path: Path) -> str:
        if Path(path).resolve() == qualification_path:
            raise PermissionError("qualification is unreadable")
        return real_sha256(path)

    monkeypatch.setattr(module, "_sha256", denied)
    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "qualification is unreadable" in activation.disabled["S16"]


def test_tampered_live_strict_s1_anchor_rejects_the_activation(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification = json.loads(
        (tmp_path / str(block["qualification"])).read_text())
    baseline_path = tmp_path / qualification["baselines"][0][
        "qualification"]["path"]
    baseline = json.loads(baseline_path.read_text())
    capture_path = tmp_path / baseline["captures"][0]["artifact"]["path"]
    capture_path.write_text("{}", encoding="utf-8")

    with pytest.raises(module.PrefillActivationError, match="strict-S1 capture"):
        module.load_activation(
            _manifest(tmp_path, [block]), deployment_root=tmp_path,
            context=1024, available_bytes=1000,
            base_resident_bytes=100, reserve_bytes=0)


def test_swapped_live_strict_s1_route_models_reject_activation(
        tmp_path: Path) -> None:
    module = _module()
    manifest = _manifest(tmp_path, [_block(tmp_path, 16)])
    value = json.loads(manifest.read_text())
    strict_s1 = value["strict_s1"]
    strict_s1["bootstrap_model"], strict_s1["canonical_decode_model"] = (
        strict_s1["canonical_decode_model"], strict_s1["bootstrap_model"])
    manifest.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(
            module.PrefillActivationError,
            match="live strict-S1 bootstrap_om"):
        module.load_activation(
            manifest, deployment_root=tmp_path, context=1024,
            available_bytes=1000, base_resident_bytes=100,
            reserve_bytes=0)


@pytest.mark.parametrize("field", ["head_model", "embedding"])
def test_no_handler_activation_still_verifies_live_token_artifacts(
        tmp_path: Path, field: str) -> None:
    module = _module()
    manifest = _manifest(tmp_path, [])
    declaration = json.loads(manifest.read_text())["strict_s1"]
    (tmp_path / declaration[field]).write_bytes(b"replaced-after-activation")

    with pytest.raises(
            module.PrefillActivationError,
            match=f"strict-S1 {'head OM' if field == 'head_model' else field}"):
        module.load_activation(
            manifest, deployment_root=tmp_path, context=1024,
            available_bytes=1000, base_resident_bytes=100,
            reserve_bytes=0)


def test_legacy_v3_release_records_are_explicitly_rejected(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification_path = tmp_path / str(block["qualification"])
    qualification = json.loads(qualification_path.read_text())
    qualification["schema"] = \
        "pico.minicpm5.prefill-block-qualification.v3"
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
    block["qualification_sha256"] = _sha(qualification_path)

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert activation.disabled["S16"] == \
        "legacy wide v3 qualification is not release activatable"


def test_activation_rereads_capture_evidence_and_runtime_binary(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16)
    qualification = json.loads(
        (tmp_path / str(block["qualification"])).read_text())
    capture = tmp_path / qualification["captures"][0]["artifact"]["path"]
    capture.write_text("{}", encoding="utf-8")

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)
    assert activation.enabled_widths == (1,)
    assert "capture artifact" in activation.disabled["S16"]

    block = _block(tmp_path, 16)
    (tmp_path / str(block["executor"])).write_bytes(b"tampered executor")
    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)
    assert activation.enabled_widths == (1,)
    assert "executor bytes/SHA" in activation.disabled["S16"]


def test_manifest_cannot_self_report_a_smaller_admission_delta(
        tmp_path: Path) -> None:
    module = _module()
    block = _block(tmp_path, 16, admission=100)
    block["admission_bytes"] = 99

    activation = module.load_activation(
        _manifest(tmp_path, [block]), deployment_root=tmp_path, context=1024,
        available_bytes=1000, base_resident_bytes=100, reserve_bytes=0)

    assert activation.enabled_widths == (1,)
    assert "clean-board MMZ" in activation.disabled["S16"]
