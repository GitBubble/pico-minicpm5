from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pico_minicpm5.contract import OFFICIAL_CONTRACT, sha256_file
from pico_minicpm5.score import (
    build_runtime_capture,
    score_head_output,
    score_public_outputs,
)


def _bound_om(
    tmp_path: Path, role: str, *, context: int = 1024
) -> tuple[Path, Path]:
    om = tmp_path / f"{role}.om"
    om.write_bytes(b"PICO" + role.encode())
    manifest = tmp_path / "build-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "pico.minicpm5.three-handle-build.v1",
                "context": context,
                "handles": 3,
                "builds": [
                    {
                        "role": role,
                        "backend": "atc",
                        "output_sha256": sha256_file(om),
                    }
                ]
            }
        )
    )
    return om, manifest


def test_public_output_score_binds_the_executed_prefill_om(tmp_path: Path) -> None:
    contract = OFFICIAL_CONTRACT
    reference = tmp_path / "reference"
    (reference / "pos0").mkdir(parents=True)
    (reference / "pos1").mkdir()

    hidden = np.linspace(0.5, 2.0, contract.hidden_size, dtype=np.float32)
    hidden.tofile(reference / "pos0" / "layer_out_23.f32.bin")
    k_rows = []
    v_rows = []
    for layer in range(contract.num_hidden_layers):
        k = (
            np.arange(contract.num_key_value_heads * contract.head_dim, dtype=np.float16)
            .reshape(contract.num_key_value_heads, 1, contract.head_dim)
            + np.float16(layer + 1)
        )
        v = k + np.float16(0.5)
        k.tofile(reference / "pos1" / f"k_cache_{layer:02d}.f16.bin")
        v.tofile(reference / "pos1" / f"v_cache_{layer:02d}.f16.bin")
        k_rows.append(k.astype(np.float32).reshape(-1))
        v_rows.append(v.astype(np.float32).reshape(-1))

    outputs = tuple(tmp_path / f"output-{index}.bin" for index in range(3))
    np.concatenate(k_rows).astype(np.float32).tofile(outputs[0])
    np.concatenate(v_rows).astype(np.float32).tofile(outputs[1])
    hidden.tofile(outputs[2])
    om, manifest = _bound_om(tmp_path, "prefill", context=2)
    capture = tmp_path / "prefill-capture.json"
    build_runtime_capture(
        runner="libinstsim",
        role="prefill",
        position=0,
        context=2,
        om=om,
        build_manifest=manifest,
        inputs={},
        outputs=outputs,
        report=capture,
    )

    report = score_public_outputs(
        output_files=outputs,
        reference=reference,
        position=0,
        om=om,
        build_manifest=manifest,
        capture_manifest=capture,
        context=2,
    )

    assert report["status"] == "PASS"
    assert report["role"] == "prefill"
    assert report["minimum_public_output"] == pytest.approx(1.0)
    assert report["execution"]["om"] == {
        "bytes": om.stat().st_size,
        "sha256": sha256_file(om),
    }
    assert report["execution"]["compiler"]["build_manifest_sha256"] == sha256_file(
        manifest
    )


def test_score_rejects_an_om_not_named_by_the_build_manifest(tmp_path: Path) -> None:
    reference = tmp_path / "unused-reference"
    outputs = tuple(tmp_path / f"output-{index}.bin" for index in range(3))
    for output in outputs:
        output.write_bytes(b"")
    om, manifest = _bound_om(tmp_path, "prefill", context=2)
    om.write_bytes(b"PICOmoved")

    with pytest.raises(ValueError, match="build manifest hash"):
        score_public_outputs(
            output_files=outputs,
            reference=reference,
            position=0,
            om=om,
            build_manifest=manifest,
            capture_manifest=tmp_path / "capture.json",
            context=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "pico.minicpm5.untrusted-build.v1"),
        ("handles", 2),
        ("context", 512),
    ],
)
def test_capture_rejects_an_invalid_build_contract(
    tmp_path: Path, field: str, value
) -> None:
    om, manifest = _bound_om(tmp_path, "prefill")
    build = json.loads(manifest.read_text())
    build[field] = value
    manifest.write_text(json.dumps(build))
    outputs = tuple(tmp_path / f"output-{index}.bin" for index in range(3))
    for output in outputs:
        output.write_bytes(b"data")
    with pytest.raises(ValueError, match="schema, handle count or context"):
        build_runtime_capture(
            runner="libinstsim",
            role="prefill",
            position=0,
            context=1024,
            om=om,
            build_manifest=manifest,
            inputs={},
            outputs=outputs,
            report=tmp_path / "capture.json",
        )


@pytest.mark.parametrize("drift", ["output", "om"])
def test_score_rejects_capture_drift(tmp_path: Path, drift: str) -> None:
    contract = OFFICIAL_CONTRACT
    reference = tmp_path / "reference"
    (reference / "pos0").mkdir(parents=True)
    (reference / "pos1").mkdir()
    hidden = np.ones(contract.hidden_size, dtype=np.float32)
    hidden.tofile(reference / "pos0" / "layer_out_23.f32.bin")
    rows = []
    for layer in range(contract.num_hidden_layers):
        row = np.full(
            (contract.num_key_value_heads, 1, contract.head_dim),
            layer + 1,
            dtype=np.float16,
        )
        row.tofile(reference / "pos1" / f"k_cache_{layer:02d}.f16.bin")
        (row + np.float16(0.5)).tofile(
            reference / "pos1" / f"v_cache_{layer:02d}.f16.bin"
        )
        rows.append(row.astype(np.float32).reshape(-1))
    outputs = tuple(tmp_path / f"captured-{index}.bin" for index in range(3))
    np.concatenate(rows).tofile(outputs[0])
    np.concatenate([row + 0.5 for row in rows]).astype(np.float32).tofile(outputs[1])
    hidden.tofile(outputs[2])
    om, manifest = _bound_om(tmp_path, "prefill", context=2)
    capture = tmp_path / "capture.json"
    build_runtime_capture(
        runner="ss928-board",
        role="prefill",
        position=0,
        context=2,
        om=om,
        build_manifest=manifest,
        inputs={},
        outputs=outputs,
        report=capture,
    )
    if drift == "output":
        outputs[0].write_bytes(outputs[0].read_bytes() + b"\x00\x00\x00\x00")
        expected = "capture output hashes"
    else:
        value = json.loads(capture.read_text())
        value["om"]["sha256"] = "f" * 64
        capture.write_text(json.dumps(value))
        expected = "runtime capture does not match"

    with pytest.raises(ValueError, match=expected):
        score_public_outputs(
            output_files=outputs,
            reference=reference,
            position=0,
            om=om,
            build_manifest=manifest,
            capture_manifest=capture,
            context=2,
        )


def test_head_score_requires_numeric_and_top1_agreement(tmp_path: Path) -> None:
    reference = tmp_path / "reference" / "pos1"
    reference.mkdir(parents=True)
    expected = np.ones(OFFICIAL_CONTRACT.vocab_size, dtype=np.float32)
    expected[42] = 2.0
    expected.tofile(reference / "logits.f32.bin")
    output = tmp_path / "logits.f32.bin"
    expected.tofile(output)
    hidden = tmp_path / "hidden.f32.bin"
    residual = tmp_path / "residual.f32.bin"
    np.linspace(0.5, 2.0, OFFICIAL_CONTRACT.hidden_size, dtype=np.float32).tofile(hidden)
    np.zeros(OFFICIAL_CONTRACT.hidden_size, dtype=np.float32).tofile(residual)
    om, manifest = _bound_om(tmp_path, "head_flat")
    capture = tmp_path / "head-capture.json"
    build_runtime_capture(
        runner="libinstsim",
        role="head_flat",
        position=1,
        context=1024,
        om=om,
        build_manifest=manifest,
        inputs={"hidden": hidden, "residual": residual},
        outputs=(output,),
        report=capture,
    )

    passing = score_head_output(
        output_file=output,
        reference=reference.parent,
        position=1,
        om=om,
        build_manifest=manifest,
        capture_manifest=capture,
        hidden_input=hidden,
        residual_input=residual,
    )
    assert passing["status"] == "PASS"
    assert passing["top1_exact"] is True
    assert passing["execution"]["role"] == "head_flat"

    bad_residual = np.zeros(OFFICIAL_CONTRACT.hidden_size, dtype=np.float32)
    bad_residual[0] = 1.0
    bad_residual.tofile(residual)
    build_runtime_capture(
        runner="libinstsim",
        role="head_flat",
        position=1,
        context=1024,
        om=om,
        build_manifest=manifest,
        inputs={"hidden": hidden, "residual": residual},
        outputs=(output,),
        report=capture,
    )
    with pytest.raises(ValueError, match="residual input must be exactly"):
        score_head_output(
            output_file=output,
            reference=reference.parent,
            position=1,
            om=om,
            build_manifest=manifest,
            capture_manifest=capture,
            hidden_input=hidden,
            residual_input=residual,
        )

    np.zeros(OFFICIAL_CONTRACT.hidden_size, dtype=np.float32).tofile(residual)
    changed = expected.copy()
    changed[41] = 3.0
    changed.tofile(output)
    build_runtime_capture(
        runner="libinstsim",
        role="head_flat",
        position=1,
        context=1024,
        om=om,
        build_manifest=manifest,
        inputs={"hidden": hidden, "residual": residual},
        outputs=(output,),
        report=capture,
    )
    failing = score_head_output(
        output_file=output,
        reference=reference.parent,
        position=1,
        om=om,
        build_manifest=manifest,
        capture_manifest=capture,
        hidden_input=hidden,
        residual_input=residual,
    )
    assert failing["cosine"] > 0.98
    assert failing["top1_exact"] is False
    assert failing["status"] == "FAIL"
