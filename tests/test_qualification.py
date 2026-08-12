from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pico_minicpm5.contract import HF_REPO_ID, HF_REVISION, sha256_file
from pico_minicpm5.qualification import MODEL_FILES, build_qualification
from pico_minicpm5.release.bundle import _qualified

ROOT = Path(__file__).resolve().parents[1]
ZERO_RESIDUAL = {
    "bytes": 1536 * 4,
    "sha256": hashlib.sha256(bytes(1536 * 4)).hexdigest(),
}


def _models(
    path: Path, *, backend: str = "atc", context: int = 1024
) -> Path:
    path.mkdir()
    builds = []
    for role, name in MODEL_FILES.items():
        model = path / name
        model.write_bytes(b"PICO" + role.encode())
        builds.append(
            {
                "role": role,
                "backend": backend,
                "output_sha256": sha256_file(model),
            }
        )
    (path / "build-manifest.json").write_text(
        json.dumps(
            {
                "schema": "pico.minicpm5.three-handle-build.v1",
                "context": context,
                "handles": 3,
                "builds": builds,
            }
        )
    )
    return path


def _execution(
    models: Path,
    role: str,
    *,
    position: int,
    outputs: list[dict],
    inputs: dict[str, dict] | None = None,
    context: int = 1024,
) -> dict:
    model = models / MODEL_FILES[role]
    return {
        "role": role,
        "om": {"bytes": model.stat().st_size, "sha256": sha256_file(model)},
        "compiler": {
            "backend": "atc",
            "build_manifest_sha256": sha256_file(models / "build-manifest.json"),
        },
        "capture": {
            "manifest_sha256": hashlib.sha256(f"capture-{role}".encode()).hexdigest(),
            "runner": "libinstsim",
            "status": "PASS",
            "position": position,
            "context": context,
            "inputs": inputs or {},
            "outputs": outputs,
        },
    }


def _score(
    path: Path,
    *,
    models: Path,
    role: str,
    position: int,
    minimum: float,
    context: int = 1024,
    logical_hidden: dict | None = None,
) -> None:
    outputs = [
        {
            "path": f"/private/work/out.{index}.bin",
            "bytes": 16,
            "sha256": f"{index + 1:064x}",
        }
        for index in range(3)
    ]
    logical_hidden = logical_hidden or {
        "bytes": 1536 * 4,
        "sha256": ("b" if role == "decode" else "c") * 64,
    }
    value = {
        "schema": "pico.minicpm5.public-output-score.v1",
        "position": position,
        "context": context,
        "role": role,
        "execution": _execution(
            models,
            role,
            position=position,
            outputs=[
                {"bytes": record["bytes"], "sha256": record["sha256"]}
                for record in outputs
            ],
            context=context,
        ),
        "threshold_exclusive": 0.98,
        "cosine": {
            "next_hidden": minimum,
            "k_cur_all": max(minimum, 0.999),
            "v_cur_all": max(minimum, 0.998),
        },
        "minimum_public_output": minimum,
        "output_order": "direct",
        "logical_outputs": {"next_hidden": logical_hidden},
        "outputs": outputs,
        "status": "PASS" if minimum > 0.98 else "FAIL",
    }
    path.write_text(json.dumps(value))


def _head_score(
    path: Path,
    *,
    models: Path,
    minimum: float = 0.997,
    top1_exact: bool = True,
    context: int = 1024,
    hidden: dict | None = None,
) -> None:
    hidden = hidden or {"bytes": 1536 * 4, "sha256": "b" * 64}
    inputs = {"hidden": hidden, "residual": ZERO_RESIDUAL}
    output = {"bytes": 64, "sha256": "a" * 64}
    value = {
        "schema": "pico.minicpm5.head-output-score.v1",
        "position": 1,
        "context": context,
        "role": "head_flat",
        "threshold_exclusive": 0.98,
        "cosine": minimum,
        "minimum_public_output": minimum,
        "top1_output": 42 if top1_exact else 41,
        "top1_reference": 42,
        "top1_exact": top1_exact,
        "execution": _execution(
            models,
            "head_flat",
            position=1,
            outputs=[output],
            inputs=inputs,
            context=context,
        ),
        "inputs": inputs,
        "output": output,
        "status": "PASS" if minimum > 0.98 and top1_exact else "FAIL",
    }
    path.write_text(json.dumps(value))


def _complete_new_qualification(*, declared: float, minimum: float) -> dict:
    build_manifest_sha = "a" * 64
    artifacts = {
        "prefill": {"bytes": 8, "sha256": "b" * 64},
        "decode": {"bytes": 8, "sha256": "c" * 64},
        "head_flat": {"bytes": 8, "sha256": "d" * 64},
    }

    def execution(role: str) -> dict:
        position = 0 if role == "prefill" else 1
        outputs = (
            [{"bytes": 64, "sha256": "e" * 64}]
            if role == "head_flat"
            else [
                {"bytes": 16, "sha256": f"{index + 1:064x}"}
                for index in range(3)
            ]
        )
        inputs = (
            {
                "hidden": {"bytes": 1536 * 4, "sha256": "b" * 64},
                "residual": ZERO_RESIDUAL,
            }
            if role == "head_flat"
            else {}
        )
        return {
            "role": role,
            "om": artifacts[role],
            "compiler": {
                "backend": "atc",
                "build_manifest_sha256": build_manifest_sha,
            },
            "capture": {
                "manifest_sha256": hashlib.sha256(
                    f"capture-{role}".encode()
                ).hexdigest(),
                "runner": "libinstsim",
                "status": "PASS",
                "position": position,
                "context": 1024,
                "inputs": inputs,
                "outputs": outputs,
            },
        }

    families = {}
    for role, position in (("prefill", 0), ("decode", 1)):
        outputs = [
            {"bytes": 16, "sha256": f"{index + 1:064x}"}
            for index in range(3)
        ]
        families[role] = {
            "position": position,
            "context": 1024,
            "cosine": {
                "next_hidden": minimum,
                "k_cur_all": minimum,
                "v_cur_all": minimum,
            },
            "minimum_public_output": minimum,
            "output_order": "direct",
            "outputs": outputs,
            "logical_outputs": {
                "next_hidden": {
                    "bytes": 1536 * 4,
                    "sha256": ("b" if role == "decode" else "c") * 64,
                }
            },
            "execution": execution(role),
        }
    return {
        "schema": "pico.minicpm5.qualification.v1",
        "model": {"repository": HF_REPO_ID, "revision": HF_REVISION},
        "target": {"soc": "Hi3403", "npu_arch": "V101", "context": 1024},
        "threshold_exclusive": declared,
        "families": families,
        "head": {
            "position": 1,
            "context": 1024,
            "cosine": minimum,
            "minimum_public_output": minimum,
            "top1_output": 42,
            "top1_reference": 42,
            "top1_exact": True,
            "output": {"bytes": 64, "sha256": "e" * 64},
            "inputs": {
                "hidden": {"bytes": 1536 * 4, "sha256": "b" * 64},
                "residual": ZERO_RESIDUAL,
            },
            "execution": execution("head_flat"),
            "source_transformer_role": "decode",
        },
        "minimum_public_output": minimum,
        "artifacts": artifacts,
        "compiler": {
            "backend": "atc",
            "build_manifest_sha256": build_manifest_sha,
        },
        "greedy": None,
        "verdict": {
            "public_output_numeric": "PASS",
            "head_numeric": "PASS",
            "overall": "PASS",
        },
    }


def _qualification_inputs(
    tmp_path: Path, *, backend: str = "atc", context: int = 1024
) -> tuple[Path, ...]:
    models = _models(tmp_path / "models", backend=backend, context=context)
    prefill = tmp_path / "prefill.json"
    decode = tmp_path / "decode.json"
    head = tmp_path / "head.json"
    _score(
        prefill,
        models=models,
        role="prefill",
        position=0,
        minimum=0.996,
        context=context,
    )
    _score(
        decode,
        models=models,
        role="decode",
        position=1,
        minimum=0.998,
        context=context,
    )
    _head_score(head, models=models, context=context)
    return models, prefill, decode, head


def test_qualification_requires_all_families_and_removes_paths(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path)
    report = build_qualification(
        prefill_score=prefill,
        decode_score=decode,
        head_score=head,
        models=models,
        output=tmp_path / "qualification.json",
    )
    assert _qualified(report)
    assert report["minimum_public_output"] == 0.996
    assert report["head"]["top1_exact"] is True
    assert report["head"]["execution"]["om"] == report["artifacts"]["head_flat"]
    assert report["head"]["source_transformer_role"] == "decode"
    assert (
        report["head"]["inputs"]["hidden"]
        == report["families"]["decode"]["logical_outputs"]["next_hidden"]
    )
    assert all("path" not in item for item in report["families"]["decode"]["outputs"])


def test_qualification_rejects_threshold_equality(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path)
    _score(prefill, models=models, role="prefill", position=0, minimum=0.98)
    with pytest.raises(ValueError, match="does not exceed"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )


def test_qualification_rejects_fake_compiler(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path, backend="fake")
    with pytest.raises(ValueError, match="not built by the ATC"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )


def test_qualification_rejects_score_bound_to_another_om(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path)
    value = json.loads(decode.read_text())
    value["execution"]["om"]["sha256"] = "f" * 64
    decode.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different OM"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )


def test_qualification_rejects_unproven_head(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path)
    _head_score(head, models=models, top1_exact=False)
    with pytest.raises(ValueError, match="head top1 token"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )


def test_qualification_rejects_a_broken_transformer_to_head_bridge(
    tmp_path: Path,
) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path)
    value = json.loads(head.read_text())
    wrong_hidden = {"bytes": 1536 * 4, "sha256": "f" * 64}
    value["inputs"]["hidden"] = wrong_hidden
    value["execution"]["capture"]["inputs"]["hidden"] = wrong_hidden
    head.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="logical next_hidden"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )


def test_qualification_rejects_a_non_1024_build(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path, context=512)
    with pytest.raises(ValueError, match="schema, handle count or context"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )


def test_release_policy_does_not_trust_a_lower_report_threshold() -> None:
    # This is otherwise a complete v1 report, so the rejection specifically
    # proves that a report cannot lower the public >0.98 release policy.
    assert _qualified(_complete_new_qualification(declared=0.1, minimum=0.99))
    assert not _qualified(_complete_new_qualification(declared=0.1, minimum=0.2))


def test_numeric_gate_rejects_json_booleans(tmp_path: Path) -> None:
    models, prefill, decode, head = _qualification_inputs(tmp_path)
    score = json.loads(prefill.read_text())
    score["cosine"] = {
        "next_hidden": True,
        "k_cur_all": True,
        "v_cur_all": True,
    }
    score["minimum_public_output"] = True
    prefill.write_text(json.dumps(score))
    with pytest.raises(ValueError, match="numeric|does not exceed"):
        build_qualification(
            prefill_score=prefill,
            decode_score=decode,
            head_score=head,
            models=models,
            output=tmp_path / "qualification.json",
        )

    qualification = _complete_new_qualification(declared=0.98, minimum=0.99)
    qualification["head"]["cosine"] = True
    qualification["head"]["minimum_public_output"] = True
    assert not _qualified(qualification)


def test_frozen_release_qualification_is_complete_and_portable() -> None:
    path = ROOT / "release" / "v0.1.0" / "qualification.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert _qualified(value)
    text = path.read_text(encoding="utf-8")
    assert "192." + "168." not in text
    assert "/" + "Users/" not in text
