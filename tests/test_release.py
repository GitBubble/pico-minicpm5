from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pico_minicpm5 import __version__
from pico_minicpm5.contract import HF_REPO_ID, HF_REVISION, sha256_file
from pico_minicpm5.release import bundle as bundle_module
from pico_minicpm5.release.manifest import artifact_record, write_checksums
from pico_minicpm5.release.bundle import FROZEN_ACCEPTED, _legal_source, verify_bundle
from pico_minicpm5.release.source import create_source_archive, source_files

ROOT = Path(__file__).resolve().parents[1]


def _qualification(artifacts: dict, *, declared: float = 0.98, minimum: float = 0.99) -> dict:
    model_evidence = {
        role: {
            key: artifacts[f"models/{name}"][key]
            for key in ("bytes", "sha256")
        }
        for role, name in (
            ("decode", "decode.om"),
            ("prefill", "prefill.om"),
            ("head_flat", "head_flat.om"),
        )
    }
    build_manifest_sha = "a" * 64
    zero_residual = {
        "bytes": 1536 * 4,
        "sha256": hashlib.sha256(bytes(1536 * 4)).hexdigest(),
    }
    logical_hidden = {
        "prefill": {"bytes": 1536 * 4, "sha256": "b" * 64},
        "decode": {"bytes": 1536 * 4, "sha256": "c" * 64},
    }

    def execution(
        role: str,
        *,
        position: int,
        outputs: list[dict],
        inputs: dict[str, dict] | None = None,
    ) -> dict:
        return {
            "role": role,
            "om": model_evidence[role],
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
                "inputs": inputs or {},
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
            "logical_outputs": {"next_hidden": logical_hidden[role]},
            "execution": execution(
                role, position=position, outputs=outputs
            ),
        }
    head_output = {"bytes": 64, "sha256": "e" * 64}
    head_inputs = {
        "hidden": logical_hidden["decode"],
        "residual": zero_residual,
    }
    return {
        "schema": "pico.minicpm5.qualification.v1",
        "model": {"repository": HF_REPO_ID, "revision": HF_REVISION},
        "target": {"soc": "SS928", "npu_arch": "V101", "context": 1024},
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
            "output": head_output,
            "inputs": head_inputs,
            "execution": execution(
                "head_flat",
                position=1,
                outputs=[head_output],
                inputs=head_inputs,
            ),
            "source_transformer_role": "decode",
        },
        "minimum_public_output": minimum,
        "artifacts": model_evidence,
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


def _rewrite_checksums(root: Path) -> None:
    sums = root / "SHA256SUMS"
    paths = [path for path in root.rglob("*") if path.is_file() and path != sums]
    write_checksums(root, paths)


def _complete_release(
    root: Path,
    monkeypatch,
    *,
    declared: float = 0.98,
    minimum: float = 0.99,
) -> dict:
    models = root / "models"
    assets = root / "assets"
    models.mkdir()
    assets.mkdir()
    artifacts = {}
    model_roles = {
        "decode.om": "decode-transformer",
        "prefill.om": "position-zero-transformer",
        "head_flat.om": "vocabulary-head",
    }
    for name, role in model_roles.items():
        path = models / name
        path.write_bytes(b"PICO-" + name.encode())
        artifacts[f"models/{name}"] = artifact_record(
            path, role=role, relationship="derived-model"
        )

    embedding = assets / "token_embedding.f16.bin"
    tokenizer = assets / "tokenizer.json"
    embedding.write_bytes(b"\x00\x3c\x00\x40")
    tokenizer.write_text('{"version":"1.0"}\n')
    asset_report = {
        "schema": "pico.minicpm5.runtime-assets.v1",
        "assets": {
            embedding.name: {
                "bytes": embedding.stat().st_size,
                "sha256": sha256_file(embedding),
                "relationship": "derived-model",
            },
            tokenizer.name: {
                "bytes": tokenizer.stat().st_size,
                "sha256": sha256_file(tokenizer),
                "relationship": "copied-upstream-model-asset",
            },
        },
    }
    assets_manifest = assets / "assets-manifest.json"
    assets_manifest.write_text(json.dumps(asset_report, sort_keys=True) + "\n")
    artifacts["assets/token_embedding.f16.bin"] = artifact_record(
        embedding, role="token-embedding", relationship="derived-model"
    )
    artifacts["assets/tokenizer.json"] = artifact_record(
        tokenizer, role="tokenizer", relationship="copied-upstream-model-asset"
    )
    monkeypatch.setattr(
        bundle_module,
        "ASSET_HASHES",
        {
            "assets/token_embedding.f16.bin": sha256_file(embedding),
            "assets/tokenizer.json": sha256_file(tokenizer),
        },
    )
    artifacts["assets/assets-manifest.json"] = artifact_record(
        assets_manifest, role="asset-manifest", relationship="release-metadata"
    )

    for name, role in (
        ("LICENSE", "software-license"),
        ("NOTICE", "software-notice"),
        ("MODEL_LICENSE_NOTICE.md", "model-license-notice"),
    ):
        path = root / name
        path.write_text(f"{name} fixture\n")
        artifacts[name] = artifact_record(
            path, role=role, relationship="project-metadata"
        )

    qualification = _qualification(
        artifacts, declared=declared, minimum=minimum
    )
    qualification_path = root / "qualification.json"
    qualification_path.write_text(json.dumps(qualification, sort_keys=True) + "\n")
    artifacts["qualification.json"] = artifact_record(
        qualification_path,
        role="numeric-qualification",
        relationship="release-metadata",
    )
    manifest = {
        "schema": "pico.minicpm5.local-model-release.v1",
        "source": {"package": "pico-minicpm5", "version": __version__},
        "model": {"repository": HF_REPO_ID, "revision": HF_REVISION},
        "target": {"soc": "SS928", "npu_arch": "V101", "context": 1024},
        "abi": {
            "handles": 3,
            "transformer_public_inputs": 5,
            "transformer_runtime_inputs": 7,
            "transformer_outputs": 3,
        },
        "qualification": qualification,
        "artifacts": artifacts,
        "external_runtime_required": True,
    }
    (root / "release-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    _rewrite_checksums(root)
    return manifest


def test_source_archive_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("open source\n")
    (project / "tool.py").write_text("print('ok')\n")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1720000000")
    first = create_source_archive(project_root=project, output_dir=tmp_path / "a")
    second = create_source_archive(project_root=project, output_dir=tmp_path / "b")
    assert first["sha256"] == second["sha256"]
    archive = tmp_path / "a" / first["path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == first["sha256"]


def test_source_denylist_rejects_model_artifacts(tmp_path: Path) -> None:
    (tmp_path / "leak.om").write_bytes(b"PICO")
    with pytest.raises(ValueError, match="denied artifact"):
        source_files(tmp_path)


def test_source_denylist_rejects_symlinks_and_credentials(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_text("private\n")
    (tmp_path / "leak").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        source_files(tmp_path)
    (tmp_path / "leak").unlink()
    (tmp_path / ".env").write_text("SECRET=value\n")
    with pytest.raises(ValueError, match="environment file"):
        source_files(tmp_path)


def test_source_ignores_generated_environment_and_egg_info(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("source\n")
    environment = tmp_path / ".venv" / "bin"
    environment.mkdir(parents=True)
    (environment / "python").symlink_to("/usr/bin/python3")
    egg = tmp_path / "src" / "pico_minicpm5.egg-info"
    egg.mkdir(parents=True)
    (egg / "PKG-INFO").write_text("generated\n")
    assert [path.relative_to(tmp_path).as_posix() for path in source_files(tmp_path)] == [
        "README.md"
    ]


def test_release_checksums_detect_extras(tmp_path: Path, monkeypatch) -> None:
    _complete_release(tmp_path, monkeypatch)
    assert verify_bundle(tmp_path)["status"] == "PASS"
    (tmp_path / "unexpected.log").write_text("not checksummed\n")
    with pytest.raises(ValueError, match="extras"):
        verify_bundle(tmp_path)


def test_release_does_not_ignore_a_nested_sha256sums(tmp_path: Path, monkeypatch) -> None:
    _complete_release(tmp_path, monkeypatch)
    assert verify_bundle(tmp_path)["status"] == "PASS"
    (tmp_path / "assets" / "SHA256SUMS").write_text("must not be invisible\n")
    with pytest.raises(ValueError, match="extras"):
        verify_bundle(tmp_path)


def test_release_requires_runtime_assets_and_legal_files(tmp_path: Path, monkeypatch) -> None:
    manifest = _complete_release(tmp_path, monkeypatch)
    assets_manifest_path = tmp_path / "assets" / "assets-manifest.json"
    assets_manifest = json.loads(assets_manifest_path.read_text())
    assets_manifest["assets"].pop("tokenizer.json")
    assets_manifest_path.write_text(json.dumps(assets_manifest, sort_keys=True) + "\n")
    (tmp_path / "assets" / "tokenizer.json").unlink()
    (tmp_path / "LICENSE").unlink()
    manifest["artifacts"].pop("assets/tokenizer.json")
    manifest["artifacts"].pop("LICENSE")
    manifest["artifacts"]["assets/assets-manifest.json"] = artifact_record(
        assets_manifest_path, role="asset-manifest", relationship="release-metadata"
    )
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    _rewrite_checksums(tmp_path)
    with pytest.raises(ValueError, match="required|missing"):
        verify_bundle(tmp_path)


def test_release_rejects_a_complete_low_threshold_qualification(
    tmp_path: Path, monkeypatch
) -> None:
    _complete_release(tmp_path, monkeypatch, declared=0.1, minimum=0.2)
    with pytest.raises(ValueError, match="qualification"):
        verify_bundle(tmp_path)


def test_project_license_is_available_to_source_bundle() -> None:
    assert _legal_source("LICENSE").read_text().startswith("                                 Apache")
    assert _legal_source("NOTICE").is_file()


def test_model_bundle_failure_is_atomic(tmp_path: Path, monkeypatch) -> None:
    def fail(**_kwargs):
        raise RuntimeError("qualification failed")

    monkeypatch.setattr(bundle_module, "_assemble_bundle_into", fail)
    output = tmp_path / "release"
    with pytest.raises(RuntimeError, match="qualification failed"):
        bundle_module.assemble_bundle(
            models=tmp_path / "models",
            model_dir=tmp_path / "checkpoint",
            output=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".release.staging-*"))


def test_frozen_release_manifest_matches_code_and_qualification() -> None:
    release = ROOT / "release" / "v0.1.0"
    manifest = json.loads((release / "release-manifest.json").read_text())
    qualification = json.loads((release / "qualification.json").read_text())
    for name, digest in FROZEN_ACCEPTED.items():
        record = manifest["artifacts"][f"models/{name}"]
        assert record["sha256"] == digest
        role = {"decode.om": "decode", "prefill.om": "prefill", "head_flat.om": "head_flat"}[name]
        assert qualification["artifacts"][role]["sha256"] == digest
        assert qualification["artifacts"][role]["bytes"] == record["bytes"]
    assert manifest["artifacts"]["assets/token_embedding.f16.bin"]["sha256"] == (
        "5a93b589f0c5920021c95e04327c0771da2721d8eec2dd7ac1b283aa0d3b7df5"
    )
    assert manifest["artifacts"]["assets/tokenizer.json"]["sha256"] == (
        "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
    )
