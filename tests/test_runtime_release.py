from __future__ import annotations

import json
from pathlib import Path
import struct
import tarfile

import pytest

from pico_minicpm5.release.runtime import create_runtime_archive


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    (root / "app/native").mkdir(parents=True)
    (root / "app/src").mkdir(parents=True)
    (root / "release/v0.1.0").mkdir(parents=True)
    (root / "app/chat.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "app/native/Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")
    (root / "app/src/minicpm_prefill_schedule.py").write_text(
        "CANDIDATE_WIDTHS = (128, 32, 16, 1)\n", encoding="utf-8")
    (root / "app/src/minicpm_prefill_activation.py").write_text(
        "SCHEMA = 'pico.minicpm5.prefill-activation.v1'\n", encoding="utf-8")
    (root / "app/src/__pycache__").mkdir(parents=True)
    (root / "app/src/__pycache__/server.cpython-311.pyc").write_bytes(b"cache")
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "MODEL_PROVENANCE.md"):
        (root / name).write_text(name + "\n", encoding="utf-8")
    elf = bytearray(64)
    elf[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", elf, 18, 183)
    executor = tmp_path / "executor"
    executor.write_bytes(elf)
    import hashlib
    manifest = {
        "runtime": {"executor_binary": {
            "bytes": len(elf), "sha256": hashlib.sha256(elf).hexdigest()}}
    }
    (root / "release/v0.1.0/release-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return root, executor


def test_runtime_archive_has_one_app_layout(tmp_path: Path) -> None:
    root, executor = _fixture(tmp_path)
    report = create_runtime_archive(
        project_root=root, executor=executor, output_dir=tmp_path / "out")
    with tarfile.open(tmp_path / "out" / report["path"], "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("/app/chat.sh") for name in names)
    assert any(name.endswith("/app/native/Makefile") for name in names)
    assert any(name.endswith("/app/src/minicpm_prefill_schedule.py") for name in names)
    assert any(name.endswith("/app/src/minicpm_prefill_activation.py") for name in names)
    assert any(name.endswith("/app/bin/pico_persistent_acl_executor.aarch64") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any("/demo/" in name or name.endswith("/chat.sh") and "/app/" not in name
                   for name in names)


def test_runtime_archive_rejects_wrong_executor(tmp_path: Path) -> None:
    root, executor = _fixture(tmp_path)
    executor.write_bytes(b"not an elf")
    with pytest.raises(ValueError, match="ELF"):
        create_runtime_archive(project_root=root, executor=executor, output_dir=tmp_path / "out")
