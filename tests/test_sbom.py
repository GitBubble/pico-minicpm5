from __future__ import annotations

import json
from pathlib import Path

from pico_minicpm5.release.sbom import generate_spdx


def test_spdx_sbom_is_portable_and_deterministic(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("source\n")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1720000000")
    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    first = generate_spdx(project_root=project, output=first_path)
    second = generate_spdx(project_root=project, output=second_path)
    assert first["sha256"] == second["sha256"]
    document = json.loads(first_path.read_text())
    assert document["spdxVersion"] == "SPDX-2.3"
    assert document["packages"][0]["licenseDeclared"] == "Apache-2.0"
    assert document["files"][0]["fileName"] == "./README.md"
    assert str(project) not in first_path.read_text()


def test_spdx_excludes_release_checksum_to_avoid_digest_cycle(tmp_path: Path) -> None:
    project = tmp_path / "project"
    checksum = project / "release" / "v0.1.0" / "SHA256SUMS"
    checksum.parent.mkdir(parents=True)
    checksum.write_text("0" * 64 + "  sbom.json\n")
    (project / "README.md").write_text("source\n")
    output = tmp_path / "sbom.json"

    generate_spdx(project_root=project, output=output)

    names = {item["fileName"] for item in json.loads(output.read_text())["files"]}
    assert "./release/v0.1.0/SHA256SUMS" not in names
