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
