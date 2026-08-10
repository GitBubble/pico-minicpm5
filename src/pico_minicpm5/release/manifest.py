from __future__ import annotations

import json
from pathlib import Path

from ..contract import sha256_file


def artifact_record(path: Path, *, role: str, relationship: str) -> dict:
    return {
        "role": role,
        "relationship": relationship,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_checksums(root: Path, paths: list[Path]) -> Path:
    entries = []
    for path in sorted(paths, key=lambda value: str(value.relative_to(root))):
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "SHA256SUMS"
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return target


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value
