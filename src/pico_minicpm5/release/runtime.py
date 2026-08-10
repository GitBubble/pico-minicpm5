"""Deterministic SS928 board-application archive."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile

from .. import __version__


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _executor_bytes(path: Path, manifest: dict) -> bytes:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"\x7fELF":
        raise ValueError("executor is not an ELF binary")
    endian = "<" if data[5] == 1 else ">" if data[5] == 2 else None
    if endian is None or struct.unpack_from(f"{endian}H", data, 18)[0] != 183:
        raise ValueError("executor is not an AArch64 ELF binary")
    expected = manifest["runtime"]["executor_binary"]
    if len(data) != int(expected["bytes"]) or _sha256(data) != expected["sha256"]:
        raise ValueError("executor does not match the qualified runtime manifest")
    return data


def create_runtime_archive(*, project_root: Path, executor: Path, output_dir: Path) -> dict:
    manifest_path = project_root / "release" / f"v{__version__}" / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary = _executor_bytes(executor, manifest)
    files: dict[str, tuple[bytes, int]] = {}
    for path in sorted((project_root / "app").rglob("*")):
        relative = path.relative_to(project_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"} \
                or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"runtime application symlink is forbidden: {path}")
        if path.is_file():
            name = relative.as_posix()
            files[name] = (path.read_bytes(), 0o755 if os.access(path, os.X_OK) else 0o644)
    files["app/bin/pico_persistent_acl_executor.aarch64"] = (binary, 0o755)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "MODEL_PROVENANCE.md"):
        files[name] = ((project_root / name).read_bytes(), 0o644)

    checksums = "".join(
        f"{_sha256(data)}  {name}\n" for name, (data, _mode) in sorted(files.items())
    ).encode("utf-8")
    files["SHA256SUMS"] = (checksums, 0o644)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"pico-minicpm5-runtime-v{__version__}.tar.gz"
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    prefix = f"pico-minicpm5-deployment-v{__version__}"
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, (data, mode) in sorted(files.items()):
                    info = tarfile.TarInfo(f"{prefix}/{name}")
                    info.size = len(data)
                    info.mtime = epoch
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mode = mode
                    archive.addfile(info, io.BytesIO(data))
    report = {
        "schema": "pico.minicpm5.runtime-release.v1",
        "version": __version__,
        "path": target.name,
        "bytes": target.stat().st_size,
        "sha256": _sha256(target.read_bytes()),
        "file_count": len(files),
        "executor_sha256": _sha256(binary),
        "status": "PASS",
    }
    (output_dir / "runtime-release.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
