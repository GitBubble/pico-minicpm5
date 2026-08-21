"""Reproducible source archive with a strict private/large-artifact denylist."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

from .. import __version__

DENIED_SUFFIXES = {
    ".om", ".onnx", ".safetensors", ".so", ".dylib", ".dll", ".bin", ".image_list",
    ".pem", ".key", ".p12", ".pfx", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".a",
    ".zip", ".tar", ".gz", ".xz", ".7z",
}
DENIED_PARTS = {
    ".git", ".venv", ".tox", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "__pycache__", "artifacts", "work", "model", "dist", "build", "node_modules",
}
ALLOWED_ROOT_FILES = {
    ".gitignore", "CHANGELOG.md", "CITATION.cff", "CONTRIBUTING.md", "LICENSE",
    "MANIFEST.in", "Makefile", "MODEL_PROVENANCE.md", "NOTICE", "README.md",
    "README.zh-CN.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
}
ALLOWED_TOP_LEVEL = {
    ".github", "app", "configs", "docs", "experimental", "release", "schemas", "src", "tests"
}
# Keep markers split in this module so the denylist does not reject its own
# source while still matching the joined private strings in candidate files.
TEXT_DENYLIST = (
    "/" + "Users/",
    "/" + "root/minicpm5_gate",
    "HF_TOKEN" + "=",
)
MAX_SOURCE_FILE = 5 << 20


def source_files(project_root: Path) -> list[Path]:
    files = []
    for path in project_root.rglob("*"):
        relative = path.relative_to(project_root)
        # Local migration snapshots are operator backups, not source inputs.
        if ".pre_unification_" in path.name:
            continue
        # The runtime archive owns qualified board executables and SDK shared
        # objects. The Python source distribution carries their rebuildable
        # source and must not duplicate binary payloads.
        if relative.parts[:2] in {("app", "bin"), ("app", "lib")}:
            continue
        # A published checksum list contains the SBOM digest. Including that
        # list in the SBOM input would create a self-referential release cycle.
        if relative.parts[:1] == ("release",) and path.name == "SHA256SUMS":
            continue
        if (
            any(part in DENIED_PARTS or part.endswith(".egg-info") for part in relative.parts)
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is forbidden in a source release: {relative}")
        if not path.is_file():
            continue
        if path.name == ".env" or path.name.startswith(".env."):
            raise ValueError(f"environment file is forbidden in a source release: {relative}")
        # Community AIfly runtime (Pegasus extras + glibc 2.39 sidecar).
        # Keep rebuildable sources; skip ELF / .so payloads.
        if relative.parts[:2] in {("app", "glibc239"), ("app", "lib-community")}:
            if path.suffix.lower() in DENIED_SUFFIXES:
                continue
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    continue
        if path.suffix.lower() in DENIED_SUFFIXES:
            raise ValueError(f"denied artifact in source tree: {relative}")
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic == b"PICO" or magic == b"\x7fELF" or magic in {
            b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe"
        }:
            raise ValueError(f"binary executable/model in source tree: {relative}")
        if len(relative.parts) == 1:
            if relative.name not in ALLOWED_ROOT_FILES:
                continue
        elif relative.parts[0] not in ALLOWED_TOP_LEVEL:
            continue
        if path.stat().st_size > MAX_SOURCE_FILE:
            raise ValueError(f"source file exceeds {MAX_SOURCE_FILE} bytes: {relative}")
        if path.suffix.lower() in {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".sh", ""}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in TEXT_DENYLIST:
                if marker in text:
                    raise ValueError(f"private-path marker {marker!r} in {relative}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(project_root).as_posix())


def create_source_archive(
    *, project_root: Path, output_dir: Path, check_only: bool = False
) -> dict:
    files = source_files(project_root)
    report = {
        "schema": "pico.minicpm5.source-release.v1",
        "version": __version__,
        "file_count": len(files),
        "status": "PASS",
    }
    if check_only:
        return report
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"pico-minicpm5-{__version__}.tar.gz"
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in files:
                    relative = path.relative_to(project_root)
                    data = path.read_bytes()
                    info = tarfile.TarInfo(
                        f"pico-minicpm5-{__version__}/{relative.as_posix()}"
                    )
                    info.size = len(data)
                    info.mtime = epoch
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                    archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (output_dir / "SHA256SUMS.source").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8"
    )
    report.update(path=target.name, bytes=target.stat().st_size, sha256=digest)
    (output_dir / "source-release.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
