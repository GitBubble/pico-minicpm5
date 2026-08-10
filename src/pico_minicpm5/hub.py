"""Pinned Hugging Face download frontend using the supported ``hf`` CLI."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

from .contract import HF_REPO_ID, HF_REVISION, verify_checkpoint


def fetch_checkpoint(
    local_dir: Path,
    *,
    revision: str = HF_REVISION,
    max_workers: int = 4,
    dry_run: bool = False,
    includes: Sequence[str] = (),
) -> dict:
    if revision != HF_REVISION:
        raise ValueError(
            f"unqualified revision {revision}; expected pinned {HF_REVISION}"
        )
    executable = shutil.which("hf")
    if executable is None:
        raise RuntimeError("hf CLI is not installed; see https://hf.co/docs/huggingface_hub/guides/cli")
    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "download",
        HF_REPO_ID,
        "--revision",
        revision,
        "--local-dir",
        str(local_dir),
        "--max-workers",
        str(max_workers),
    ]
    for pattern in includes:
        command += ["--include", pattern]
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        tail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"hf download failed:\n{tail}")
    if dry_run:
        return {
            "schema": "pico.minicpm5.hf-download-plan.v1",
            "repository": HF_REPO_ID,
            "revision": revision,
            "local_dir": str(local_dir),
            "command": command,
            "stdout": completed.stdout,
        }
    report = verify_checkpoint(local_dir)
    (local_dir / "pico-checkpoint-verification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
