"""Convert compact float-reference binaries into deterministic ATC image lists."""
from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO, Sequence

from .contract import OFFICIAL_CONTRACT

CALIBRATION_VALID_SLOTS = (1, 2, 3, 8, 32, 128, 512, 0)
CALIBRATION_ROPE_POSITIONS = (0, 1, 2, 7, 31, 127, 511, 1023)


def _write_values(stream: BinaryIO, values, *, first: bool) -> bool:
    import numpy as np

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if not first:
        stream.write(b" ")
    flat.tofile(stream, sep=" ", format="%.9g")
    return False


def _write_rows(path: Path, rows: Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            _write_values(stream, row, first=True)
            stream.write(b"\n")


def rope_matrix(position: int):
    import numpy as np

    c = OFFICIAL_CONTRACT
    inv = 1.0 / (
        c.rope_theta
        ** (np.arange(0, c.head_dim, 2, dtype=np.float64) / c.head_dim)
    )
    theta = np.concatenate((inv, inv)) * position
    cosine, sine = np.cos(theta), np.sin(theta)
    half = c.head_dim // 2
    matrix = np.zeros((1, 1, c.head_dim, c.head_dim), dtype=np.float32)
    for index in range(c.head_dim):
        matrix[0, 0, index, index] = cosine[index]
    for index in range(half):
        matrix[0, 0, index + half, index] = -sine[index]
        matrix[0, 0, index, index + half] = sine[index + half]
    return matrix


def _available_positions(reference: Path) -> list[int]:
    result = []
    for entry in reference.glob("pos*"):
        if entry.is_dir() and entry.name[3:].isdigit():
            result.append(int(entry.name[3:]))
    return sorted(result)


def _write_packed_cache(
    path: Path,
    *,
    reference: Path,
    sample_positions: Sequence[int],
    role: str,
) -> None:
    import numpy as np

    c = OFFICIAL_CONTRACT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for position in sample_positions:
            first = True
            for layer in range(c.num_hidden_layers):
                source = reference / f"pos{position}" / f"{role}_cache_{layer:02d}.f16.bin"
                values = np.fromfile(source, dtype=np.float16).astype(np.float32)
                first = _write_values(stream, values, first=first)
            stream.write(b"\n")


def build_calibration_bundle(
    *,
    reference: Path,
    output: Path,
    family: str,
    context: int = 1024,
    stem: str | None = None,
) -> dict:
    import numpy as np

    if family not in ("prefill", "decode"):
        raise ValueError("family must be prefill or decode")
    c = OFFICIAL_CONTRACT
    positions = _available_positions(reference)
    required = [0] if family == "prefill" else [value for value in positions if value > 0]
    if not required:
        raise ValueError(f"reference has no {family} positions")
    stem = stem or f"minicpm5_merge{c.num_hidden_layers}_ctx{context}"
    output.mkdir(parents=True, exist_ok=True)
    sample_positions = [required[index % len(required)] for index in range(8)]

    hidden_rows = [
        np.fromfile(reference / f"pos{position}" / "layer_in_00.f32.bin", np.float32)
        .reshape(1, c.hidden_size, 1, 1)
        for position in sample_positions
    ]
    _write_rows(output / f"{stem}.hidden.image_list", hidden_rows)

    masks = []
    for valid in CALIBRATION_VALID_SLOTS:
        mask = np.full((1, 1, 1, context), -64.0, dtype=np.float32)
        mask[..., : context if valid == 0 else min(valid, context - 1)] = 0.0
        mask[..., -1] = 0.0
        masks.append(mask)
    _write_rows(output / f"{stem}.attention_mask.image_list", masks)
    rope_positions = [min(value, context - 1) for value in CALIBRATION_ROPE_POSITIONS]
    _write_rows(
        output / f"{stem}.rope_r.image_list",
        [rope_matrix(position) for position in rope_positions],
    )
    for role in ("k", "v"):
        _write_packed_cache(
            output / f"{stem}.{role}_cache_all.image_list",
            reference=reference,
            sample_positions=sample_positions,
            role=role,
        )

    # The accepted dense head exposes an inert residual input for runtime ABI
    # compatibility. Its calibration value must stay exactly zero.
    head_hidden = [
        np.fromfile(reference / f"pos{position}" / "layer_out_23.f32.bin", np.float32)
        .reshape(1, c.hidden_size, 1, 1)
        for position in sample_positions[:4]
    ]
    _write_rows(output / "minicpm5_native_head.hidden.image_list", head_hidden)
    _write_rows(
        output / "minicpm5_native_head.residual.image_list",
        [np.zeros_like(row) for row in head_hidden],
    )
    report = {
        "schema": "pico.minicpm5.calibration.v1",
        "family": family,
        "context": context,
        "stem": stem,
        "reference": str(reference.resolve()),
        "sample_positions": sample_positions,
        "mask_valid_slots": list(CALIBRATION_VALID_SLOTS),
        "rope_positions": rope_positions,
        "inputs": ["hidden", "attention_mask", "rope_r", "k_cache_all", "v_cache_all"],
        "status": "GENERATED_REQUIRES_NUMERIC_REQUALIFICATION",
        "note": (
            "The accepted OM used a frozen historical donor corpus. This regenerated "
            "open corpus preserves the semantic calibration contract but is not claimed "
            "byte-identical; a >0.98 numeric gate is mandatory."
        ),
    }
    (output / f"{family}.calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
