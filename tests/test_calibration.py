from __future__ import annotations

from pathlib import Path

import numpy as np

from pico_minicpm5.calibration import _write_rows, rope_matrix


def test_text_rows_are_deterministic_and_parseable(tmp_path: Path) -> None:
    target = tmp_path / "fixture.image_list"
    _write_rows(target, [np.asarray([1.25, -2.5], np.float32), np.asarray([0.0], np.float32)])
    assert target.read_text().splitlines() == ["1.25 -2.5", "0"]


def test_rope_position_zero_is_identity() -> None:
    matrix = rope_matrix(0)[0, 0]
    np.testing.assert_array_equal(matrix, np.eye(128, dtype=np.float32))
