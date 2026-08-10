from __future__ import annotations

from pathlib import Path

import numpy as np

from pico_minicpm5.assets import _write_bf16_as_f16


def test_chunked_bf16_to_f16_conversion(tmp_path: Path) -> None:
    values = np.asarray([0.0, 1.0, -2.5, 0.33333334, 65504.0], np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype("<u2")
    source, target = tmp_path / "source.bin", tmp_path / "target.bin"
    source.write_bytes(b"prefix!!" + bf16.tobytes())
    _write_bf16_as_f16(
        source, target, byte_offset=8, elements=values.size, chunk_elements=2
    )
    restored_bf16 = (bf16.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(
        np.fromfile(target, np.float16), restored_bf16.astype(np.float16)
    )
