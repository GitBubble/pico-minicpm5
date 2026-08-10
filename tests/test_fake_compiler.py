from __future__ import annotations

from pathlib import Path

import pytest

from pico_minicpm5.compiler.fake import FakeCompiler
from pico_minicpm5.pipeline import build_three_handle, transformer_request


def _touch_calibration(root: Path, role: str) -> None:
    stem = "minicpm5_merge24_ctx1024"
    target = root / role
    target.mkdir(parents=True)
    for name in ("hidden", "attention_mask", "rope_r", "k_cache_all", "v_cache_all"):
        (target / f"{stem}.{name}.image_list").write_text(f"{role}-{name}\n")
    if role == "prefill":
        for name in ("hidden", "residual"):
            (target / f"minicpm5_native_head.{name}.image_list").write_text(f"head-{name}\n")


def test_fake_three_handle_build_is_deterministic(tmp_path: Path) -> None:
    models = tmp_path / "onnx"
    models.mkdir()
    for name in ("decode.onnx", "prefill.onnx", "head.onnx"):
        (models / name).write_bytes(b"fixture-" + name.encode())
    calibration = tmp_path / "calibration"
    _touch_calibration(calibration, "decode")
    _touch_calibration(calibration, "prefill")
    first = build_three_handle(
        compiler=FakeCompiler(),
        decode_onnx=models / "decode.onnx",
        prefill_onnx=models / "prefill.onnx",
        head_onnx=models / "head.onnx",
        calibration_root=calibration,
        output=tmp_path / "build-a",
    )
    second = build_three_handle(
        compiler=FakeCompiler(),
        decode_onnx=models / "decode.onnx",
        prefill_onnx=models / "prefill.onnx",
        head_onnx=models / "head.onnx",
        calibration_root=calibration,
        output=tmp_path / "build-b",
    )
    assert len(first["builds"]) == len(second["builds"]) == 3
    for name in ("decode.om", "prefill.om", "head_flat.om"):
        left = (tmp_path / "build-a" / name).read_bytes()
        right = (tmp_path / "build-b" / name).read_bytes()
        assert left == right and left.startswith(b"PICO")


def test_fake_compiler_checks_calibration_contract(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    request = transformer_request(
        role="decode",
        model=model,
        output=tmp_path / "decode.om",
        calibration_dir=tmp_path / "missing",
        context=1024,
    )
    with pytest.raises(FileNotFoundError):
        FakeCompiler().compile(request)
