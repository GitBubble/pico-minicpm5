"""High-level build orchestration with an injectable compiler backend."""
from __future__ import annotations

import json
from pathlib import Path

from .compiler.base import CompileRequest
from .contract import OFFICIAL_CONTRACT


def transformer_request(
    *, role: str, model: Path, output: Path, calibration_dir: Path, context: int
) -> CompileRequest:
    if role not in ("decode", "prefill"):
        raise ValueError("transformer role must be decode or prefill")
    c = OFFICIAL_CONTRACT
    stem = f"minicpm5_merge{c.num_hidden_layers}_ctx{context}"
    return CompileRequest(
        role=role,
        model=model,
        output=output,
        calibration_dir=calibration_dir,
        calibration_stem=stem,
        inputs=("hidden", "attention_mask", "rope_r", "k_cache_all", "v_cache_all"),
        input_shape=(
            f"hidden:1,{c.hidden_size},1,1;"
            f"attention_mask:1,1,1,{context};"
            f"rope_r:1,1,{c.head_dim},{c.head_dim};"
            f"k_cache_all:1,{c.num_hidden_layers * c.num_key_value_heads},"
            f"{context - 1},{c.head_dim};"
            f"v_cache_all:1,{c.num_hidden_layers * c.num_key_value_heads},"
            f"{context - 1},{c.head_dim}"
        ),
        input_type=(
            "hidden:FP32;attention_mask:FP32;rope_r:FP32;"
            "k_cache_all:FP16;v_cache_all:FP16"
        ),
    )


def head_request(*, model: Path, output: Path, calibration_dir: Path) -> CompileRequest:
    hidden = OFFICIAL_CONTRACT.hidden_size
    return CompileRequest(
        role="head_flat",
        model=model,
        output=output,
        calibration_dir=calibration_dir,
        calibration_stem="minicpm5_native_head",
        inputs=("hidden", "residual"),
        input_shape=f"hidden:1,{hidden},1,1;residual:1,{hidden},1,1",
        input_type="hidden:FP32;residual:FP32",
    )


def build_three_handle(
    *,
    compiler,
    decode_onnx: Path,
    prefill_onnx: Path,
    head_onnx: Path,
    calibration_root: Path,
    output: Path,
    context: int = 1024,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    requests = (
        transformer_request(
            role="decode",
            model=decode_onnx,
            output=output / "decode.om",
            calibration_dir=calibration_root / "decode",
            context=context,
        ),
        transformer_request(
            role="prefill",
            model=prefill_onnx,
            output=output / "prefill.om",
            calibration_dir=calibration_root / "prefill",
            context=context,
        ),
        head_request(
            model=head_onnx,
            output=output / "head_flat.om",
            calibration_dir=calibration_root / "prefill",
        ),
    )
    builds = [compiler.compile(request) for request in requests]
    report = {
        "schema": "pico.minicpm5.three-handle-build.v1",
        "context": context,
        "handles": 3,
        "builds": builds,
    }
    (output / "build-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
