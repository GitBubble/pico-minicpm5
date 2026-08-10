"""Export runtime assets from the separately downloaded checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .contract import (
    OFFICIAL_CONTRACT,
    SAFETENSORS_HEADER_BYTES,
    WEIGHT_SHARD,
    read_safetensors_header,
    sha256_file,
    verify_checkpoint,
)

TOKEN_EMBEDDING_BYTES = 401_080_320
TOKEN_EMBEDDING_SHA256 = "5a93b589f0c5920021c95e04327c0771da2721d8eec2dd7ac1b283aa0d3b7df5"


def _write_bf16_as_f16(
    source: Path,
    target: Path,
    *,
    byte_offset: int,
    elements: int,
    chunk_elements: int = 8 << 20,
) -> None:
    import numpy as np

    remaining = elements
    with source.open("rb") as input_stream, target.open("wb") as output_stream:
        input_stream.seek(byte_offset)
        while remaining:
            count = min(remaining, chunk_elements)
            raw = np.fromfile(input_stream, dtype=np.dtype("<u2"), count=count)
            if raw.size != count:
                raise ValueError("truncated BF16 tensor payload")
            bits = raw.astype(np.uint32) << 16
            bits.view(np.float32).astype(np.float16).tofile(output_stream)
            remaining -= count


def export_runtime_assets(*, model_dir: Path, output: Path, full_hash: bool = False) -> dict:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install pico-minicpm5[onnx]") from exc
    verify_checkpoint(model_dir, full_hash=full_hash)
    output.mkdir(parents=True, exist_ok=True)
    embedding_path = output / "token_embedding.f16.bin"
    shard = model_dir / WEIGHT_SHARD
    header = read_safetensors_header(shard)
    entry = header["model.embed_tokens.weight"]
    expected_shape = (OFFICIAL_CONTRACT.vocab_size, OFFICIAL_CONTRACT.hidden_size)
    shape = tuple(int(value) for value in entry["shape"])
    if shape != expected_shape or entry.get("dtype") != "BF16":
        raise ValueError(f"embedding contract drift: dtype={entry.get('dtype')} shape={shape}")
    begin, end = (int(value) for value in entry["data_offsets"])
    elements = int(np.prod(expected_shape, dtype=np.int64))
    if end - begin != elements * 2:
        raise ValueError("embedding BF16 byte span drift")
    temporary = embedding_path.with_suffix(".tmp")
    _write_bf16_as_f16(
        shard,
        temporary,
        byte_offset=8 + SAFETENSORS_HEADER_BYTES + begin,
        elements=elements,
    )
    temporary.replace(embedding_path)
    if embedding_path.stat().st_size != TOKEN_EMBEDDING_BYTES:
        raise ValueError("token embedding size drift")
    embedding_digest = sha256_file(embedding_path)
    if embedding_digest != TOKEN_EMBEDDING_SHA256:
        raise ValueError("token embedding SHA256 drift")
    tokenizer_path = output / "tokenizer.json"
    shutil.copyfile(model_dir / "tokenizer.json", tokenizer_path)
    report = {
        "schema": "pico.minicpm5.runtime-assets.v1",
        "assets": {
            embedding_path.name: {
                "bytes": embedding_path.stat().st_size,
                "sha256": embedding_digest,
                "relationship": "derived-model",
            },
            tokenizer_path.name: {
                "bytes": tokenizer_path.stat().st_size,
                "sha256": sha256_file(tokenizer_path),
                "relationship": "copied-upstream-model-asset",
            },
        },
    }
    (output / "assets-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
