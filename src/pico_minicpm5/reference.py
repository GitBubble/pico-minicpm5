"""Capture per-position, per-layer float reference data from the checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .contract import OFFICIAL_CONTRACT, verify_checkpoint

DEFAULT_PROMPT_TOKENS = (0, 608, 4894, 304, 6918, 357, 13)


def _kv_pair(cache, layer: int):
    if hasattr(cache, "layers"):
        entry = cache.layers[layer]
        for key_name, value_name in (("keys", "values"), ("key_cache", "value_cache")):
            if hasattr(entry, key_name):
                return getattr(entry, key_name), getattr(entry, value_name)
    for key_name, value_name in (("key_cache", "value_cache"), ("keys", "values")):
        if hasattr(cache, key_name):
            return getattr(cache, key_name)[layer], getattr(cache, value_name)[layer]
    return cache[layer]


def capture_reference(
    *,
    model_dir: Path,
    output: Path,
    context: int = 1024,
    prompt_tokens: Sequence[int] = DEFAULT_PROMPT_TOKENS,
    dtype_name: str = "float64",
) -> dict:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install pico-minicpm5[reference]") from exc
    verify_checkpoint(model_dir)
    c = OFFICIAL_CONTRACT
    if not prompt_tokens:
        raise ValueError("at least one prompt token is required")
    if len(prompt_tokens) >= context:
        raise ValueError("reference prompt must be shorter than context")
    dtype = getattr(torch, dtype_name, None)
    if dtype_name not in ("float64", "float32") or dtype is None:
        raise ValueError("dtype must be float64 or float32")
    output.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=dtype, device_map="cpu", local_files_only=True
    ).eval()
    inner = model.model
    captured: dict[int, object] = {}
    embedded: dict[str, object] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, result):
            hidden = result[0] if isinstance(result, tuple) else result
            captured[index] = hidden.detach()[0, -1].clone()

        return hook

    for index, layer in enumerate(inner.layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    handles.append(
        inner.embed_tokens.register_forward_hook(
            lambda _module, _inputs, result: embedded.__setitem__(
                "value", result.detach()[0, -1].clone()
            )
        )
    )
    past = context - 1
    manifest = {
        "schema": "pico.minicpm5.reference.v1",
        "model_dir": str(model_dir.resolve()),
        "dtype": dtype_name,
        "context": context,
        "past": past,
        "prompt_token_ids": [int(value) for value in prompt_tokens],
        "positions": len(prompt_tokens),
        "layers": c.num_hidden_layers,
        "cache_layout": "(kv_heads,past,head_dim) FP16; unseen rows are zero",
        "boundaries": {},
    }
    cache = None
    try:
        for position, token in enumerate(prompt_tokens):
            position_dir = output / f"pos{position}"
            position_dir.mkdir(parents=True, exist_ok=True)
            for layer in range(c.num_hidden_layers):
                key, value = (None, None) if cache is None else _kv_pair(cache, layer)
                for tag, tensor in (("k", key), ("v", value)):
                    window = np.zeros(
                        (c.num_key_value_heads, past, c.head_dim), dtype=np.float16
                    )
                    if tensor is not None:
                        seen = int(tensor.shape[2])
                        if seen > past:
                            raise ValueError(f"position {position} exceeds ctx{context}")
                        window[:, :seen, :] = (
                            tensor[0].detach().to(torch.float32).cpu().numpy().astype(np.float16)
                        )
                    window.tofile(position_dir / f"{tag}_cache_{layer:02d}.f16.bin")
            ids = torch.tensor([[int(token)]], dtype=torch.long)
            with torch.no_grad():
                result = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            previous = embedded["value"]
            layer_report = {}
            for layer in range(c.num_hidden_layers):
                current = captured[layer]
                previous.detach().to(torch.float32).contiguous().cpu().numpy().tofile(
                    position_dir / f"layer_in_{layer:02d}.f32.bin"
                )
                current.detach().to(torch.float32).contiguous().cpu().numpy().tofile(
                    position_dir / f"layer_out_{layer:02d}.f32.bin"
                )
                layer_report[str(layer)] = {
                    "input_norm": float(previous.norm()),
                    "output_norm": float(current.norm()),
                }
                previous = current
            logits = result.logits[0, -1].to(torch.float64)
            logits.to(torch.float32).contiguous().cpu().numpy().tofile(
                position_dir / "logits.f32.bin"
            )
            top = torch.argsort(logits, descending=True)[:5]
            manifest["boundaries"][str(position)] = {
                "token_id": int(token),
                "open_slots": position + 1,
                "top5_token_ids": [int(value) for value in top],
                "top5_logits": [float(logits[value]) for value in top],
                "layers": layer_report,
            }
    finally:
        for handle in handles:
            handle.remove()
    (output / "reference.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
