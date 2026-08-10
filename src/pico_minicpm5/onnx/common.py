from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contract import OFFICIAL_CONTRACT, WEIGHT_SHARD


def require_onnx():
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install pico-minicpm5[onnx]") from exc
    return np, onnx, helper, numpy_helper


def load_layer_weights(model_dir: Path, layer: int) -> dict[str, Any]:
    try:
        import numpy as np
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("install pico-minicpm5[onnx]") from exc
    if not 0 <= layer < OFFICIAL_CONTRACT.num_hidden_layers:
        raise ValueError(f"layer must be in [0,{OFFICIAL_CONTRACT.num_hidden_layers - 1}]")
    keys = {
        "in_norm": f"model.layers.{layer}.input_layernorm.weight",
        "q": f"model.layers.{layer}.self_attn.q_proj.weight",
        "k": f"model.layers.{layer}.self_attn.k_proj.weight",
        "v": f"model.layers.{layer}.self_attn.v_proj.weight",
        "o": f"model.layers.{layer}.self_attn.o_proj.weight",
        "post_norm": f"model.layers.{layer}.post_attention_layernorm.weight",
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up": f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }
    shard = model_dir / WEIGHT_SHARD
    result: dict[str, Any] = {}
    with safe_open(str(shard), framework="np") as handle:
        available = set(handle.keys())
        for role, symbol in keys.items():
            if symbol not in available:
                raise KeyError(f"checkpoint is missing {symbol}")
            result[role] = np.ascontiguousarray(handle.get_tensor(symbol), dtype=np.float32)
    return result


def fold_projection(weight, gamma):
    import numpy as np

    return np.ascontiguousarray(weight * gamma.reshape(1, -1), dtype=np.float32)


def append_conv(nodes, initializers, name: str, source: str, target: str, weight) -> None:
    np, _onnx, helper, numpy_helper = require_onnx()
    initializers.append(
        numpy_helper.from_array(
            np.ascontiguousarray(weight[:, :, None, None], dtype=np.float32),
            name=f"{name}_w",
        )
    )
    nodes.append(
        helper.make_node(
            "Conv", (source, f"{name}_w"), (target,), name=name, kernel_shape=(1, 1)
        )
    )


def check_with_custom_ops(model) -> None:
    """Run the ONNX checker after replacing known custom ops with Identity."""
    _np, onnx, _helper, _numpy_helper = require_onnx()
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    for node in probe.graph.node:
        if node.op_type == "ExtendRMSNorm":
            node.op_type = "Identity"
            del node.input[1:]
            del node.attribute[:]
    onnx.checker.check_model(probe)


def save_model(model, path: Path, *, external_data: bool) -> list[Path]:
    _np, onnx, _helper, _numpy_helper = require_onnx()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not external_data:
        if path.exists():
            raise FileExistsError(path)
        onnx.save_model(model, str(path))
        return []
    existing = list(path.parent.iterdir())
    if existing:
        raise FileExistsError(
            "external-data ONNX requires a dedicated empty output directory; "
            f"found {[entry.name for entry in existing[:4]]} in {path.parent}"
        )
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=False,
        size_threshold=4096,
        convert_attribute=False,
    )
    stored = onnx.load(str(path), load_external_data=False)
    locations: list[str] = []
    for initializer in stored.graph.initializer:
        metadata = {entry.key: entry.value for entry in initializer.external_data}
        if "location" not in metadata:
            continue
        if int(metadata.get("offset", "0")) != 0:
            raise ValueError(f"external tensor {initializer.name} does not start at offset zero")
        location = Path(metadata["location"])
        if location.is_absolute() or ".." in location.parts:
            raise ValueError(f"unsafe external tensor location: {location}")
        locations.append(location.as_posix())
    if len(locations) != len(set(locations)):
        raise ValueError("external initializers must not share a data file")
    files = [path.parent / location for location in locations]
    missing = [entry for entry in files if not entry.is_file()]
    if missing:
        raise FileNotFoundError(f"missing external tensor files: {missing[:4]}")
    return sorted(files, key=lambda entry: entry.name)
