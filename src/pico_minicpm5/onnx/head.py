"""Build the accepted dense MiniCPM5 final RMSNorm + vocabulary head graph."""
from __future__ import annotations

from pathlib import Path

from ..contract import OFFICIAL_CONTRACT, WEIGHT_SHARD, verify_checkpoint
from .common import append_conv, check_with_custom_ops, fold_projection, require_onnx, save_model


def build_head_model(model_dir: Path):
    np, onnx, helper, numpy_helper = require_onnx()
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install pico-minicpm5[onnx]") from exc
    c = OFFICIAL_CONTRACT
    with safe_open(str(model_dir / WEIGHT_SHARD), framework="np") as handle:
        gamma = np.asarray(handle.get_tensor("model.norm.weight"), np.float32)
        head = np.asarray(handle.get_tensor("lm_head.weight"), np.float32)
    nodes: list = [
        helper.make_node(
            "Add", ("hidden", "residual"), ("residual_hidden",), name="residual_add"
        )
    ]
    initializers: list = [
        numpy_helper.from_array(
            np.ones((1, c.hidden_size, 1, 1), np.float32), name="identity_gamma"
        )
    ]
    nodes.append(
        helper.make_node(
            "ExtendRMSNorm",
            ("residual_hidden", "identity_gamma"),
            ("normed_raw",),
            name="final_rmsnorm",
            epsilon=c.rms_norm_eps,
        )
    )
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1, c.hidden_size, 1, 1], np.int64), name="anchor_shape"
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ("normed_raw", "anchor_shape"), ("normed_shaped",), name="final_anchor"
        )
    )
    bound = float(np.sqrt(c.hidden_size))
    initializers.extend(
        (
            numpy_helper.from_array(np.asarray(-bound, np.float32), name="final_clip_lo"),
            numpy_helper.from_array(np.asarray(bound, np.float32), name="final_clip_hi"),
        )
    )
    nodes.append(
        helper.make_node(
            "Clip",
            ("normed_shaped", "final_clip_lo", "final_clip_hi"),
            ("normed",),
            name="final_clip",
        )
    )
    append_conv(
        nodes,
        initializers,
        "lm_head",
        "normed",
        "physical_logits",
        fold_projection(head, gamma),
    )
    nodes.append(
        helper.make_node(
            "Flatten", ("physical_logits",), ("dense_logits",), name="logits_flatten", axis=1
        )
    )
    f32 = onnx.TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        "minicpm5_native_head",
        [
            helper.make_tensor_value_info("hidden", f32, (1, c.hidden_size, 1, 1)),
            helper.make_tensor_value_info("residual", f32, (1, c.hidden_size, 1, 1)),
        ],
        [helper.make_tensor_value_info("dense_logits", f32, (1, c.vocab_size))],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=(helper.make_opsetid("", 13),))
    model.ir_version = 7
    check_with_custom_ops(model)
    return model


def export_head(*, model_dir: Path, output: Path, external_data: bool = True) -> dict:
    verify_checkpoint(model_dir)
    model = build_head_model(model_dir)
    external_files = save_model(model, output, external_data=external_data)
    return {
        "schema": "pico.minicpm5.head-onnx.v1",
        "onnx": str(output.resolve()),
        "nodes": len(model.graph.node),
        "graph_inputs": len(model.graph.input),
        "graph_outputs": len(model.graph.output),
        "external_data": [path.name for path in external_files],
        "onnx_bytes_total": output.stat().st_size
        + sum(path.stat().st_size for path in external_files),
    }
