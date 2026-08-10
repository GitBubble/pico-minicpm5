"""Build one qualified MiniCPM5 decoder layer from official weights.

The graph mirrors the accepted family: ExtendRMSNorm with a shape anchor and
provable clip, gamma folded into following projections, matrix-form RoPE,
in-graph KV append, batched GQA, and per-family calibration clips.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..contract import OFFICIAL_CONTRACT, verify_checkpoint
from .common import (
    append_conv,
    check_with_custom_ops,
    fold_projection,
    load_layer_weights,
    require_onnx,
    save_model,
)


def load_clip_preset(path: Path, family: str, layer: int) -> dict[str, float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("family") != family:
        raise ValueError(f"clip preset family mismatch: {value.get('family')} != {family}")
    layers = value.get("layers", {})
    if str(layer) not in layers:
        raise ValueError(f"clip preset has no layer {layer}")
    bounds = layers[str(layer)].get("bounds")
    if not isinstance(bounds, dict) or not bounds:
        raise ValueError(f"clip preset layer {layer} has no bounds")
    return {str(name): float(bound) for name, bound in bounds.items()}


def _extend_rmsnorm(nodes, initializers, *, tag: str, source: str, target: str, scale: float):
    np, _onnx, helper, numpy_helper = require_onnx()
    c = OFFICIAL_CONTRACT
    norm_source = source
    if scale != 1.0:
        gain = f"{tag}_pre_gain"
        norm_source = f"{tag}_pre_out"
        initializers.append(
            numpy_helper.from_array(np.asarray([scale], np.float32), name=gain)
        )
        nodes.append(helper.make_node("Mul", (source, gain), (norm_source,), name=f"{tag}_pre"))
    raw, shaped = f"{tag}_raw", f"{tag}_shaped"
    nodes.append(
        helper.make_node(
            "ExtendRMSNorm",
            (norm_source, "identity_gamma"),
            (raw,),
            name=tag,
            epsilon=c.rms_norm_eps,
        )
    )
    shape_name = f"{tag}_anchor_shape"
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1, c.hidden_size, 1, 1], np.int64), name=shape_name
        )
    )
    nodes.append(helper.make_node("Reshape", (raw, shape_name), (shaped,), name=f"{tag}_anchor"))
    bound = float(np.sqrt(c.hidden_size))
    lo, hi = f"{tag}_clip_lo", f"{tag}_clip_hi"
    initializers.extend(
        (
            numpy_helper.from_array(np.asarray(-bound, np.float32), name=lo),
            numpy_helper.from_array(np.asarray(bound, np.float32), name=hi),
        )
    )
    nodes.append(helper.make_node("Clip", (shaped, lo, hi), (target,), name=f"{tag}_clip"))


def _insert_clip(graph, produced: str, bound: float) -> None:
    np, _onnx, helper, numpy_helper = require_onnx()
    consumers = [
        (node, slot)
        for node in graph.node
        for slot, value in enumerate(node.input)
        if value == produced
    ]
    if not consumers:
        raise ValueError(f"clip target {produced!r} has no graph consumer")
    lo, hi = f"{produced}_range_lo", f"{produced}_range_hi"
    graph.initializer.extend(
        (
            numpy_helper.from_array(np.asarray(-bound, np.float32), name=lo),
            numpy_helper.from_array(np.asarray(bound, np.float32), name=hi),
        )
    )
    target = f"{produced}_range_clipped"
    clip = helper.make_node("Clip", (produced, lo, hi), (target,), name=f"{produced}_range_clip")
    first = min(next(index for index, candidate in enumerate(graph.node) if candidate is node)
                for node, _slot in consumers)
    for node, slot in consumers:
        node.input[slot] = target
    graph.node.insert(first, clip)


def build_layer_model(
    *,
    model_dir: Path,
    layer: int,
    context: int,
    clips: Mapping[str, float],
):
    np, onnx, helper, numpy_helper = require_onnx()
    c = OFFICIAL_CONTRACT
    if not 2 <= context <= c.max_position_embeddings:
        raise ValueError("context must be in [2,max_position_embeddings]")
    weights = load_layer_weights(model_dir, layer)
    nodes: list = []
    initializers: list = [
        numpy_helper.from_array(
            np.ones((1, c.hidden_size, 1, 1), np.float32), name="identity_gamma"
        )
    ]
    scale = 32.0 if layer == 0 else 1.0
    _extend_rmsnorm(
        nodes,
        initializers,
        tag="in_rmsnorm",
        source="hidden",
        target="normed",
        scale=scale,
    )
    for role, name, target in (
        ("q", "q_proj", "q_flat"),
        ("k", "k_proj", "k_flat"),
        ("v", "v_proj", "v_flat"),
    ):
        append_conv(
            nodes,
            initializers,
            name,
            "normed",
            target,
            fold_projection(weights[role], weights["in_norm"]),
        )

    def rope(tag: str, source: str, heads: int, target: str) -> None:
        shape = f"{tag}_shape"
        initializers.append(
            numpy_helper.from_array(
                np.asarray([1, heads, 1, c.head_dim], np.int64), name=shape
            )
        )
        nodes.append(helper.make_node("Reshape", (source, shape), (f"{tag}_h",), name=f"{tag}_reshape"))
        nodes.append(helper.make_node("MatMul", (f"{tag}_h", "rope_r"), (target,), name=f"{tag}_rope"))

    rope("q", "q_flat", c.num_attention_heads, "q_rope")
    rope("k", "k_flat", c.num_key_value_heads, "k_cur")
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1, c.num_key_value_heads, 1, c.head_dim], np.int64),
            name="v_shape",
        )
    )
    nodes.append(helper.make_node("Reshape", ("v_flat", "v_shape"), ("v_cur",), name="reshape_v"))
    nodes.append(helper.make_node("Concat", ("k_cache", "k_cur"), ("full_k",), name="k_append", axis=2))
    nodes.append(helper.make_node("Concat", ("v_cache", "v_cur"), ("full_v",), name="v_append", axis=2))

    groups = c.num_attention_heads // c.num_key_value_heads
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1, c.num_key_value_heads, groups, c.head_dim], np.int64),
            name="q_group_shape",
        )
    )
    nodes.append(helper.make_node("Reshape", ("q_rope", "q_group_shape"), ("q_grouped",), name="reshape_q"))
    nodes.append(helper.make_node("Transpose", ("full_k",), ("kt",), name="kt", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", ("q_grouped", "kt"), ("scores",), name="qk"))
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1.0 / np.sqrt(c.head_dim)], np.float32), name="attn_scale"
        )
    )
    nodes.append(helper.make_node("Mul", ("scores", "attn_scale"), ("scaled",), name="scale"))
    nodes.append(helper.make_node("Add", ("scaled", "attention_mask"), ("masked",), name="mask_add"))
    nodes.append(helper.make_node("Softmax", ("masked",), ("probability",), name="softmax", axis=-1))
    nodes.append(helper.make_node("MatMul", ("probability", "full_v"), ("context_grouped",), name="pv"))
    initializers.append(
        numpy_helper.from_array(
            np.asarray([1, c.query_width, 1, 1], np.int64), name="context_shape"
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ("context_grouped", "context_shape"), ("context_flat",), name="reshape_context"
        )
    )
    append_conv(nodes, initializers, "o_proj", "context_flat", "o_out", weights["o"])
    nodes.append(helper.make_node("Add", ("hidden", "o_out"), ("attn_residual",), name="attn_residual"))
    _extend_rmsnorm(
        nodes,
        initializers,
        tag="post_rmsnorm",
        source="attn_residual",
        target="post_normed",
        scale=scale,
    )
    for role, name, target in (
        ("gate", "gate_proj", "gate_out"),
        ("up", "up_proj", "up_out"),
    ):
        append_conv(
            nodes,
            initializers,
            name,
            "post_normed",
            target,
            fold_projection(weights[role], weights["post_norm"]),
        )
    nodes.append(helper.make_node("Sigmoid", ("gate_out",), ("gate_sig",), name="swish_sigmoid"))
    nodes.append(helper.make_node("Mul", ("gate_out", "gate_sig"), ("swish",), name="swish"))
    nodes.append(helper.make_node("Mul", ("swish", "up_out"), ("swiglu",), name="swiglu"))
    append_conv(nodes, initializers, "down_proj", "swiglu", "down_out", weights["down"])
    nodes.append(helper.make_node("Add", ("attn_residual", "down_out"), ("next_hidden",), name="residual_add"))

    f32 = onnx.TensorProto.FLOAT
    past = context - 1
    graph = helper.make_graph(
        nodes,
        f"minicpm5_layer{layer}_ctx{context}",
        [
            helper.make_tensor_value_info("hidden", f32, (1, c.hidden_size, 1, 1)),
            helper.make_tensor_value_info("k_cache", f32, (1, c.num_key_value_heads, past, c.head_dim)),
            helper.make_tensor_value_info("v_cache", f32, (1, c.num_key_value_heads, past, c.head_dim)),
            helper.make_tensor_value_info("attention_mask", f32, (1, 1, 1, context)),
            helper.make_tensor_value_info("rope_r", f32, (1, 1, c.head_dim, c.head_dim)),
        ],
        [
            helper.make_tensor_value_info("next_hidden", f32, (1, c.hidden_size, 1, 1)),
            helper.make_tensor_value_info("k_cur", f32, (1, c.num_key_value_heads, 1, c.head_dim)),
            helper.make_tensor_value_info("v_cur", f32, (1, c.num_key_value_heads, 1, c.head_dim)),
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=(helper.make_opsetid("", 13),))
    model.ir_version = 7
    for name, bound in clips.items():
        # sqrt(H) is already proven by the anchored RMSNorm clip. Repeating the
        # exact same bound is redundant; tighter family bounds remain useful.
        if name in ("normed", "post_normed") and abs(bound - np.sqrt(c.hidden_size)) < 1e-5:
            continue
        _insert_clip(model.graph, name, float(bound))
    check_with_custom_ops(model)
    return model


def export_layer(
    *,
    model_dir: Path,
    layer: int,
    family: str,
    context: int,
    preset: Path,
    output: Path,
) -> dict:
    verify_checkpoint(model_dir)
    clips = load_clip_preset(preset, family, layer)
    model = build_layer_model(
        model_dir=model_dir, layer=layer, context=context, clips=clips
    )
    save_model(model, output, external_data=False)
    return {
        "schema": "pico.minicpm5.layer-onnx.v1",
        "family": family,
        "layer": layer,
        "context": context,
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "nodes": len(model.graph.node),
        "clips": clips,
    }
