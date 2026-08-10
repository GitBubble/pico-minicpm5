"""Compose per-layer ONNX graphs into one depth-N transformer graph."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..contract import OFFICIAL_CONTRACT
from .common import check_with_custom_ops, require_onnx, save_model

SHARED_INPUTS = ("attention_mask", "rope_r")
PER_LAYER_INPUTS = ("k_cache", "v_cache")


def _prefix_graph(model, index: int, hidden_source: str):
    _np, onnx, _helper, _numpy_helper = require_onnx()
    prefix = f"L{index}_"
    rename: dict[str, str] = {}

    def renamed(name: str) -> str:
        if not name:
            return ""
        if name == "hidden":
            return hidden_source
        if name in SHARED_INPUTS:
            return name
        rename.setdefault(name, prefix + name)
        return rename[name]

    graph = model.graph
    nodes = []
    for offset, node in enumerate(graph.node):
        clone = onnx.NodeProto()
        clone.CopyFrom(node)
        clone.name = prefix + (node.name or f"{node.op_type}_{offset}")
        for slot, value in enumerate(node.input):
            clone.input[slot] = renamed(value)
        for slot, value in enumerate(node.output):
            clone.output[slot] = renamed(value)
        nodes.append(clone)
    initializers = []
    for initializer in graph.initializer:
        clone = onnx.TensorProto()
        clone.CopyFrom(initializer)
        clone.name = renamed(initializer.name)
        initializers.append(clone)
    value_info = []
    for info in list(graph.value_info) + list(graph.output):
        clone = onnx.ValueInfoProto()
        clone.CopyFrom(info)
        clone.name = renamed(info.name)
        value_info.append(clone)
    return nodes, initializers, value_info


def compose_models(
    models: Sequence,
    *,
    context: int,
    hidden_size: int,
    kv_heads: int,
    head_dim: int,
    pack_input_kv: bool,
    pack_output_kv: bool,
):
    np, onnx, helper, numpy_helper = require_onnx()
    if not models:
        raise ValueError("at least one layer model is required")
    depth = len(models)
    past = context - 1
    all_nodes, all_initializers, all_value_info = [], [], []
    for index, model in enumerate(models):
        actual_inputs = {value.name for value in model.graph.input}
        required_inputs = {"hidden", "k_cache", "v_cache", *SHARED_INPUTS}
        actual_outputs = {value.name for value in model.graph.output}
        if not required_inputs.issubset(actual_inputs):
            raise ValueError(f"layer {index} input ABI drift: {sorted(actual_inputs)}")
        if not {"next_hidden", "k_cur", "v_cur"}.issubset(actual_outputs):
            raise ValueError(f"layer {index} output ABI drift: {sorted(actual_outputs)}")
        source = "hidden" if index == 0 else f"L{index - 1}_next_hidden"
        nodes, initializers, value_info = _prefix_graph(model, index, source)
        all_nodes.extend(nodes)
        all_initializers.extend(initializers)
        all_value_info.extend(value_info)

    f32 = onnx.TensorProto.FLOAT
    graph_inputs = [
        helper.make_tensor_value_info("hidden", f32, (1, hidden_size, 1, 1)),
        helper.make_tensor_value_info("attention_mask", f32, (1, 1, 1, context)),
        helper.make_tensor_value_info("rope_r", f32, (1, 1, head_dim, head_dim)),
    ]
    outputs = [
        helper.make_tensor_value_info(
            f"L{depth - 1}_next_hidden", f32, (1, hidden_size, 1, 1)
        )
    ]
    prologue, epilogue = [], []

    for role in ("k", "v"):
        if pack_input_kv:
            packed = f"{role}_cache_all"
            graph_inputs.append(
                helper.make_tensor_value_info(
                    packed, f32, (1, depth * kv_heads, past, head_dim)
                )
            )
            for index in range(depth):
                tag = f"slice_{role}{index}"
                for suffix, value in (
                    ("starts", index * kv_heads),
                    ("ends", (index + 1) * kv_heads),
                    ("axes", 1),
                ):
                    all_initializers.append(
                        numpy_helper.from_array(
                            np.asarray([value], np.int64), name=f"{tag}_{suffix}"
                        )
                    )
                prologue.append(
                    helper.make_node(
                        "Slice",
                        (packed, f"{tag}_starts", f"{tag}_ends", f"{tag}_axes"),
                        (f"L{index}_{role}_cache",),
                        name=tag,
                    )
                )
        else:
            for index in range(depth):
                graph_inputs.append(
                    helper.make_tensor_value_info(
                        f"L{index}_{role}_cache",
                        f32,
                        (1, kv_heads, past, head_dim),
                    )
                )

        if pack_output_kv:
            packed = f"{role}_cur_all"
            epilogue.append(
                helper.make_node(
                    "Concat",
                    tuple(f"L{index}_{role}_cur" for index in range(depth)),
                    (packed,),
                    name=f"concat_{role}_cur",
                    axis=1,
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    packed, f32, (1, depth * kv_heads, 1, head_dim)
                )
            )
        else:
            for index in range(depth):
                outputs.append(
                    helper.make_tensor_value_info(
                        f"L{index}_{role}_cur", f32, (1, kv_heads, 1, head_dim)
                    )
                )

    published = {value.name for value in outputs}
    all_value_info = [value for value in all_value_info if value.name not in published]
    graph = helper.make_graph(
        prologue + all_nodes + epilogue,
        f"minicpm5_merge{depth}_ctx{context}",
        graph_inputs,
        outputs,
        all_initializers,
        value_info=all_value_info,
    )
    model = helper.make_model(graph, opset_imports=(helper.make_opsetid("", 13),))
    model.ir_version = 7
    check_with_custom_ops(model)
    return model


def compose_files(
    *,
    layer_paths: Sequence[Path],
    output: Path,
    family: str,
    context: int,
    pack_input_kv: bool = True,
    pack_output_kv: bool = True,
    external_data: bool = True,
) -> dict:
    _np, onnx, _helper, _numpy_helper = require_onnx()
    models = [onnx.load(str(path.resolve()), load_external_data=True) for path in layer_paths]
    c = OFFICIAL_CONTRACT
    model = compose_models(
        models,
        context=context,
        hidden_size=c.hidden_size,
        kv_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        pack_input_kv=pack_input_kv,
        pack_output_kv=pack_output_kv,
    )
    external_files = save_model(model, output, external_data=external_data)
    total_bytes = output.stat().st_size + sum(path.stat().st_size for path in external_files)
    report = {
        "schema": "pico.minicpm5.composed-onnx.v1",
        "family": family,
        "depth": len(layer_paths),
        "context": context,
        "pack_input_kv": pack_input_kv,
        "pack_output_kv": pack_output_kv,
        "graph_inputs": len(model.graph.input),
        "graph_outputs": len(model.graph.output),
        "nodes": len(model.graph.node),
        "onnx": str(output.resolve()),
        "onnx_bytes_total": total_bytes,
        "external_data": [path.name for path in external_files],
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
