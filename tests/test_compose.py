from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper

from pico_minicpm5.onnx.compose import compose_models


def _tiny_layer(seed: float):
    f32 = onnx.TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("hidden", f32, (1, 4, 1, 1)),
        helper.make_tensor_value_info("attention_mask", f32, (1, 1, 1, 4)),
        helper.make_tensor_value_info("rope_r", f32, (1, 1, 2, 2)),
        helper.make_tensor_value_info("k_cache", f32, (1, 1, 3, 2)),
        helper.make_tensor_value_info("v_cache", f32, (1, 1, 3, 2)),
    ]
    outputs = [
        helper.make_tensor_value_info("next_hidden", f32, (1, 4, 1, 1)),
        helper.make_tensor_value_info("k_cur", f32, (1, 1, 1, 2)),
        helper.make_tensor_value_info("v_cur", f32, (1, 1, 1, 2)),
    ]
    nodes = [
        helper.make_node("Add", ("hidden", "bias"), ("next_hidden",), name="hidden_add"),
        helper.make_node("Identity", ("k_cache",), ("k_cur",), name="k_identity"),
        helper.make_node("Identity", ("v_cache",), ("v_cur",), name="v_identity"),
    ]
    initializers = [numpy_helper.from_array(np.full((1, 4, 1, 1), seed, np.float32), "bias")]
    graph = helper.make_graph(nodes, "tiny_layer", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=(helper.make_opsetid("", 13),))
    model.ir_version = 7
    return model


def test_two_layer_packed_composition_contract() -> None:
    model = compose_models(
        [_tiny_layer(1.0), _tiny_layer(2.0)],
        context=4,
        hidden_size=4,
        kv_heads=1,
        head_dim=2,
        pack_input_kv=True,
        pack_output_kv=True,
    )
    assert [value.name for value in model.graph.input] == [
        "hidden", "attention_mask", "rope_r", "k_cache_all", "v_cache_all"
    ]
    assert [value.name for value in model.graph.output] == [
        "L1_next_hidden", "k_cur_all", "v_cur_all"
    ]
    assert sum(node.op_type == "Slice" for node in model.graph.node) == 4
    assert sum(node.op_type == "Concat" for node in model.graph.node) == 2
    layer1 = next(node for node in model.graph.node if node.name == "L1_hidden_add")
    assert layer1.input[0] == "L0_next_hidden"
    initializer_names = [value.name for value in model.graph.initializer]
    assert "L0_bias" in initializer_names and "L1_bias" in initializer_names
    assert len(initializer_names) == len(set(initializer_names))
