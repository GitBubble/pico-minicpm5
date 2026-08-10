from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest

from pico_minicpm5.onnx.common import save_model


def _model(fill: float):
    f32 = onnx.TensorProto.FLOAT
    x = helper.make_tensor_value_info("x", f32, (2048,))
    y = helper.make_tensor_value_info("y", f32, (2048,))
    bias = numpy_helper.from_array(np.full(2048, fill, np.float32), "bias")
    graph = helper.make_graph(
        [helper.make_node("Add", ("x", "bias"), ("y",))], "fixture", [x], [y], [bias]
    )
    return helper.make_model(graph, opset_imports=(helper.make_opsetid("", 13),))


def test_external_data_is_one_file_at_offset_zero(tmp_path: Path) -> None:
    output = tmp_path / "model" / "model.onnx"
    files = save_model(_model(1.0), output, external_data=True)
    assert [path.name for path in files] == ["bias"]
    stored = onnx.load(str(output), load_external_data=False)
    metadata = {entry.key: entry.value for entry in stored.graph.initializer[0].external_data}
    assert metadata["location"] == "bias"
    assert int(metadata.get("offset", "0")) == 0


def test_external_data_rejects_shared_output_directory(tmp_path: Path) -> None:
    directory = tmp_path / "shared"
    save_model(_model(1.0), directory / "decode.onnx", external_data=True)
    with pytest.raises(FileExistsError, match="dedicated empty"):
        save_model(_model(2.0), directory / "prefill.onnx", external_data=True)
