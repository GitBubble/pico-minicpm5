from __future__ import annotations

from pathlib import Path

import onnx
from onnx import helper, numpy_helper
import numpy as np
import pytest

from pico_minicpm5.compiler.atc import AtcCompiler


def _external_model(path: Path, location: str) -> None:
    tensor = numpy_helper.from_array(np.ones(8, np.float32), name="weight")
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in (("location", location), ("offset", "0"), ("length", "32")):
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    graph = helper.make_graph([], "external", [], [], [tensor])
    onnx.save_model(helper.make_model(graph), str(path))


@pytest.mark.parametrize("location", ["../secret", "/private/secret", ""])
def test_external_locations_reject_path_escape(tmp_path: Path, location: str) -> None:
    model = tmp_path / "model.onnx"
    _external_model(model, location)
    with pytest.raises(ValueError, match="unsafe"):
        AtcCompiler._external_locations(model)


def test_external_locations_accept_nested_relative_path(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    _external_model(model, "weights/layer0.bin")
    assert AtcCompiler._external_locations(model) == ["weights/layer0.bin"]
