from __future__ import annotations

import json
from pathlib import Path

import pytest

from pico_minicpm5.onnx.layer import load_clip_preset

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA = ROOT / "src" / "pico_minicpm5" / "data"


@pytest.mark.parametrize("family", ["decode", "prefill"])
def test_all_24_layer_clip_contracts_are_present(family: str) -> None:
    path = ROOT / "configs" / "calibration" / f"{family}-clips.json"
    packaged = PACKAGE_DATA / f"{family}-clips.json"
    assert path.read_bytes() == packaged.read_bytes()
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["family"] == family
    assert set(value["layers"]) == {str(index) for index in range(24)}
    for layer in range(24):
        bounds = load_clip_preset(path, family, layer)
        assert bounds
        assert all(bound > 0 for bound in bounds.values())
        assert {"k_cur", "v_cur"}.issubset(bounds)


@pytest.mark.parametrize("name", ["minicpm5-1b.json", "ss928-ctx1024.json"])
def test_packaged_contract_configs_match_source(name: str) -> None:
    assert (ROOT / "configs" / name).read_bytes() == (PACKAGE_DATA / name).read_bytes()
