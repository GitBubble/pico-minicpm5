from __future__ import annotations

from dataclasses import asdict

import pytest

from pico_minicpm5.contract import (
    OFFICIAL_CONTRACT,
    contract_from_hf_config,
    expected_weight_shapes,
)


def _official_config() -> dict:
    values = asdict(OFFICIAL_CONTRACT)
    return {
        "architectures": [values.pop("architecture")],
        "model_type": values.pop("model_type"),
        **values,
        "rope_scaling": None,
    }


def test_official_config_round_trips() -> None:
    assert contract_from_hf_config(_official_config()) == OFFICIAL_CONTRACT
    shapes = expected_weight_shapes()
    assert len(shapes) == 3 + 24 * 9
    assert shapes["model.layers.23.self_attn.q_proj.weight"] == (2048, 1536)


def test_geometry_drift_fails_closed() -> None:
    config = _official_config()
    config["num_hidden_layers"] = 23
    with pytest.raises(ValueError, match="config drift"):
        contract_from_hf_config(config)


def test_rope_scaling_drift_fails_closed() -> None:
    config = _official_config()
    config["rope_scaling"] = {"type": "linear", "factor": 2.0}
    with pytest.raises(ValueError, match="rope_scaling"):
        contract_from_hf_config(config)
