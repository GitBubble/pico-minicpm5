from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"
SOURCE = APP_SRC / "minicpm_prefill_runtime.py"


def _module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_prefill_runtime_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_activation_manifest_and_live_mmz_are_plumbed_exactly(
        monkeypatch, tmp_path: Path) -> None:
    runtime = _module()
    manifest = tmp_path / "activation.json"
    root = tmp_path / "deploy"
    calls = []

    class Activation:
        enabled_widths = (128, 16, 1)
        disabled = {"S32": "not admitted"}

        def to_dict(self):
            return {"schema": "fake.activation.v4", "enabled_widths": [128, 16, 1]}

    def load_activation(path, **kwargs):
        calls.append((path, kwargs))
        return Activation()

    monkeypatch.setattr(
        runtime.activation_contract, "load_activation", load_activation)
    registry = runtime.load_runtime_registry(
        activation_manifest=manifest,
        deployment_root=root,
        context=4096,
        available_bytes=900,
        base_resident_bytes=400,
        reserve_bytes=200,
    )

    assert calls == [(manifest, {
        "deployment_root": root,
        "context": 4096,
        "available_bytes": 900,
        "base_resident_bytes": 400,
        "reserve_bytes": 200,
    })]
    assert registry.qualified_widths == (128, 16, 1)
    assert registry.enabled_widths == (1,)
    assert registry.unavailable["S16"] == (
        "release-qualified but no wide-handle executor is registered")
    assert registry.to_dict()["status"] == \
        "qualified-wide-handler-unavailable"
    assert registry.plan(0, 33).to_dict()["counts"] == {
        "S128": 0, "S32": 0, "S16": 0, "S1": 33}


def test_no_manifest_is_strict_s1_and_mmz_tuple_is_all_or_nothing(
        tmp_path: Path) -> None:
    runtime = _module()
    registry = runtime.load_runtime_registry(
        activation_manifest=None,
        deployment_root=tmp_path,
        context=1024,
        available_bytes=None,
        base_resident_bytes=None,
        reserve_bytes=None,
    )

    assert registry.qualified_widths == (1,)
    assert registry.enabled_widths == (1,)
    assert registry.to_dict()["status"] == "strict-s1-default"
    with pytest.raises(runtime.PrefillRuntimeError, match="require an activation"):
        runtime.load_runtime_registry(
            activation_manifest=None,
            deployment_root=tmp_path,
            context=1024,
            available_bytes=100,
            base_resident_bytes=None,
            reserve_bytes=None,
        )
    with pytest.raises(runtime.PrefillRuntimeError, match="requires live"):
        runtime.load_runtime_registry(
            activation_manifest=tmp_path / "activation.json",
            deployment_root=tmp_path,
            context=1024,
            available_bytes=100,
            base_resident_bytes=50,
            reserve_bytes=None,
        )


def test_invalid_top_level_strict_s1_is_not_downgraded_to_fallback(
        monkeypatch, tmp_path: Path) -> None:
    runtime = _module()

    def reject(*_args, **_kwargs):
        raise runtime.activation_contract.PrefillActivationError(
            "strict-S1 qualification SHA-256 mismatch")

    monkeypatch.setattr(runtime.activation_contract, "load_activation", reject)
    with pytest.raises(
            runtime.activation_contract.PrefillActivationError,
            match="strict-S1 qualification"):
        runtime.load_runtime_registry(
            activation_manifest=tmp_path / "activation.json",
            deployment_root=tmp_path,
            context=1024,
            available_bytes=1000,
            base_resident_bytes=500,
            reserve_bytes=100,
        )


def test_registry_cannot_label_wide_without_a_handler() -> None:
    runtime = _module()
    with pytest.raises(runtime.PrefillRuntimeError, match="runtime handler"):
        runtime.PrefillRuntimeRegistry(
            context=1024,
            activation_manifest=Path("activation.json"),
            activation_report={"schema": "fake"},
            qualified_widths=(16, 1),
            enabled_widths=(16, 1),
            handler_widths=(),
            unavailable={},
        )
