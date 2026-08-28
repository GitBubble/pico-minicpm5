from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"
PROFILES = PROJECT / "app" / "profiles"


def _module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_runtime_profile_test", APP_SRC / "minicpm_profile.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_builtin_profile_matrix_and_limits(tmp_path: Path) -> None:
    profiles = _module()
    loaded = {
        name: profiles.load_runtime_profile(name, PROFILES)
        for name in (
            "ctx128", "ctx1024", "ctx4096", "ctx8192", "ctx10240",
            "ctx16384")
    }

    assert [loaded[name].context for name in loaded] == [
        128, 1024, 4096, 8192, 10240, 16384]
    assert loaded["ctx128"].chat and not loaded["ctx128"].agent
    assert all(loaded[name].agent for name in (
        "ctx1024", "ctx4096", "ctx8192", "ctx10240", "ctx16384"))
    assert loaded["ctx1024"].status == "qualified"
    assert loaded["ctx4096"].status == "qualified"
    assert loaded["ctx8192"].status == "qualified"
    assert loaded["ctx10240"].status == "pending"
    # Qualified 2026-08-27: release/contexts/ctx16384.qualification.json
    # passes with overall PASS on the 2026-08-19 probrenorm-ranked build.
    assert loaded["ctx16384"].status == "qualified"
    assert loaded["ctx4096"].max_new_limit == 512
    assert loaded["ctx8192"].max_new_limit == 1024
    assert loaded["ctx10240"].max_new_limit == 1024
    assert loaded["ctx16384"].max_new_limit == 2048
    assert loaded["ctx1024"].kv_output_slots == (0, 1)
    assert loaded["ctx1024"].hidden_output_slot == 2
    assert loaded["ctx4096"].model_paths(tmp_path)["decode"] == (
        tmp_path / "models/ctx4096/decode.om")
    # Mixed prefill-window contract: the extended contexts bootstrap
    # position 0 with the qualified ctx1024 prefill binary.
    assert [loaded[name].prefill_window for name in loaded] == [
        128, 1024, 1024, 1024, 1024, 1024]
    for name in ("ctx4096", "ctx8192", "ctx10240", "ctx16384"):
        assert loaded[name].model_paths(tmp_path)["prefill"] == (
            tmp_path / "models/prefill.om")


def test_ctx128_agent_fails_before_pending_status() -> None:
    profiles = _module()
    ctx128 = profiles.load_runtime_profile("ctx128", PROFILES)

    with pytest.raises(profiles.ProfileError, match="ctx128 is chat-only"):
        ctx128.require_mode("agent")
    with pytest.raises(profiles.ProfileError, match="status is 'pending'"):
        ctx128.require_mode("chat")
    ctx128.require_mode("chat", allow_unqualified=True)


def test_pending_long_context_needs_explicit_development_override() -> None:
    profiles = _module()
    profile = profiles.load_runtime_profile("ctx10240", PROFILES)
    with pytest.raises(profiles.ProfileError,
                       match="allow-unqualified-profile"):
        profile.require_mode("agent")
    profile.require_mode("agent", allow_unqualified=True)


def test_qualified_ctx16384_needs_no_override() -> None:
    profiles = _module()
    profiles.load_runtime_profile("ctx16384", PROFILES).require_mode("agent")


def test_qualified_ctx8192_needs_no_override() -> None:
    profiles = _module()
    profiles.load_runtime_profile("ctx8192", PROFILES).require_mode("agent")


def test_qualified_mixed_contract_profile_needs_no_override() -> None:
    profiles = _module()
    ctx4096 = profiles.load_runtime_profile("ctx4096", PROFILES)

    ctx4096.require_mode("agent")
    assert ctx4096.prefill_window == 1024


def test_profile_rejects_prefill_window_contract_drift(tmp_path: Path) -> None:
    profiles = _module()

    source = json.loads((PROFILES / "ctx1024.json").read_text())
    source["context"]["prefill_window"] = 512
    malformed = tmp_path / "unknown-window.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="prefill_window"):
        profiles.load_runtime_profile(str(malformed), PROFILES)

    source = json.loads((PROFILES / "ctx1024.json").read_text())
    source["context"]["prefill_window"] = 4096
    malformed = tmp_path / "window-over-capacity.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="prefill_window"):
        profiles.load_runtime_profile(str(malformed), PROFILES)

    source = json.loads((PROFILES / "ctx1024.json").read_text())
    del source["context"]["prefill_window"]
    malformed = tmp_path / "missing-window.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="prefill_window"):
        profiles.load_runtime_profile(str(malformed), PROFILES)


def test_profile_rejects_context_and_path_contract_drift(tmp_path: Path) -> None:
    profiles = _module()
    source = json.loads((PROFILES / "ctx1024.json").read_text())
    source["context"]["past_length"] = 1000
    malformed = tmp_path / "bad-past.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="capacity - 1"):
        profiles.load_runtime_profile(str(malformed), PROFILES)

    source = json.loads((PROFILES / "ctx1024.json").read_text())
    source["models"]["decode"] = "../outside.om"
    malformed = tmp_path / "bad-path.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="deployment root"):
        profiles.load_runtime_profile(str(malformed), PROFILES)


def test_profile_rejects_boolean_numeric_gate(tmp_path: Path) -> None:
    profiles = _module()
    source = json.loads((PROFILES / "ctx1024.json").read_text())
    source["numeric_gate"]["minimum_cosine_exclusive"] = True
    malformed = tmp_path / "boolean-gate.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(profiles.ProfileError, match="real number"):
        profiles.load_runtime_profile(str(malformed), PROFILES)


def test_profile_rejects_ambiguous_transformer_output_slots(tmp_path: Path) -> None:
    profiles = _module()
    source = json.loads((PROFILES / "ctx1024.json").read_text())
    source["transformer_outputs"] = {"k": 0, "v": 0, "hidden": 2}
    malformed = tmp_path / "duplicate-output-slot.json"
    malformed.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(profiles.ProfileError, match="distinct and cover"):
        profiles.load_runtime_profile(str(malformed), PROFILES)
