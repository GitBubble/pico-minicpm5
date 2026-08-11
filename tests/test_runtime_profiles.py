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
        for name in ("ctx128", "ctx1024", "ctx4096", "ctx8192")
    }

    assert [loaded[name].context for name in loaded] == [128, 1024, 4096, 8192]
    assert loaded["ctx128"].chat and not loaded["ctx128"].agent
    assert all(loaded[name].agent for name in ("ctx1024", "ctx4096", "ctx8192"))
    assert loaded["ctx1024"].status == "qualified"
    assert loaded["ctx4096"].max_new_limit == 512
    assert loaded["ctx8192"].max_new_limit == 1024
    assert loaded["ctx4096"].model_paths(tmp_path)["decode"] == (
        tmp_path / "models/ctx4096/decode.om")


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
    ctx4096 = profiles.load_runtime_profile("ctx4096", PROFILES)

    with pytest.raises(profiles.ProfileError, match="allow-unqualified-profile"):
        ctx4096.require_mode("agent")
    ctx4096.require_mode("agent", allow_unqualified=True)


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
