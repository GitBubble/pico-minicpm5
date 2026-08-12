from __future__ import annotations

import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
BOARD = PROJECT / "release" / "perf" / "perf-board.json"
RENDERED = PROJECT / "release" / "perf" / "README.md"
FROZEN = PROJECT / "release" / "v0.1.0" / "release-manifest.json"
FROZEN_QUALIFICATION = PROJECT / "release" / "v0.1.0" / "qualification.json"
CONTEXTS = PROJECT / "release" / "contexts"
PROFILES = PROJECT / "app" / "profiles"

_LOCAL_LEAK = re.compile(r"/(?:Users|home|root)/|(?:\d{1,3}\.){3}\d{1,3}")


def _board() -> dict:
    return json.loads(BOARD.read_text(encoding="utf-8"))


def _entry(profile: str) -> dict:
    for record in _board()["contexts"]:
        if record["profile"] == profile:
            return record
    raise AssertionError(f"perf board has no {profile} entry")


def test_board_covers_every_profile_with_an_om_and_stays_portable() -> None:
    board = _board()
    assert board["schema"] == "pico.minicpm5.performance-board.v1"
    assert [record["profile"] for record in board["contexts"]] == [
        "ctx1024", "ctx4096", "ctx8192"]
    assert _LOCAL_LEAK.search(BOARD.read_text(encoding="utf-8")) is None
    assert _LOCAL_LEAK.search(RENDERED.read_text(encoding="utf-8")) is None
    assert board["not_measured"]


def test_ctx1024_phases_equal_the_frozen_qualification_record() -> None:
    frozen = json.loads(
        FROZEN_QUALIFICATION.read_text(encoding="utf-8"))["performance"]
    entry = _entry("ctx1024")
    phases = entry["steady_state_p50_ms"]
    for key, source in (
            ("prepare", "prepare"), ("transformer", "transformer"),
            ("kv", "kv_scatter"), ("head", "head_execute"),
            ("argmax", "argmax"), ("total", "total")):
        assert phases[key] == frozen["phase_median_ms"][source], key
    assert entry["tokens_per_second"]["range"] == \
        frozen["merged_tokens_per_second_range"]
    assert entry["tokens_per_second"]["baseline_49_handle_range"] == \
        frozen["accepted_49handle_tokens_per_second_range"]
    assert entry["tokens_per_second"]["speedup_over_baseline"] == \
        frozen["conservative_throughput_speedup"]


def test_extended_contexts_match_their_gate_records() -> None:
    for profile in ("ctx4096", "ctx8192"):
        entry = _entry(profile)
        record = json.loads(
            (CONTEXTS / f"{profile}.qualification.json").read_text(
                encoding="utf-8"))
        performance = record["gates"]["performance"]
        # The gate record publishes one median scope; the board carries it
        # verbatim and reports its own phase-breakdown scope alongside.
        gate_reported = entry["gate_reported"]
        assert round(gate_reported["total_p50_ms"], 3) == \
            performance["total_p50_ms"], profile
        assert round(gate_reported["tokens_per_second"], 3) == \
            performance["tokens_per_second"], profile
        assert entry["steady_state_scope"], profile
        # The two scopes must agree closely; a real divergence means one of
        # them was transcribed from the wrong run.
        assert abs(entry["steady_state_p50_ms"]["total"]
                   - gate_reported["total_p50_ms"]) < 1.0, profile
        reported = entry["tokens_per_second"]["steady_state"]
        assert abs(reported - performance["tokens_per_second"]) < 0.01, profile
        assert entry["context"] == record["target"]["context"]
        assert entry["prefill_window"] == record["target"]["prefill_window"]
        assert entry["tail"]["position"] == record["target"]["context"] - 1
        assert entry["evidence"]["source_sha256"] == \
            record["evidence"]["source_sha256"], profile


def test_gate_binding_matches_profiles_records_and_frozen_hashes() -> None:
    board = _board()
    artifacts = json.loads(FROZEN.read_text(encoding="utf-8"))["artifacts"]
    frozen_prefill = artifacts["models/prefill.om"]["sha256"]
    bindings = {entry["profile"]: entry
                for entry in board["gate_binding"]["entries"]}
    assert set(bindings) == {"ctx1024", "ctx4096", "ctx8192"}

    assert bindings["ctx1024"]["decode_om_sha256"] == \
        artifacts["models/decode.om"]["sha256"]
    for profile, binding in bindings.items():
        assert binding["prefill_om_sha256"] == frozen_prefill, profile
        entry = _entry(profile)
        assert binding["gate_record"] == entry["gate_record"], profile
        status = json.loads(
            (PROFILES / f"{profile}.json").read_text(
                encoding="utf-8"))["status"]
        expected = "qualified" if binding["gate_verdict"] == "PASS" \
            else "pending"
        assert status == expected == entry["status"], profile

    for profile in ("ctx4096", "ctx8192"):
        record = json.loads(
            (CONTEXTS / f"{profile}.qualification.json").read_text(
                encoding="utf-8"))
        assert bindings[profile]["decode_om_sha256"] == \
            record["contract"]["decode"]["sha256"], profile
        assert bindings[profile]["gate_verdict"] == \
            record["verdict"]["overall"], profile
        assert bindings[profile]["minimum_public_output"] == \
            record["numeric_gate"]["minimum_public_output"], profile


def test_phase_sums_and_mode_table_are_internally_consistent() -> None:
    board = _board()
    for record in board["contexts"]:
        phases = record["steady_state_p50_ms"]
        parts = sum(phases[key] for key in
                    ("prepare", "transformer", "kv", "head", "argmax"))
        # Medians are taken per phase, so the parts need not sum exactly to
        # the median total, but a large gap would mean a transcription error.
        assert abs(parts - phases["total"]) < 2.0, record["profile"]

    ab = board["executor_mode_ab"]
    modes = {mode["mode"]: mode for mode in ab["modes"]}
    assert set(modes) == {"cached", "nocache", "per-model-mixed",
                          "zero-once", "promptskip+zero-once"}
    assert modes["cached"]["verdict"] == "baseline"
    assert modes["nocache"]["verdict"] == "rejected"
    assert modes["zero-once"]["verdict"] == "adopted"
    baseline = modes["cached"]["total_p50_ms"]
    assert modes["nocache"]["total_p50_ms"] > baseline
    for name in ("per-model-mixed", "zero-once", "promptskip+zero-once"):
        assert modes[name]["total_p50_ms"] < baseline, name
    for mode in ab["modes"]:
        derived = 1000.0 / mode["total_p50_ms"]
        assert abs(derived - mode["tokens_per_second"]) < 0.02, mode["mode"]

    # The adopted A/B mode is the ctx8192 steady state reported above.
    assert modes["zero-once"]["total_p50_ms"] == \
        _entry("ctx8192")["steady_state_p50_ms"]["total"]
    assert ab["context"] == 8192


def test_mixed_contract_shows_one_shared_prefill_handle_cost() -> None:
    ctx4096 = _entry("ctx4096")["position_zero_ms"]
    ctx8192 = _entry("ctx8192")["position_zero_ms"]
    # Both extended contexts bootstrap on the same ctx1024 prefill binary, so
    # their position-zero transformer times must agree closely and both must
    # be cheaper than their own steady decode step.
    assert abs(ctx4096["transformer"] - ctx8192["transformer"]) < 1.0
    assert ctx4096["transformer"] < \
        _entry("ctx4096")["steady_state_p50_ms"]["transformer"]
    assert ctx8192["transformer"] < \
        _entry("ctx8192")["steady_state_p50_ms"]["transformer"]
    assert ctx8192["with_prompt_head_skip"] < ctx8192["total"]
