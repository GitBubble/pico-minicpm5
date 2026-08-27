from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pico_minicpm5 import context_profiles

PROJECT = Path(__file__).resolve().parents[1]
CONTEXTS = PROJECT / "release" / "contexts"
PROFILES = PROJECT / "app" / "profiles"
MANIFEST = PROJECT / "release" / "v0.1.0" / "release-manifest.json"


def _record(name: str) -> dict:
    return json.loads(
        (CONTEXTS / f"{name}.qualification.json").read_text(encoding="utf-8"))


def _profile(name: str) -> dict:
    return json.loads(
        (PROFILES / f"{name}.json").read_text(encoding="utf-8"))


def test_checked_in_context_records_validate_and_split_verdicts() -> None:
    ctx4096 = context_profiles.validate_record(_record("ctx4096"))
    ctx8192 = context_profiles.validate_record(_record("ctx8192"))
    ctx16384 = context_profiles.validate_record(_record("ctx16384"))

    assert ctx4096["passes"] is True
    assert ctx4096["minimum_public_output"] == 0.9908199813
    assert ctx8192["passes"] is True
    assert ctx8192["overall"] == "PASS"
    assert ctx8192["minimum_public_output"] == 0.9860760661
    # ctx16384 is a candidate: the calibration is donor-zero-extended from
    # 8192 and no position-16383 capture exists, so it must not pass.
    assert ctx16384["passes"] is False
    assert ctx16384["overall"] == "CANDIDATE_CALIBRATION_NOT_NATIVE"
    assert ctx16384["minimum_public_output"] == 0.9981029868187341


def test_ctx16384_candidate_cannot_claim_numeric_pass_without_tail() -> None:
    record = _record("ctx16384")
    assert record["numeric_gate"]["board_tail_byte_exact_position"] is None
    assert record["verdict"]["public_output_numeric"] != "PASS"

    claimed = copy.deepcopy(record)
    claimed["verdict"]["public_output_numeric"] = "PASS"
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="last valid"):
        context_profiles.validate_record(claimed)

    tail_bound = copy.deepcopy(record)
    tail_bound["numeric_gate"]["board_tail_byte_exact_position"] = 16383
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="last valid"):
        context_profiles.validate_record(tail_bound)


def test_profile_status_matches_context_record_verdict() -> None:
    for name in ("ctx4096", "ctx8192", "ctx16384"):
        profile = _profile(name)
        summary = context_profiles.validate_record(_record(name))
        expected = "qualified" if summary["passes"] else "pending"
        assert profile["status"] == expected, name
        assert profile["context"]["prefill_window"] == \
            summary["prefill_window"]


def test_records_bind_frozen_v010_prefill_and_head_and_profile_paths() -> None:
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    for name in ("ctx4096", "ctx8192", "ctx16384"):
        record = _record(name)
        profile = _profile(name)
        contract = record["contract"]
        for role, manifest_key in (
                ("prefill", "models/prefill.om"),
                ("head", "models/head_flat.om")):
            frozen = artifacts[manifest_key]
            assert contract[role]["sha256"] == frozen["sha256"], (name, role)
            assert contract[role]["bytes"] == frozen["bytes"], (name, role)
            assert contract[role]["deployment_path"] == manifest_key
        assert contract["decode"]["deployment_path"] == \
            profile["models"]["decode"]
        assert contract["prefill"]["deployment_path"] == \
            profile["models"]["prefill"]


def test_validator_rejects_low_threshold_and_boundary_equality() -> None:
    record = _record("ctx4096")
    low = copy.deepcopy(record)
    low["numeric_gate"]["cosine_threshold_exclusive"] = 0.9
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="floor"):
        context_profiles.validate_record(low)

    equal = copy.deepcopy(record)
    equal["numeric_gate"]["public_outputs"][0]["next_hidden"] = 0.98
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="strictly above"):
        context_profiles.validate_record(equal)


def test_validator_rejects_verdict_and_portability_holes() -> None:
    record = _record("ctx8192")
    fake_pass = copy.deepcopy(record)
    fake_pass["verdict"]["eos"] = "FAIL"
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="every gate PASS"):
        context_profiles.validate_record(fake_pass)

    leaky = copy.deepcopy(_record("ctx4096"))
    # Split the literal the way release/source.py splits its own
    # denylist, so this test does not trip the source-archive guard.
    leaky["evidence"]["location"] = "/" + "Users/nobody/evidence"
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="leaks"):
        context_profiles.validate_record(leaky)

    wrong_minimum = copy.deepcopy(_record("ctx4096"))
    wrong_minimum["numeric_gate"]["minimum_public_output"] = 0.999
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="smallest reported"):
        context_profiles.validate_record(wrong_minimum)

    missing_tail = copy.deepcopy(_record("ctx4096"))
    missing_tail["numeric_gate"]["public_outputs"] = [
        row for row in missing_tail["numeric_gate"]["public_outputs"]
        if row["position"] != 4095
    ]
    with pytest.raises(context_profiles.ContextQualificationError,
                       match="last valid"):
        context_profiles.validate_record(missing_tail)


def test_cli_verifies_context_records(capsys) -> None:
    from pico_minicpm5 import cli

    assert cli.main([
        "qualify-context-profile", "--record",
        str(CONTEXTS / "ctx4096.qualification.json")]) == 0
    assert json.loads(capsys.readouterr().out)["passes"] is True
    assert cli.main([
        "qualify-context-profile", "--record",
        str(CONTEXTS / "ctx8192.qualification.json")]) == 0
    assert json.loads(capsys.readouterr().out)["passes"] is True
    # The candidate record validates but does not pass, so the CLI
    # exits non-zero and cannot be mistaken for a qualification.
    assert cli.main([
        "qualify-context-profile", "--record",
        str(CONTEXTS / "ctx16384.qualification.json")]) == 1
    assert json.loads(capsys.readouterr().out)["passes"] is False
