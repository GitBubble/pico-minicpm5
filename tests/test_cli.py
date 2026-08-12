from __future__ import annotations

import json

from pico_minicpm5.cli import _clip_preset, main


def test_doctor_is_available_without_vendor_sdk(capsys) -> None:
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "pico.minicpm5.doctor.v1"
    assert report["atc_is_external"] is True


def test_clip_presets_resolve_in_source_or_wheel_layout() -> None:
    assert _clip_preset("decode").name == "decode-clips.json"
    assert _clip_preset("prefill").name == "prefill-clips.json"


def test_native_prefill_qualification_cli_is_discoverable() -> None:
    from pico_minicpm5.cli import build_parser

    args = build_parser().parse_args([
        "qualify-prefill-block", "--evidence", "evidence.json",
        "--out", "qualification.json",
    ])
    assert args.command == "qualify-prefill-block"

    release = build_parser().parse_args([
        "qualify-prefill-block-release", "--evidence", "release-evidence.json",
        "--out", "release-qualification.json",
    ])
    assert release.command == "qualify-prefill-block-release"

    strict_s1 = build_parser().parse_args([
        "qualify-prefill-s1-release", "--evidence", "s1-evidence.json",
        "--out", "s1-qualification.json",
    ])
    assert strict_s1.command == "qualify-prefill-s1-release"
