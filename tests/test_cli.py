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
