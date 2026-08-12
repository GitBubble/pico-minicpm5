from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "app" / "src" / "minicpm_prefill_schedule.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_prefill_schedule_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_largest_first_s128_s32_s16_s1_tail() -> None:
    schedule = _module().plan_prefill(
        0, 433, context=1024, enabled_widths=(1, 16, 128, 32))

    assert schedule.counts() == {"S128": 3, "S32": 1, "S16": 1, "S1": 1}
    assert schedule.invocation_count == 6
    assert [(item.width, item.start, item.stop) for item in schedule.segments] == [
        (1, 0, 1), (128, 1, 385), (32, 385, 417), (16, 417, 433)]


def test_position_zero_is_strict_s1_before_any_wide_block() -> None:
    schedule = _module().plan_prefill(
        0, 129, context=1024, enabled_widths=(128, 32, 16, 1))

    assert [(item.width, item.start, item.stop) for item in schedule.segments] == [
        (1, 0, 1), (128, 1, 129)]
    assert schedule.counts() == {"S128": 1, "S32": 0, "S16": 0, "S1": 1}


def test_resident_prefix_range_keeps_absolute_positions() -> None:
    schedule = _module().plan_prefill(
        643, 810, context=1024, enabled_widths=(128, 32, 16, 1))

    assert schedule.counts() == {"S128": 1, "S32": 1, "S16": 0, "S1": 7}
    assert schedule.segments[0].start == 643
    assert schedule.segments[-1].stop == 810


def test_hard_boundary_restarts_largest_first_without_crossing() -> None:
    schedule = _module().plan_prefill(
        0, 36, context=128, enabled_widths=(16, 1),
        hard_boundaries=(20,))

    assert schedule.hard_boundaries == (20,)
    assert [(item.width, item.count, item.start, item.stop)
            for item in schedule.segments] == [
        (1, 1, 0, 1),
        (16, 1, 1, 17),
        (1, 3, 17, 20),
        (16, 1, 20, 36),
    ]
    assert schedule.to_dict()["hard_boundaries"] == [20]


def test_current_release_is_strict_s1_only() -> None:
    schedule = _module().plan_prefill(643, 810, context=1024)

    assert schedule.enabled_widths == (1,)
    assert schedule.counts() == {"S128": 0, "S32": 0, "S16": 0, "S1": 167}
    assert schedule.invocation_count == 167


@pytest.mark.parametrize("enabled, match", [
    ((128, 32, 16), "S1 fallback"),
    ((64, 1), "unsupported"),
    ((16, 16, 1), "must not repeat"),
    ((True, 1), "must be integers"),
])
def test_unqualified_or_ambiguous_widths_fail_closed(enabled, match) -> None:
    module = _module()
    with pytest.raises(module.PrefillScheduleError, match=match):
        module.plan_prefill(0, 16, context=1024, enabled_widths=enabled)


@pytest.mark.parametrize("start, stop, context", [
    (-1, 1, 1024), (0, 1025, 1024), (4, 3, 1024), (0, 1, 1),
])
def test_range_must_stay_inside_context(start, stop, context) -> None:
    module = _module()
    with pytest.raises(module.PrefillScheduleError, match="context"):
        module.plan_prefill(start, stop, context=context)


def test_empty_range_has_no_invocations() -> None:
    schedule = _module().plan_prefill(
        17, 17, context=128, enabled_widths=(128, 32, 16, 1))
    assert schedule.invocation_count == 0
    assert schedule.segments == ()


@pytest.mark.parametrize(
    "stop", (0, 1, 15, 16, 17, 31, 32, 33, 47, 48, 127, 128, 129))
def test_startup_and_width_boundaries_cover_every_token_exactly(stop: int) -> None:
    schedule = _module().plan_prefill(
        0, stop, context=256, enabled_widths=(128, 32, 16, 1))

    assert schedule.token_count == stop
    assert sum(item.width * item.count for item in schedule.segments) == stop
    if stop:
        assert schedule.segments[0].width == 1
        assert schedule.segments[0].start == 0
        assert schedule.segments[0].stop == 1
