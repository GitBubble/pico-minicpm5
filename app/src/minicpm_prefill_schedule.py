#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed largest-first MiniCPM prompt-prefill scheduling contract.

The runtime may only enable a block width after the corresponding OM has
passed its descriptor, public-output cosine, prefill-to-decode handoff and
board token-exact gates.  Width one is the permanent strict fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


SCHEMA = "pico.minicpm5.prefill-schedule.v1"
CANDIDATE_WIDTHS = (128, 32, 16, 1)


class PrefillScheduleError(ValueError):
    """A prefill range or enabled block family violates the runtime contract."""


@dataclass(frozen=True)
class PrefillSegment:
    width: int
    count: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.width not in CANDIDATE_WIDTHS or self.count < 1:
            raise PrefillScheduleError("prefill segment width/count is invalid")
        if self.start < 0 or self.stop != self.start + self.width * self.count:
            raise PrefillScheduleError("prefill segment range is not contiguous")


@dataclass(frozen=True)
class PrefillSchedule:
    schema: str
    policy: str
    start: int
    stop: int
    context: int
    enabled_widths: tuple[int, ...]
    segments: tuple[PrefillSegment, ...]
    hard_boundaries: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.policy != "largest_first_strict_s1":
            raise PrefillScheduleError("prefill schedule identity drift")
        if not 0 <= self.start <= self.stop <= self.context:
            raise PrefillScheduleError("prefill schedule escapes context")
        if self.hard_boundaries != tuple(sorted(set(self.hard_boundaries))) \
                or any(type(value) is not int \
                       or not self.start < value < self.stop
                       for value in self.hard_boundaries):
            raise PrefillScheduleError("prefill hard boundaries are invalid")
        boundary_set = set(self.hard_boundaries)
        cursor = self.start
        previous = None
        for index, segment in enumerate(self.segments):
            if segment.start != cursor:
                raise PrefillScheduleError("prefill schedule has a gap or overlap")
            if segment.width not in self.enabled_widths:
                raise PrefillScheduleError("prefill schedule used a disabled width")
            if segment.start in boundary_set:
                previous = None
            startup_s1 = (
                index == 0 and self.start == 0 and segment.start == 0
                and segment.stop == 1 and segment.width == 1
                and any(width > 1 for width in self.enabled_widths)
            )
            if startup_s1:
                previous = None
            elif previous is not None and segment.width >= previous:
                raise PrefillScheduleError("prefill segments are not largest-first")
            cursor = segment.stop
            if not startup_s1:
                previous = segment.width
        if cursor != self.stop:
            raise PrefillScheduleError("prefill schedule does not cover its range")

    @property
    def token_count(self) -> int:
        return self.stop - self.start

    @property
    def invocation_count(self) -> int:
        return sum(segment.count for segment in self.segments)

    def counts(self) -> dict[str, int]:
        present: dict[int, int] = {}
        for segment in self.segments:
            present[segment.width] = present.get(segment.width, 0) + segment.count
        return {f"S{width}": present.get(width, 0)
                for width in CANDIDATE_WIDTHS}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy": self.policy,
            "start": self.start,
            "stop": self.stop,
            "context": self.context,
            "token_count": self.token_count,
            "candidate_widths": list(CANDIDATE_WIDTHS),
            "enabled_widths": list(self.enabled_widths),
            "hard_boundaries": list(self.hard_boundaries),
            "counts": self.counts(),
            "invocation_count": self.invocation_count,
            "segments": [asdict(segment) for segment in self.segments],
        }


def _enabled_widths(values: Iterable[int]) -> tuple[int, ...]:
    supplied = tuple(values)
    if not supplied:
        raise PrefillScheduleError("at least the strict S1 fallback is required")
    if any(type(value) is not int for value in supplied):
        raise PrefillScheduleError("prefill widths must be integers")
    if len(set(supplied)) != len(supplied):
        raise PrefillScheduleError("prefill widths must not repeat")
    unsupported = set(supplied) - set(CANDIDATE_WIDTHS)
    if unsupported:
        raise PrefillScheduleError(
            f"unsupported prefill widths: {sorted(unsupported)}")
    if 1 not in supplied:
        raise PrefillScheduleError("strict S1 fallback must remain enabled")
    return tuple(width for width in CANDIDATE_WIDTHS if width in supplied)


def plan_prefill(
    start: int,
    stop: int,
    *,
    context: int,
    enabled_widths: Iterable[int] = (1,),
    hard_boundaries: Iterable[int] = (),
) -> PrefillSchedule:
    """Cover ``[start, stop)`` greedily with qualified block families.

    Ordering in ``enabled_widths`` cannot change policy: the canonical
    S128→S32→S16→S1 order is always used.  Missing wider artifacts simply fall
    through to the next qualified width, and S1 guarantees exact coverage.
    """
    if type(start) is not int or type(stop) is not int or type(context) is not int:
        raise PrefillScheduleError("prefill positions/context must be integers")
    if context < 2 or not 0 <= start <= stop <= context:
        raise PrefillScheduleError("prefill range must stay inside context")
    widths = _enabled_widths(enabled_widths)
    boundaries = tuple(hard_boundaries)
    if any(type(value) is not int for value in boundaries) \
            or len(boundaries) != len(set(boundaries)) \
            or boundaries != tuple(sorted(boundaries)) \
            or any(not start < value < stop for value in boundaries):
        raise PrefillScheduleError(
            "prefill hard boundaries must be sorted unique interior positions")
    segments: list[PrefillSegment] = []
    points = (start, *boundaries, stop)
    for interval_index, (interval_start, interval_stop) in enumerate(
            zip(points, points[1:])):
        cursor = interval_start
        remaining = interval_stop - interval_start
        # Wide artifacts are steady-only until they have a separately
        # qualified startup quantization contract.  Preserve position zero on
        # S1, then restart largest-first policy after every hard boundary.
        if interval_index == 0 and interval_start == 0 and remaining \
                and any(width > 1 for width in widths):
            segments.append(PrefillSegment(
                width=1, count=1, start=0, stop=1))
            cursor = 1
            remaining -= 1
        for width in widths:
            count, remaining = divmod(remaining, width)
            if not count:
                continue
            segment = PrefillSegment(
                width=width, count=count, start=cursor,
                stop=cursor + count * width)
            segments.append(segment)
            cursor = segment.stop
        if remaining:
            raise PrefillScheduleError(
                "strict S1 fallback failed to cover bounded interval")
    return PrefillSchedule(
        schema=SCHEMA,
        policy="largest_first_strict_s1",
        start=start,
        stop=stop,
        context=context,
        enabled_widths=widths,
        segments=tuple(segments),
        hard_boundaries=boundaries,
    )
