#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact per-segment accounting for an assembled MiniCPM5 prompt.

A prompt token costs `79.49 ms` to ingest on the qualified ctx1024 board, and
the whole conversation has to fit in 1024 of them, so the question "what is
consuming the context" has a measurable answer that the runtime should not have
to guess at. This module produces it.

Attribution is exact rather than estimated. The runtime already encodes the
rendered prompt once in order to feed it to the model; this module reuses that
encoding's character offsets and assigns each token to the segment holding its
first character. Encoding each segment separately would be both slower and
wrong, because the tokenizer merges across a segment boundary and the parts
would not sum to the whole.

Nothing here imports the tokenizer, the agent or the runtime. The input is a
string, a segment table and one `(start, end)` pair per token; the output is a
plain dictionary. The scan is a single pass over the offsets with a cursor into
the sorted segment table, so a future Rust implementation can be diffed against
this one through the recorded schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


SCHEMA = "pico.minicpm5.context-ledger.v1"

#: Segment kinds, in the order they appear in a rendered prompt. `preamble`
#: covers the leading `<s>`; `boundary` never appears as a segment kind and is
#: reserved for the diagnostic count of tokens that straddle two segments.
KINDS = (
    "preamble",
    "system",
    "tool_schema",
    "history_user",
    "history_assistant",
    "tool_result",
    "current_user",
    "generation_prompt",
)

_TURN = re.compile(r"<\|im_start\|>(?P<role>[a-z]+)\n")
_TOOLS = re.compile(r"<tools>\n.*?\n</tools>\n?", re.DOTALL)
_TOOL_RESPONSE = "<tool_response>"


class LedgerError(ValueError):
    """A segment table does not describe the text it claims to describe."""


@dataclass(frozen=True)
class Segment:
    """A half-open `[start, end)` character span of the rendered prompt."""

    kind: str
    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise LedgerError(f"unknown segment kind {self.kind!r}")
        if self.start < 0 or self.end < self.start:
            raise LedgerError(f"segment {self.label!r} has a negative span")


def segment_prompt(text: str) -> tuple[Segment, ...]:
    """Split a rendered prompt into segments along the MiniCPM5 wire format.

    The wire format is the contract `render_chat` implements, so reading it back
    keeps this module independent of how the prompt was built — including by a
    renderer written in another language. Every character lands in exactly one
    segment; `measure` reports any that do not.

    :param text: the rendered prompt, exactly as it will be tokenised.
    :returns: segments in ascending order, tiling the whole string.
    """
    starts = [match.start() for match in _TURN.finditer(text)]
    if not starts:
        return (Segment("system", "whole prompt", 0, len(text)),) if text else ()

    segments: list[Segment] = []
    if starts[0] > 0:
        segments.append(Segment("preamble", "bos", 0, starts[0]))

    bounds = starts + [len(text)]
    turn_index = 0
    for index, start in enumerate(starts):
        end = bounds[index + 1]
        role = _TURN.match(text, start).group("role")
        block = text[start:end]
        if role == "system":
            segments.extend(_split_system(block, start))
            continue
        if role == "assistant":
            if index == len(starts) - 1 and "<|im_end|>" not in block:
                segments.append(Segment("generation_prompt", "assistant", start, end))
            else:
                turn_index += 1
                segments.append(
                    Segment("history_assistant", f"turn {turn_index}", start, end))
            continue
        # Everything else is a user-role block: a real turn, or the tool
        # responses the contract also delivers under the user role.
        if _TOOL_RESPONSE in block:
            kind, label = "tool_result", _tool_label(block)
        elif _is_last_user(starts, index, text):
            kind, label = "current_user", "current"
        else:
            turn_index += 1
            kind, label = "history_user", f"turn {turn_index}"
        segments.append(Segment(kind, label, start, end))
    return tuple(segments)


def _split_system(block: str, offset: int) -> list[Segment]:
    """Carve the `<tools>` schema block out of the system message."""
    match = _TOOLS.search(block)
    if match is None:
        return [Segment("system", "system", offset, offset + len(block))]
    tools_start = offset + match.start()
    tools_end = offset + match.end()
    parts = [Segment("system", "system", offset, tools_start),
             Segment("tool_schema", _schema_label(match.group()),
                     tools_start, tools_end)]
    if tools_end < offset + len(block):
        parts.append(Segment("system", "system tail",
                             tools_end, offset + len(block)))
    return parts


def _schema_label(block: str) -> str:
    """Name a tool-schema block by the tools it declares."""
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', block)
    if not names:
        return "tools"
    return f"{len(names)} tools: " + ", ".join(names)


def _tool_label(block: str) -> str:
    """Name a tool-result block by the payload's first line."""
    body = block.split(_TOOL_RESPONSE, 1)[1].strip()
    first = body.splitlines()[0] if body else ""
    return first[:40] or "tool response"


def _is_last_user(starts: list[int], index: int, text: str) -> bool:
    """True when no later block carries a user turn."""
    for later in starts[index + 1:]:
        if _TURN.match(text, later).group("role") == "user":
            return False
    return True


def measure(text, segments, offsets, *, resident_tokens=0,
            prompt_token_ms=None, capacity=None, reserve_tokens=0):
    """Attribute every token to a segment and price the ones not yet resident.

    :param text: the rendered prompt the offsets index into.
    :param segments: segments from `segment_prompt`, or an equivalent table.
    :param offsets: one `(start, end)` character pair per token, in token
        order, as produced by the tokenizer that encoded `text`.
    :param resident_tokens: leading tokens already in the board's K/V cache.
        The resident prefix is always a prefix by token position, so the split
        is exact rather than apportioned.
    :param prompt_token_ms: measured ingestion cost of one prompt token on the
        active profile. `None` omits every millisecond field.
    :param capacity: the profile's compiled context, for the pressure ratio.
    :param reserve_tokens: response headroom excluded from the pressure budget.
    :returns: a `SCHEMA` record.
    """
    ordered = sorted(segments, key=lambda segment: segment.start)
    _check_tiling(text, ordered)
    total = len(offsets)
    if resident_tokens < 0 or resident_tokens > total:
        raise LedgerError(
            f"resident_tokens {resident_tokens} is outside [0, {total}]")

    tallies = {id(segment): [0, 0] for segment in ordered}
    unattributed = [0, 0]
    boundary = 0
    cursor = 0
    for index, (start, end) in enumerate(offsets):
        while cursor < len(ordered) and ordered[cursor].end <= start:
            cursor += 1
        owner = None
        if cursor < len(ordered) and ordered[cursor].start <= start:
            owner = ordered[cursor]
            if end > owner.end:
                boundary += 1
        bucket = tallies[id(owner)] if owner is not None else unattributed
        bucket[0] += 1
        if index >= resident_tokens:
            bucket[1] += 1

    rows = []
    for segment in ordered:
        tokens, new = tallies[id(segment)]
        if not tokens:
            continue
        rows.append(_row(segment.kind, segment.label, tokens, new, total,
                         prompt_token_ms))
    if unattributed[0]:
        rows.append(_row("unattributed", "unattributed", unattributed[0],
                         unattributed[1], total, prompt_token_ms))

    new_total = total - resident_tokens
    record = {
        "schema": SCHEMA,
        "total_tokens": total,
        "resident_tokens": resident_tokens,
        "new_tokens": new_total,
        "boundary_tokens": boundary,
        "unattributed_tokens": unattributed[0],
        "segments": rows,
    }
    if prompt_token_ms is not None:
        record["prompt_token_ms"] = prompt_token_ms
        record["ingest_ms"] = round(new_total * prompt_token_ms, 3)
    if capacity is not None:
        budget = capacity - reserve_tokens
        if budget <= 0:
            raise LedgerError(
                f"capacity {capacity} leaves no budget after "
                f"{reserve_tokens} reserved tokens")
        record["capacity"] = capacity
        record["reserve_tokens"] = reserve_tokens
        record["budget_tokens"] = budget
        record["pressure"] = round(total / budget, 4)
    return record


def _row(kind, label, tokens, new, total, prompt_token_ms):
    row = {
        "kind": kind,
        "label": label,
        "tokens": tokens,
        "new_tokens": new,
        "share": round(tokens / total, 4) if total else 0.0,
    }
    if prompt_token_ms is not None:
        row["ingest_ms"] = round(new * prompt_token_ms, 3)
    return row


def _check_tiling(text: str, ordered: list[Segment]) -> None:
    """Reject a segment table that overlaps itself or overruns the text."""
    previous = 0
    for segment in ordered:
        if segment.start < previous:
            raise LedgerError(
                f"segment {segment.label!r} overlaps the previous segment")
        previous = segment.end
    if ordered and ordered[-1].end > len(text):
        raise LedgerError("a segment ends past the end of the text")


def render(record, *, width=64):
    """Format a record as the fixed-width table the REPL prints for `/context`.

    :param record: a `SCHEMA` record from `measure`.
    :param width: total width of the bar column plus its label.
    :returns: the rendered table, without a trailing newline.
    """
    rows = record["segments"]
    if not rows:
        return "context ledger: empty prompt"
    label_width = max(len(row["label"]) for row in rows)
    label_width = min(label_width, max(8, width - 40))
    lines = []
    for row in sorted(rows, key=lambda item: -item["tokens"]):
        bar = "#" * max(1, round(row["share"] * 24))
        line = (f"{row['label'][:label_width]:<{label_width}}  "
                f"{row['tokens']:>5} tok  {row['share'] * 100:5.1f}%  "
                f"{bar:<24}")
        if "ingest_ms" in row and row["new_tokens"]:
            line += f"  {row['ingest_ms'] / 1000:6.2f}s new"
        lines.append(line.rstrip())
    footer = (f"total {record['total_tokens']} tokens, "
              f"{record['resident_tokens']} resident, "
              f"{record['new_tokens']} new")
    if "ingest_ms" in record:
        footer += f", {record['ingest_ms'] / 1000:.2f}s to ingest"
    if "pressure" in record:
        footer += (f"; {record['pressure'] * 100:.1f}% of the "
                   f"{record['budget_tokens']}-token budget")
    lines.append(footer)
    if record["unattributed_tokens"]:
        lines.append(f"warning: {record['unattributed_tokens']} tokens "
                     "were not attributed to any segment")
    return "\n".join(lines)
