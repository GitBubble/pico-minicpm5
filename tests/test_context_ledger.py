from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "app" / "src" / "context_ledger.py"
AGENT_PATH = PROJECT / "app" / "src" / "minicpm_agent.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ledger():
    return _load(MODULE_PATH, "context_ledger_test")


def _agent():
    return _load(AGENT_PATH, "minicpm_agent_for_ledger_test")


def _offsets(text: str, size: int = 4):
    """A deterministic stand-in for a tokenizer: fixed-width character runs."""
    return [(start, min(start + size, len(text)))
            for start in range(0, len(text), size)]


PROMPT = (
    "<s>"
    "<|im_start|>system\nYou are an agent.\n"
    "<tools>\n"
    '{"type":"function","function":{"name":"list_directory"}}\n'
    '{"type":"function","function":{"name":"read_file"}}\n'
    "</tools>\n"
    "Use a tool when one applies.<|im_end|>\n"
    "<|im_start|>user\nlist the directory<|im_end|>\n"
    "<|im_start|>user\n<tool_response>\nentries: a b c\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\nThere are three entries.<|im_end|>\n"
    "<|im_start|>user\nhello<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def test_segments_tile_the_prompt_exactly() -> None:
    ledger = _ledger()
    segments = ledger.segment_prompt(PROMPT)

    assert segments
    assert segments[0].start == 0
    assert segments[-1].end == len(PROMPT)
    for earlier, later in zip(segments, segments[1:]):
        assert earlier.end == later.start, "segments must leave no gap"


def test_every_wire_role_is_recognised() -> None:
    ledger = _ledger()
    kinds = [segment.kind for segment in ledger.segment_prompt(PROMPT)]

    assert kinds[0] == "preamble"
    assert "system" in kinds
    assert "tool_schema" in kinds
    assert "history_user" in kinds
    assert "tool_result" in kinds
    assert "history_assistant" in kinds
    assert kinds[-1] == "generation_prompt"
    # The final user block is the turn being asked, not history.
    assert kinds.count("current_user") == 1


def test_tool_schema_is_labelled_by_the_tools_it_declares() -> None:
    ledger = _ledger()
    schema = next(segment for segment in ledger.segment_prompt(PROMPT)
                  if segment.kind == "tool_schema")

    assert schema.label == "2 tools: list_directory, read_file"


def test_tokens_are_conserved_across_segments() -> None:
    ledger = _ledger()
    offsets = _offsets(PROMPT)
    record = ledger.measure(PROMPT, ledger.segment_prompt(PROMPT), offsets)

    counted = sum(row["tokens"] for row in record["segments"])
    assert counted == record["total_tokens"] == len(offsets)
    assert record["unattributed_tokens"] == 0


def test_resident_prefix_splits_exactly() -> None:
    ledger = _ledger()
    offsets = _offsets(PROMPT)
    record = ledger.measure(PROMPT, ledger.segment_prompt(PROMPT), offsets,
                            resident_tokens=20)

    assert record["new_tokens"] == len(offsets) - 20
    assert sum(row["new_tokens"] for row in record["segments"]) \
        == record["new_tokens"]


def test_a_token_belongs_to_the_segment_holding_its_first_character() -> None:
    ledger = _ledger()
    text = "AAAABBBB"
    segments = [ledger.Segment("system", "a", 0, 4),
                ledger.Segment("current_user", "b", 4, 8)]
    # One token straddles the boundary: characters 3..5.
    offsets = [(0, 3), (3, 5), (5, 8)]
    record = ledger.measure(text, segments, offsets)

    rows = {row["label"]: row for row in record["segments"]}
    assert rows["a"]["tokens"] == 2, "the straddling token goes to its start"
    assert rows["b"]["tokens"] == 1
    assert record["boundary_tokens"] == 1


def test_cost_and_pressure_use_the_measured_board_rate() -> None:
    ledger = _ledger()
    text = "AAAABBBB"
    segments = [ledger.Segment("system", "a", 0, 8)]
    record = ledger.measure(text, segments, _offsets(text, 2),
                            resident_tokens=1, prompt_token_ms=79.49,
                            capacity=1024, reserve_tokens=24)

    assert record["total_tokens"] == 4
    assert record["new_tokens"] == 3
    assert record["ingest_ms"] == pytest.approx(238.47)
    assert record["budget_tokens"] == 1000
    assert record["pressure"] == pytest.approx(0.004)


def test_unattributed_tokens_are_reported_not_absorbed() -> None:
    ledger = _ledger()
    text = "AAAABBBB"
    segments = [ledger.Segment("system", "a", 0, 4)]
    record = ledger.measure(text, segments, _offsets(text, 2))

    assert record["unattributed_tokens"] == 2
    assert any(row["kind"] == "unattributed" for row in record["segments"])
    assert sum(row["tokens"] for row in record["segments"]) == 4


def test_a_malformed_segment_table_fails_closed() -> None:
    ledger = _ledger()
    text = "AAAABBBB"

    with pytest.raises(ledger.LedgerError):
        ledger.measure(text, [ledger.Segment("system", "a", 0, 6),
                              ledger.Segment("system", "b", 4, 8)],
                       _offsets(text))
    with pytest.raises(ledger.LedgerError):
        ledger.measure(text, [ledger.Segment("system", "a", 0, 8)],
                       _offsets(text), resident_tokens=99)
    with pytest.raises(ledger.LedgerError):
        ledger.measure(text, [ledger.Segment("system", "a", 0, 8)],
                       _offsets(text), capacity=16, reserve_tokens=16)
    with pytest.raises(ledger.LedgerError):
        ledger.Segment("nonsense", "a", 0, 8)


def test_render_names_the_largest_consumer_first() -> None:
    ledger = _ledger()
    offsets = _offsets(PROMPT)
    record = ledger.measure(PROMPT, ledger.segment_prompt(PROMPT), offsets,
                            prompt_token_ms=79.49, capacity=1024,
                            reserve_tokens=24)
    table = ledger.render(record)

    first = table.splitlines()[0]
    largest = max(record["segments"], key=lambda row: row["tokens"])
    assert first.startswith(largest["label"][:20])
    assert "total" in table.splitlines()[-1]


def test_the_real_wire_format_segments_without_leftovers() -> None:
    """The ledger reads back exactly what render_chat writes."""
    agent, ledger = _agent(), _ledger()
    rendered = agent.render_chat(
        [{"role": "system", "content": "Be useful."},
         {"role": "user", "content": "list the directory"},
         {"role": "assistant", "content": "Here it is."},
         {"role": "user", "content": "thanks"}],
        [{"type": "function", "function": {
            "name": "list_directory", "description": "List", "parameters": {}}}],
        enable_thinking=False,
    )
    segments = ledger.segment_prompt(rendered)

    assert "".join(rendered[s.start:s.end] for s in segments) == rendered
    kinds = [segment.kind for segment in segments]
    assert "tool_schema" in kinds
    assert kinds.count("current_user") == 1
    assert kinds[-1] == "generation_prompt"


def _server():
    return _load(PROJECT / "app" / "src" / "merged_board_server.py",
                 "merged_board_server_for_ledger_test")


def test_prompt_rate_is_measured_from_the_head_skipping_positions() -> None:
    server = _server()
    observed = {"ms": None, "tokens": 0, "total_ms": 0.0}
    steps = [
        {"head_skipped": True, "token_count": 1, "total_ms": 80.0},
        {"head_skipped": True, "token_count": 1, "total_ms": 78.0},
        {"head_skipped": False, "token_count": 1, "total_ms": 100.0},
    ]

    assert server.observe_prompt_token_ms(steps, observed) == pytest.approx(79.0)
    assert observed["tokens"] == 2, "generation positions must not be counted"


def test_a_step_without_a_width_counts_as_one_position() -> None:
    """The board REPL lineage records no `token_count`; wide prefill does."""
    server = _server()
    observed = {"ms": None, "tokens": 0, "total_ms": 0.0}
    steps = [{"head_skipped": True, "total_ms": 79.49}]

    assert server.observe_prompt_token_ms(steps, observed) \
        == pytest.approx(79.49)


def test_prompt_rate_is_none_before_anything_was_observed() -> None:
    server = _server()
    observed = {"ms": None, "tokens": 0, "total_ms": 0.0}

    assert server.observe_prompt_token_ms([], observed) is None
    assert server.observe_prompt_token_ms(
        [{"head_skipped": False, "total_ms": 100.0}], observed) is None


def test_a_full_snapshot_cache_slides_instead_of_failing() -> None:
    """The executor holds eight input snapshots; a ninth profile is normal.

    Raising here used to end a live board session over a cache condition, and
    it capped tool disclosure at eight profiles because the snapshot key is
    the profile name. The window now slides: the least recently used slot is
    overwritten and its profile re-ingests next time.
    """
    server = _server()
    session = object.__new__(server.Merged)
    session._prefix_snapshots = {}
    session._next_prefix_snapshot_id = 1
    session._resident_tokens = [7, 7, 7]
    session.timeout = 60.0
    session.last_prefix_metrics = server.Merged._empty_prefix_metrics()
    saved = []
    session._save_input_snapshot = lambda slot, length: saved.append(
        (slot, length))

    slots = server.PREFIX_SNAPSHOT_SLOTS
    for index in range(slots):
        assert session._maybe_create_prefix_snapshot(f"p{index}", (7, 7, 7)) == 3
    assert len(session._prefix_snapshots) == slots
    assert [slot for slot, _ in saved] == list(range(1, slots + 1))

    # Touch the oldest so it is no longer the eviction candidate.
    session._prefix_snapshots["p0"] = session._prefix_snapshots.pop("p0")
    assert session._maybe_create_prefix_snapshot("p8", (7, 7, 7)) == 3

    assert "p1" not in session._prefix_snapshots, "the LRU entry is evicted"
    assert "p0" in session._prefix_snapshots, "a touched entry survives"
    assert len(session._prefix_snapshots) == slots, "the window does not grow"
    assert session._prefix_snapshots["p8"][0] == 2, "it reuses the freed slot"
    assert session.last_prefix_metrics["prefix_snapshot_evicted"] == 1


def test_an_evicted_prefix_is_reported_as_a_miss_not_a_stale_hit() -> None:
    server = _server()
    session = object.__new__(server.Merged)
    session._prefix_snapshots = {}
    session._next_prefix_snapshot_id = 1
    session._resident_tokens = [1, 2]
    session.timeout = 60.0
    session.last_prefix_metrics = server.Merged._empty_prefix_metrics()
    session._save_input_snapshot = lambda slot, length: None

    session._maybe_create_prefix_snapshot("gone", (1, 2))
    session._prefix_snapshots.pop("gone")

    session._resident_tokens = []
    hit, _ = session._prepare_prefix_snapshot("gone", (1, 2), (1, 2, 3))
    assert hit == 0, "an evicted key must re-ingest, not restore a dead slot"
