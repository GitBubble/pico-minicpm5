# SPDX-License-Identifier: Apache-2.0
"""Eager streaming prefill vs the non-eager agent path.

MOCK executor (no board, no NPU): records every fed token with position and
wall-clock time.  The proven invariant, for every scenario: the tokens
ultimately resident and consumed by generate are BYTE-IDENTICAL to the
non-eager path, and any transiently wrong eager feed is covered by an explicit
rewind/reset event (never silent).

These run against the LANDED modules in ``app/src`` (``eager_prefill``,
``eager_agent``, the streaming ``minicpm_agent.WorkspaceTools._run_shell`` and
``eager_server.EagerSessionMixin``), not against a staged prototype copy.

Run:  cd projects/pico-minicpm5 && python -m pytest tests/test_eager_prefill.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

import minicpm_agent as AGENT                    # noqa: E402
import eager_agent                               # noqa: E402
import eager_server                              # noqa: E402
from eager_prefill import (                      # noqa: E402
    EagerCapacityReached, EagerPreconditionError, EagerPrefillEngine,
    ProvisionalPromptBuilder, RunShellContentPlanner, StableTokenFeed,
    Utf8LineGate, check_preconditions, common_prefix_len, require_preconditions)

HOLDBACK = 6


# -------------------------------------------------------------------------
# adversarial mock tokenizer: multi-char merges make trailing tokens unstable
# -------------------------------------------------------------------------
_MERGES = ("\n\n", "exit=", "aa", "  ")


def encode(text: str) -> list[str]:
    tokens, i = [], 0
    while i < len(text):
        for merge in _MERGES:
            if text.startswith(merge, i):
                tokens.append(merge)
                i += len(merge)
                break
        else:
            tokens.append(text[i])
            i += 1
    return tokens


# -------------------------------------------------------------------------
# mock executor session
# -------------------------------------------------------------------------
class MockSession:
    """Records (op, position, token, time) for every cache mutation."""

    resident_kv = True

    def __init__(self, past: int = 10**9):
        self.resident: list = []
        self.events: list[tuple] = []
        self.past = past

    def resident_tokens(self):
        return tuple(self.resident)

    def prefill_extend(self, tokens):
        for token in tokens:
            if len(self.resident) >= self.past:
                raise EagerCapacityReached(len(self.resident))
            self.resident.append(token)
            self.events.append(
                ("feed", len(self.resident) - 1, token, time.perf_counter()))

    def rewind_to(self, count):
        assert 0 <= count <= len(self.resident)
        self.events.append(("rewind", count, None, time.perf_counter()))
        del self.resident[count:]

    def reset_kv(self):
        self.events.append(("reset", 0, None, time.perf_counter()))
        self.resident = []

    def generate_consume(self, ids, reuse_prefix=True):
        """Mirror of Merged.generate prompt ingestion + _prefix_plan."""
        ids = list(ids)
        previous = list(self.resident)
        common = common_prefix_len(previous, ids)
        if reuse_prefix and previous:
            start = min(common, max(0, len(ids) - 1))
        else:
            start = 0
            self.events.append(
                ("reset_prefix", 0, None, time.perf_counter()))
            self.resident = []
        del self.resident[start:]
        for pos in range(start, len(ids)):
            self.resident.append(ids[pos])
            self.events.append(
                ("consume", pos, ids[pos], time.perf_counter()))
        return start


def wait_drained(session, timeout=3.0):
    """Wait until the engine worker stops mutating the mock session."""
    deadline = time.time() + timeout
    last, stable_since = len(session.events), time.time()
    while time.time() < deadline:
        time.sleep(0.03)
        now = len(session.events)
        if now != last:
            last, stable_since = now, time.time()
        elif time.time() - stable_since > 0.15:
            return
    return


# -------------------------------------------------------------------------
# shared fixtures
# -------------------------------------------------------------------------
def make_workspace_tools(tmpdir, max_output_chars=800):
    return AGENT.WorkspaceTools(Path(tmpdir), max_output_chars=max_output_chars)


def base_messages():
    system = {"role": "system", "content": "You are MiniCPM Agent."}
    return [
        system,
        {"role": "user", "content": "run the build"},
        {"role": "assistant", "content":
         '<function name="run_shell"><param name="command">make</param>'
         '</function>'},
    ]


def final_tool_content(wt, output_string):
    """EXACT production assembly: WorkspaceTools.result + for_model."""
    result_json = wt.result("run_shell", True, output_string)
    return AGENT.WorkspaceTools.for_model(result_json)


def production_output_string(stdout_bytes, exit_code=0, stderr_text=""):
    """EXACT production _run_shell string for a finished tool."""
    stdout = AGENT._decode_text_mode(stdout_bytes)
    combined = stdout
    if stderr_text:
        combined += ("\n" if combined else "") + stderr_text
    return f"exit={exit_code}\n{combined.strip()}"


class EagerRound:
    """Drive one eager tool round against a MockSession, chunk by chunk."""

    def __init__(self, session, wt, messages, *, thinking=False,
                 optimistic_truncation=True, holdback=HOLDBACK,
                 batch_tokens=4):
        self.session, self.wt, self.messages = session, wt, messages
        self.thinking = thinking
        self.tools_defs = wt.definitions
        self.planner = RunShellContentPlanner(
            ref=wt.next_ref(),
            max_output_chars=wt.max_output_chars,
            optimistic_truncation=optimistic_truncation)
        self.builder = ProvisionalPromptBuilder(
            AGENT, messages, self.tools_defs, enable_thinking=thinking)
        self.engine = EagerPrefillEngine(
            session, encode, holdback=holdback, batch_tokens=batch_tokens)
        self.engine.begin(self._render)

    def _render(self):
        content, frozen = self.planner.provisional()
        if content is None:
            return self.builder.open_text()
        return self.builder.text(content, closed=frozen)

    def stream(self, chunks, delay=0.0):
        for chunk in chunks:
            self.planner.on_stdout(chunk)
            self.engine.notify()
            if delay:
                time.sleep(delay)

    def finish(self, stdout_bytes, *, exit_code=0, stderr_text=""):
        """Tool exited: assemble the REAL prompt, finalize, consume."""
        self.exit_time = time.perf_counter()
        output = production_output_string(
            stdout_bytes, exit_code, stderr_text)
        content = final_tool_content(self.wt, output)
        self.final_content = content
        final_messages = self.messages + [{"role": "tool", "content": content}]
        rendered = AGENT.render_chat(
            final_messages, self.tools_defs, add_generation_prompt=True,
            enable_thinking=self.thinking)
        self.final_ids = encode(rendered)
        self.metrics = self.engine.finalize(self.final_ids)
        self.report = self.engine.report()
        self.start = self.session.generate_consume(
            self.final_ids, reuse_prefix=True)
        return self.final_ids, self.metrics


class EagerPrefillInvariantTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    # -- helpers -----------------------------------------------------------
    def assert_byte_identical_vs_noneager(self, round_):
        session, ids = round_.session, round_.final_ids
        # 1. eager path final resident == full assembled prompt
        self.assertEqual(list(session.resident), ids)
        # 2. non-eager reference path over the same prompt
        reference = MockSession()
        reference.generate_consume(ids, reuse_prefix=True)
        self.assertEqual(reference.resident, list(session.resident))
        # 3. event-log integrity: every feed appended at the then-current
        #    length; wrong tokens only ever removed by explicit rewind/reset.
        replay = []
        for op, pos, token, _t in session.events:
            if op == "feed":
                self.assertEqual(pos, len(replay))
                replay.append(token)
            elif op == "rewind":
                self.assertLessEqual(pos, len(replay))
                del replay[pos:]
            elif op in ("reset", "reset_prefix"):
                replay = []
            elif op == "consume":
                if pos < len(replay):
                    replay[pos] = token
                else:
                    self.assertEqual(pos, len(replay))
                    replay.append(token)
        self.assertEqual(replay, ids)

    # -- scenarios ---------------------------------------------------------
    def test_multiline_streaming_output(self):
        wt = make_workspace_tools(self._tmp.name)
        stdout = "".join(f"line {i} of the build\n" for i in range(12))
        chunks = [line.encode() for line in stdout.splitlines(keepends=True)]
        round_ = EagerRound(MockSession(), wt, base_messages())
        round_.stream(chunks, delay=0.01)
        wait_drained(round_.session)
        round_.finish(stdout.encode())
        self.assertFalse(round_.metrics["failclosed"], round_.metrics)
        self.assert_byte_identical_vs_noneager(round_)
        # eager feeding actually happened before the tool exited
        eager_feeds = [e for e in round_.session.events
                       if e[0] == "feed" and e[3] < round_.exit_time]
        self.assertGreater(len(eager_feeds), 20)
        # and the finalize tail is much smaller than the whole prompt
        tail = len(round_.final_ids) - round_.metrics["resident_at_finalize"]
        self.assertLess(tail, len(round_.final_ids) // 3)
        # the report block the server writes carries the deciding fields
        for key in ("tokens_fed_eagerly", "managed_rollbacks", "failclosed",
                    "recovery", "recovery_reason", "rewound_to",
                    "common_prefix_len", "finalize_ms"):
            self.assertIn(key, round_.report)

    def test_output_ends_midline(self):
        wt = make_workspace_tools(self._tmp.name)
        stdout = "complete line\nsecond line\nunterminated tail without nl"
        chunks = [stdout[:20].encode(), stdout[20:].encode()]
        round_ = EagerRound(MockSession(), wt, base_messages())
        round_.stream(chunks, delay=0.02)
        wait_drained(round_.session)
        round_.finish(stdout.encode())
        self.assertFalse(round_.metrics["failclosed"], round_.metrics)
        # the held-back final line must not have been eagerly fed
        fed_text = "".join(
            t for t in round_.session.resident[
                :round_.metrics["resident_at_finalize"]])
        self.assertNotIn("unterminated tail without nl", fed_text)
        self.assert_byte_identical_vs_noneager(round_)

    def test_ansi_and_partial_utf8_chunks(self):
        wt = make_workspace_tools(self._tmp.name)
        stdout_bytes = ("\x1b[31m红色告警\x1b[0m\nplain\r\ncrlf line\n"
                        "tail 中文 no nl").encode()
        # split inside the 3-byte UTF-8 char, inside the ANSI escape, and
        # between \r and \n
        cuts = [4, 7, 11, 30, 41]
        chunks = [stdout_bytes[a:b] for a, b in
                  zip([0] + cuts, cuts + [len(stdout_bytes)])]
        round_ = EagerRound(MockSession(), wt, base_messages())
        round_.stream(chunks, delay=0.02)
        wait_drained(round_.session)
        round_.finish(stdout_bytes)
        self.assertFalse(round_.metrics["failclosed"], round_.metrics)
        self.assert_byte_identical_vs_noneager(round_)

    def test_truncation_cap_crossing_managed_rollback(self):
        wt = make_workspace_tools(self._tmp.name, max_output_chars=64)
        stdout = "".join(f"noisy output line number {i}\n" for i in range(9))
        chunks = [line.encode() for line in stdout.splitlines(keepends=True)]
        round_ = EagerRound(MockSession(), wt, base_messages())
        round_.stream(chunks, delay=0.02)
        wait_drained(round_.session)
        round_.finish(stdout.encode())
        # crossing the cap rewrites the optimistic header -> managed rollback
        self.assertGreaterEqual(round_.metrics["rollbacks"], 1)
        self.assertEqual(round_.report["managed_rollbacks"],
                         round_.metrics["rollbacks"])
        self.assertFalse(round_.metrics["failclosed"], round_.metrics)
        self.assert_byte_identical_vs_noneager(round_)

    def test_tool_retry_rollback_failclosed_then_clean_retry(self):
        wt = make_workspace_tools(self._tmp.name)
        session = MockSession()
        stdout = "attempt one\npartial work\n"
        round_ = EagerRound(session, wt, base_messages())
        round_.stream([stdout.encode()], delay=0.02)
        wait_drained(session)
        # tool FAILS (exit 3): optimistic exit=0 prediction is wrong
        round_.finish(stdout.encode(), exit_code=3)
        self.assertTrue(round_.metrics["failclosed"])
        # recovery is a managed rewind, NOT a full cache reset
        self.assertEqual(round_.metrics["recovery"], "rewind")
        self.assertNotIn("reset", [e[0] for e in session.events])
        self.assert_byte_identical_vs_noneager(round_)
        # agent retries: history replays round 1's REAL tool content (same
        # ref r1); the second round's planner predicts the next ref r2 off
        # the shared WorkspaceTools counter
        retry_messages = base_messages() + [
            {"role": "tool", "content": round_.final_content},
            {"role": "assistant", "content":
             '<function name="run_shell"><param name="command">make -k'
             '</param></function>'},
        ]
        stdout2 = "attempt two\nall good now\n"
        round2 = EagerRound(session, wt, retry_messages)
        round2.stream([stdout2[:9].encode(), stdout2[9:].encode()],
                      delay=0.02)
        wait_drained(session)
        round2.finish(stdout2.encode())
        self.assertFalse(round2.metrics["failclosed"], round2.metrics)
        self.assert_byte_identical_vs_noneager(round2)

    def test_interleaved_generate_request_mutual_exclusion(self):
        wt = make_workspace_tools(self._tmp.name)
        session = MockSession()
        round_ = EagerRound(session, wt, base_messages())
        stdout = "".join(f"slow line {i}\n" for i in range(40))
        stop = threading.Event()

        def feeder():
            for line in stdout.splitlines(keepends=True):
                if stop.is_set():
                    return
                round_.planner.on_stdout(line.encode())
                round_.engine.notify()
                time.sleep(0.01)

        thread = threading.Thread(target=feeder, daemon=True)
        thread.start()
        time.sleep(0.08)
        with round_.engine.exclusive():
            window_start = time.perf_counter()
            time.sleep(0.12)          # a generate/decode would run here
            window_end = time.perf_counter()
        stop.set()
        thread.join()
        wait_drained(session)
        round_.finish(stdout.encode())
        inside = [e for e in session.events
                  if e[0] == "feed" and window_start < e[3] < window_end]
        self.assertEqual(inside, [])
        self.assertFalse(round_.metrics["failclosed"], round_.metrics)
        self.assert_byte_identical_vs_noneager(round_)

    def test_capacity_reached_stops_gracefully(self):
        wt = make_workspace_tools(self._tmp.name)
        session = MockSession(past=40)
        stdout = "".join(f"row {i}\n" for i in range(30))
        round_ = EagerRound(session, wt, base_messages())
        round_.stream([stdout.encode()], delay=0.02)
        wait_drained(session)
        self.assertLessEqual(len(session.resident), 40)
        output = production_output_string(stdout.encode())
        content = final_tool_content(wt, output)
        rendered = AGENT.render_chat(
            base_messages() + [{"role": "tool", "content": content}],
            wt.definitions, add_generation_prompt=True,
            enable_thinking=False)
        final_ids = encode(rendered)
        metrics = round_.engine.finalize(final_ids)
        self.assertFalse(metrics["failclosed"], metrics)
        self.assertTrue(round_.engine.report()["capacity_reached"])
        self.assertEqual(list(session.resident),
                         final_ids[:len(session.resident)])

    # -- fail-closed cost (defect 2, board-diagnosed 2026-08-16) -----------
    def _prime_pretool_prefix(self, session, messages, wt):
        """Resident state the NON-eager path also starts the tool round with.

        Mirrors the board driver: the planning ``generate`` that emitted the
        tool call leaves the pre-tool prompt resident, and the baseline arm
        reuses it (measured prefix_cache_hit=355).  The mock starts empty, so
        without this the fail-closed cost question cannot even be posed.
        """
        base_ids = encode(AGENT.render_chat(
            messages, wt.definitions, add_generation_prompt=False,
            enable_thinking=False))
        session.prefill_extend(base_ids)
        return base_ids

    def test_failclosed_rewind_keeps_pretool_prefix_and_costs_baseline(self):
        wt = make_workspace_tools(self._tmp.name)
        session = MockSession()
        messages = base_messages()
        base_ids = self._prime_pretool_prefix(session, messages, wt)
        stdout = "attempt one\npartial work\n"
        round_ = EagerRound(session, wt, messages)
        round_.stream([stdout.encode()], delay=0.02)
        wait_drained(session)
        round_.finish(stdout.encode(), exit_code=3)     # exit=0 predicted
        metrics = round_.metrics
        self.assertTrue(metrics["failclosed"])
        self.assertEqual(metrics["recovery"], "rewind")
        self.assertNotIn("reset", [e[0] for e in session.events])
        # the prefix the baseline reuses survived the fail-closed path
        self.assertGreaterEqual(metrics["rewound_to"], len(base_ids))
        self.assertEqual(metrics["rewound_to"], metrics["common_prefix_len"])
        self.assert_byte_identical_vs_noneager(round_)
        # ... and the post-exit work is NOT larger than the non-eager path
        reference = MockSession()
        reference.prefill_extend(base_ids)
        reference.events.clear()
        reference.generate_consume(round_.final_ids, reuse_prefix=True)
        eager_rows = len([e for e in session.events
                          if e[0] == "consume" and e[3] > round_.exit_time])
        base_rows = len([e for e in reference.events if e[0] == "consume"])
        self.assertLessEqual(eager_rows, base_rows)

    def test_failclosed_full_reset_when_worker_state_uncertain(self):
        class ExplodingSession(MockSession):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def prefill_extend(self, tokens):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("device fault mid-batch")
                super().prefill_extend(tokens)

        wt = make_workspace_tools(self._tmp.name)
        session = ExplodingSession()
        stdout = "".join(f"line {i}\n" for i in range(12))
        round_ = EagerRound(session, wt, base_messages())
        round_.stream([stdout.encode()], delay=0.02)
        wait_drained(session)
        round_.finish(stdout.encode())
        self.assertTrue(round_.metrics["failclosed"])
        self.assertIn("worker-error", round_.metrics["reason"])
        self.assertEqual(round_.metrics["recovery"], "reset")
        self.assertIn("reset", [e[0] for e in session.events])
        self.assert_byte_identical_vs_noneager(round_)

    def test_failclosed_full_reset_when_session_cannot_rewind(self):
        class NoRewindSession(MockSession):
            rewind_to = None

        wt = make_workspace_tools(self._tmp.name)
        session = NoRewindSession()
        stdout = "attempt one\npartial work\n"
        round_ = EagerRound(session, wt, base_messages())
        round_.stream([stdout.encode()], delay=0.02)
        wait_drained(session)
        round_.finish(stdout.encode(), exit_code=3)
        self.assertTrue(round_.metrics["failclosed"])
        self.assertEqual(round_.metrics["recovery"], "reset")
        self.assertIn("no rewind_to", round_.metrics["recovery_reason"])
        self.assert_byte_identical_vs_noneager(round_)

    def test_failclosed_full_reset_when_rewind_does_not_verify(self):
        class LyingRewindSession(MockSession):
            def rewind_to(self, count):
                super().rewind_to(max(0, count - 3))   # rewinds too far

        wt = make_workspace_tools(self._tmp.name)
        session = LyingRewindSession()
        stdout = "attempt one\npartial work\n"
        round_ = EagerRound(session, wt, base_messages())
        round_.stream([stdout.encode()], delay=0.02)
        wait_drained(session)
        round_.finish(stdout.encode(), exit_code=3)
        self.assertTrue(round_.metrics["failclosed"])
        self.assertEqual(round_.metrics["recovery"], "reset")
        self.assertIn("verify failed", round_.metrics["recovery_reason"])
        self.assert_byte_identical_vs_noneager(round_)

    def test_forced_prediction_corruption_fails_closed(self):
        wt = make_workspace_tools(self._tmp.name)
        session = MockSession()
        round_ = EagerRound(session, wt, base_messages())
        # sabotage the planner AFTER begin: predictions no longer match
        round_.planner.optimistic_exit = 7
        round_.stream([b"some output\nmore output\n"], delay=0.02)
        wait_drained(session)
        round_.finish(b"some output\nmore output\n", exit_code=0)
        self.assertTrue(round_.metrics["failclosed"])
        self.assertIn("mismatch", round_.metrics["reason"])
        self.assert_byte_identical_vs_noneager(round_)


class ComponentTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_builder_matches_render_chat_bytes(self):
        wt = make_workspace_tools(self._tmp.name)
        for thinking in (False, True):
            for trail in ([], [{"role": "tool", "content": "prior result"}]):
                messages = base_messages() + trail
                builder = ProvisionalPromptBuilder(
                    AGENT, messages, wt.definitions,
                    enable_thinking=thinking)
                content = "Tool run_shell succeeded. ref=r1 " \
                          "type=shell_output.\nexit=0\nhello"
                expected = AGENT.render_chat(
                    messages + [{"role": "tool", "content": content}],
                    wt.definitions, add_generation_prompt=True,
                    enable_thinking=thinking)
                self.assertEqual(
                    builder.text(content, closed=True), expected)
                self.assertTrue(expected.startswith(
                    builder.text(content[:20], closed=False)))
                self.assertTrue(expected.startswith(builder.open_text()))

    def test_planner_prediction_is_prefix_of_production_content(self):
        for cap, stdout in (
                (800, "abc\ndef line\n"),
                (800, "  leading ws\nmid\n"),
                (64, "".join(f"long line {i}\n" for i in range(8))),
                (800, "no newline at all"),
                (800, "")):
            with tempfile.TemporaryDirectory() as tmp:
                wt = make_workspace_tools(tmp, max_output_chars=cap)
                planner = RunShellContentPlanner(
                    ref="r1", max_output_chars=cap)
                planner.on_stdout(stdout.encode())
                content, _frozen = planner.provisional()
                production = final_tool_content(
                    wt, production_output_string(stdout.encode(), 0))
                if content is not None:
                    self.assertTrue(
                        production.startswith(content),
                        f"cap={cap}\npred={content!r}\nprod={production!r}")

    def test_utf8_line_gate_holds_partials(self):
        gate = Utf8LineGate()
        gate.feed("汉".encode()[:2])
        self.assertEqual(gate.committed_text(), "")
        gate.feed("汉".encode()[2:] + b"x\r")
        gate.feed(b"\nrest")
        self.assertEqual(gate.committed_text(), "汉x\n")
        self.assertEqual(gate.pending_text(), "rest")

    def test_stable_feed_rollback_on_merge(self):
        feed = StableTokenFeed(encode, holdback=1)
        plan = feed.plan("a\n")
        feed.planned.extend(plan.new_tokens)     # fed "a" (holdback ate \n)
        plan = feed.plan("a\n\nb\n")             # "\n\n" merge appears
        self.assertEqual(plan.rollback_to, None)  # "a" still stable
        feed.planned = ["a", "\n"]               # pretend "\n" was fed
        plan = feed.plan("a\n\nb\n")
        self.assertEqual(plan.rollback_to, 1)    # "\n" -> "\n\n" rewrite

    def test_streaming_executor_matches_production_run_shell(self):
        """The two _run_shell branches are indistinguishable to a caller."""
        cases = [
            "printf 'a\\nb'; echo err >&2; exit 3",
            "echo one; sleep 0.05; echo two",
            "printf 'no trailing newline'",
            "printf '  padded  \\n\\n'",
            "true",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            production = AGENT.WorkspaceTools(Path(tmp))
            eager = AGENT.WorkspaceTools(Path(tmp), max_output_chars=800)
            chunks = []
            eager.stdout_listener = chunks.append
            for command in cases:
                args = {"command": command}
                self.assertIsNone(production.stdout_listener)
                self.assertEqual(eager._run_shell(dict(args)),
                                 production._run_shell(dict(args)),
                                 command)

    def test_pump_streams_small_chunks_before_tool_exit(self):
        """Regression for defect 1 (board-diagnosed 2026-08-16).

        ``BufferedReader.read(4096)`` blocks until 4096 bytes OR EOF, so a
        tool whose whole output is smaller than one buffer delivers exactly
        one chunk, at exit -- the eager engine then has nothing to feed.  The
        pump must read the raw fd and surface whatever is ready.
        """
        with tempfile.TemporaryDirectory() as tmp:
            arrivals = []
            tools = AGENT.WorkspaceTools(Path(tmp), max_output_chars=8000)
            tools.stdout_listener = lambda chunk: arrivals.append(
                (time.perf_counter(), len(chunk)))
            began = time.perf_counter()
            output = tools._run_shell({
                "command": "for i in 1 2 3 4 5 6 7 8; do echo short line $i;"
                           " sleep 0.08; done"})
            wall = time.perf_counter() - began
            total = sum(size for _t, size in arrivals)
            self.assertLess(total, 4096,
                            "fixture must stay inside one buffer to be a test")
            self.assertGreaterEqual(len(arrivals), 6, arrivals)
            self.assertLess(arrivals[0][0] - began, wall * 0.5,
                            "first chunk arrived only near EOF")
            self.assertEqual(output.count("short line"), 8)

    def test_end_to_end_real_subprocess_with_mock_session(self):
        session = MockSession()
        with tempfile.TemporaryDirectory() as tmp:
            wt = AGENT.WorkspaceTools(Path(tmp), max_output_chars=800)
            messages = base_messages()
            call = AGENT.ToolCall(name="run_shell", arguments={
                "command":
                "for i in 1 2 3 4 5 6; do echo streaming line $i; "
                "sleep 0.05; done"})
            self.assertTrue(eager_agent.eligible(call))
            launched = time.perf_counter()
            tool_result, engine, _planner = \
                eager_agent.run_tool_call_with_eager_prefill(
                    session=session, encode=encode, workspace_tools=wt,
                    call=call, messages=messages,
                    tool_definitions=wt.definitions,
                    enable_thinking=False, approve=lambda _p: True,
                    holdback=HOLDBACK, batch_tokens=4)
            exit_time = time.perf_counter()
            # the listener must be detached again after the call
            self.assertIsNone(wt.stdout_listener)
            content = AGENT.WorkspaceTools.for_model(tool_result)
            final_messages = messages + [
                {"role": "tool", "content": content}]
            rendered = AGENT.render_chat(
                final_messages, wt.definitions,
                add_generation_prompt=True, enable_thinking=False)
            final_ids = encode(rendered)
            metrics = engine.finalize(final_ids)
            session.generate_consume(final_ids, reuse_prefix=True)
            self.assertFalse(metrics["failclosed"], metrics)
            self.assertEqual(list(session.resident), final_ids)
            eager_feeds = [e for e in session.events
                           if e[0] == "feed" and launched < e[3] < exit_time]
            self.assertGreater(len(eager_feeds), 10,
                               "no overlap between tool and prefill")


class HardPreconditionTests(unittest.TestCase):
    """Condition 3: refuse loudly at startup, never degrade at runtime."""

    def test_accepts_resident_kv_plus_prefix_reuse(self):
        self.assertEqual(
            check_preconditions(session=MockSession(), reuse_prefix=True), [])
        self.assertTrue(
            require_preconditions(session=MockSession(), reuse_prefix=True))

    def test_refuses_without_prefix_reuse(self):
        with self.assertRaises(EagerPreconditionError) as caught:
            require_preconditions(session=MockSession(), reuse_prefix=False)
        self.assertIn("--reuse-session-kv", str(caught.exception))

    def test_refuses_without_resident_kv(self):
        session = MockSession()
        session.resident_kv = False
        with self.assertRaises(EagerPreconditionError) as caught:
            require_preconditions(session=session, reuse_prefix=True)
        self.assertIn("resident KV", str(caught.exception))

    def test_refuses_outside_agent_mode(self):
        with self.assertRaises(EagerPreconditionError) as caught:
            require_preconditions(session=MockSession(), reuse_prefix=True,
                                  agent_mode=False)
        self.assertIn("--agent", str(caught.exception))

    def test_refuses_when_session_lacks_the_protocol(self):
        class Bare:
            resident_kv = True

        unmet = check_preconditions(session=Bare(), reuse_prefix=True)
        self.assertEqual(len(unmet), 4, unmet)
        for name in ("resident_tokens", "prefill_extend", "rewind_to",
                     "reset_kv"):
            self.assertTrue(any(name in item for item in unmet), unmet)

    def test_all_unmet_conditions_are_reported_together(self):
        session = MockSession()
        session.resident_kv = False
        unmet = check_preconditions(session=session, reuse_prefix=False)
        self.assertEqual(len(unmet), 2, unmet)


class LandedWiringTests(unittest.TestCase):
    """Condition 2: real modules on the real classes, not an overlay."""

    @staticmethod
    def _server_module():
        sys.path.insert(0, str(APP_SRC))
        spec = importlib.util.spec_from_file_location(
            "merged_board_server_eager_test",
            APP_SRC / "merged_board_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_modules_are_regular_files_not_symlinks(self):
        for name in ("eager_prefill.py", "eager_agent.py", "eager_server.py"):
            path = APP_SRC / name
            self.assertTrue(path.is_file(), path)
            self.assertFalse(path.is_symlink(), f"{path} is a symlink")

    def test_merged_inherits_the_eager_session_protocol(self):
        server = self._server_module()
        self.assertTrue(
            issubclass(server.Merged, eager_server.EagerSessionMixin))
        for name in ("resident_tokens", "prefill_extend", "rewind_to",
                     "reset_kv"):
            self.assertTrue(callable(getattr(server.Merged, name, None)), name)

    def test_eager_and_generate_share_one_mask_builder(self):
        """The eager row must be byte-identical to the non-eager row."""
        server = self._server_module()
        helper = getattr(server.Merged, "_attention_mask_bytes", None)
        self.assertTrue(callable(helper))
        # the helper reproduces the mask formula generate used inline
        import struct
        for position, width in ((0, 8), (3, 8), (7, 8), (0, 4)):
            expected = bytearray(
                struct.pack("<f", server.gc.MASK_NEG) * width)
            for slot in range(position):
                struct.pack_into("<f", expected, slot * 4, 0.0)
            struct.pack_into("<f", expected, (width - 1) * 4, 0.0)
            self.assertEqual(helper(object(), position, width),
                             bytes(expected))
        # ... and BOTH callers now go through it
        source = (APP_SRC / "merged_board_server.py").read_text()
        generate_body = source.split("def generate(", 1)[1].split(
            "\n    def ", 1)[0]
        self.assertNotIn("gc.MASK_NEG", generate_body)
        self.assertIn("_attention_mask_bytes", generate_body)
        self.assertIn(
            "_attention_mask_bytes",
            (APP_SRC / "eager_server.py").read_text())

    def test_flag_is_default_off_and_plumbed(self):
        source = (APP_SRC / "merged_board_server.py").read_text()
        self.assertIn('"--eager-tool-prefill", action="store_true"', source)
        for guard in (
                "--eager-tool-prefill requires --agent",
                "--eager-tool-prefill requires resident KV",
                "--eager-tool-prefill requires --reuse-session-kv"):
            self.assertIn(guard, source)
        chat = (PROJECT / "app" / "chat.sh").read_text()
        self.assertIn("EAGER_TOOL_PREFILL=${EAGER_TOOL_PREFILL:-0}", chat)
        self.assertIn("--eager-tool-prefill", chat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
