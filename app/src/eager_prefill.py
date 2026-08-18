#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Eager streaming prefill core.

While a tool subprocess streams stdout the NPU is idle for seconds.  This
module drains that idle time by prefilling the *stable* token prefix of the
prompt the agent loop will assemble once the tool exits.  Correctness
invariant: the token sequence ultimately resident and consumed by
``generate`` is BYTE-IDENTICAL to the non-eager path.  Anything uncertain is
held back or rolled back; any finalize-time mismatch fails closed into
``reset_kv`` + full re-prefill.

Board-proven 2026-08-16 (11 rounds x 2 arms on SS928): fed ids, generated
ids and tool strings byte-identical between the eager and non-eager arms;
perceived tool-exit-to-first-token down 86-96% on rounds where the tool
outlives the decode tail; the fail-closed round costs 0.85x baseline.

The session side lives in ``eager_server.EagerSessionMixin`` (inherited by
``merged_board_server.Merged``), the agent side in ``eager_agent``.  The
feature is gated behind ``--eager-tool-prefill`` and is default OFF.
"""
from __future__ import annotations

import codecs
from contextlib import contextmanager
from dataclasses import dataclass, field
import io
import threading
import time

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"


class EagerCapacityReached(Exception):
    """Resident cache cannot accept more eager rows (position >= context-1)."""


class EagerPreconditionError(RuntimeError):
    """Eager prefill was requested without the state that makes it useful."""


def check_preconditions(*, session, reuse_prefix, agent_mode=True):
    """Return the list of unmet hard preconditions for eager tool prefill.

    Eager rows are appended to the RESIDENT prefix and are only ever consumed
    by a following ``generate(..., reuse_prefix=True)``.  Without resident KV
    the rows cannot exist at all; without prefix reuse ``generate`` resets the
    prefix and every eagerly fed row is discarded -- correct, but pointless,
    and it buys the complexity for nothing.  Both are therefore checked and
    REFUSED at startup rather than silently degraded at runtime.
    """
    unmet = []
    if not agent_mode:
        unmet.append("agent mode (--agent): eager prefill only has a tool loop "
                     "to overlap with in agent mode")
    if not getattr(session, "resident_kv", False):
        unmet.append("resident KV: eager rows live in the resident prefix; "
                     "--host-kv discards them")
    if not reuse_prefix:
        unmet.append("prefix reuse (--reuse-session-kv): without it generate "
                     "resets the prefix and every eagerly fed row is discarded")
    for name in ("resident_tokens", "prefill_extend", "rewind_to", "reset_kv"):
        if not callable(getattr(session, name, None)):
            unmet.append(f"session protocol: {name}() is missing")
    return unmet


def require_preconditions(*, session, reuse_prefix, agent_mode=True):
    """Raise ``EagerPreconditionError`` listing every unmet precondition."""
    unmet = check_preconditions(
        session=session, reuse_prefix=reuse_prefix, agent_mode=agent_mode)
    if unmet:
        raise EagerPreconditionError(
            "--eager-tool-prefill requires:\n  - " + "\n  - ".join(unmet))
    return True


# --------------------------------------------------------------------------
# byte -> text discipline
# --------------------------------------------------------------------------
class Utf8LineGate:
    """Incremental bytes -> text with newline translation and line holdback.

    Mirrors the production decode contract (``subprocess.run(text=True)``):
    strict UTF-8 plus universal-newline translation.  A partial multi-byte
    sequence or a lone ``\\r`` at a chunk boundary is held by the incremental
    decoders, never surfaced.  ``committed_text`` ends at the last complete
    line; the unterminated tail is only visible via ``pending_text`` and is
    never eagerly fed (it is re-tokenized when complete).
    """

    def __init__(self) -> None:
        self._decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")(errors="strict"),
            translate=True)
        self._text = ""

    def feed(self, chunk: bytes) -> None:
        self._text += self._decoder.decode(chunk)

    def committed_text(self) -> str:
        cut = self._text.rfind("\n")
        return "" if cut < 0 else self._text[:cut + 1]

    def pending_text(self) -> str:
        cut = self._text.rfind("\n")
        return self._text[cut + 1:]

    def close(self) -> str:
        self._text += self._decoder.decode(b"", final=True)
        return self._text


# --------------------------------------------------------------------------
# tool-content prediction (run_shell flavour)
# --------------------------------------------------------------------------
class RunShellContentPlanner:
    """Predict the stable prefix of the tool message the model will see.

    The production content is ``WorkspaceTools.for_model(result(...))``:

        Tool run_shell succeeded. ref=rN type=shell_output.[ offset clause]
        exit=RC
        <combined stdout/stderr, .strip()ed, char-capped, '<'-escaped>

    Three quantities are unknown while the tool runs and are handled as:

    * ``ok``/``exit=RC``   -> optimistic (``succeeded``/``exit=0``); a wrong
      guess is caught at finalize and fails closed (cost == today's path).
    * truncation/offset    -> optimistic "not truncated" until the char cap
      is crossed; crossing forces one managed rollback of the tool message
      region, after which the content is frozen (fully determined).
    * stderr               -> never eagerly fed (appended after stdout in the
      production format); it lands in the finalize tail.
    """

    def __init__(self, *, ref: str, max_output_chars: int,
                 tool_name: str = "run_shell",
                 result_type: str = "shell_output",
                 optimistic_exit: int = 0,
                 optimistic_truncation: bool = True) -> None:
        self.ref = ref
        self.max_output_chars = int(max_output_chars)
        self.tool_name = tool_name
        self.result_type = result_type
        self.optimistic_exit = int(optimistic_exit)
        self.optimistic_truncation = bool(optimistic_truncation)
        self._gate = Utf8LineGate()
        self._lock = threading.Lock()

    def on_stdout(self, chunk: bytes) -> None:
        with self._lock:
            self._gate.feed(chunk)

    def _stable_stream(self) -> str:
        committed = self._gate.committed_text()
        stream = committed.lstrip()          # final content is .strip()ed
        return stream[:len(stream.rstrip())]  # withhold unstable trailing ws

    def provisional(self) -> tuple[str | None, bool]:
        """Return (content-or-None, frozen).

        ``None``: nothing of the tool message may be fed yet.
        ``frozen``: the content is fully determined under the optimistic
        assumptions, so the closing template + generation prompt may be fed.
        """
        with self._lock:
            raw = f"exit={self.optimistic_exit}\n" + self._stable_stream()
            cap = self.max_output_chars
            if len(raw) > cap:
                shown, truncated, frozen = raw[:cap], True, True
            else:
                if not self.optimistic_truncation:
                    return None, False
                shown, truncated, frozen = raw, False, False
            header = (f"Tool {self.tool_name} succeeded. "
                      f"ref={self.ref} type={self.result_type}.")
            if truncated:
                header += (f" More is available at offset={len(shown)} "
                           "with read_result_page.")
            return header + "\n" + shown.replace("<", "\\u003c"), frozen


# --------------------------------------------------------------------------
# provisional prompt text (must be an exact text prefix of render_chat output)
# --------------------------------------------------------------------------
class ProvisionalPromptBuilder:
    """Build render_chat-identical text with a growing final tool message.

    ``base_messages`` must end with the assistant message that carries the
    streaming tool call (prior *complete* tool messages may precede it or
    trail after it in the same user block).  A byte-equality unit test pins
    this builder against the real ``minicpm_agent.render_chat``.
    """

    def __init__(self, agent_module, base_messages: list[dict],
                 tool_definitions: list[dict], *, enable_thinking: bool) -> None:
        self._enable_thinking = bool(enable_thinking)
        trail: list[str] = []
        head = list(base_messages)
        while head and head[-1].get("role") == "tool":
            trail.insert(0, str(head[-1].get("content", "")))
            head.pop()
        base_text = agent_module.render_chat(
            head, tool_definitions, add_generation_prompt=False,
            enable_thinking=False)
        prefix = base_text + IM_START + "user"
        for content in trail:
            prefix += ("\n" + TOOL_RESPONSE_OPEN + "\n" + content
                       + "\n" + TOOL_RESPONSE_CLOSE)
        self._prefix = prefix

    def open_text(self) -> str:
        return self._prefix + "\n" + TOOL_RESPONSE_OPEN + "\n"

    def text(self, content: str, *, closed: bool) -> str:
        out = self.open_text() + content
        if closed:
            out += ("\n" + TOOL_RESPONSE_CLOSE + IM_END + "\n"
                    + IM_START + "assistant\n")
            out += ("<think>\n" if self._enable_thinking
                    else "<think>\n\n</think>\n\n")
        return out


# --------------------------------------------------------------------------
# token boundary discipline
# --------------------------------------------------------------------------
def common_prefix_len(left, right) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


@dataclass
class FeedPlan:
    rollback_to: int | None   # absolute resident length to rewind to, or None
    new_tokens: list = field(default_factory=list)


class StableTokenFeed:
    """Track fed tokens; hold back a trailing margin against BPE re-merges.

    Text-level discipline (held-back unterminated line) bounds how far a new
    chunk can rewrite the rendered text; the token ``holdback`` additionally
    protects against tokenizer merges across the committed boundary (e.g.
    ``"\\n"+"\\n" -> "\\n\\n"``).  Every plan is recomputed from a full
    re-encode, so a diverged prefix is always detected and turned into an
    explicit rollback -- never silently overwritten.
    """

    def __init__(self, encode, *, holdback: int = 8) -> None:
        if holdback < 1:
            raise ValueError("holdback must be >= 1")
        self.encode = encode
        self.holdback = int(holdback)
        self.planned: list = []

    def seed(self, tokens) -> None:
        self.planned = list(tokens)

    def plan(self, text: str) -> FeedPlan:
        tokens = list(self.encode(text))
        stable = tokens[:max(0, len(tokens) - self.holdback)]
        common = common_prefix_len(self.planned, stable)
        rollback_to = common if common < len(self.planned) else None
        return FeedPlan(rollback_to, stable[common:])


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class EagerPrefillEngine:
    """Drain an eager-prefill queue while a tool subprocess streams stdout.

    Session protocol (implemented by ``eager_server.EagerMerged`` and the
    test mock):

        resident_tokens() -> tuple      # host view of device-resident prefix
        prefill_extend(tokens) -> None  # append rows at resident length,
                                        # head skipped (raises
                                        # EagerCapacityReached at context-1)
        rewind_to(n) -> None            # truncate host metadata; stale device
                                        # rows stay causally masked
        reset_kv() -> None              # full invalidation (fail-closed)

    Threading: one worker thread; every device interaction happens under
    ``session_mutex``, in batches of ``batch_tokens``, so ``exclusive()``
    (used by any interleaved generate/decode request) blocks feeding with
    bounded latency.  ``finalize`` joins the worker before verification, so
    the follow-up ``generate`` call never races the feeder.
    """

    def __init__(self, session, encode, *, holdback: int = 8,
                 batch_tokens: int = 8, clock=time.perf_counter) -> None:
        self.session = session
        self.session_mutex = threading.RLock()
        self._feed = StableTokenFeed(encode, holdback=holdback)
        self.batch_tokens = int(batch_tokens)
        self._clock = clock
        self._cond = threading.Condition()
        self._dirty = False
        self._stop = False
        self._paused = False
        self._worker: threading.Thread | None = None
        self._render = None
        self._error: BaseException | None = None
        self._capacity = False
        self.metrics = {
            "eager_fed_tokens": 0, "rollbacks": 0, "rolled_back_tokens": 0,
            "failclosed": False, "reason": "", "resident_at_finalize": 0,
            "seed_resident": 0, "feed_timestamps": [], "finalize_ms": 0.0,
            # fail-closed recovery bookkeeping (see finalize)
            "recovery": "", "recovery_reason": "", "rewound_to": None,
            "common_prefix_len": None,
        }

    # -- lifecycle ---------------------------------------------------------
    def begin(self, render_provisional) -> None:
        if self._worker is not None:
            raise RuntimeError("engine already started")
        self._render = render_provisional
        self._feed.seed(self.session.resident_tokens())
        self.metrics["seed_resident"] = len(self._feed.planned)
        self._worker = threading.Thread(
            target=self._run, name="eager-prefill", daemon=True)
        self._worker.start()
        self.notify()

    def notify(self) -> None:
        with self._cond:
            self._dirty = True
            self._cond.notify_all()

    @contextmanager
    def exclusive(self):
        """Strict mutual exclusion for generate/decode against the feeder."""
        with self._cond:
            self._paused = True
        try:
            with self.session_mutex:
                yield self.session
        finally:
            with self._cond:
                self._paused = False
                self._cond.notify_all()

    def finalize(self, final_ids) -> dict:
        """Stop feeding, verify against the assembled prompt, fail closed.

        On success the resident prefix is a strict prefix of ``final_ids``
        and the caller's ordinary ``generate(final_ids, reuse_prefix=True)``
        prefills exactly the missing tail (``_prefix_plan`` picks the prefix
        up).

        On a *mismatch* the eager feed is un-done to the longest verified
        common prefix with ``rewind_to`` -- the same managed-rollback
        contract the in-flight feeder already uses -- so the round costs the
        NON-eager path and nothing more.  ``reset_kv`` discards the whole
        resident prefix INCLUDING the pre-tool prefix that the non-eager path
        reuses, which is strictly worse than baseline (board-measured
        2026-08-16: fail-closed round 39874 ms vs baseline 11951 ms,
        prefix_cache_hit 0 vs 355).  It is therefore kept only for the cases
        where the common prefix cannot be established with certainty:
        a worker exception (device rows may be half-written under an
        exception that escaped ``prefill_extend``), a ``rewind_to`` that
        raises, or a post-rewind state that does not verify.
        """
        began = self._clock()
        try:
            with self._cond:
                self._stop = True
                self._cond.notify_all()
            if self._worker is not None:
                self._worker.join()
                self._worker = None
            final_ids = list(final_ids)
            resident = list(self.session.resident_tokens())
            self.metrics["resident_at_finalize"] = len(resident)
            if self._error is not None:
                self.metrics.update(
                    failclosed=True, reason=f"worker-error: {self._error!r}",
                    recovery_reason="worker state uncertain")
                self._full_reset()
                return dict(self.metrics)
            if resident != final_ids[:len(resident)]:
                keep = common_prefix_len(resident, final_ids)
                self.metrics.update(
                    failclosed=True, reason="fed-tokens/final-prompt mismatch",
                    common_prefix_len=keep)
                self._recover_to_prefix(keep, final_ids)
                return dict(self.metrics)
            self.metrics["reason"] = "verified-prefix"
            return dict(self.metrics)
        finally:
            self.metrics["finalize_ms"] = (self._clock() - began) * 1000.0

    def report(self) -> dict:
        """Canonical per-round metric block for the server's report file."""
        metrics = self.metrics
        return {
            "tokens_fed_eagerly": int(metrics["eager_fed_tokens"]),
            "managed_rollbacks": int(metrics["rollbacks"]),
            "rolled_back_tokens": int(metrics["rolled_back_tokens"]),
            "failclosed": bool(metrics["failclosed"]),
            "reason": metrics["reason"],
            "recovery": metrics["recovery"],
            "recovery_reason": metrics["recovery_reason"],
            "rewound_to": metrics["rewound_to"],
            "common_prefix_len": metrics["common_prefix_len"],
            "resident_at_finalize": int(metrics["resident_at_finalize"]),
            "seed_resident": int(metrics["seed_resident"]),
            "capacity_reached": bool(self._capacity),
            "finalize_ms": round(float(metrics["finalize_ms"]), 3),
        }

    def _full_reset(self) -> None:
        self.metrics["recovery"] = "reset"
        self.metrics["rewound_to"] = 0
        self.session.reset_kv()

    def _recover_to_prefix(self, keep: int, final_ids: list) -> None:
        """Rewind to a VERIFIED common prefix; full reset if unverifiable."""
        rewind = getattr(self.session, "rewind_to", None)
        if not callable(rewind):
            self.metrics["recovery_reason"] = "session has no rewind_to"
            self._full_reset()
            return
        try:
            with self.session_mutex:
                rewind(keep)
                after = list(self.session.resident_tokens())
        except BaseException as error:
            self.metrics["recovery_reason"] = f"rewind failed: {error!r}"
            self._full_reset()
            return
        if len(after) != keep or after != final_ids[:keep]:
            # Cannot certify what the device holds -> fall back, fail closed.
            self.metrics["recovery_reason"] = (
                f"post-rewind verify failed (len={len(after)}, want={keep})")
            self._full_reset()
            return
        self.metrics["recovery"] = "rewind"
        self.metrics["rewound_to"] = keep
        self.metrics["recovery_reason"] = "verified common prefix"

    # -- worker ------------------------------------------------------------
    def _wait_for_work(self) -> bool:
        with self._cond:
            while not self._dirty and not self._stop:
                self._cond.wait(timeout=0.05)
            if self._stop:
                return False
            self._dirty = False
            return True

    def _run(self) -> None:
        try:
            while self._wait_for_work():
                text = self._render()
                if text is None:
                    continue
                plan = self._feed.plan(text)
                if plan.rollback_to is not None:
                    with self.session_mutex:
                        self.session.rewind_to(plan.rollback_to)
                    self.metrics["rollbacks"] += 1
                    self.metrics["rolled_back_tokens"] += (
                        len(self._feed.planned) - plan.rollback_to)
                    self._feed.planned = self._feed.planned[:plan.rollback_to]
                queue = plan.new_tokens
                at = 0
                while at < len(queue):
                    with self._cond:
                        if self._stop:
                            return
                        if self._paused:
                            self._cond.wait(timeout=0.05)
                            continue
                        if self._dirty:
                            break   # newer text available: re-plan
                    batch = queue[at:at + self.batch_tokens]
                    try:
                        with self.session_mutex:
                            self.session.prefill_extend(batch)
                    except EagerCapacityReached:
                        self._capacity = True
                        return
                    self._feed.planned.extend(batch)
                    self.metrics["eager_fed_tokens"] += len(batch)
                    self.metrics["feed_timestamps"].append(
                        (self._clock(), len(batch)))
                    at += len(batch)
        except BaseException as error:  # fail closed at finalize
            self._error = error
