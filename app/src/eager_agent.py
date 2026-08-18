#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Agent-side eager-prefill orchestration: one streaming tool round.

First-class sibling of ``minicpm_agent`` -- imported by name, no path loading
and no subclassing at import time.  The streaming shell executor itself lives
in the real ``minicpm_agent.WorkspaceTools`` (attach a ``stdout_listener`` and
``_run_shell`` pumps the raw fds); the returned string is produced by the
ORDINARY ``execute``/``result``/``for_model`` path, so the prompt the agent
loop assembles after the tool exits is byte-identical to the non-eager path by
construction.  The eager engine only predicts a stable prefix of it.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import minicpm_agent as agent            # noqa: E402
from eager_prefill import (              # noqa: E402
    EagerPrefillEngine, ProvisionalPromptBuilder, RunShellContentPlanner)

# Only shell output streams incrementally; every other tool returns in one
# shot and has nothing for the engine to overlap with.
STREAMING_TOOLS = ("run_shell",)


def eligible(call) -> bool:
    """True when this call is worth streaming into the eager engine."""
    return getattr(call, "name", None) in STREAMING_TOOLS


def run_tool_call_with_eager_prefill(
        *, session, encode, workspace_tools, call, messages,
        tool_definitions, enable_thinking, approve=None, allowed_names=None,
        holdback=8, batch_tokens=8, engine_factory=EagerPrefillEngine):
    """Execute ONE streaming tool call while eagerly prefilling its output.

    ``messages`` must already contain the assistant message carrying the tool
    call (prior complete tool messages may trail it).  Returns
    ``(tool_result_json, engine, planner)``; the caller appends the tool
    message exactly as the non-eager loop does, re-renders the prompt ids and
    calls ``engine.finalize(final_ids)`` BEFORE
    ``session.generate(final_ids, ..., reuse_prefix=True)``.
    """
    planner = RunShellContentPlanner(
        ref=workspace_tools.next_ref(),
        max_output_chars=workspace_tools.max_output_chars,
        tool_name=call.name)
    builder = ProvisionalPromptBuilder(
        agent, messages, tool_definitions, enable_thinking=enable_thinking)
    engine = engine_factory(session, encode,
                            holdback=holdback, batch_tokens=batch_tokens)

    def render_provisional():
        content, frozen = planner.provisional()
        if content is None:
            return builder.open_text()
        return builder.text(content, closed=frozen)

    def on_stdout(chunk: bytes):
        planner.on_stdout(chunk)
        engine.notify()

    workspace_tools.stdout_listener = on_stdout
    engine.begin(render_provisional)
    try:
        tool_result = workspace_tools.execute(
            call, approve=approve, allowed_names=allowed_names)
    finally:
        workspace_tools.stdout_listener = None
    return tool_result, engine, planner
