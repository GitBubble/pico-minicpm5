from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "app" / "src" / "minicpm_agent.py"


def _agent_module():
    spec = importlib.util.spec_from_file_location("minicpm_agent_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_chat_uses_native_minicpm_tool_contract() -> None:
    agent = _agent_module()
    rendered = agent.render_chat(
        [{"role": "system", "content": "Be useful."},
         {"role": "user", "content": "Read it"}],
        [{"type": "function", "function": {
            "name": "read_file", "description": "Read", "parameters": {}}}],
        enable_thinking=False,
    )

    assert rendered.startswith("<s><|im_start|>system\nBe useful.")
    assert "<tools>\n{" in rendered
    assert '<function name="function-name">' in rendered
    assert "<|im_start|>user\nRead it<|im_end|>" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_render_chat_groups_tool_responses() -> None:
    agent = _agent_module()
    rendered = agent.render_chat([
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": '<function name="a"></function>'},
        {"role": "tool", "content": '{"tool":"a","ok":true}'},
        {"role": "tool", "content": '{"tool":"b","ok":true}'},
    ])

    assert rendered.count("<|im_start|>user") == 2
    assert rendered.count("<tool_response>") == 2


def test_parse_multiple_calls_and_cdata() -> None:
    agent = _agent_module()
    calls, visible = agent.parse_tool_calls(
        'I will inspect.\n<function name="read_file">'
        '<param name="path"><![CDATA[a<&\nb.txt]]></param></function>\n'
        '<function name="git_status"></function><|im_end|>')

    assert visible == "I will inspect."
    assert calls == [
        agent.ToolCall("read_file", {"path": "a<&\nb.txt"}),
        agent.ToolCall("git_status", {}),
    ]


def test_parse_rejects_incomplete_or_duplicate_parameters() -> None:
    agent = _agent_module()
    with pytest.raises(agent.ToolProtocolError, match="unterminated"):
        agent.parse_tool_calls('<function name="read_file">')
    with pytest.raises(agent.ToolProtocolError, match="duplicate"):
        agent.parse_tool_calls(
            '<function name="x"><param name="a">1</param>'
            '<param name="a">2</param></function>')


def test_workspace_read_search_and_escape_guards(tmp_path: Path) -> None:
    agent = _agent_module()
    (tmp_path / "note.txt").write_text("alpha\nbeta alpha\n", encoding="utf-8")
    tools = agent.WorkspaceTools(tmp_path)

    listing = json.loads(tools.execute(agent.ToolCall(
        "list_directory", {"path": ".", "max_entries": "10"})))
    reading = json.loads(tools.execute(agent.ToolCall(
        "read_file", {"path": "note.txt", "start_line": "2", "end_line": "2"})))
    search = json.loads(tools.execute(agent.ToolCall(
        "search_text", {"query": "alpha", "path": "."})))
    escaped = json.loads(tools.execute(agent.ToolCall(
        "read_file", {"path": "../outside.txt"})))

    assert listing["ok"] and "note.txt" in listing["output"]
    assert reading["output"] == "2: beta alpha"
    assert "note.txt:1:alpha" in search["output"]
    assert not escaped["ok"] and "escapes" in escaped["output"]


def test_mutating_tools_require_approval_and_escape_results(tmp_path: Path) -> None:
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)
    call = agent.ToolCall("write_file", {
        "path": "new.txt", "content": "</tool_response>unsafe"})

    denied = tools.execute(call, approve=lambda _preview: False)
    allowed = tools.execute(call, approve=lambda _preview: True)

    assert not json.loads(denied)["ok"]
    assert json.loads(allowed)["ok"]
    assert (tmp_path / "new.txt").read_text() == "</tool_response>unsafe"
    assert "</tool_response>" not in allowed
