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


def test_render_chat_can_enable_thinking_prefix() -> None:
    agent = _agent_module()
    rendered = agent.render_chat(
        [{"role": "user", "content": "Plan it"}],
        enable_thinking=True)

    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    assert "</think>" not in rendered


def test_stable_stream_buffers_incomplete_cjk_token() -> None:
    agent = _agent_module()

    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens
            return {1: "秦汉的兵马\ufffd", 2: "秦汉的兵马俑"}[len(ids)]

    stream = agent.StableTextStream(Tokenizer())
    assert stream.update([1]) == "秦汉的兵马"
    assert stream.update([1, 2]) == "俑"
    assert stream.finish([1, 2]) == ("", True)
    assert stream.text == "秦汉的兵马俑"


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


@pytest.mark.parametrize(("query", "previous", "path"), [
    ("列出当前路径的文件。", "", "."),
    ("list files in the workspace", "", "."),
    ("root", "", "/root"),
    ("路径是/root", "请先告诉我当前路径", "/root"),
])
def test_route_obvious_directory_intent(
        query: str, previous: str, path: str) -> None:
    agent = _agent_module()
    decision = agent.route_obvious_read_only(query, previous)

    assert decision is not None
    assert decision.mode == "DIRECT_TOOL"
    assert decision.response_policy == "DIRECT_FORMATTED"
    call = decision.tool_calls[0]
    assert call == agent.ToolCall(
        "list_directory", {"path": path, "max_entries": "10"})
    assert agent.parse_tool_calls(agent.format_tool_call(call))[0] == [call]


def test_read_only_router_does_not_guess_general_requests() -> None:
    agent = _agent_module()
    assert agent.route_obvious_read_only("介绍一下 root 用户") is None
    assert agent.route_obvious_read_only("修改 README") is None
    assert agent.route_obvious_read_only("explain README.md") is None

    transformed = agent.route_obvious_read_only("读取 README.md 并总结")
    assert transformed is not None
    assert transformed.mode == "TOOL_THEN_MODEL"
    assert transformed.response_policy == "MODEL_SUMMARY"
    assert transformed.schema_profile == "result_page"
    assert transformed.tool_calls == (agent.ToolCall(
        "read_file", {"path": "README.md", "start_line": "1",
                      "end_line": "40"}),)


@pytest.mark.parametrize(("query", "name", "arguments"), [
    ("当前工作目录是什么？", "current_directory", {}),
    ("pwd", "current_directory", {}),
    ("查看 Git 仓库状态", "git_status", {}),
    ("git status", "git_status", {}),
    ("读取 README.md 前 20 行", "read_file", {
        "path": "README.md", "start_line": "1", "end_line": "20"}),
    ("读取 README.md 前两行", "read_file", {
        "path": "README.md", "start_line": "1", "end_line": "2"}),
    ("show lines 10-15 of `docs/README.md`", "read_file", {
        "path": "docs/README.md", "start_line": "10", "end_line": "15"}),
    ("read runtime.py", "read_file", {
        "path": "runtime.py", "start_line": "1", "end_line": "40"}),
    ('在工作区搜索 "transformer"', "search_text", {
        "query": "transformer", "path": ".", "max_matches": "15"}),
    ("find `needle`", "search_text", {
        "query": "needle", "path": ".", "max_matches": "15"}),
])
def test_route_obvious_general_read_only_intents(
        query: str, name: str, arguments: dict[str, str]) -> None:
    agent = _agent_module()
    decision = agent.route_obvious_read_only(query)

    assert decision is not None
    assert decision.mode == "DIRECT_TOOL"
    assert decision.response_policy == "DIRECT_FORMATTED"
    assert decision.permission == "automatic"
    assert decision.tool_calls == (agent.ToolCall(name, arguments),)


def test_workspace_read_search_and_escape_guards(tmp_path: Path) -> None:
    agent = _agent_module()
    (tmp_path / "note.txt").write_text("alpha\nbeta alpha\n", encoding="utf-8")
    tools = agent.WorkspaceTools(tmp_path)

    current = json.loads(tools.execute(agent.ToolCall("current_directory", {})))
    listing = json.loads(tools.execute(agent.ToolCall(
        "list_directory", {"path": ".", "max_entries": "10"})))
    reading = json.loads(tools.execute(agent.ToolCall(
        "read_file", {"path": "note.txt", "start_line": "2", "end_line": "2"})))
    search = json.loads(tools.execute(agent.ToolCall(
        "search_text", {"query": "alpha", "path": "."})))
    escaped = json.loads(tools.execute(agent.ToolCall(
        "read_file", {"path": "../outside.txt"})))

    assert current["ok"] and current["output"] == str(tmp_path)
    assert listing["ok"] and "note.txt" in listing["output"]
    assert reading["output"] == "2: beta alpha"
    assert "note.txt:1:alpha" in search["output"]
    assert not escaped["ok"] and "escapes" in escaped["output"]


def test_search_does_not_follow_workspace_symlinks(tmp_path: Path) -> None:
    agent = _agent_module()
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("unique-secret-value\n", encoding="utf-8")
    (tmp_path / "leak.txt").symlink_to(outside)
    tools = agent.WorkspaceTools(tmp_path)

    result = json.loads(tools.execute(agent.ToolCall(
        "search_text", {"query": "unique-secret-value", "path": "."})))
    assert result["ok"]
    assert result["output"] == "[no matches]"


@pytest.mark.parametrize(("query", "profile"), [
    ("读取 README 并总结", "read_only"),
    ("修改 README 的标题", "read_write"),
    ("运行测试", "read_shell"),
    ("修复测试失败", "all"),
    ("help me with this project", "all"),
])
def test_progressive_tool_schema_selection(query: str, profile: str) -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == profile


@pytest.mark.parametrize("query", [
    "第二行有什么作用？",
    "这个是什么意思？",
    "what does that line do?",
])
def test_contextual_follow_up_omits_unrelated_tool_schemas(query: str) -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(
        query, has_context=True) == "none"
    assert agent.WorkspaceTools.select_schema_profile(query) == "all"


def test_contextual_mutation_or_shell_intent_remains_fail_closed() -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(
        "修改这一行", has_context=True) == "read_write"
    assert agent.WorkspaceTools.select_schema_profile(
        "运行这个命令", has_context=True) == "read_shell"


def test_context_rebase_preserves_recent_turns_and_omits_raw_tool_output() -> None:
    agent = _agent_module()
    messages = [{"role": "system", "content": "system"}]
    messages.extend([
        {"role": "user", "content": "old request one"},
        {"role": "assistant", "content": (
            '<function name="read_file"><param name="path">a</param>'
            "</function>")},
        {"role": "tool", "content": (
            "Tool read_file succeeded. ref=r1 type=text_lines.\n"
            "RAW-TOOL-OUTPUT-MUST-DISAPPEAR")},
        {"role": "assistant", "content": "old answer one"},
        {"role": "user", "content": "old request two"},
        {"role": "assistant", "content": "old answer two"},
        {"role": "user", "content": "recent request"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": "current request"},
    ])

    compacted, info = agent.compact_agent_messages(
        messages, keep_recent_turns=1, memory_chars=500)

    assert info["changed"] is True
    assert info["turns_compacted"] == 2
    assert compacted[0] == messages[0]
    assert compacted[-3:] == messages[-3:]
    rendered = json.dumps(compacted, ensure_ascii=False)
    assert "ref=r1 type=text_lines" in rendered
    assert "RAW-TOOL-OUTPUT-MUST-DISAPPEAR" not in rendered
    assert '<function name="read_file">' not in compacted[1]["content"]


def test_context_rebase_is_noop_without_old_complete_turns() -> None:
    agent = _agent_module()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "current"},
    ]
    compacted, info = agent.compact_agent_messages(
        messages, keep_recent_turns=1)
    assert compacted == messages
    assert info == {
        "changed": False, "turns_compacted": 0, "memory_chars": 0}


def test_progressive_tool_schema_is_fail_closed(tmp_path: Path) -> None:
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    read_names = tools.names_for_profile("read_only")
    assert read_names == (
        "current_directory", "list_directory", "read_file", "search_text",
        "git_status", "read_result_page")
    definitions = tools.definitions_for_profile("read_only")
    assert [item["function"]["name"] for item in definitions] == list(read_names)

    hidden_write = tools.execute(
        agent.ToolCall("write_file", {"path": "blocked.txt", "content": "x"}),
        approve=lambda _preview: True, allowed_names=read_names)
    assert not json.loads(hidden_write)["ok"]
    assert "not disclosed" in json.loads(hidden_write)["output"]
    assert not (tmp_path / "blocked.txt").exists()


def test_default_tool_results_fit_ctx1024_budget(tmp_path: Path) -> None:
    agent = _agent_module()
    for index in range(80):
        (tmp_path / f"very_long_generated_entry_name_{index:03d}").write_text("x")
    tools = agent.WorkspaceTools(tmp_path)

    listing = tools.execute(agent.ToolCall("list_directory", {}))

    decoded = json.loads(listing)
    assert decoded["ok"]
    assert len(decoded["output"].splitlines()) == 10
    assert len(decoded["output"]) <= agent.MAX_TOOL_OUTPUT_CHARS + 32

    model_result = tools.for_model(listing)
    assert model_result.startswith("Tool list_directory succeeded.")
    assert "ref=r1 type=directory_entries" in model_result
    assert "\\n" not in model_result
    direct_result = tools.for_direct(
        agent.ToolCall("list_directory", {"path": "."}), listing)
    assert direct_result.startswith("目录内容（.）：\n")
    assert "very_long_generated_entry_name_000" in direct_result


def test_typed_tool_results_can_be_paged(tmp_path: Path) -> None:
    agent = _agent_module()
    (tmp_path / "long.txt").write_text("x" * 200 + "\n", encoding="utf-8")
    tools = agent.WorkspaceTools(tmp_path, max_output_chars=64)
    call = agent.ToolCall(
        "read_file", {"path": "long.txt", "start_line": "1", "end_line": "1"})

    first_json = tools.execute(call)
    first = json.loads(first_json)
    assert first["ok"] and first["type"] == "text_lines"
    assert first["ref"] == "r1" and first["truncated"]
    assert first["next_offset"] == 64 and len(first["output"]) == 64
    assert "read_result_page" in tools.for_model(first_json)
    assert "结果引用 r1" in tools.for_direct(call, first_json)

    page = json.loads(tools.execute(agent.ToolCall(
        "read_result_page", {"ref": "r1", "offset": "64", "max_chars": "64"})))
    assert page["ok"] and page["ref"] == "r1"
    assert page["offset"] == 64 and page["next_offset"] == 128
    assert len(page["output"]) == 64

    missing = json.loads(tools.execute(agent.ToolCall(
        "read_result_page", {"ref": "missing"})))
    assert not missing["ok"] and "expired" in missing["output"]


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
