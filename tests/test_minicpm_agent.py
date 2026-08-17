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
    ("修改 README 的标题", "write"),
    ("运行测试", "shell"),
    ("修复测试失败", "all"),
    # Was "all" while every unclassified turn fell through to the whole
    # schema. "project" is a workspace referent, but the turn carries no
    # mutation, shell or broad-work verb, so nothing justifies charging 117
    # ctx1024 prompt tokens for write_file and run_shell before the model has
    # looked at anything. The Chinese broad-work siblings 修复/实现/开发/调试
    # still return "all"; "help" is deliberately not one of them, because
    # \bhelp\b would also capture "帮我看看 README" and every help-phrased read.
    ("help me with this project", "read_only"),
])
def test_progressive_tool_schema_selection(query: str, profile: str) -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == profile


@pytest.mark.parametrize(("query", "profile", "tool"), [
    ("把结果写进 out.txt", "write", "write_file"),
    ("把结果写到 out.txt", "write", "write_file"),
    ("结果存到 notes.md", "write", "write_file"),
    ("记录到 log.txt 里", "write", "write_file"),
    # A delete verb also reaches run_shell: the registry has no delete
    # tool, so "clear it out" needs both the writer and the shell.
    ("帮我把 TODO 清掉", "write_shell", "write_file"),
    ("删除 a.txt", "write_shell", "run_shell"),
    ("把 swish(2) 的结果写进 out.txt", "write", "write_file"),
    ("启动服务", "shell", "run_shell"),
])
def test_mutation_verbs_beyond_the_dictionary_form(
        query: str, profile: str, tool: str) -> None:
    # 写入/创建/追加 are the dictionary spellings; 写进/写到/存到/清掉 are what
    # people type. Without them a "put it in a file" turn was disclosed a
    # profile with no write_file, and the escalation round cannot rescue that:
    # the model would have to emit a well-formed call to a tool it never saw.
    # Disclosure is still not authorization -- both tools ask per call.
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == profile
    assert tool in agent.WorkspaceTools.names_for_profile(profile)


@pytest.mark.parametrize("query", [
    "第二行有什么作用？",
    "这个是什么意思？",
    "what does that line do?",
])
def test_contextual_follow_up_omits_unrelated_tool_schemas(query: str) -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(
        query, has_context=True) == "none"
    # Was "all". That assertion encoded the old fallback rather than a
    # property of the input: as the FIRST message of a session there is no
    # line to inspect, no path and no file, so the eight tool schemas cannot
    # help and disclosing them costs 675 prompt tokens before the model can
    # ask "which line?". The has_context=True case above is unchanged, and so
    # is the fail-closed test below, which is what proves the contextual
    # branch still does not swallow mutation intent.
    assert agent.WorkspaceTools.select_schema_profile(query) == "none"


@pytest.mark.parametrize("query", [
    "你好，喵",
    "讲个笑话",
    "现在几点",
    "给我讲讲 transformer",
    "解释一下什么是梯度下降",
    "2 x 0.8808 是多少",
])
def test_untooled_turn_discloses_no_schema(query: str) -> None:
    # Deliberately unrelated chit-chat rather than greetings: detection is the
    # absence of tool evidence, not a 你好 whitelist. On the measured board a
    # greeting used to pay 675 fixed-prefix tokens (53.7 s) for eight schemas
    # it cannot call. The last case is the arithmetic the board measured this
    # model doing correctly in 14 tokens, so it stays with the model.
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == "none"
    assert agent.WorkspaceTools.names_for_profile("none") == ()


@pytest.mark.parametrize("query", [
    "列出当前目录的内容",
    "列出这个仓库里都有什么",
    "看看 src 里面都写了些什么",
    "帮我看一下项目结构",
    "工作区里有哪些东西",
    "检查一下配置",
    "ls",
    "list everything here",
    "what is in /root",
])
def test_read_intent_covers_list_and_look_phrasings(query: str) -> None:
    # 列出/看看/打开/检索 were already precedented inside route_obvious_read_only;
    # the planner's read group had simply fallen out of sync with it, so every
    # phrasing the deterministic router declines used to reach the model with
    # the whole schema attached.
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == "read_only"


def test_measured_directory_listing_stays_read_only() -> None:
    agent = _agent_module()
    decision = agent.route_obvious_read_only("列出当前目录的内容")

    assert decision is not None
    assert decision.mode == "DIRECT_TOOL"
    assert decision.tool_calls == (agent.ToolCall(
        "list_directory", {"path": ".", "max_entries": "10"}),)
    assert agent.WorkspaceTools.select_schema_profile(
        "列出当前目录的内容") == "read_only"


@pytest.mark.parametrize(("query", "profile"), [
    ("总结一下 README", "read_only"),
    ("分析这段代码", "read_only"),
    ("解释一下什么是梯度下降", "none"),
    ("翻译这段话：hello world", "none"),
])
def test_transform_needs_a_referent(query: str, profile: str) -> None:
    # 总结/分析/explain are transformations, not reads: they imply a tool only
    # once the turn names something a tool can reach, so they were moved out
    # of the unconditional read group and onto the referent rule.
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == profile


@pytest.mark.parametrize("query", [
    "请计算 swish(2)",
    "sigmoid(2) 是多少？",
    "sqrt(2) 等于多少",
    "求 ln(7) 保留四位小数",
    "2 的 30 次方",
    "123456789 * 987654 是多少",
    "20 的阶乘是多少",
    # A look verb with nothing to look at is arithmetic phrased as a read.
    # These used to select read_only: 625 fixed-prefix tokens (49.7 s) where
    # 265 (21.1 s) suffice, on turns that reach no file at all.
    "查看 swish(2) 的值",
    "显示 2^32 的结果",
    "show me sigmoid(2)",
    "find the value of gelu(1)",
])
def test_transcendental_math_is_dispatched(query: str) -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == "calculate"
    assert agent.WorkspaceTools.names_for_profile("calculate") == ("calculate",)


@pytest.mark.parametrize("query", [
    "日志里出现了 sigmoid(2) 这一行吗",
    "1024 * 768 的图片在哪个目录",
    "查看日志里的 sigmoid(2)",
])
def test_a_workspace_referent_outranks_a_number_in_the_same_turn(
        query: str) -> None:
    # 日志/目录 are workspace referents, so these turns need search_text and
    # list_directory; classifying them as pure arithmetic would have taken
    # every file tool away. read_only carries calculate, so the compute half
    # is served either way and only the file half is at stake.
    agent = _agent_module()
    profile = agent.WorkspaceTools.select_schema_profile(query)

    assert profile == "read_only"
    assert "calculate" in agent.WorkspaceTools.names_for_profile(profile)


@pytest.mark.parametrize(("query", "profile"), [
    ("那 gelu(2) 呢", "calculate"),
    ("这个保留四位小数", "calculate"),
    ("第二行有什么作用？", "none"),
    ("这个是什么意思？", "none"),
])
def test_contextual_follow_up_still_buys_the_digits(
        query: str, profile: str) -> None:
    # The measured swish(2) failure was itself a follow-up. Returning "none"
    # here saves 154 tokens and buys a wrong answer, because the model then
    # recalls the value instead of computing it -- which is how "1.7284" was
    # invented. Follow-ups that name no value keep the empty schema.
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(
        query, has_context=True) == profile


@pytest.mark.parametrize("query", [
    "2 加 3 等于几",
    "2 x 0.8808 是多少",
    "什么是 sigmoid 函数",
    "标准差是什么意思",
    "求 sigmoid 的导数公式",
    "22/7 约等于多少",
])
def test_algebra_and_concept_questions_stay_with_the_model(query: str) -> None:
    # The maintainer's split: where the model computes correctly it computes.
    # A named function with no numeric operand is a concept question, and the
    # board measured this model producing 2 x 0.8808 = 1.7616 correctly.
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(query) == "none"


@pytest.mark.parametrize(("query", "expression"), [
    ("请计算 swish(2)", "swish(2)"),
    ("sigmoid(2) 等于多少", "sigmoid(2)"),
    ("计算 2*0.8808", "2*0.8808"),
    ("what is 2^10", "2**10"),
    ("3.14 * 2", "3.14 * 2"),
])
def test_route_numeric_evaluation_to_the_host(
        query: str, expression: str) -> None:
    # The primary math path never calls the model, so no generated token can
    # carry a computed digit: 675 fixed-prefix tokens become zero.
    agent = _agent_module()
    decision = agent.route_obvious_read_only(query)

    assert decision is not None
    assert decision.mode == "DIRECT_TOOL"
    assert decision.response_policy == "DIRECT_FORMATTED"
    assert decision.schema_profile == "none"
    assert decision.permission == "automatic"
    assert decision.tool_calls == (agent.ToolCall(
        "calculate", {"expression": expression}),)


@pytest.mark.parametrize("query", [
    "为什么 swish 比 relu 平滑",
    "解释 2+2",
    "如何计算 2+2",
    "计算 2+2 并写入 result.txt",
    "计算 1/3 的结果，写进 out.txt",
    "算一下 1/3 然后存到 out.txt",
    "把 swish(2) 的结果记录到 log.txt",
    "运行测试并计算通过率",
    "search for 2+2",
    "读取 README.md 前 40 行",
    "第 1-40 行是什么",
    "你好，喵",
])
def test_symbolic_or_non_numeric_turns_are_not_calculated(query: str) -> None:
    # Conceptual mathematics is the model's own work, and the write/shell
    # tables disqualify the branch outright so a turn that also mutates never
    # has its mutation silently dropped in favour of the arithmetic. This is a
    # DIRECT_TOOL branch: whatever it answers IS the turn, so a dropped clause
    # is dropped in silence.
    agent = _agent_module()
    decision = agent.route_obvious_read_only(query)

    assert decision is None or decision.tool_calls[0].name != "calculate"


@pytest.mark.parametrize(("query", "path"), [
    ("读取 a.txt 并计算 2+2", "a.txt"),
    ("计算 2+2 并读取 a.txt", "a.txt"),
    ("打开 notes.txt 看看 3*4 是多少", "notes.txt"),
])
def test_a_named_file_outranks_arithmetic_in_the_same_turn(
        query: str, path: str) -> None:
    # Arithmetic is routed after the search and file-window routes, so the
    # standing precedence rule -- a named file wins -- applies to it too. The
    # opposite order answered the sum and never opened the file.
    agent = _agent_module()
    decision = agent.route_obvious_read_only(query)

    assert decision is not None
    assert decision.tool_calls == (agent.ToolCall(
        "read_file", {"path": path, "start_line": "1", "end_line": "40"}),)


@pytest.mark.parametrize("query", ["1-40", "3-5", "-1", "2+2"])
def test_a_bare_span_is_not_evaluated_on_its_shape_alone(query: str) -> None:
    # "1-40" is a line range answering the agent's own question far more often
    # than it is a subtraction, and "-1" is a literal rather than a
    # computation. A whole-turn span is evaluated only when a call, a decimal,
    # a power or a second operator leaves no other reading; anything else
    # needs one of the 计算/是多少 cues, and plain 2+2 stays with the model,
    # whose algebra the board measured as exact.
    agent = _agent_module()
    decision = agent.route_obvious_read_only(query, "请问要看哪几行？")

    assert decision is None


@pytest.mark.parametrize(("expression", "expected"), [
    ("swish(2)", "swish(2) = 1.7615941559557646"),
    ("silu(2)", "silu(2) = 1.7615941559557646"),
    ("sigmoid(2)", "sigmoid(2) = 0.8807970779778823"),
    ("gelu(2)", "gelu(2) = 1.9544997361036416"),
    ("gelu_tanh(2)", "gelu_tanh(2) = 1.954597694087775"),
    ("relu(-3)", "relu(-3) = 0.0"),
    ("2 * 0.8808", "2 * 0.8808 = 1.7616"),
    ("1234567 * 7654321", "1234567 * 7654321 = 9449772114007"),
    ("2^10", "2 ** 10 = 1024"),
    ("1+2^3", "1 + 2 ** 3 = 9"),
    ("GELU(2)", "GELU(2) = 1.9544997361036416"),
    # The planner dispatches 保留四位小数/精确到 turns here, so the vocabulary
    # has to contain the rounding they ask for; refusing it pushes the model
    # back onto the truncation habit that produced "1.7284".
    ("round(ln(7), 4)", "round(ln(7), 4) = 1.9459"),
    ("log(8, 2)", "log(8, 2) = 3.0"),
    # A Chinese IME left in fullwidth mode types this and means 1+1.
    ("１＋１", "1 + 1 = 2"),
])
def test_calculate_is_exact_where_the_model_is_not(
        expression: str, expected: str, tmp_path: Path) -> None:
    # swish(2) is the measured failure: the model answered 1.728 and then
    # invented "1.7284". gelu and gelu_tanh differ in the 4th decimal, so the
    # echoed expression is what keeps the two definitions distinguishable.
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    result = json.loads(tools.execute(
        agent.ToolCall("calculate", {"expression": expression})))

    assert result["ok"] and result["type"] == "number"
    assert result["output"] == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('id')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd')",
    "runtime.py",
    "[x for x in (1, 2)]",
    "lambda: 1",
    "2 ** 10 ** 9",
    "pow(2, 10 ** 9)",
    "(-8) ** 0.5",
    "1 / 0",
    "exp(10000)",
    "1e400",
    "'abc'",
    "1 & 2",
])
def test_calculate_refuses_everything_outside_arithmetic(
        expression: str, tmp_path: Path) -> None:
    # The classic escape needs Attribute and Subscript, and both are refused
    # during the tree walk before a single operation is evaluated. The two
    # bombs are here so that any future relaxation of _power is caught.
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    result = json.loads(tools.execute(
        agent.ToolCall("calculate", {"expression": expression})))

    assert not result["ok"]


def test_calculate_never_leaves_the_tool_error_contract(tmp_path: Path) -> None:
    # SyntaxError, ZeroDivisionError and OverflowError are not in execute()'s
    # except tuple, so the evaluator owns its own error taxonomy rather than
    # widening error handling for the eight other tools.
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    missing = json.loads(tools.execute(agent.ToolCall("calculate", {})))
    syntax = json.loads(tools.execute(
        agent.ToolCall("calculate", {"expression": "1 +"})))

    assert not missing["ok"] and "empty" in missing["output"]
    assert not syntax["ok"] and "cannot parse" in syntax["output"]


def test_calculate_is_auto_approved_and_direct_rendered(tmp_path: Path) -> None:
    # A rejecting approver still yields a result: this pins calculate to the
    # automatic side, so an edit that flips it to permission="ask" fails here.
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)
    call = agent.ToolCall("calculate", {"expression": "swish(2)"})

    result = tools.execute(call, approve=lambda _preview: False)

    assert json.loads(result)["ok"]
    assert tools.for_direct(call, result) == (
        "计算结果：swish(2) = 1.7615941559557646")
    assert tools.for_model(result).splitlines()[-1] == (
        "swish(2) = 1.7615941559557646")

    hidden = json.loads(tools.execute(
        call, allowed_names=tools.names_for_profile("result_page")))
    assert not hidden["ok"] and "not disclosed" in hidden["output"]


@pytest.mark.parametrize(("profile", "requested", "widened"), [
    ("none", ("list_directory",), "read_only"),
    ("read_only", ("write_file",), "read_write"),
    ("read_write", ("run_shell",), "all"),
    ("calculate", ("read_result_page",), "read_only"),
    ("read_only", ("read_file",), None),
    ("all", ("run_shell",), None),
])
def test_schema_escalation_widens_to_the_narrowest_superset(
        profile: str, requested: tuple[str, ...], widened: str | None) -> None:
    # Widening takes the union, so a read_write turn that then asks for a
    # shell escalates to "all" and never demotes to read_shell: a transcript
    # never loses a schema it has already used.
    agent = _agent_module()
    assert agent.WorkspaceTools.escalate_schema_profile(
        profile, requested) == widened
    with pytest.raises(ValueError, match="unknown tool schema profile"):
        agent.WorkspaceTools.escalate_schema_profile("bogus", requested)


def test_more_profiles_than_snapshot_slots_is_allowed() -> None:
    # The snapshot key is the profile name and the executor holds eight input
    # snapshots, so this used to cap the profile table at eight. It capped the
    # wrong thing: disclosure granularity is a correctness property -- nine
    # tools offered for "write pi.py then run it" was measured selecting a
    # tool that does not exist -- while a snapshot is only a cache. The board
    # now declines to cache the ninth prefix instead of ending the session,
    # so the table may exceed the slot count.
    agent = _agent_module()
    tools = agent.WorkspaceTools(Path("."))
    assert len(agent.WorkspaceTools._PROFILES) > 8
    for name in agent.WorkspaceTools._PROFILES:
        assert set(tools.names_for_profile(name)).issubset(tools.names)


def test_contextual_mutation_or_shell_intent_remains_fail_closed() -> None:
    agent = _agent_module()
    assert agent.WorkspaceTools.select_schema_profile(
        "修改这一行", has_context=True) == "write"
    assert agent.WorkspaceTools.select_schema_profile(
        "运行这个命令", has_context=True) == "shell"


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
    # This tuple is the inventory of the auto-approved set, and "calculate"
    # now genuinely belongs to it: it reaches no file, spawns no process and
    # resolves no name outside its own closed table, so it is strictly weaker
    # than read_file. The fail-closed property this test is named for is
    # asserted below and is unchanged.
    assert read_names == (
        "current_directory", "list_directory", "read_file", "search_text",
        "git_status", "read_result_page", "calculate")
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


def test_compacted_memory_is_background_not_a_turn_to_continue() -> None:
    """Compaction must not hand the model a line it can answer with.

    Delivered as a user turn with a synthetic assistant reply, the compacted
    transcript reads as conversation: one board session answered by
    reproducing the transcript, and another by repeating the acknowledgement
    text verbatim. Background belongs in a system block, and an assistant
    line the host wrote is a template the model will copy.
    """
    agent = _agent_module()
    messages = [{"role": "system", "content": "system"}]
    for index in range(6):
        messages.extend([
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ])
    compacted, report = agent.compact_agent_messages(
        messages, keep_recent_turns=1, memory_chars=400)

    assert report["changed"]
    assert compacted[0]["role"] == "system", "the fixed prefix is untouched"
    assert compacted[1]["role"] == "system", "memory is background, not a turn"
    assert not any(
        message["role"] == "assistant"
        and "Compacted session memory" in message["content"]
        for message in compacted), "no host-written assistant line to copy"
    assert compacted[-2:] == messages[-2:], "the recent turn is kept verbatim"


def test_compaction_leaves_the_snapshot_prefix_byte_identical() -> None:
    """A second system block must not disturb the fixed-prefix snapshot.

    The snapshot renders only the first system message and the tool schema.
    If compaction changed those bytes the executor would reject the restore
    for a key whose token content changed, and every rebase would cost a full
    re-ingest instead of the snapshot it was built to avoid.
    """
    agent = _agent_module()
    tools = [{"type": "function", "function": {
        "name": "read_file", "description": "Read", "parameters": {}}}]
    messages = [{"role": "system", "content": "system"}]
    for index in range(6):
        messages.extend([
            {"role": "user", "content": f"request {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ])
    before = agent.render_chat([messages[0]], tools,
                               add_generation_prompt=False)
    compacted, _ = agent.compact_agent_messages(
        messages, keep_recent_turns=1, memory_chars=400)
    after = agent.render_chat([compacted[0]], tools,
                              add_generation_prompt=False)

    assert after == before


@pytest.mark.parametrize(("raw", "written", "repaired"), [
    # What the board actually produced, asked for a Python script: literal
    # backslash-n instead of newlines. The file collapsed to one comment
    # line, ran, exited zero, and the model reported success.
    ("# a\\n# b\\nprint(1)", "# a\n# b\nprint(1)", True),
    ("```python\nprint(1)\n```", "print(1)", True),
    ("```py\\nprint(1)\\n```", "print(1)", True),
    ("```\nplain\n```", "plain", True),
    # Content that is already a file must survive untouched.
    ("print(1)\nprint(2)", "print(1)\nprint(2)", False),
    ("", "", False),
    # A real newline means the escapes are deliberate string literals.
    ("path = 'a\\nb'\nprint(path)", "path = 'a\\nb'\nprint(path)", False),
    # A fence that does not enclose the whole content is left alone: the
    # narrow rule fails visibly instead of guessing which part is the file.
    ("intro\n```python\nprint(1)\n```", "intro\n```python\nprint(1)\n```", False),
])
def test_write_file_repairs_only_what_it_can_prove(
        raw: str, written: str, repaired: bool) -> None:
    agent = _agent_module()
    content, repairs = agent._repair_written_content(raw)

    assert content == written
    assert bool(repairs) is repaired


def test_the_write_receipt_records_every_repair(tmp_path) -> None:
    """A repaired write must say so: the transcript is the audit trail."""
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    receipt = tools._write_file(
        {"path": "t.py", "content": "```python\\nprint(1)\\n```"})

    assert "decoded literal escape sequences" in receipt
    assert "removed an enclosing markdown fence" in receipt
    assert (tmp_path / "t.py").read_text() == "print(1)"


@pytest.mark.parametrize(("raw", "expected"), [
    ("'t.py'", "t.py"),
    ('"t.py"', "t.py"),
    ('"a/b.txt"', "a/b.txt"),
    ("t.py", "t.py"),
    ("  't.py'  ", "t.py"),
])
def test_a_quoted_path_is_the_model_quoting_itself(
        tmp_path, raw: str, expected: str) -> None:
    """Quotes around a path argument are not part of the filename.

    The argument arrives already parsed out of XML, so a surrounding quote is
    the model copying its own prose. Left alone it becomes the filename: one
    board turn wrote thirty bytes to a file literally named '' before writing
    the file it had been asked for.
    """
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    assert tools._path(raw) == (tmp_path / expected)


@pytest.mark.parametrize("raw", ["''", '""', "``", "  ''  "])
def test_a_path_that_is_only_quotes_is_refused(tmp_path, raw: str) -> None:
    agent = _agent_module()
    tools = agent.WorkspaceTools(tmp_path)

    with pytest.raises(agent.ToolExecutionError):
        tools._path(raw, for_write=True)


def test_a_read_request_never_widens_into_a_mutation_tool() -> None:
    """Escalation widens towards least privilege, not towards fewest tools.

    Ranking by size alone tied write with shell at four tools and broke the
    tie alphabetically, so a model asking to read a file from a turn that
    disclosed nothing was handed run_shell.
    """
    agent = _agent_module()
    tools = agent.WorkspaceTools(Path("."))
    for requested in ("read_file", "current_directory", "list_directory"):
        for start in ("none", "calculate", "result_page"):
            widened = agent.WorkspaceTools.escalate_schema_profile(
                start, (requested,))
            assert widened is not None
            names = set(tools.names_for_profile(widened))
            assert requested in names
            assert not names & {"write_file", "run_shell"}, (
                f"{start} + {requested} disclosed a mutation tool")


@pytest.mark.parametrize("query", [
    "读取 README.md 并把结果保存为 out.txt",
    "看看 a.txt 然后写到 b.txt",
    "显示 config.json 并追加一行",
])
def test_a_read_that_also_mutates_is_never_answered_without_the_model(
        query: str) -> None:
    """A direct route answers with one read and stops.

    The guard used to be a hand-kept word list that had drifted from the
    planner's tables, so a read-then-write turn was direct-routed to the read
    half and the write was dropped without a word to the operator.
    """
    agent = _agent_module()
    assert agent.route_obvious_read_only(query) is None
