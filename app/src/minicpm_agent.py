#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5 native chat/tool protocol and a small local tool registry.

The renderer follows OpenBMB's ``chat_template.jinja`` contract: JSON function
signatures live inside ``<tools>`` and the assistant emits XML ``<function>``
calls.  This module deliberately has no third-party runtime dependencies so it
can run in the minimal Python environment shipped on Hi3403 boards.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
from typing import Callable
import xml.etree.ElementTree as ET


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
BOS = "<s>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
# Default tool budget for legacy ctx1024 launches. Runtime profiles may supply
# a different bound; every profile must leave room for the schema, call,
# conversation and final response.
MAX_TOOL_OUTPUT_CHARS = 800


def _decode_text_mode(data: bytes) -> str:
    """Replicate ``subprocess.run(text=True)``: strict UTF-8 + universal NL."""
    text = data.decode("utf-8", errors="strict")
    return text.replace("\r\n", "\n").replace("\r", "\n")


class ToolProtocolError(ValueError):
    """Generated tool XML is incomplete or violates the MiniCPM contract."""


class ToolExecutionError(RuntimeError):
    """A requested tool could not be executed safely."""


class StableTextStream:
    """Decode growing token prefixes without exposing incomplete UTF-8.

    Byte-level tokenizers may temporarily decode half of a CJK character as
    U+FFFD and replace it once the next token arrives.  Emitting that temporary
    replacement makes an append-only terminal stream impossible to repair.
    Keep trailing replacement characters buffered until they become stable.
    """

    def __init__(self, tokenizer, *, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.text = ""

    def update(self, token_ids) -> str:
        rendered = self.tokenizer.decode(
            token_ids, skip_special_tokens=self.skip_special_tokens)
        stable = rendered.rstrip("\ufffd")
        if not stable.startswith(self.text):
            # Keep the last known-good prefix. A later token may make the
            # tokenizer's decoded prefix stable again.
            return ""
        suffix = stable[len(self.text):]
        self.text = stable
        return suffix

    def finish(self, token_ids) -> tuple[str, bool]:
        """Return the final append-only suffix and whether it reconciled."""
        rendered = self.tokenizer.decode(
            token_ids, skip_special_tokens=self.skip_special_tokens)
        if not rendered.startswith(self.text):
            return "", False
        suffix = rendered[len(self.text):]
        self.text = rendered
        return suffix, True


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    confidence: float
    tool_calls: tuple[ToolCall, ...]
    response_policy: str
    schema_profile: str
    permission: str
    reason: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict[str, str]], str]
    permission: str = "read"

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _tool_definitions(system: str, tools: list[dict]) -> str:
    definitions = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        for tool in tools)
    block = (
        "# Tools\n\n"
        "You are provided with function signatures within <tools></tools> "
        "XML tags:\n<tools>\n"
        f"{definitions}\n"
        "</tools>\n\n"
        "Tool usage guidelines:\n"
        "- You may call zero or more functions. If none are needed, answer "
        "normally and do not emit a <function>.\n"
        "- Call a function with <function name=\"function-name\">"
        "<param name=\"param-name\">param-value</param></function>.\n"
        "- Wrap parameter values containing <, & or newlines in CDATA."
    )
    if "<tool_def_sep>" in system:
        return system.replace("<tool_def_sep>", block)
    return f"{system}\n\n{block}" if system else block


def render_chat(
    messages: list[dict], tools: list[dict] | None = None, *,
    add_generation_prompt: bool = True, enable_thinking: bool = False,
) -> str:
    """Render the official MiniCPM5 chat/tool wire format.

    ``enable_thinking=False`` is intentional for the ctx1024 board release: it
    emits the trained empty-think prefix while preserving budget for tools and
    the final answer.
    """
    tools = tools or []
    system = ""
    start = 0
    if messages and messages[0].get("role") == "system":
        system = str(messages[0].get("content", ""))
        start = 1

    output = [BOS]
    if tools:
        output.extend((IM_START, "system\n",
                       _tool_definitions(system, tools), IM_END, "\n"))
    elif system:
        output.extend((IM_START, "system\n", system, IM_END, "\n"))

    index = start
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        content = str(message.get("content", ""))
        if role in {"user", "system"}:
            output.extend((IM_START, role, "\n", content, IM_END, "\n"))
        elif role == "assistant":
            output.extend((IM_START, "assistant\n", content, IM_END, "\n"))
        elif role == "tool":
            output.extend((IM_START, "user"))
            while index < len(messages) and messages[index].get("role") == "tool":
                tool_content = str(messages[index].get("content", ""))
                output.extend(("\n", TOOL_RESPONSE_OPEN, "\n", tool_content,
                               "\n", TOOL_RESPONSE_CLOSE))
                index += 1
            output.extend((IM_END, "\n"))
            continue
        else:
            raise ValueError(f"unsupported chat role: {role!r}")
        index += 1

    if add_generation_prompt:
        output.extend((IM_START, "assistant\n"))
        if enable_thinking:
            output.append("<think>\n")
        else:
            output.append("<think>\n\n</think>\n\n")
    return "".join(output)


def clean_generated(text: str) -> str:
    """Remove transport terminators while retaining XML tool tokens."""
    return clean_stream_generated(text).strip()


def clean_stream_generated(text: str) -> str:
    """Remove complete trailing transport tokens without trimming content."""
    value = text
    for token in (IM_END, "</s>"):
        while value.endswith(token):
            value = value[:-len(token)]
    return value


_FUNCTION_START = re.compile(r"<function(?:\s|>)")


def _function_blocks(text: str) -> tuple[list[str], str]:
    blocks: list[str] = []
    visible: list[str] = []
    cursor = 0
    while True:
        match = _FUNCTION_START.search(text, cursor)
        if match is None:
            visible.append(text[cursor:])
            break
        visible.append(text[cursor:match.start()])
        scan = match.start()
        while True:
            close = text.find("</function>", scan)
            if close < 0:
                raise ToolProtocolError("unterminated <function> tool call")
            cdata = text.find("<![CDATA[", scan)
            if cdata >= 0 and cdata < close:
                cdata_end = text.find("]]>", cdata + 9)
                if cdata_end < 0:
                    raise ToolProtocolError("unterminated CDATA in tool call")
                scan = cdata_end + 3
                continue
            end = close + len("</function>")
            blocks.append(text[match.start():end])
            cursor = end
            break
    return blocks, "".join(visible).strip()


def parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Parse one or more native MiniCPM XML calls from generated text."""
    blocks, visible = _function_blocks(clean_generated(text))
    calls: list[ToolCall] = []
    for block in blocks:
        try:
            root = ET.fromstring(block)
        except ET.ParseError as error:
            raise ToolProtocolError(f"invalid tool XML: {error}") from error
        if root.tag != "function" or set(root.attrib) != {"name"}:
            raise ToolProtocolError("tool call must be <function name=...>")
        name = root.attrib["name"].strip()
        if not name:
            raise ToolProtocolError("tool name is empty")
        arguments: dict[str, str] = {}
        for child in root:
            if child.tag != "param" or set(child.attrib) != {"name"}:
                raise ToolProtocolError("function children must be <param name=...>")
            param_name = child.attrib["name"].strip()
            if not param_name or param_name in arguments:
                raise ToolProtocolError(f"invalid or duplicate parameter {param_name!r}")
            arguments[param_name] = "".join(child.itertext())
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls, visible


_ABSOLUTE_PATH = re.compile(r"(/[^\s，。！？；]*)")
_FILE_PATH = re.compile(
    r"(?<![\w.-])((?:\.{0,2}/|/)?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9_-]+)")
_MODEL_TRANSFORM = (
    "总结", "概括", "解释", "分析", "比较", "翻译", "summarize", "summary",
    "explain", "analyse", "analyze", "compare", "translate",
)
# A hand-kept list of mutation words drifted from the planner's own tables:
# "保存为", "写到", "追加", "save", "append" and "store" were absent, so
# "read README.md and save the result as out.txt" was direct-routed to a bare
# read with the model skipped and the save silently dropped. The guard is now
# derived from the same tables the planner uses, so the two cannot disagree
# about what may be answered without the model. Defined after them; see
# _unsafe_for_direct_route.
# Planning-stage intent tables, all matched through _has_term: ASCII terms need
# word boundaries, CJK terms are substrings. Mutation and shell intent are read
# by both the planner and the deterministic router so the two never disagree
# about what may be answered without the model.
_BROAD_WORK = (
    "修复", "实现", "开发", "调试", "fix", "implement", "develop", "debug",
    "refactor",
)
_WRITE_INTENT = (
    "修改", "改成", "改为", "换成", "写入", "写进", "写到", "创建", "新建",
    "保存", "存到", "存进", "存为", "另存", "追加", "更新", "删除", "删掉",
    "清空", "清掉", "替换", "重命名", "记录到", "输出到", "edit", "modify",
    "write", "create", "save", "append", "update", "delete", "remove",
    "replace", "rename", "mkdir", "touch",
)
# The registry has no delete tool: write_file only writes UTF-8 content. A
# delete verb therefore needs the shell as well, and "clear the TODOs out of
# this file" needs write_file, so both are disclosed and the operator approves
# whichever the model actually calls. Routing these to "write" alone offered a
# tool that cannot do the job.
_DELETE_INTENT = (
    "删除", "删掉", "清空", "清掉", "delete", "remove", "rm",
)
_SHELL_INTENT = (
    "执行", "运行", "启动", "跑", "命令", "测试", "构建", "编译", "安装",
    "shell", "command", "run", "test", "build", "compile", "install", "bash",
    "python3", "pytest",
)
_READ_INTENT = (
    "读取", "查看", "看看", "看下", "看一下", "显示", "列出", "列举", "罗列",
    "打开", "浏览", "检查", "搜索", "查找", "检索", "有哪些", "都有什么",
    "read", "show", "view", "open", "print", "display", "list", "ls",
    "inspect", "search", "find", "grep", "pwd", "cwd", "git status",
    "git log", "git diff",
)


def _unsafe_for_direct_route(lowered: str) -> bool:
    """Whether a turn must reach the model rather than a deterministic tool.

    A direct route answers with one read and stops. That is only safe when the
    turn asks for nothing else, so any mutation, shell or broad-work intent
    disqualifies it -- otherwise the read is performed, the model is skipped,
    and the rest of the request is dropped without a word to the operator.
    """
    return (_has_term(lowered, _WRITE_INTENT)
            or _has_term(lowered, _SHELL_INTENT)
            or _has_term(lowered, _BROAD_WORK))
# Something a workspace tool can actually reach. A transformation such as
# 总结/explain implies a tool only once the turn names one of these.
_WORKSPACE_NOUN = (
    "文件", "文件夹", "目录", "路径", "工作区", "仓库", "分支", "提交", "代码",
    "项目", "脚本", "日志", "配置", "file", "files", "folder", "directory",
    "dir", "path", "workspace", "repo", "repository", "branch", "commit",
    "code", "project", "script", "log", "logs", "config", "readme",
)
# A turn that asks for a value rather than about one. The deterministic router
# reads this as its evaluation cue; the planner never does, because a cue on
# its own does not say whether the model can be trusted with the arithmetic.
_MATH_REQUEST = (
    "计算", "算一下", "算下", "算出", "求值", "求出", "等于", "是多少",
    "得多少", "结果是", "calculate", "compute", "evaluate", "what is",
    "how much",
)
# Mathematics that is about a function rather than a value of it. These turns
# are the model's own work and must never reach the calculator.
_MATH_CONCEPTUAL = (
    "为什么", "为何", "原理", "推导", "证明", "区别", "对比", "怎么", "如何",
    "介绍", "定义", "含义", "why", "how does", "how do", "derive", "proof",
    "prove", "difference", "definition", "meaning", "intuition",
)
# Arithmetic this model evaluates wrongly on its own. Its algebra is exact --
# it reproduces 2 x 0.8808 = 1.7616 -- so only transcendental, high-precision
# and large-magnitude work is planned as a tool call.
_DISPATCHED_MATH = (
    "sigmoid", "swish", "gelu", "silu", "relu", "softplus", "factorial",
    "logarithm", "square root", "cube root", "arctan", "阶乘", "开方",
    "平方根", "立方根", "对数", "正弦", "余弦", "正切",
)
_DISPATCHED_SHAPE = re.compile(
    r"\d\s*(?:\*\*|\^)\s*\d"
    r"|\d{4,}\s*[+\-*/×÷]|[+\-*/×÷]\s*\d{4,}"
    r"|[\d一二三四五六七八九十]+\s*次方"
    r"|保留\s*[\d一二三四五六七八九十]+\s*位小数"
    r"|精确到|\d+\s*位小数|decimal places")
# A path referent, unlike _ABSOLUTE_PATH, must open a token: "22/7" and the
# decimal literal "3.14" are arithmetic, not workspace evidence.
_PATH_REFERENT = re.compile(r"(?:^|[\s\"'`(（])/[\w.-]")
_NAMED_EXTENSION = re.compile(r"\.[A-Za-z]")


def _has_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(
        re.search(rf"\b{re.escape(term)}\b", text) is not None
        if term.isascii() else term in text
        for term in terms)


def _explicit_file_path(text: str) -> str | None:
    """Return a concrete file path; never infer a file from general prose."""
    for pattern in (
            r"`([^`\r\n]+)`", r"[\"“]([^\"”\r\n]+)[\"”]",
            r"'([^'\r\n]+)'"):
        match = re.search(pattern, text)
        if match and _FILE_PATH.fullmatch(match.group(1).strip()):
            return match.group(1).strip()
    match = _FILE_PATH.search(text)
    return match.group(1) if match else None


def _line_window(text: str) -> tuple[str, str]:
    patterns = (
        r"第\s*(\d+)\s*(?:到|至|[-:])\s*(\d+)\s*行",
        r"lines?\s+(\d+)\s*(?:to|through|[-:])\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if 1 <= start <= end <= start + 79:
                return str(start), str(end)
            return "1", "40"
    first = re.search(r"前\s*(\d+)\s*行", text)
    if first is None:
        chinese = re.search(r"前\s*([一二两三四五六七八九十])\s*行", text)
        if chinese:
            values = {
                "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            return "1", str(values[chinese.group(1)])
    if first is None:
        first = re.search(r"first\s+(\d+)\s+lines?", text, re.IGNORECASE)
    if first:
        end = min(max(int(first.group(1)), 1), 80)
        return "1", str(end)
    return "1", "40"


def _search_query(text: str) -> str | None:
    match = re.search(
        r"(?:搜索|查找|检索|search(?:\s+for)?|find)\s*"
        r"(?:[\"“']([^\"”'\r\n]+)[\"”']|`([^`\r\n]+)`)",
        text, re.IGNORECASE)
    if match is None:
        return None
    query = next(value for value in match.groups() if value is not None).strip()
    return query or None


_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _sigmoid(value: float) -> float:
    """Logistic 1/(1+e^-x), taken in the branch that cannot overflow."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


def _silu(value: float) -> float:
    """SiLU, also published as swish: x*sigmoid(x)."""
    return value * _sigmoid(value)


def _gelu(value: float) -> float:
    """Exact GELU x*Phi(x) = 0.5x(1+erf(x/sqrt(2))), not the tanh fit."""
    return 0.5 * value * (1.0 + math.erf(value / _SQRT_2))


def _gelu_tanh(value: float) -> float:
    """The tanh GELU approximation many inference kernels ship instead."""
    inner = _SQRT_2_OVER_PI * (value + 0.044715 * value * value * value)
    return 0.5 * value * (1.0 + math.tanh(inner))


def _softplus(value: float) -> float:
    """ln(1+e^x), taken in the branch that cannot overflow."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _relu(value: float) -> float:
    """max(x, 0) in the float domain the activation is defined over."""
    return float(value) if value > 0.0 else 0.0


#: Beyond this the digit count cannot change any value the evaluator can
#: hold: integers are capped at 1024 bits and floats saturate at 1e308.
_MAX_ROUND_DIGITS = 323


def _round(value, ndigits=None):
    """round() with a bounded digit count.

    CPython's ``int.__round__`` materialises ``10 ** abs(ndigits)`` to do the
    rounding, so ``round(1, -10**7)`` builds a 33-million-bit integer from a
    sixteen-character expression: eight AST nodes, well under every limit this
    module checks, five seconds of CPU here and quadratically worse beyond.
    Nothing else in the language reaches it -- ``_power`` rejects a large base
    first -- and the deterministic router evaluates arithmetic with no model
    and no operator in the loop, so the guard has to live on the function.
    """
    if ndigits is None:
        return round(value)
    if isinstance(ndigits, bool) or not isinstance(ndigits, int):
        raise ToolExecutionError("round takes an integer digit count")
    if not -_MAX_ROUND_DIGITS <= ndigits <= _MAX_ROUND_DIGITS:
        raise ToolExecutionError(
            f"round takes a digit count in "
            f"[-{_MAX_ROUND_DIGITS}, {_MAX_ROUND_DIGITS}]")
    return round(value, ndigits)


# A closed expression language: no attribute, no subscript, no import and no
# name the tables below do not define. Every result is a real number short
# enough to print. Keeping the cost bounded needs care as well as closure:
# a builtin can allocate far more than its arguments suggest, which is why
# `round` is wrapped below rather than registered directly.
_MAX_EXPRESSION_CHARS = 256
_MAX_EXPRESSION_NODES = 128
_MAX_INT_BITS = 1024
_MAX_FACTORIAL = 170
_MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _finite(value: int | float) -> int | float:
    """Reject any value that is complex, unprintable or not finite."""
    if type(value) is int:
        if value.bit_length() > _MAX_INT_BITS:
            raise ToolExecutionError(
                f"integer magnitude exceeds {_MAX_INT_BITS} bits")
        return value
    if type(value) is not float:
        raise ToolExecutionError("result is not a real number")
    if math.isnan(value) or math.isinf(value):
        raise ToolExecutionError("result is not finite")
    return value


def _power(base: int | float, exponent: int | float) -> int | float:
    """Raise to a power, refusing the integer results too large to hold.

    Only ``int ** nonnegative int`` can grow without bound, and its size is
    known from the operands, so the bound is checked before the multiply.
    """
    if type(base) is int and type(exponent) is int and exponent >= 0:
        if base.bit_length() * exponent > _MAX_INT_BITS:
            raise ToolExecutionError(
                f"integer magnitude exceeds {_MAX_INT_BITS} bits")
    return _finite(base ** exponent)


def _factorial(value: int | float) -> int:
    """Factorial of a small nonnegative integer, refusing the rest."""
    if type(value) is not int or not 0 <= value <= _MAX_FACTORIAL:
        raise ToolExecutionError(
            f"factorial takes an integer in [0, {_MAX_FACTORIAL}]")
    return math.factorial(value)


# Each entry carries the argument counts its function accepts, because the
# planner dispatches precision requests here: a turn asking for four decimals
# must be able to say round(ln(7), 4) rather than be refused into the rounding
# habit the tool exists to replace.
_MATH_FUNCTIONS = {
    "abs": (abs, (1,)), "min": (min, (2, 3)), "max": (max, (2, 3)),
    "round": (_round, (1, 2)), "floor": (math.floor, (1,)),
    "ceil": (math.ceil, (1,)), "pow": (_power, (2,)),
    "exp": (math.exp, (1,)), "ln": (math.log, (1,)), "log": (math.log, (1, 2)),
    "log2": (math.log2, (1,)), "log10": (math.log10, (1,)),
    "sqrt": (math.sqrt, (1,)), "sin": (math.sin, (1,)),
    "cos": (math.cos, (1,)), "tan": (math.tan, (1,)),
    "asin": (math.asin, (1,)), "acos": (math.acos, (1,)),
    "atan": (math.atan, (1,)), "sinh": (math.sinh, (1,)),
    "cosh": (math.cosh, (1,)), "tanh": (math.tanh, (1,)),
    "erf": (math.erf, (1,)), "gamma": (math.gamma, (1,)),
    "factorial": (_factorial, (1,)), "sigmoid": (_sigmoid, (1,)),
    "silu": (_silu, (1,)), "swish": (_silu, (1,)), "gelu": (_gelu, (1,)),
    "gelu_tanh": (_gelu_tanh, (1,)), "relu": (_relu, (1,)),
    "softplus": (_softplus, (1,)),
}
_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: _power,
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda operand: +operand,
    ast.USub: lambda operand: -operand,
}
# Mathematical notation rather than Python source: the caret is exponentiation
# and the typographic operators are their ASCII spellings. Rewriting the text
# instead of the parsed tree keeps Python's precedence, so 1+2^3 is 9. A
# Chinese IME left in fullwidth mode types the digits that way too, so they
# fold here as well and １＋１ parses like 1+1.
_MATH_ALIASES = str.maketrans({
    "×": "*", "·": "*", "＊": "*", "÷": "/", "／": "/", "＋": "+", "－": "-",
    "−": "-", "（": "(", "）": ")", "．": ".", "０": "0", "１": "1", "２": "2",
    "３": "3", "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})


def _normalized_math(text: str) -> str:
    return text.translate(_MATH_ALIASES).replace("^", "**")


def _evaluate_node(node: ast.AST) -> int | float:
    """Evaluate one whitelisted node; anything unlisted is refused here."""
    if type(node) is ast.Constant:
        if type(node.value) not in (int, float):
            raise ToolExecutionError("only real number literals are allowed")
        return _finite(node.value)
    if type(node) is ast.Name:
        try:
            return _MATH_CONSTANTS[node.id.lower()]
        except KeyError:
            raise ToolExecutionError(f"unknown name: {node.id}") from None
    if type(node) is ast.UnaryOp:
        operator = _UNARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ToolExecutionError("unsupported unary operator")
        return _finite(operator(_evaluate_node(node.operand)))
    if type(node) is ast.BinOp:
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ToolExecutionError("unsupported operator")
        return _finite(operator(
            _evaluate_node(node.left), _evaluate_node(node.right)))
    if type(node) is ast.Call:
        if type(node.func) is not ast.Name or node.keywords:
            raise ToolExecutionError("only positional calls to named functions")
        entry = _MATH_FUNCTIONS.get(node.func.id.lower())
        if entry is None:
            raise ToolExecutionError(
                f"unknown function: {node.func.id}; available: "
                + " ".join(sorted(_MATH_FUNCTIONS)))
        function, arity = entry
        if len(node.args) not in arity:
            raise ToolExecutionError(
                f"{node.func.id} takes "
                + " or ".join(str(count) for count in arity) + " argument(s)")
        return _finite(function(
            *(_evaluate_node(argument) for argument in node.args)))
    raise ToolExecutionError(
        f"unsupported expression element: {type(node).__name__}")


def evaluate_expression(expression: str) -> str:
    """Evaluate a mathematical expression and echo what was computed.

    The echoed expression is unparsed from the tree that was evaluated, so the
    answer always states the operator and function that produced it -- ``2^10``
    echoes ``2 ** 10``, and ``gelu`` is never confused with ``gelu_tanh``.  The
    value is printed at full precision, because truncating it here is the very
    habit this tool exists to replace.
    """
    source = _normalized_math(str(expression).strip())
    if not source:
        raise ToolExecutionError("expression is empty")
    if len(source) > _MAX_EXPRESSION_CHARS:
        raise ToolExecutionError(
            f"expression exceeds {_MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        raise ToolExecutionError(f"cannot parse expression: {error}") from None
    # Bounds the depth of _evaluate_node's recursion as well as its width.
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        raise ToolExecutionError(
            f"expression exceeds {_MAX_EXPRESSION_NODES} nodes")
    try:
        value = _evaluate_node(tree.body)
    except ToolExecutionError:
        raise
    except ZeroDivisionError:
        raise ToolExecutionError("division by zero") from None
    except (ArithmeticError, ValueError, TypeError, MemoryError) as error:
        raise ToolExecutionError(f"cannot evaluate: {error}") from None
    return f"{ast.unparse(tree)} = {value!r}"


# Read from the one namespace the host can evaluate, so the planner never
# advertises a function the calculator would refuse.
_DISPATCHED_CALL = re.compile(
    r"\b(?:" + "|".join(
        sorted(_MATH_FUNCTIONS, key=len, reverse=True)) + r")\s*\(")
_EXPRESSION_RUN = re.compile(r"[0-9A-Za-z_.,+\-*/%()\s]+")
_OPERATOR_MARKS = "+-*/%("
_SENTENCE_TAIL = " \t?!=.。？！"
# A span that is the whole turn is an evaluation request only when nothing
# else reads it that way: "1-40" answers "which lines?" far more often than it
# subtracts. A call, a decimal, a power or a second operator settles it.
_SELF_EVIDENT_EXPRESSION = re.compile(
    r"\(|\d\.\d|\*\*|[+\-*/%][^+\-*/%]*[+\-*/%]")


def _dispatched_math(text: str, lowered: str) -> bool:
    """Report arithmetic this model evaluates wrongly on its own.

    A named function without an operand is a concept question the model
    answers from its own knowledge; only an evaluation is dispatched.
    """
    if _DISPATCHED_CALL.search(lowered) is not None:
        return True
    if _DISPATCHED_SHAPE.search(text) is not None:
        return True
    return _has_term(lowered, _DISPATCHED_MATH) \
        and re.search(r"\d", text) is not None


def _expression_edges(head: str, tail: str) -> bool:
    """Reject windows that cannot open or close an arithmetic expression."""
    return ((head[0].isdigit() or head[0] in "(+-." or "(" in head)
            and (tail[-1].isdigit() or tail[-1] == ")"))


def _applies_an_operation(source: str) -> bool:
    """Report whether a candidate computes something rather than restates it.

    Read from the parsed tree rather than the text, so a sign is not mistaken
    for arithmetic: ``-1`` is a literal, ``2-1`` is a subtraction.
    """
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return any(type(node) in (ast.BinOp, ast.Call) for node in ast.walk(tree))


def _math_expression(text: str) -> str | None:
    """Return the longest span of the text this host can evaluate itself.

    A bare literal is not an evaluation request, so a candidate must apply at
    least one operator or function, read from its tree rather than its text.
    Candidates are then accepted by evaluating them, which keeps the router
    from ever proposing a form the calculator would refuse.
    """
    best: str | None = None
    for run in _EXPRESSION_RUN.findall(_normalized_math(text)):
        tokens = run.split()[:24]
        for start in range(len(tokens)):
            for stop in range(len(tokens), start, -1):
                candidate = " ".join(tokens[start:stop]).rstrip(",.")
                if not candidate or (
                        best is not None and len(candidate) <= len(best)):
                    continue
                if not _expression_edges(tokens[start], candidate):
                    continue
                if not any(mark in candidate for mark in _OPERATOR_MARKS):
                    continue
                if not _applies_an_operation(candidate):
                    continue
                try:
                    evaluate_expression(candidate)
                except (ToolExecutionError, ArithmeticError, ValueError,
                        TypeError, RecursionError):
                    continue
                best = candidate
                break
    return best


def route_obvious_read_only(
    user_text: str, previous_assistant: str = "",
) -> RouteDecision | None:
    """Route unambiguous, presentation-free read-only intents without an LLM.

    MiniCPM5 remains the general tool selector. A request that needs summary,
    explanation, transformation, mutation or a shell is deliberately returned
    to the model even when it also contains a recognizable read operation.
    """
    text = user_text.strip()
    lowered = text.lower()
    if not text or _unsafe_for_direct_route(lowered):
        return None
    needs_model = _has_term(lowered, _MODEL_TRANSFORM)

    def route(call: ToolCall, reason: str, confidence: float = 0.99):
        if needs_model:
            return RouteDecision(
                mode="TOOL_THEN_MODEL", confidence=confidence,
                tool_calls=(call,), response_policy="MODEL_SUMMARY",
                schema_profile="result_page", permission="automatic",
                reason=f"{reason} with model transformation")
        return RouteDecision(
            mode="DIRECT_TOOL", confidence=confidence, tool_calls=(call,),
            response_policy="DIRECT_FORMATTED", schema_profile="none",
            permission="automatic", reason=reason)

    current_directory = (
        lowered in {"pwd", "cwd"}
        or text in {"当前路径", "当前目录", "工作目录", "工作区路径"}
        or re.fullmatch(
            r"(?:what(?:'s| is)\s+)?(?:the\s+)?current\s+"
            r"(?:working\s+)?directory\??", lowered) is not None
        or (any(term in text for term in ("当前路径", "当前目录", "工作目录", "工作区路径"))
            and any(term in text for term in ("什么", "显示", "查看", "告诉", "输出", "?", "？")))
    )
    if current_directory:
        return route(
            ToolCall("current_directory", {}),
            "explicit current-directory request")

    git_status = (
        re.fullmatch(r"(?:show\s+)?git\s+status[.!?]?", lowered) is not None
        or ("状态" in text and any(term in lowered for term in ("git", "仓库")))
    )
    if git_status:
        return route(ToolCall("git_status", {}), "explicit git-status request")

    query = _search_query(text)
    if query is not None:
        return route(ToolCall(
            "search_text", {"query": query, "path": ".", "max_matches": "15"}),
            "explicit literal-search request", confidence=0.98)

    file_path = _explicit_file_path(text)
    read_intent = (
        any(term in text for term in ("读取", "打开", "显示", "查看", "看下", "看看"))
        or re.search(r"\b(?:read|show|open|view|print)\b", lowered) is not None
    )
    if read_intent and file_path is not None:
        start, end = _line_window(text)
        return route(ToolCall(
            "read_file", {"path": file_path, "start_line": start,
                          "end_line": end}),
            "explicit file-window request", confidence=0.98)

    # Arithmetic is routed only after the search and file-window routes have
    # declined, so a named file keeps winning: a turn that reads and computes
    # must not have its read half dropped in favour of the digits.
    if not needs_model and not _has_term(lowered, _MATH_CONCEPTUAL) \
            and not _has_term(lowered, _WRITE_INTENT) \
            and not _has_term(lowered, _SHELL_INTENT):
        expression = _math_expression(text)
        # Either the turn says it wants a value, or it is nothing but an
        # expression no other reading fits. Anything else keeps its own intent.
        asks_for_value = expression is not None and (
            _has_term(lowered, _MATH_REQUEST)
            or (expression == _normalized_math(text).strip(_SENTENCE_TAIL)
                and _SELF_EVIDENT_EXPRESSION.search(expression) is not None))
        if asks_for_value:
            # The host produces the digits and prints them verbatim: the model
            # never sees this turn, so it cannot round or invent one.
            return RouteDecision(
                mode="DIRECT_TOOL", confidence=0.99,
                tool_calls=(ToolCall("calculate", {"expression": expression}),),
                response_policy="DIRECT_FORMATTED", schema_profile="none",
                permission="automatic",
                reason="explicit numeric evaluation request")

    path_match = _ABSOLUTE_PATH.search(text)
    path = path_match.group(1) if path_match else "."
    chinese_list = (
        any(word in text for word in ("列出", "查看", "显示", "看看", "有哪些"))
        and any(word in text for word in ("文件", "目录", "工作区", "路径")))
    english_list = (
        re.search(r"\b(?:ls|list|show)\b", lowered) is not None
        and re.search(r"\b(?:files?|director(?:y|ies)|folders?|workspace)\b",
                      lowered) is not None)
    path_reply = (
        ("路径" in previous_assistant or "path" in previous_assistant.lower())
        and (path_match is not None or lowered in {"root", "/root", "."}))
    root_shortcut = lowered in {"root", "/root"}
    if not (chinese_list or english_list or path_reply or root_shortcut):
        return None
    if root_shortcut and path_match is None:
        path = "/root"
    call = ToolCall("list_directory", {"path": path, "max_entries": "10"})
    return route(call, "explicit directory listing request")


def format_tool_call(call: ToolCall) -> str:
    """Serialize a host-routed call using MiniCPM5's native XML contract."""
    root = ET.Element("function", {"name": call.name})
    for name, value in call.arguments.items():
        child = ET.SubElement(root, "param", {"name": name})
        child.text = value
    return ET.tostring(root, encoding="unicode", short_empty_elements=False)


def _compact_memory_text(value: object, limit: int) -> str:
    text = " ".join(str(value).replace("\x00", "\\x00").split())
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)] + "…"


def compact_agent_messages(
    messages: list[dict], *, keep_recent_turns: int = 1,
    memory_chars: int = 800,
) -> tuple[list[dict], dict[str, int | bool]]:
    """Deterministically compact old complete Agent turns.

    The current user/tool exchange and a bounded number of recent turns remain
    byte-for-byte. Older raw tool output is replaced by its audited first-line
    reference plus clipped user/final-assistant text. No model is invoked and
    the system message is never rewritten.
    """
    if type(keep_recent_turns) is not int or keep_recent_turns < 0:
        raise ValueError("keep_recent_turns must be a nonnegative integer")
    if type(memory_chars) is not int or memory_chars < 128:
        raise ValueError("memory_chars must be an integer >= 128")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("agent messages must start with one system message")

    turns: list[list[dict]] = []
    current: list[dict] = []
    for message in messages[1:]:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(dict(message))
    if current:
        turns.append(current)
    retained = keep_recent_turns + 1  # Always preserve the current turn.
    if len(turns) <= retained:
        return list(messages), {
            "changed": False, "turns_compacted": 0,
            "memory_chars": 0}

    older = turns[:-retained]
    recent = turns[-retained:]
    blocks: list[str] = []
    used = 0
    for turn in reversed(older):
        lines: list[str] = []
        user = next((item for item in turn if item.get("role") == "user"), None)
        if user is not None:
            lines.append(
                "user: " + _compact_memory_text(user.get("content", ""), 180))
        for item in turn:
            if item.get("role") == "tool":
                header = str(item.get("content", "")).splitlines()[0]
                lines.append("tool: " + _compact_memory_text(header, 180))
        final_assistant = next((
            item for item in reversed(turn)
            if item.get("role") == "assistant"
            and "<function" not in str(item.get("content", ""))), None)
        if final_assistant is not None:
            lines.append("assistant: " + _compact_memory_text(
                final_assistant.get("content", ""), 220))
        block = "\n".join(lines) or "[older turn omitted]"
        if len(block) + used > memory_chars:
            remaining = memory_chars - used
            if remaining >= 32:
                blocks.insert(0, _compact_memory_text(block, remaining))
            break
        blocks.insert(0, block)
        used += len(block) + 1

    memory = (
        "Earlier turns in this session, compacted by the host; raw tool "
        "output omitted. This is background, not a request:\n"
        + "\n".join(blocks))
    # The memory is a second system block, not a user turn with a synthetic
    # assistant reply. Delivered as conversation it reads as something to
    # continue: one board session reproduced the compacted transcript as its
    # answer, and another answered with the acknowledgement text verbatim.
    # A system block is background, and there is no assistant line to copy.
    # The fixed-prefix snapshot is unaffected -- it renders only the first
    # system message and the tool schema, both of which are unchanged.
    compacted = [dict(messages[0]), {"role": "system", "content": memory}]
    for turn in recent:
        compacted.extend(turn)
    return compacted, {
        "changed": True,
        "turns_compacted": len(older),
        "memory_chars": len(memory),
    }


_FENCE = re.compile(
    r"\A\s*```[A-Za-z0-9_+.-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?```\s*\Z",
    re.DOTALL)
_ESCAPE = re.compile(r"\\[nrt]")
_UNESCAPE = {"\\n": "\n", "\\r": "\r", "\\t": "\t"}


def _repair_written_content(content: str) -> tuple[str, list[str]]:
    """Undo two things a small model reliably does to a file parameter.

    Asked for a Python script, MiniCPM5 wrote the two-character sequence
    backslash-n instead of newlines and wrapped the result in a markdown
    fence. The file collapsed to a single comment line, ran, exited zero, and
    the model reported success. Both are mechanical and both are repairable
    here, which is cheaper and more reliable than asking the model again.

    The repairs are deliberately narrow. Escapes are decoded only when the
    content carries no real newline at all, which no hand-written multi-line
    file does, and the fence is stripped only when it encloses the whole
    content. Anything the model did that these two rules do not cover is left
    alone and fails visibly, which is better than being silently mangled.

    :param content: the raw ``content`` argument.
    :returns: the content to write and a list of repairs, for the receipt.
    """
    repairs: list[str] = []
    if "\n" not in content and _ESCAPE.search(content) is not None:
        content = _ESCAPE.sub(lambda m: _UNESCAPE[m.group()], content)
        repairs.append("decoded literal escape sequences")
    match = _FENCE.match(content)
    if match is not None:
        content = match.group("body")
        repairs.append("removed an enclosing markdown fence")
    return content, repairs


def _integer(arguments: dict[str, str], name: str, default: int,
             minimum: int, maximum: int) -> int:
    raw = arguments.get(name)
    value = default if raw in (None, "") else int(raw)
    if value < minimum or value > maximum:
        raise ToolExecutionError(f"{name} must be in [{minimum}, {maximum}]")
    return value


class WorkspaceTools:
    """Workspace-confined tools with explicit approval for mutation/shell."""

    _READ_ONLY = (
        "current_directory", "list_directory", "read_file", "search_text",
        "git_status", "read_result_page", "calculate")
    # Seven of the executor's eight fixed-prefix snapshot ids; arithmetic is a
    # cross-cutting capability rather than a task category, so it rides with
    # the read set instead of claiming a profile of its own per filesystem
    # combination. Measured price of riding along, ctx1024 tokens: read_only
    # 558 -> 625, read_write 616 -> 683, read_shell 617 -> 684, all 675 -> 742,
    # i.e. +67 tokens (+5.3 s of prefill) on every read-bearing turn, paid so
    # that a read-then-compute turn never needs a widening round.
    # A turn that names a file to write, or a command to run, needs the tool
    # that does it and a way to say where -- not the read set as well. The
    # superset is not merely 675 ctx1024 tokens of prefill: disclosing nine
    # tools for "write pi.py, then run it" was measured selecting `math.pi`,
    # a tool that does not exist, four rounds running. Narrow disclosure is a
    # correctness measure before it is a latency one.
    # read_file rides along because the system prompt tells the model to
    # inspect before changing, and paying 72 tokens for it beats the widening
    # round it replaces. current_directory does not: the system prompt already
    # names the workspace root in its second sentence, so a tool that returns
    # it again is 35 tokens, 2.8 s, for an answer the model has been given.
    #: Tools that ask the operator on every call. Escalation widens towards
    #: fewer of these before it widens towards fewer tools.
    _PRIVILEGED = frozenset({"write_file", "run_shell"})
    _MUTATION_BASE = ("read_file",)
    #: calculate joins a mutating turn only when that turn computes something.
    #: It must be there when it does -- asked to write swish(2) to a file the
    #: model answers 1.728 against a true 1.7616, and that lands on disk -- but
    #: a plain write pays 67 tokens, 5.3 s, for a tool it never calls.
    _MATH_SUFFIX = "+calc"
    _WRITE_SET = (*_MUTATION_BASE, "write_file")
    _SHELL_SET = (*_MUTATION_BASE, "run_shell")
    _PROFILES = {
        "none": (),
        "calculate": ("calculate",),
        "result_page": ("read_result_page",),
        "write": _WRITE_SET,
        "write+calc": (*_MUTATION_BASE, "calculate", "write_file"),
        "shell": _SHELL_SET,
        "shell+calc": (*_MUTATION_BASE, "calculate", "run_shell"),
        "write_shell": (*_MUTATION_BASE, "write_file", "run_shell"),
        "write_shell+calc": (*_MUTATION_BASE, "calculate", "write_file",
                             "run_shell"),
        "read_only": _READ_ONLY,
        "read_write": (*_READ_ONLY, "write_file"),
        "read_shell": (*_READ_ONLY, "run_shell"),
        "all": (*_READ_ONLY, "write_file", "run_shell"),
    }
    _RESULT_TYPES = {
        "current_directory": "path",
        "list_directory": "directory_entries",
        "read_file": "text_lines",
        "search_text": "search_matches",
        "git_status": "git_status",
        "calculate": "number",
        "write_file": "write_receipt",
        "run_shell": "shell_output",
    }
    _MAX_RESULTS = 16
    _MAX_STORED_RESULT_CHARS = 65_536

    def __init__(self, root: Path, *, max_output_chars: int = MAX_TOOL_OUTPUT_CHARS):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"agent workspace is not a directory: {self.root}")
        if type(max_output_chars) is not int or max_output_chars < 64:
            raise ValueError("max_output_chars must be an integer >= 64")
        self.max_output_chars = max_output_chars
        self._results: dict[str, tuple[str, str, bool]] = {}
        self._result_counter = 0
        # Optional ``bytes -> None`` sink fed with raw stdout chunks while a
        # shell tool runs.  ``None`` (the default) keeps the plain
        # ``subprocess.run`` path; eager tool prefill attaches one for the
        # duration of a single call.  See ``_run_shell``.
        self.stdout_listener: Callable[[bytes], None] | None = None
        obj = {"type": "object", "additionalProperties": False}
        self._tools = {
            tool.name: tool for tool in (
                Tool("current_directory", "Return the configured workspace path.", {
                    **obj, "properties": {},
                }, self._current_directory),
                Tool("list_directory", (
                    "List files under the workspace. Use path='.' for the "
                    "workspace root; never ask the user for the current path."), {
                    **obj, "properties": {
                        "path": {"type": "string", "default": "."},
                        "max_entries": {"type": "integer", "default": 10}},
                }, self._list_directory),
                Tool("read_file", "Read a UTF-8 text file by line range.", {
                    **obj, "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "default": 1},
                        "end_line": {"type": "integer", "default": 40}},
                    "required": ["path"],
                }, self._read_file),
                Tool("search_text", "Find literal text in workspace files.", {
                    **obj, "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "max_matches": {"type": "integer", "default": 15}},
                    "required": ["query"],
                }, self._search_text),
                Tool("git_status", "Show concise git branch and worktree status.", {
                    **obj, "properties": {},
                }, self._git_status),
                Tool("read_result_page", (
                    "Read another character page from a prior typed tool result."
                ), {
                    **obj, "properties": {
                        "ref": {"type": "string"},
                        "offset": {"type": "integer", "default": 0},
                        "max_chars": {"type": "integer", "default": 800}},
                    "required": ["ref"],
                }, self._read_result_page),
                Tool("calculate", (
                    "Evaluate arithmetic exactly. Use it for any decimal, "
                    "power, root, log or activation value instead of computing "
                    "one yourself, and report its digits verbatim."), {
                    **obj, "properties": {
                        "expression": {"type": "string"}},
                    "required": ["expression"],
                }, self._calculate),
                Tool("write_file", "Write a small UTF-8 file in the workspace.", {
                    **obj, "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}},
                    "required": ["path", "content"],
                }, self._write_file, permission="ask"),
                Tool("run_shell", "Run a shell command in the workspace.", {
                    **obj, "properties": {
                        "command": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "default": 20}},
                    "required": ["command"],
                }, self._run_shell, permission="ask"),
            )
        }

    @property
    def definitions(self) -> list[dict]:
        return [tool.definition() for tool in self._tools.values()]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @classmethod
    def names_for_profile(cls, profile: str) -> tuple[str, ...]:
        try:
            return cls._PROFILES[profile]
        except KeyError as error:
            raise ValueError(f"unknown tool schema profile: {profile}") from error

    def definitions_for_profile(self, profile: str) -> list[dict]:
        return [self._tools[name].definition()
                for name in self.names_for_profile(profile)]

    @classmethod
    def _workspace_reference(cls, text: str, lowered: str) -> bool:
        """Report whether the turn names something a workspace tool reaches."""
        if _has_term(lowered, _WORKSPACE_NOUN):
            return True
        if _has_term(lowered, cls._PROFILES["all"]):
            return True
        if _PATH_REFERENT.search(text) is not None:
            return True
        candidate = _explicit_file_path(text)
        return candidate is not None \
            and _NAMED_EXTENSION.search(candidate) is not None

    @classmethod
    def select_schema_profile(
        cls, user_text: str, *, has_context: bool = False,
    ) -> str:
        """Disclose the narrowest tool set this turn gives evidence for.

        The schema is a fixed prompt prefix, so an unused tool is charged in
        ctx1024 prefill tokens on every fresh profile while an undisclosed one
        costs a single widening round (:meth:`escalate_schema_profile`).  A
        turn with no evidence of tool intent therefore gets no schema at all.
        Planning also adapts to the model's arithmetic: the short algebra it
        reproduces stays with it, and transcendental or high-precision work is
        planned as a call to ``calculate``.  Disclosure is not authorization --
        write_file and run_shell still ask the operator on every call,
        whichever profile names them.
        """
        text = str(user_text)
        lowered = text.lower()
        if _has_term(lowered, _BROAD_WORK):
            return "all"
        needs_delete = _has_term(lowered, _DELETE_INTENT)
        needs_write = needs_delete or _has_term(lowered, _WRITE_INTENT)
        needs_shell = needs_delete or _has_term(lowered, _SHELL_INTENT)
        # A mutating turn is disclosed the mutating tools. It gets the read
        # set as well only when it also asks to look at something: the file
        # named in "save it as pi.py" is the write target, not a referent to
        # be read, so the referent rule must not widen this branch.
        also_reads = _has_term(lowered, _READ_INTENT)
        calc = cls._MATH_SUFFIX if _dispatched_math(text, lowered) else ""
        if needs_write and needs_shell:
            return "all" if also_reads else "write_shell" + calc
        if needs_write:
            return "read_write" if also_reads else "write" + calc
        if needs_shell:
            # The read set carries the calculator already, so a shell turn
            # that also reads needs no suffix.
            return "read_shell" if also_reads else "shell" + calc
        if _has_term(lowered, _READ_INTENT):
            # A read-then-compute turn takes its numbers from the file, and
            # the read set carries the calculator for exactly that reason.
            # A look verb with nothing to look at is arithmetic phrased as a
            # read, and the read schema would cost it 360 tokens it cannot use.
            if _dispatched_math(text, lowered) \
                    and not cls._workspace_reference(text, lowered):
                return "calculate"
            return "read_only"
        contextual_follow_up = (
            re.match(
                r"^(?:第\s*[一二两三四五六七八九十\d]+\s*(?:行|项|个)|"
                r"这(?:一|个|些)?|那(?:一|个|些)?|它|上面|前面|刚才)",
                lowered) is not None
            or re.match(
                r"^(?:what|why|how)\b.*\b(?:it|that|this|above|previous|"
                r"line|item)\b", lowered) is not None)
        if has_context and contextual_follow_up \
                and not cls._workspace_reference(text, lowered):
            # The prior transcript already contains the evidence. Avoid
            # spending hundreds of ctx1024 tokens on unrelated tool schemas --
            # but a follow-up that names a value still needs the digits: 265
            # fixed-prefix tokens against 111 buys them exactly. A follow-up
            # that names a FILE is not answerable from the transcript at all,
            # so the referent test has to run first: "这个文件 README.md 里写了
            # 什么" was reaching this branch and getting no schema, and with no
            # tools disclosed it also lost the system prompt that would have
            # told it to call one.
            return "calculate" if _dispatched_math(text, lowered) else "none"
        if cls._workspace_reference(text, lowered):
            # A workspace referent outranks a number in the same turn: the
            # read set carries the calculator either way, so the compute half
            # is served while the file half would otherwise be undisclosed.
            return "read_only"
        if _dispatched_math(text, lowered):
            return "calculate"
        return "none"

    @classmethod
    def escalate_schema_profile(
        cls, profile: str, requested: tuple[str, ...],
    ) -> str | None:
        """Return the narrowest profile adding ``requested``, else ``None``.

        Widening keeps every tool the turn already disclosed, so a transcript
        never loses a schema it has already used.  ``None`` means the current
        profile already discloses the request.
        """
        needed = set(cls.names_for_profile(profile))
        if needed.issuperset(requested):
            return None
        needed.update(requested)
        # Fewest privileged tools first, then fewest tools. Ranking by size
        # alone tied "write" with "shell" at four tools and broke the tie
        # alphabetically, so a model asking to read a file from a turn that
        # disclosed nothing was handed run_shell. Disclosure is not
        # authorization, but it is still the wrong direction to widen in, and
        # paying a few hundred tokens for the read set is the right trade.
        candidates = sorted(
            (sum(1 for tool in names if tool in cls._PRIVILEGED),
             len(names), name)
            for name, names in cls._PROFILES.items() if needed.issubset(names))
        return candidates[0][2] if candidates else None

    #: Quote characters a model wraps a path in when it copies its own prose.
    _PATH_QUOTES = ("'", '"', "`", "\u2018", "\u2019", "\u201c", "\u201d")

    def _path(self, value: str, *, for_write: bool = False) -> Path:
        # A path argument arrives already parsed out of XML, so quotes around
        # it are the model quoting itself. Left alone they become the
        # filename: one board turn wrote thirty bytes to a file literally
        # named '' before writing the file it had been asked for.
        cleaned = str(value or "").strip()
        while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] \
                and cleaned[0] in self._PATH_QUOTES:
            cleaned = cleaned[1:-1].strip()
        if value and not cleaned:
            raise ToolExecutionError("path is empty once its quotes are removed")
        raw = Path(cleaned or ".")
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ToolExecutionError("path escapes the configured workspace") from error
        if for_write and candidate.is_symlink():
            raise ToolExecutionError("writing through a symlink is forbidden")
        return resolved

    def _clip(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        return value[:self.max_output_chars] + "\n...[tool output truncated]"

    def preview(self, call: ToolCall) -> str:
        shown = []
        for key, value in call.arguments.items():
            if key == "content":
                shown.append(f"content=<{len(value)} chars>")
            else:
                compact = value.replace("\n", "\\n")
                shown.append(f"{key}={compact[:160]!r}")
        return f"{call.name}({', '.join(shown)})"

    def execute(self, call: ToolCall,
                approve: Callable[[str], bool] | None = None,
                allowed_names: tuple[str, ...] | None = None) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return self.result(call.name, False, f"unknown tool; available={self.names}")
        if allowed_names is not None and call.name not in allowed_names:
            return self.result(
                call.name, False,
                "tool was not disclosed for this request; restate the task "
                "with an explicit write or command intent")
        if tool.permission == "ask":
            if approve is None or not approve(self.preview(call)):
                return self.result(call.name, False, "permission denied by user")
        try:
            if call.name == "read_result_page":
                return self._page_result(call.arguments)
            output = tool.handler(call.arguments)
            return self.result(call.name, True, output)
        except (OSError, ValueError, ToolExecutionError, subprocess.SubprocessError) as error:
            return self.result(call.name, False, str(error))

    def next_ref(self) -> str:
        """Ref the NEXT successful result will be given.

        Eager tool prefill has to predict the ``ref=rN`` clause of the tool
        message before the tool has finished, so the counter that ``result``
        will bump is exposed rather than re-derived.
        """
        return f"r{self._result_counter + 1}"

    def result(self, name: str, ok: bool, output: str) -> str:
        # Prevent a file or command from closing the surrounding template tag.
        payload = {"tool": name, "ok": bool(ok), "output": output}
        if ok:
            source_complete = len(output) <= self._MAX_STORED_RESULT_CHARS
            stored = output[:self._MAX_STORED_RESULT_CHARS]
            self._result_counter += 1
            ref = f"r{self._result_counter}"
            while len(self._results) >= self._MAX_RESULTS:
                self._results.pop(next(iter(self._results)))
            result_type = self._RESULT_TYPES.get(name, "text")
            self._results[ref] = (stored, result_type, source_complete)
            shown = stored[:self.max_output_chars]
            truncated = len(stored) > len(shown) or not source_complete
            payload.update({
                "type": result_type, "ref": ref, "output": shown,
                "truncated": truncated,
                "next_offset": len(shown) if truncated else None,
                "source_complete": source_complete,
            })
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))
        return encoded.replace("<", "\\u003c")

    @staticmethod
    def for_model(result_json: str) -> str:
        """Convert audited JSON evidence into compact model-readable text."""
        result = json.loads(result_json)
        status = "succeeded" if result["ok"] else "failed"
        output = str(result["output"]).replace("<", "\\u003c")
        reference = ""
        if result.get("ref"):
            reference = (
                f" ref={result['ref']} type={result.get('type', 'text')}.")
            if result.get("next_offset") is not None:
                reference += (
                    f" More is available at offset={result['next_offset']} "
                    "with read_result_page.")
        return f"Tool {result['tool']} {status}.{reference}\n{output}"

    @staticmethod
    def for_direct(call: ToolCall, result_json: str) -> str:
        """Render an audited result without invoking MiniCPM5."""
        result = json.loads(result_json)

        def terminal_safe(value: str) -> str:
            rendered = []
            for character in value:
                code = ord(character)
                if character == "\n":
                    rendered.append(character)
                elif code < 32 or code == 127:
                    rendered.append(f"\\x{code:02x}")
                else:
                    rendered.append(character)
            return "".join(rendered)

        output = terminal_safe(str(result["output"]))
        if not result["ok"]:
            return f"{call.name} 失败：{output}"

        def with_page_hint(value: str) -> str:
            if result.get("next_offset") is None:
                return value
            return (
                f"{value}\n[结果引用 {result['ref']}；继续读取 "
                f"offset={result['next_offset']}]")

        if call.name == "list_directory":
            path = terminal_safe(call.arguments.get("path", "."))
            lines = output.splitlines() or ["[empty directory]"]
            return with_page_hint(
                f"目录内容（{path}）：\n" + "\n".join(
                    f"  {line}" for line in lines))
        if call.name == "current_directory":
            return with_page_hint(f"当前工作目录：{output}")
        if call.name == "read_file":
            path = terminal_safe(call.arguments.get("path", ""))
            return with_page_hint(f"文件内容（{path}）：\n{output}")
        if call.name == "search_text":
            query = terminal_safe(call.arguments.get("query", ""))
            return with_page_hint(f"搜索结果（{query}）：\n{output}")
        if call.name == "git_status":
            return with_page_hint(f"Git 状态：\n{output}")
        if call.name == "calculate":
            return with_page_hint(f"计算结果：{output}")
        return with_page_hint(output)

    def _page_slice(self, arguments: dict[str, str]):
        ref = arguments.get("ref", "")
        if ref not in self._results:
            raise ToolExecutionError("unknown or expired result reference")
        source, result_type, source_complete = self._results[ref]
        offset = _integer(arguments, "offset", 0, 0, len(source))
        limit = _integer(
            arguments, "max_chars", self.max_output_chars, 64,
            self.max_output_chars)
        end = min(offset + limit, len(source))
        return ref, source[offset:end], result_type, source_complete, offset, end

    def _read_result_page(self, arguments: dict[str, str]) -> str:
        return self._page_slice(arguments)[1]

    def _page_result(self, arguments: dict[str, str]) -> str:
        ref, output, result_type, source_complete, offset, end = \
            self._page_slice(arguments)
        truncated = end < len(self._results[ref][0]) or not source_complete
        payload = {
            "tool": "read_result_page", "ok": True,
            "type": result_type, "ref": ref, "offset": offset,
            "output": output, "truncated": truncated,
            "next_offset": end if truncated else None,
            "source_complete": source_complete,
        }
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")).replace(
                "<", "\\u003c")

    def _current_directory(self, _arguments: dict[str, str]) -> str:
        return str(self.root)

    def _calculate(self, arguments: dict[str, str]) -> str:
        # One canonical line, so the first line of this result is already the
        # complete answer wherever the host displays or remembers it.
        return evaluate_expression(arguments.get("expression", ""))

    def _list_directory(self, arguments: dict[str, str]) -> str:
        path = self._path(arguments.get("path", "."))
        limit = _integer(arguments, "max_entries", 10, 1, 200)
        if not path.is_dir():
            raise ToolExecutionError(f"not a directory: {path.relative_to(self.root)}")
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: item.name)[:limit]:
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
        return "\n".join(entries) if entries else "[empty directory]"

    def _read_file(self, arguments: dict[str, str]) -> str:
        path = self._path(arguments["path"])
        start = _integer(arguments, "start_line", 1, 1, 1_000_000)
        end = _integer(arguments, "end_line", min(start + 39, 1_000_000),
                       start, start + 79)
        if not path.is_file():
            raise ToolExecutionError(f"not a file: {path.relative_to(self.root)}")
        if path.stat().st_size > (2 << 20):
            raise ToolExecutionError("file is larger than 2 MiB")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start - 1:end]
        return "\n".join(f"{number}: {line}" for number, line in
                         enumerate(selected, start=start))

    def _search_text(self, arguments: dict[str, str]) -> str:
        query = arguments["query"]
        if not query:
            raise ToolExecutionError("query is empty")
        base = self._path(arguments.get("path", "."))
        limit = _integer(arguments, "max_matches", 15, 1, 100)
        native = self._grep_search(query, base, limit)
        if native is not None:
            return "\n".join(native) if native else "[no matches]"
        matches = []
        def candidates():
            if base.is_file():
                yield base
                return
            if not base.is_dir():
                raise ToolExecutionError("search path does not exist")
            for root, dirs, files in os.walk(base):
                dirs[:] = [name for name in dirs
                           if name not in {".git", "__pycache__"}]
                for name in files:
                    candidate = Path(root) / name
                    if not candidate.is_symlink():
                        yield candidate

        for path in candidates():
            try:
                if path.stat().st_size > (1 << 20):
                    continue
                for number, line in enumerate(
                        path.read_text(encoding="utf-8", errors="strict").splitlines(), 1):
                    if query in line:
                        relative = path.relative_to(self.root)
                        matches.append(f"{relative}:{number}:{line[:240]}")
                        if len(matches) >= limit:
                            return "\n".join(matches)
            except (OSError, UnicodeError, ValueError):
                continue
        return "\n".join(matches) if matches else "[no matches]"

    def _grep_search(self, query: str, base: Path,
                     limit: int) -> list[str] | None:
        """Use GNU/BSD grep when available; do not follow workspace symlinks."""
        try:
            relative = base.relative_to(self.root)
        except ValueError:
            return None
        target = "." if not relative.parts else str(relative)
        recursive = ["-r"] if base.is_dir() else []
        command = [
            "grep", *recursive, "-nF", "--binary-files=without-match",
            "--exclude-dir=.git", "--exclude-dir=__pycache__",
            "--exclude=*.om", "--exclude=*.bin", "--exclude=*.pyc",
            "--exclude=*.so", "--exclude=*.aarch64", "--", query, target,
        ]
        process = None
        try:
            process = subprocess.Popen(
                command, cwd=self.root, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            if process.stdout is None:
                return None
            matches = []
            for raw in process.stdout:
                line = raw.rstrip("\r\n")
                if line.startswith("./"):
                    line = line[2:]
                parts = line.split(":", 2)
                if len(parts) == 3:
                    line = f"{parts[0]}:{parts[1]}:{parts[2][:240]}"
                matches.append(line)
                if len(matches) >= limit:
                    process.terminate()
                    break
            try:
                return_code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
                return None
            if return_code not in {0, 1, -15}:
                return None
            return matches
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            if process is not None and process.poll() is None:
                process.kill()
            return None

    def _git_status(self, _arguments: dict[str, str]) -> str:
        completed = subprocess.run(
            ["git", "status", "--short", "--branch"], cwd=self.root,
            text=True, capture_output=True, timeout=10, check=False)
        if completed.returncode:
            raise ToolExecutionError(completed.stderr.strip() or "git status failed")
        return completed.stdout.strip() or "[clean]"

    def _write_file(self, arguments: dict[str, str]) -> str:
        path = self._path(arguments["path"], for_write=True)
        if path.is_dir():
            # os.replace onto a directory raises Errno 21 from deep inside the
            # write, and the model is handed a temporary filename it has never
            # seen. A model that omits the filename should be told that, in
            # the terms it asked in.
            raise ToolExecutionError(
                "path names a directory; give the file to write")
        content, repairs = _repair_written_content(arguments["content"])
        if len(content.encode("utf-8")) > 16_384:
            raise ToolExecutionError("write_file content exceeds 16 KiB")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.minicpm-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        written = (f"wrote {len(content.encode('utf-8'))} bytes to "
                   f"{path.relative_to(self.root)}")
        return written + (f" ({'; '.join(repairs)})" if repairs else "")

    def _run_shell(self, arguments: dict[str, str]) -> str:
        command = arguments["command"].strip()
        if not command:
            raise ToolExecutionError("command is empty")
        timeout = _integer(arguments, "timeout_seconds", 20, 1, 60)
        if self.stdout_listener is not None:
            return self._run_shell_streaming(command, timeout)
        completed = subprocess.run(
            command, cwd=self.root, shell=True, text=True,
            capture_output=True, timeout=timeout, check=False)
        combined = completed.stdout
        if completed.stderr:
            combined += ("\n" if combined else "") + completed.stderr
        return f"exit={completed.returncode}\n{combined.strip()}"

    def _run_shell_streaming(self, command: str, timeout: int) -> str:
        """``_run_shell`` with stdout chunks surfaced while the tool runs.

        Returns the SAME string as the ``subprocess.run`` branch above --
        including the stderr-after-stdout concatenation, the text-mode
        newline translation and the final ``.strip()`` -- so a caller can
        never tell which branch produced a result.  A unit test pins the two
        against each other over a set of shell cases.
        """
        process = subprocess.Popen(
            command, cwd=self.root, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_buf, err_buf = bytearray(), bytearray()
        listener = self.stdout_listener

        def pump(stream, buffer, listen):
            # Read the RAW fd, never BufferedReader.read(n): the buffered
            # call blocks until n bytes OR EOF, so a tool emitting less than
            # n bytes streams NOTHING until it exits.  Board-measured on
            # SS928 (2026-08-16): a 42-line / 1386 B / 10 s tool delivered
            # one chunk at t=10.24 s under stream.read(4096) and 42 chunks
            # starting at t=0.23 s under os.read(fd, 4096).
            fileno = stream.fileno()
            while True:
                try:
                    chunk = os.read(fileno, 4096)
                except OSError:
                    return                      # fd closed/reset under us
                if not chunk:
                    return                      # EOF
                buffer.extend(chunk)
                if listen is not None:
                    listen(chunk)

        readers = [
            threading.Thread(target=pump, daemon=True,
                             args=(process.stdout, out_buf, listener)),
            threading.Thread(target=pump, daemon=True,
                             args=(process.stderr, err_buf, None)),
        ]
        for reader in readers:
            reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            for reader in readers:
                reader.join(timeout=2)
            raise
        for reader in readers:
            reader.join()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        stdout = _decode_text_mode(bytes(out_buf))
        stderr = _decode_text_mode(bytes(err_buf))
        combined = stdout
        if stderr:
            combined += ("\n" if combined else "") + stderr
        return f"exit={returncode}\n{combined.strip()}"
