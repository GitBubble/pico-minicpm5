#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5 native chat/tool protocol and a small local tool registry.

The renderer follows OpenBMB's ``chat_template.jinja`` contract: JSON function
signatures live inside ``<tools>`` and the assistant emits XML ``<function>``
calls.  This module deliberately has no third-party runtime dependencies so it
can run in the minimal Python environment shipped on SS928 boards.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable
import xml.etree.ElementTree as ET


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
BOS = "<s>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
# The deployed model has a ctx1024 contract. Tool results must leave room for
# the schemas, call, conversation and final response; callers can request a
# narrower follow-up window when more data is needed.
MAX_TOOL_OUTPUT_CHARS = 800


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


def route_obvious_read_only(
    user_text: str, previous_assistant: str = "",
) -> ToolCall | None:
    """Route only unambiguous directory-listing intents without an LLM.

    MiniCPM5 remains the general tool selector.  This narrow fallback covers
    the CLI-critical case where a small model asks the user for a path that the
    workspace tool already knows.  It never routes mutation or shell commands.
    """
    text = user_text.strip()
    lowered = text.lower()
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
    return ToolCall("list_directory", {"path": path, "max_entries": "10"})


def format_tool_call(call: ToolCall) -> str:
    """Serialize a host-routed call using MiniCPM5's native XML contract."""
    root = ET.Element("function", {"name": call.name})
    for name, value in call.arguments.items():
        child = ET.SubElement(root, "param", {"name": name})
        child.text = value
    return ET.tostring(root, encoding="unicode", short_empty_elements=False)


def _integer(arguments: dict[str, str], name: str, default: int,
             minimum: int, maximum: int) -> int:
    raw = arguments.get(name)
    value = default if raw in (None, "") else int(raw)
    if value < minimum or value > maximum:
        raise ToolExecutionError(f"{name} must be in [{minimum}, {maximum}]")
    return value


class WorkspaceTools:
    """Workspace-confined tools with explicit approval for mutation/shell."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"agent workspace is not a directory: {self.root}")
        obj = {"type": "object", "additionalProperties": False}
        self._tools = {
            tool.name: tool for tool in (
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

    def _path(self, value: str, *, for_write: bool = False) -> Path:
        raw = Path(value or ".")
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ToolExecutionError("path escapes the configured workspace") from error
        if for_write and candidate.is_symlink():
            raise ToolExecutionError("writing through a symlink is forbidden")
        return resolved

    @staticmethod
    def _clip(value: str) -> str:
        if len(value) <= MAX_TOOL_OUTPUT_CHARS:
            return value
        return value[:MAX_TOOL_OUTPUT_CHARS] + "\n...[tool output truncated]"

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
                approve: Callable[[str], bool] | None = None) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return self.result(call.name, False, f"unknown tool; available={self.names}")
        if tool.permission == "ask":
            if approve is None or not approve(self.preview(call)):
                return self.result(call.name, False, "permission denied by user")
        try:
            output = tool.handler(call.arguments)
            return self.result(call.name, True, self._clip(output))
        except (OSError, ValueError, ToolExecutionError, subprocess.SubprocessError) as error:
            return self.result(call.name, False, str(error))

    @staticmethod
    def result(name: str, ok: bool, output: str) -> str:
        # Prevent a file or command from closing the surrounding template tag.
        encoded = json.dumps(
            {"tool": name, "ok": bool(ok), "output": output},
            ensure_ascii=False, separators=(",", ":"))
        return encoded.replace("<", "\\u003c")

    @staticmethod
    def for_model(result_json: str) -> str:
        """Convert audited JSON evidence into compact model-readable text."""
        result = json.loads(result_json)
        status = "succeeded" if result["ok"] else "failed"
        output = str(result["output"]).replace("<", "\\u003c")
        return f"Tool {result['tool']} {status}.\n{output}"

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
                    yield Path(root) / name

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

    def _git_status(self, _arguments: dict[str, str]) -> str:
        completed = subprocess.run(
            ["git", "status", "--short", "--branch"], cwd=self.root,
            text=True, capture_output=True, timeout=10, check=False)
        if completed.returncode:
            raise ToolExecutionError(completed.stderr.strip() or "git status failed")
        return completed.stdout.strip() or "[clean]"

    def _write_file(self, arguments: dict[str, str]) -> str:
        path = self._path(arguments["path"], for_write=True)
        content = arguments["content"]
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
        return f"wrote {len(content.encode('utf-8'))} bytes to {path.relative_to(self.root)}"

    def _run_shell(self, arguments: dict[str, str]) -> str:
        command = arguments["command"].strip()
        if not command:
            raise ToolExecutionError("command is empty")
        timeout = _integer(arguments, "timeout_seconds", 20, 1, 60)
        completed = subprocess.run(
            command, cwd=self.root, shell=True, text=True,
            capture_output=True, timeout=timeout, check=False)
        combined = completed.stdout
        if completed.stderr:
            combined += ("\n" if combined else "") + completed.stderr
        return f"exit={completed.returncode}\n{combined.strip()}"
