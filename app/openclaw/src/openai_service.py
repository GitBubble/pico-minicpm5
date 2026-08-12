#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible HTTP service for the native MiniCPM5 runner.

The service owns chat templating and tokenization.  The native C runner owns
the model and its packed KV state; no ModelZoo-style per-layer KV tensors are
part of this boundary.  A request and response are one UTF-8 JSON object per
line over either child-process stdin/stdout or an AF_UNIX stream socket.

Runner request (generated token IDs are not included in ``input_ids``)::

    {"protocol":"pico.minicpm5.runner.v1","request_id":"...",
     "op":"generate","model":"minicpm5-1b","input_ids":[0, ...],
     "max_new_tokens":64,"temperature":0.0,"eos_token_ids":[1,130073],
     "reset_kv":true}

Successful runner response (``output_ids`` contains generated IDs only)::

    {"protocol":"pico.minicpm5.runner.v1","request_id":"...","ok":true,
     "output_ids":[...],"finish_reason":"stop"}

Runner error response::

    {"protocol":"pico.minicpm5.runner.v1","request_id":"...","ok":false,
     "error":{"code":"runner_error","message":"..."}}

The default service remains deliberately narrow: one in-flight generation,
temperature zero, text-only messages, and either a normal JSON response or one
data-bearing SSE event followed by ``[DONE]``.  Structured tool calling is an
explicit opt-in.  In that mode the service renders the checkpoint's official
tool template and translates only model-emitted, strictly validated MiniCPM5
XML into OpenAI ``tool_calls``; it never executes a tool itself.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from types import TracebackType
from typing import Mapping, Protocol, Sequence, TextIO
from urllib.parse import urlsplit
import uuid
import xml.etree.ElementTree as ElementTree


MODEL_ID = "minicpm5-1b"
MODEL_OWNER = "openbmb"
RUNNER_PROTOCOL = "pico.minicpm5.runner.v1"
SUPPORTS_TOOLS = False
EOS_TOKEN_IDS = (1, 130073)
MAX_HTTP_BODY_BYTES = 1 << 20
MAX_RUNNER_LINE_BYTES = 16 << 20
MAX_MODEL_CONTEXT = 131072
TOKENIZER_SHA256 = (
    "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81")
CHAT_TEMPLATE_SHA256 = (
    "7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c")
MAX_TOOLS = 64
# The native Agent contract intentionally permits one tool-call block per
# active user turn and one call in that block.  MiniCPM5-1B otherwise tends to
# repeat a successful in-process tool after its result is re-fed.
MAX_TOOL_CALLS = 1
TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
TOOL_CALL_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_TOOL_XML_MARKERS = ("<function", "</function>", "<param", "</param>")
FINAL_ONLY_SYSTEM_DIRECTIVE = (
    "The tool-call budget for this user turn is exhausted. Do not emit "
    "function XML and do not request another tool. Use the tool result already "
    "present in the conversation and return the final answer as plain text only."
)


class RunnerBackend(Protocol):
    """Transport-neutral native runner interface."""

    def generate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Return one protocol response for one protocol request."""


class RunnerProtocolError(RuntimeError):
    """The native runner transport or response violated the JSON contract."""


class BusyError(RuntimeError):
    """Another generation already owns the single inference slot."""


@dataclass(frozen=True)
class APIError(Exception):
    status: int
    message: str
    error_type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            },
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _template_tojson(
        value: object, ensure_ascii: bool = False, **_: object) -> str:
    """Transformers-compatible ``tojson`` for the pinned external template.

    Stock Jinja's filter accepts ``indent`` but not the template's
    ``ensure_ascii=False`` keyword.  Installing this filter before compiling
    the template preserves the official template bytes while matching the
    Transformers rendering contract used by the checkpoint.
    """
    return json.dumps(value, ensure_ascii=bool(ensure_ascii))


class MiniCPM5Tokenizer:
    """Pinned MiniCPM5 tokenizer plus the checkpoint's external chat template."""

    def __init__(self, tokenizer_json: Path, chat_template: Path) -> None:
        tokenizer_json = Path(tokenizer_json)
        chat_template = Path(chat_template)
        if _sha256(tokenizer_json) != TOKENIZER_SHA256:
            raise ValueError("MiniCPM5 tokenizer.json hash drift")
        if _sha256(chat_template) != CHAT_TEMPLATE_SHA256:
            raise ValueError("MiniCPM5 chat_template.jinja hash drift")

        try:
            from jinja2 import Undefined
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - board packaging guard
            raise RuntimeError(
                "tokenizers and jinja2 are required by the MiniCPM5 service") \
                from exc

        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        environment = ImmutableSandboxedEnvironment(
            undefined=Undefined, autoescape=False)
        environment.filters["tojson"] = _template_tojson
        self._template = environment.from_string(
            chat_template.read_text(encoding="utf-8"))
        self.tokenizer_json = tokenizer_json.resolve()
        self.chat_template = chat_template.resolve()

        expected_specials = {
            "<s>": 0,
            "</s>": 1,
            "<|im_start|>": 130072,
            "<|im_end|>": 130073,
        }
        actual = {
            token: self._tokenizer.token_to_id(token)
            for token in expected_specials
        }
        if actual != expected_specials or self._tokenizer.get_vocab_size() != 130560:
            raise ValueError("MiniCPM5 tokenizer vocabulary contract drift")

    def render(
            self, messages: Sequence[Mapping[str, object]],
            tools: Sequence[Mapping[str, object]] | None = None) -> str:
        """Render the pinned official template, optionally with tools."""
        return self._template.render(
            messages=list(messages),
            bos_token="<s>",
            eos_token="</s>",
            tools=list(tools) if tools else None,
            add_generation_prompt=True,
            enable_thinking=False,
            has_tool_sep=False,
        )

    def encode_messages(
            self, messages: Sequence[Mapping[str, object]],
            tools: Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[str, list[int]]:
        prompt = self.render(messages, tools)
        # The template already emits <s>; adding the tokenizer post-processor
        # here would duplicate BOS.
        ids = self._tokenizer.encode(prompt, add_special_tokens=False).ids
        return prompt, ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            [int(token_id) for token_id in token_ids], skip_special_tokens=True)

    def decode_structured(self, token_ids: Sequence[int]) -> str:
        """Decode model text without deleting MiniCPM5's XML special tokens."""
        ids = [int(token_id) for token_id in token_ids]
        while ids and ids[-1] in EOS_TOKEN_IDS:
            ids.pop()
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())


def _read_json_line(reader: TextIO) -> Mapping[str, object]:
    line = reader.readline(MAX_RUNNER_LINE_BYTES + 1)
    if not line:
        raise RunnerProtocolError("native runner closed before responding")
    if len(line.encode("utf-8")) > MAX_RUNNER_LINE_BYTES or not line.endswith("\n"):
        raise RunnerProtocolError("native runner response line is too large or unterminated")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RunnerProtocolError("native runner returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RunnerProtocolError("native runner response must be a JSON object")
    return payload


def _write_json_line(writer: TextIO, payload: Mapping[str, object]) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    if len(line.encode("utf-8")) > MAX_RUNNER_LINE_BYTES:
        raise RunnerProtocolError("native runner request line is too large")
    writer.write(line)
    writer.flush()


class JsonLineStreamBackend:
    """JSON-lines backend over an already-open stdin/stdout-like pair."""

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        self._reader = reader
        self._writer = writer
        self._lock = threading.Lock()

    def generate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        with self._lock:
            _write_json_line(self._writer, request)
            return _read_json_line(self._reader)


class SubprocessRunnerBackend(JsonLineStreamBackend):
    """Spawn a C runner and use its stdout/stdin as a JSON-lines transport."""

    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ValueError("runner command must not be empty")
        process = subprocess.Popen(
            [str(part) for part in command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:  # pragma: no cover
            process.terminate()
            raise RuntimeError("failed to create native runner pipes")
        self._process = process
        super().__init__(process.stdout, process.stdin)

    def close(self) -> None:
        # JSONL runners own resident device handles.  Closing stdin first lets
        # their EOF/finally path send the native shutdown opcode and release
        # MMZ cleanly; SIGTERM-first can orphan the executor and its handles.
        with self._lock:
            if self._process.poll() is None:
                try:
                    assert self._process.stdin is not None
                    self._process.stdin.close()
                except (OSError, ValueError):
                    # The process or another orderly close may have closed
                    # the text pipe first.
                    pass
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            finally:
                if self._process.stdout is not None:
                    self._process.stdout.close()

    def __enter__(self) -> "SubprocessRunnerBackend":
        return self

    def __exit__(
            self, exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None) -> None:
        self.close()


class UnixSocketRunnerBackend:
    """Open one AF_UNIX connection per generation and exchange one JSON line."""

    def __init__(self, path: Path, timeout_seconds: float = 600.0) -> None:
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("runner socket timeout must be positive")

    def generate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.path))
            with (connection.makefile("r", encoding="utf-8") as reader,
                  connection.makefile("w", encoding="utf-8") as writer):
                _write_json_line(writer, request)
                return _read_json_line(reader)


@dataclass(frozen=True)
class ToolChoice:
    mode: str
    name: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ValidatedChatRequest:
    messages: list[dict[str, object]]
    tools: list[dict[str, object]]
    tool_schemas: dict[str, dict[str, object]]
    tool_choice: ToolChoice
    final_only: bool
    max_tokens: int
    stream: bool


@dataclass(frozen=True)
class ChatResult:
    completion_id: str
    created: int
    text: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class MiniCPM5Service:
    """Validation, tokenization, single-flight admission, and runner bridge."""

    def __init__(
            self, tokenizer: MiniCPM5Tokenizer, backend: RunnerBackend,
            context_window: int, default_max_tokens: int = 256, *,
            enable_tools: bool = False) -> None:
        context_window = int(context_window)
        default_max_tokens = int(default_max_tokens)
        if not 1 <= context_window <= MAX_MODEL_CONTEXT:
            raise ValueError(
                f"context_window must be in the range 1..{MAX_MODEL_CONTEXT}")
        if not 1 <= default_max_tokens <= context_window:
            raise ValueError(
                "default_max_tokens must be positive and no larger than context_window")
        self.tokenizer = tokenizer
        self.backend = backend
        self.context_window = context_window
        self.default_max_tokens = default_max_tokens
        self.enable_tools = bool(enable_tools)
        self._inference_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._inference_lock.locked()

    @staticmethod
    def _validate_messages(value: object) -> list[dict[str, str]]:
        if not isinstance(value, list) or not value:
            raise APIError(
                HTTPStatus.BAD_REQUEST, "messages must be a non-empty array",
                param="messages", code="invalid_messages")
        messages: list[dict[str, str]] = []
        allowed_roles = {"system", "user", "assistant"}
        for index, message in enumerate(value):
            if not isinstance(message, dict):
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    f"messages[{index}] must be an object",
                    param="messages", code="invalid_messages")
            extra = set(message) - {"role", "content"}
            if extra:
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    f"messages[{index}] has unsupported keys: {sorted(extra)}",
                    param="messages", code="unsupported_message_fields")
            role = message.get("role")
            content = message.get("content")
            if role not in allowed_roles or not isinstance(content, str):
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    f"messages[{index}] requires a supported role and string content",
                    param="messages", code="invalid_messages")
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _request_error(message: str, param: str, code: str) -> APIError:
        return APIError(
            HTTPStatus.BAD_REQUEST, message, param=param, code=code)

    @classmethod
    def _validate_tools(
            cls, value: object,
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
        if value in (None, []):
            return [], {}
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_TOOLS:
            raise cls._request_error(
                f"tools must be an array with at most {MAX_TOOLS} entries",
                "tools", "invalid_tools")

        tools: list[dict[str, object]] = []
        schemas: dict[str, dict[str, object]] = {}
        for index, tool in enumerate(value):
            label = f"tools[{index}]"
            if (not isinstance(tool, dict) or set(tool) != {"type", "function"}
                    or tool.get("type") != "function"):
                raise cls._request_error(
                    f"{label} must be an OpenAI function tool", "tools",
                    "invalid_tools")
            function = tool.get("function")
            allowed_function_keys = {
                "name", "description", "parameters", "strict"}
            if (not isinstance(function, dict) or
                    not set(function) <= allowed_function_keys):
                raise cls._request_error(
                    f"{label}.function has an invalid shape", "tools",
                    "invalid_tools")
            name = function.get("name")
            if not isinstance(name, str) or TOOL_NAME_RE.fullmatch(name) is None:
                raise cls._request_error(
                    f"{label}.function.name is invalid", "tools",
                    "invalid_tools")
            if name in schemas:
                raise cls._request_error(
                    f"duplicate tool name {name!r}", "tools",
                    "duplicate_tool")
            description = function.get("description")
            if description is not None and not isinstance(description, str):
                raise cls._request_error(
                    f"{label}.function.description must be a string", "tools",
                    "invalid_tools")
            strict = function.get("strict")
            if strict is not None and not isinstance(strict, bool):
                raise cls._request_error(
                    f"{label}.function.strict must be a boolean", "tools",
                    "invalid_tools")

            parameters = function.get(
                "parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, dict):
                raise cls._request_error(
                    f"{label}.function.parameters must be an object", "tools",
                    "invalid_tools")
            schema_type = parameters.get("type", "object")
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            if schema_type != "object" or not isinstance(properties, dict):
                raise cls._request_error(
                    f"{label}.function.parameters must describe an object",
                    "tools", "invalid_tools")
            if (not isinstance(required, list) or
                    any(not isinstance(item, str) for item in required) or
                    len(set(required)) != len(required)):
                raise cls._request_error(
                    f"{label}.function.parameters.required is invalid",
                    "tools", "invalid_tools")
            for parameter_name, property_schema in properties.items():
                if (not isinstance(parameter_name, str) or
                        TOOL_NAME_RE.fullmatch(parameter_name) is None or
                        not isinstance(property_schema, dict)):
                    raise cls._request_error(
                        f"{label}.function.parameters.properties is invalid",
                        "tools", "invalid_tools")
                property_type = property_schema.get("type", "string")
                if property_type not in {
                        "string", "integer", "number", "boolean", "object",
                        "array", "null"}:
                    raise cls._request_error(
                        f"{label} parameter {parameter_name!r} has an "
                        "unsupported type", "tools", "invalid_tools")
                enum = property_schema.get("enum")
                if enum is not None and not isinstance(enum, list):
                    raise cls._request_error(
                        f"{label} parameter {parameter_name!r} has an invalid "
                        "enum", "tools", "invalid_tools")
            if any(item not in properties for item in required):
                raise cls._request_error(
                    f"{label}.function.parameters.required names an unknown "
                    "property", "tools", "invalid_tools")

            # Copy through JSON to detach request-owned mutable containers and
            # to guarantee that the official template receives JSON values.
            normalized = json.loads(json.dumps(tool))
            if "parameters" not in normalized["function"]:
                normalized["function"]["parameters"] = parameters
            tools.append(normalized)
            schemas[name] = json.loads(json.dumps(parameters))
        return tools, schemas

    @classmethod
    def _validate_tool_choice(
            cls, value: object, tool_schemas: Mapping[str, object],
    ) -> ToolChoice:
        if value is None:
            return ToolChoice("auto" if tool_schemas else "none")
        if isinstance(value, str):
            if value not in {"none", "auto", "required"}:
                raise cls._request_error(
                    "tool_choice must be none, auto, required, or a named "
                    "function", "tool_choice", "invalid_tool_choice")
            if not tool_schemas and value != "none":
                raise cls._request_error(
                    "tool_choice requires at least one tool", "tool_choice",
                    "invalid_tool_choice")
            return ToolChoice(value)
        if not isinstance(value, dict) or set(value) != {"type", "function"}:
            raise cls._request_error(
                "tool_choice has an invalid shape", "tool_choice",
                "invalid_tool_choice")
        function = value.get("function")
        if (value.get("type") != "function" or not isinstance(function, dict)
                or set(function) != {"name"}):
            raise cls._request_error(
                "tool_choice has an invalid function selector", "tool_choice",
                "invalid_tool_choice")
        name = function.get("name")
        if not isinstance(name, str) or name not in tool_schemas:
            raise cls._request_error(
                "tool_choice names an unknown function", "tool_choice",
                "invalid_tool_choice")
        return ToolChoice("function", name)

    @staticmethod
    def _validate_schema_value(
            value: object, schema: Mapping[str, object], *, source: str) -> None:
        expected = schema.get("type", "string")
        valid = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": (isinstance(value, (int, float)) and
                       not isinstance(value, bool) and
                       math.isfinite(float(value))),
            "boolean": isinstance(value, bool),
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "null": value is None,
        }[str(expected)]
        if not valid:
            raise RunnerProtocolError(
                f"{source} does not match schema type {expected!r}")
        enum = schema.get("enum")
        if isinstance(enum, list) and value not in enum:
            raise RunnerProtocolError(f"{source} is outside its enum")

    @classmethod
    def _validate_arguments_object(
            cls, name: str, arguments: object,
            tool_schemas: Mapping[str, Mapping[str, object]], *,
            source: str) -> dict[str, object]:
        if not isinstance(arguments, dict):
            raise RunnerProtocolError(f"{source} must be a JSON object")
        schema = tool_schemas[name]
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        assert isinstance(properties, dict)
        assert isinstance(required, list)
        unknown = set(arguments) - set(properties)
        missing = set(required) - set(arguments)
        if unknown:
            raise RunnerProtocolError(
                f"{source} has unknown parameters: {sorted(unknown)}")
        if missing:
            raise RunnerProtocolError(
                f"{source} is missing required parameters: {sorted(missing)}")
        for key, value in arguments.items():
            property_schema = properties[key]
            assert isinstance(property_schema, dict)
            cls._validate_schema_value(
                value, property_schema, source=f"{source}.{key}")
        return json.loads(json.dumps(arguments, ensure_ascii=False))

    @classmethod
    def _validate_tool_messages(
            cls, value: object,
            tool_schemas: Mapping[str, Mapping[str, object]],
    ) -> tuple[list[dict[str, object]], bool]:
        if not isinstance(value, list) or not value:
            raise cls._request_error(
                "messages must be a non-empty array", "messages",
                "invalid_messages")
        normalized: list[dict[str, object]] = []
        pending: dict[str, str] = {}
        seen_call_ids: set[str] = set()
        active_user_seen = False
        active_tool_blocks = 0
        active_round_finalized = False
        for index, message in enumerate(value):
            label = f"messages[{index}]"
            if not isinstance(message, dict):
                raise cls._request_error(
                    f"{label} must be an object", "messages",
                    "invalid_messages")
            role = message.get("role")
            if pending and role != "tool":
                raise cls._request_error(
                    f"{label} appears before all tool results", "messages",
                    "invalid_tool_history")
            if role in {"system", "user"}:
                if set(message) != {"role", "content"} or not isinstance(
                        message.get("content"), str):
                    raise cls._request_error(
                        f"{label} requires role and string content", "messages",
                        "invalid_messages")
                normalized.append({
                    "role": str(role), "content": str(message["content"])})
                if role == "user":
                    if not active_user_seen:
                        active_user_seen = True
                    elif active_round_finalized:
                        # Only an assistant plain-text final closes the prior
                        # active turn.  A synthetic/compaction user message
                        # immediately after a tool result must not mint a new
                        # call budget and bypass the loop guard.
                        active_tool_blocks = 0
                        active_round_finalized = False
                continue
            if role == "assistant":
                allowed = {"role", "content", "tool_calls"}
                if not set(message) <= allowed:
                    raise cls._request_error(
                        f"{label} has unsupported keys", "messages",
                        "unsupported_message_fields")
                content = message.get("content")
                calls = message.get("tool_calls")
                if calls is None:
                    if not isinstance(content, str):
                        raise cls._request_error(
                            f"{label}.content must be a string", "messages",
                            "invalid_messages")
                    normalized.append({"role": "assistant", "content": content})
                    if active_user_seen:
                        active_round_finalized = True
                    continue
                if (not isinstance(calls, list) or not calls or
                        len(calls) > MAX_TOOL_CALLS or
                        content is not None and not isinstance(content, str)):
                    raise cls._request_error(
                        f"{label}.tool_calls is invalid", "messages",
                        "invalid_tool_history")
                if not active_user_seen:
                    raise cls._request_error(
                        f"{label}.tool_calls has no active user turn",
                        "messages", "invalid_tool_history")
                if active_round_finalized:
                    raise cls._request_error(
                        f"{label}.tool_calls appears after the active user "
                        "turn's final assistant response", "messages",
                        "invalid_tool_history")
                if active_tool_blocks >= 1:
                    raise cls._request_error(
                        "the active user turn already contains its one "
                        "allowed tool-call block", "messages",
                        "tool_call_budget_exceeded")
                active_tool_blocks += 1
                normalized_calls: list[dict[str, object]] = []
                for call_index, call in enumerate(calls):
                    call_label = f"{label}.tool_calls[{call_index}]"
                    if (not isinstance(call, dict) or
                            set(call) != {"id", "type", "function"} or
                            call.get("type") != "function"):
                        raise cls._request_error(
                            f"{call_label} has an invalid shape", "messages",
                            "invalid_tool_history")
                    call_id = call.get("id")
                    function = call.get("function")
                    if (not isinstance(call_id, str) or
                            TOOL_CALL_ID_RE.fullmatch(call_id) is None or
                            call_id in seen_call_ids or
                            not isinstance(function, dict) or
                            set(function) != {"name", "arguments"}):
                        raise cls._request_error(
                            f"{call_label} has invalid or duplicate metadata",
                            "messages", "invalid_tool_history")
                    name = function.get("name")
                    serialized = function.get("arguments")
                    if not isinstance(name, str) or name not in tool_schemas:
                        raise cls._request_error(
                            f"{call_label} names an unknown function", "messages",
                            "invalid_tool_history")
                    if not isinstance(serialized, str):
                        raise cls._request_error(
                            f"{call_label}.function.arguments must be a JSON "
                            "string", "messages", "invalid_tool_history")
                    try:
                        parsed_arguments = json.loads(serialized)
                        arguments = cls._validate_arguments_object(
                            name, parsed_arguments, tool_schemas,
                            source=f"{call_label}.function.arguments")
                    except (json.JSONDecodeError, RunnerProtocolError) as exc:
                        raise cls._request_error(
                            f"{call_label}.function.arguments is invalid: {exc}",
                            "messages", "invalid_tool_history") from exc
                    seen_call_ids.add(call_id)
                    pending[call_id] = name
                    normalized_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    })
                normalized.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": normalized_calls,
                })
                continue
            if role == "tool":
                allowed = {"role", "content", "tool_call_id", "name"}
                if (not set(message) <= allowed or
                        not isinstance(message.get("content"), str)):
                    raise cls._request_error(
                        f"{label} has an invalid tool result shape", "messages",
                        "invalid_tool_history")
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    raise cls._request_error(
                        f"{label}.tool_call_id does not match a pending call",
                        "messages", "invalid_tool_history")
                name = message.get("name")
                if name is not None and name != pending[call_id]:
                    raise cls._request_error(
                        f"{label}.name does not match its pending call",
                        "messages", "invalid_tool_history")
                normalized_message: dict[str, object] = {
                    "role": "tool",
                    "content": str(message["content"]),
                    "tool_call_id": call_id,
                }
                if name is not None:
                    normalized_message["name"] = name
                normalized.append(normalized_message)
                del pending[call_id]
                continue
            raise cls._request_error(
                f"{label} has an unsupported role", "messages",
                "invalid_messages")
        if pending:
            raise cls._request_error(
                "messages end before all tool results are supplied", "messages",
                "invalid_tool_history")
        return normalized, active_tool_blocks == 1

    def _validate_request(
            self, payload: object) -> ValidatedChatRequest:
        if not isinstance(payload, dict):
            raise APIError(
                HTTPStatus.BAD_REQUEST, "request body must be a JSON object",
                code="invalid_json")
        if payload.get("model") != MODEL_ID:
            raise APIError(
                HTTPStatus.NOT_FOUND, f"model must be {MODEL_ID!r}",
                param="model", code="model_not_found")

        temperature = payload.get("temperature", 0)
        if (isinstance(temperature, bool) or
                not isinstance(temperature, (int, float)) or
                float(temperature) != 0.0):
            raise APIError(
                HTTPStatus.BAD_REQUEST, "only temperature=0 is supported",
                param="temperature", code="unsupported_temperature")

        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise APIError(
                HTTPStatus.BAD_REQUEST, "stream must be a boolean",
                param="stream", code="invalid_stream")

        if "max_tokens" in payload and "max_completion_tokens" in payload:
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                "max_tokens and max_completion_tokens are mutually exclusive",
                param="max_completion_tokens", code="conflicting_parameters")
        max_tokens_param = (
            "max_completion_tokens"
            if "max_completion_tokens" in payload else "max_tokens")
        max_tokens = payload.get(
            max_tokens_param, self.default_max_tokens)
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or
                not 1 <= max_tokens <= self.default_max_tokens):
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                f"{max_tokens_param} must be in the range "
                f"1..{self.default_max_tokens}",
                param=max_tokens_param, code="invalid_max_tokens")

        if not self.enable_tools:
            tools = payload.get("tools")
            if tools not in (None, []):
                raise APIError(
                    HTTPStatus.BAD_REQUEST, "tools are not supported",
                    param="tools", code="tools_not_supported")
            if payload.get("tool_choice") not in (None, "none"):
                raise APIError(
                    HTTPStatus.BAD_REQUEST, "tool_choice is not supported",
                    param="tool_choice", code="tools_not_supported")
            return ValidatedChatRequest(
                messages=list(self._validate_messages(payload.get("messages"))),
                tools=[], tool_schemas={}, tool_choice=ToolChoice("none"),
                final_only=False, max_tokens=max_tokens, stream=stream)

        tools, tool_schemas = self._validate_tools(payload.get("tools"))
        tool_choice = self._validate_tool_choice(
            payload.get("tool_choice"), tool_schemas)
        messages, tool_budget_exhausted = self._validate_tool_messages(
            payload.get("messages"), tool_schemas)
        if (tool_budget_exhausted and
                tool_choice.mode in {"required", "function"}):
            raise self._request_error(
                "tool_choice cannot require another call after the active "
                "user turn exhausted its one-call budget", "tool_choice",
                "tool_call_budget_exceeded")
        if tool_budget_exhausted:
            # Keep schemas for validating the historical tool call, but hide
            # the catalog from the next prompt and require a final response.
            tool_choice = ToolChoice("none")
        return ValidatedChatRequest(
            messages=messages, tools=tools, tool_schemas=tool_schemas,
            tool_choice=tool_choice, final_only=tool_budget_exhausted,
            max_tokens=max_tokens, stream=stream)

    def complete(self, payload: object) -> tuple[ChatResult, bool]:
        request = self._validate_request(payload)
        if not self._inference_lock.acquire(blocking=False):
            raise BusyError("the single native inference slot is busy")
        try:
            render_tools: list[dict[str, object]] | None = None
            render_messages = request.messages
            if request.final_only:
                render_messages = [
                    *request.messages,
                    {"role": "system", "content": FINAL_ONLY_SYSTEM_DIRECTIVE},
                ]
            if request.tool_choice.mode in {"auto", "required"}:
                render_tools = request.tools
            elif request.tool_choice.mode == "function":
                render_tools = [
                    tool for tool in request.tools
                    if tool["function"]["name"] == request.tool_choice.name]
            _, input_ids = self.tokenizer.encode_messages(
                render_messages, render_tools)
            prompt_tokens = len(input_ids)
            if prompt_tokens + request.max_tokens > self.context_window:
                raise APIError(
                    HTTPStatus.BAD_REQUEST,
                    "prompt tokens plus max_tokens exceed the compiled context window",
                    param="messages", code="context_length_exceeded")

            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            runner_request: dict[str, object] = {
                "protocol": RUNNER_PROTOCOL,
                "request_id": completion_id,
                "op": "generate",
                "model": MODEL_ID,
                "input_ids": input_ids,
                "max_new_tokens": request.max_tokens,
                "temperature": 0.0,
                "eos_token_ids": list(EOS_TOKEN_IDS),
                # KV never crosses this service boundary.  The native runner
                # resets and manages its packed internal state for this prompt.
                "reset_kv": True,
            }
            response = self.backend.generate(runner_request)
            output_ids, finish_reason = self._validate_runner_response(
                response, completion_id, request.max_tokens)
            text: str | None
            tool_calls: tuple[ToolCall, ...]
            if self.enable_tools:
                raw_text = self.tokenizer.decode_structured(output_ids)
                if request.final_only:
                    text = self._final_only_text(
                        raw_text, request.tool_schemas, completion_id)
                    tool_calls = ()
                    # An exhausted tool turn must terminate even if the small
                    # model repeats XML, emits nothing, or reaches its token
                    # cap.  Returning another tool call or ``length`` lets
                    # OpenClaw re-enter the loop we are guarding.
                    finish_reason = "stop"
                else:
                    text, tool_calls = self._interpret_tool_output(
                        raw_text, finish_reason, completion_id,
                        request.tool_schemas, request.tool_choice)
                    if tool_calls:
                        finish_reason = "tool_calls"
            else:
                text = self.tokenizer.decode(output_ids)
                tool_calls = ()
            return ChatResult(
                completion_id=completion_id,
                created=int(time.time()),
                text=text,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(output_ids),
            ), request.stream
        finally:
            self._inference_lock.release()

    @classmethod
    def _final_only_text(
            cls, raw_text: str,
            tool_schemas: Mapping[str, Mapping[str, object]],
            completion_id: str) -> str:
        """Return safe final text after the current turn used its tool call.

        Valid repeated function XML is removed rather than exposed as text.
        Malformed or unknown XML is also reduced to an empty final response:
        no second call can cross the OpenAI boundary, and a protocol error
        cannot provoke OpenClaw into retrying the same tool round.
        """
        try:
            content, fragments = cls._split_tool_output(raw_text)
            if fragments:
                cls._parse_tool_fragments(
                    fragments, tool_schemas, completion_id)
                return content or ""
        except RunnerProtocolError:
            return ""
        return raw_text

    @staticmethod
    def _xml_parameter_value(raw: str, schema: Mapping[str, object]) -> object:
        expected = schema.get("type", "string")
        if expected == "string":
            return raw
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunnerProtocolError(
                f"tool parameter is not valid JSON for type {expected!r}") from exc
        MiniCPM5Service._validate_schema_value(
            value, schema, source="tool parameter")
        return value

    @classmethod
    def _parse_tool_xml(
            cls, raw_text: str,
            tool_schemas: Mapping[str, Mapping[str, object]],
            completion_id: str) -> tuple[ToolCall, ...]:
        _, fragments = cls._split_tool_output(raw_text)
        return cls._parse_tool_fragments(
            fragments, tool_schemas, completion_id)

    @staticmethod
    def _function_fragment_end(text: str, start: int) -> int:
        """Find one complete function element while respecting CDATA text."""
        cursor = start + len("<function")
        while cursor < len(text):
            cdata = text.find("<![CDATA[", cursor)
            nested = text.find("<function", cursor)
            closing = text.find("</function>", cursor)
            candidates = [
                position for position in (cdata, nested, closing)
                if position >= 0]
            if not candidates:
                raise RunnerProtocolError("model emitted incomplete tool XML")
            position = min(candidates)
            if position == cdata:
                cdata_end = text.find("]]>", cdata + len("<![CDATA["))
                if cdata_end < 0:
                    raise RunnerProtocolError("model emitted incomplete CDATA")
                cursor = cdata_end + len("]]>")
                continue
            if position == nested:
                raise RunnerProtocolError("nested tool XML is forbidden")
            return closing + len("</function>")
        raise RunnerProtocolError("model emitted incomplete tool XML")

    @classmethod
    def _split_tool_output(
            cls, raw_text: str) -> tuple[str | None, list[str]]:
        """Separate ordinary assistant content from complete function XML.

        Function and parameter markers are structural in tool mode.  Every
        such marker must belong to a complete function fragment; this permits
        normal narration around calls without accepting truncated XML as text.
        """
        if not any(marker in raw_text for marker in _TOOL_XML_MARKERS):
            return raw_text, []
        if ("<?" in raw_text or "<!--" in raw_text or
                re.search(r"<!(?!\[CDATA\[)", raw_text) is not None):
            raise RunnerProtocolError("tool XML contains a forbidden declaration")

        content_parts: list[str] = []
        fragments: list[str] = []
        cursor = 0
        while True:
            start = raw_text.find("<function", cursor)
            if start < 0:
                content_parts.append(raw_text[cursor:])
                break
            content_parts.append(raw_text[cursor:start])
            end = cls._function_fragment_end(raw_text, start)
            fragments.append(raw_text[start:end])
            cursor = end
        if not fragments:
            raise RunnerProtocolError("tool XML contains no function call")
        content = "".join(content_parts)
        pollution = _TOOL_XML_MARKERS + ("<![CDATA[", "]]>")
        if any(marker in content for marker in pollution):
            raise RunnerProtocolError("tool XML marker appears outside a function")
        content = content.strip()
        return (content if content else None), fragments

    @classmethod
    def _parse_tool_fragments(
            cls, fragments: Sequence[str],
            tool_schemas: Mapping[str, Mapping[str, object]],
            completion_id: str) -> tuple[ToolCall, ...]:
        if not fragments:
            return ()
        parsed: list[tuple[str, dict[str, object]]] = []
        seen_calls: set[tuple[str, str]] = set()
        if len(fragments) > MAX_TOOL_CALLS:
            raise RunnerProtocolError("model emitted too many tool calls")
        for fragment in fragments:
            try:
                function = ElementTree.fromstring(fragment)
            except ElementTree.ParseError as exc:
                raise RunnerProtocolError("model emitted malformed tool XML") from exc
            if function.tag != "function" or set(function.attrib) != {"name"}:
                raise RunnerProtocolError("tool XML has an invalid function element")
            if function.text and function.text.strip():
                raise RunnerProtocolError("function XML contains unexpected text")
            name = function.attrib["name"]
            if name not in tool_schemas:
                raise RunnerProtocolError(
                    f"model emitted unknown function {name!r}")
            schema = tool_schemas[name]
            properties = schema.get("properties", {})
            assert isinstance(properties, dict)
            arguments: dict[str, object] = {}
            for parameter in function:
                if parameter.tag != "param" or set(parameter.attrib) != {"name"}:
                    raise RunnerProtocolError(
                        "tool XML has an invalid parameter element")
                if list(parameter):
                    raise RunnerProtocolError("nested tool XML is forbidden")
                if parameter.tail and parameter.tail.strip():
                    raise RunnerProtocolError(
                        "function XML contains unexpected text")
                parameter_name = parameter.attrib["name"]
                if parameter_name not in properties:
                    raise RunnerProtocolError(
                        f"function {name!r} has unknown parameter "
                        f"{parameter_name!r}")
                if parameter_name in arguments:
                    raise RunnerProtocolError(
                        f"function {name!r} repeats parameter "
                        f"{parameter_name!r}")
                property_schema = properties[parameter_name]
                assert isinstance(property_schema, dict)
                arguments[parameter_name] = cls._xml_parameter_value(
                    parameter.text or "", property_schema)
            arguments = cls._validate_arguments_object(
                name, arguments, tool_schemas,
                source=f"function {name!r} arguments")
            serialized = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":"),
                sort_keys=True)
            signature = (name, serialized)
            if signature in seen_calls:
                raise RunnerProtocolError("model emitted a duplicate tool call")
            seen_calls.add(signature)
            parsed.append((name, arguments))

        nonce = completion_id.removeprefix("chatcmpl-")[:24]
        return tuple(
            ToolCall(
                id=f"call_{nonce}_{index}", name=name,
                arguments=json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":"),
                    sort_keys=True))
            for index, (name, arguments) in enumerate(parsed)
        )

    @classmethod
    def _interpret_tool_output(
            cls, raw_text: str, runner_finish_reason: str,
            completion_id: str,
            tool_schemas: Mapping[str, Mapping[str, object]],
        tool_choice: ToolChoice,
    ) -> tuple[str | None, tuple[ToolCall, ...]]:
        content, fragments = cls._split_tool_output(raw_text)
        tool_calls = cls._parse_tool_fragments(
            fragments, tool_schemas, completion_id)
        if tool_calls and runner_finish_reason != "stop":
            raise RunnerProtocolError(
                "tool XML was emitted without a stop completion")
        if tool_choice.mode == "none" and tool_calls:
            raise RunnerProtocolError("model violated tool_choice='none'")
        if tool_choice.mode in {"required", "function"} and not tool_calls:
            raise RunnerProtocolError("model did not emit the required tool call")
        if (tool_choice.mode == "function" and
                any(call.name != tool_choice.name for call in tool_calls)):
            raise RunnerProtocolError("model violated the named tool_choice")
        if tool_calls:
            # This adapter exposes one tool call per active user turn.  Do not
            # persist model narration that surrounds a function fragment: on
            # MiniCPM5-1B OpenClaw feeds assistant.content back beside the tool
            # result, and the model can echo that pre-call narration instead
            # of producing the required final answer.  OpenAI permits null
            # assistant content for tool-call turns; the structured call is
            # the complete semantic result of this phase.
            return None, tool_calls
        return raw_text, ()

    def _validate_runner_response(
            self, response: Mapping[str, object], request_id: str,
            max_tokens: int) -> tuple[list[int], str]:
        if (response.get("protocol") != RUNNER_PROTOCOL or
                response.get("request_id") != request_id):
            raise RunnerProtocolError("native runner response identity mismatch")
        if response.get("ok") is not True:
            error = response.get("error")
            message = error.get("message") if isinstance(error, dict) else None
            raise RunnerProtocolError(
                str(message) if message else "native runner reported an error")
        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list) or len(output_ids) > max_tokens:
            raise RunnerProtocolError("native runner output_ids length is invalid")
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) or
               not 0 <= token_id < self.tokenizer.vocab_size
               for token_id in output_ids):
            raise RunnerProtocolError("native runner output_ids contain invalid IDs")
        finish_reason = response.get("finish_reason")
        if finish_reason not in ("stop", "length"):
            raise RunnerProtocolError("native runner finish_reason is invalid")
        return output_ids, str(finish_reason)

    def model_list(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [{
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": MODEL_OWNER,
            }],
        }

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "busy": self.busy,
            "supportsTools": self.enable_tools,
            "context_window": self.context_window,
        }


def _normal_completion(result: ChatResult) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant", "content": result.text}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in result.tool_calls
        ]
    return {
        "id": result.completion_id,
        "object": "chat.completion",
        "created": result.created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result.finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def _stream_completion(result: ChatResult) -> bytes:
    delta: dict[str, object] = {
        "role": "assistant", "content": result.text}
    if result.tool_calls:
        delta = {
            "role": "assistant",
            "content": result.text,
            "tool_calls": [
                {
                    "index": index,
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for index, call in enumerate(result.tool_calls)
            ],
        }
    chunk = {
        "id": result.completion_id,
        "object": "chat.completion.chunk",
        "created": result.created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": result.finish_reason,
        }],
    }
    event = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
    return f"data: {event}\n\ndata: [DONE]\n\n".encode("utf-8")


class MiniCPM5HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
            self, server_address: tuple[str, int], service: MiniCPM5Service) -> None:
        self.service = service
        super().__init__(server_address, MiniCPM5RequestHandler)


class MiniCPM5RequestHandler(BaseHTTPRequestHandler):
    server: MiniCPM5HTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        # Board deployments can wrap stderr logging explicitly.  The minimal
        # service does not write access logs by default.
        del format, args

    def _send_bytes(
            self, status: int, body: bytes, content_type: str,
            extra_headers: Mapping[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
            self, status: int, payload: object,
            extra_headers: Mapping[str, str] | None = None) -> None:
        body = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) +
                "\n").encode("utf-8")
        self._send_bytes(
            status, body, "application/json; charset=utf-8", extra_headers)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, self.server.service.health())
        elif path == "/v1/models":
            self._send_json(HTTPStatus.OK, self.server.service.model_list())
        else:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                APIError(
                    HTTPStatus.NOT_FOUND, "endpoint not found",
                    code="not_found").payload())

    def _read_json_body(self) -> object:
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            raise APIError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type must be application/json",
                code="unsupported_media_type")
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError as exc:
            raise APIError(
                HTTPStatus.BAD_REQUEST, "invalid Content-Length",
                code="invalid_content_length") from exc
        if length < 0:
            raise APIError(
                HTTPStatus.LENGTH_REQUIRED, "Content-Length is required",
                code="length_required")
        if length > MAX_HTTP_BODY_BYTES:
            raise APIError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large",
                code="request_too_large")
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise APIError(
                HTTPStatus.BAD_REQUEST, "request body is not valid JSON",
                code="invalid_json") from exc

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlsplit(self.path).path != "/v1/chat/completions":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                APIError(
                    HTTPStatus.NOT_FOUND, "endpoint not found",
                    code="not_found").payload())
            return
        try:
            payload = self._read_json_body()
            result, stream = self.server.service.complete(payload)
        except APIError as exc:
            self._send_json(exc.status, exc.payload())
            return
        except BusyError as exc:
            error = APIError(
                HTTPStatus.TOO_MANY_REQUESTS, str(exc),
                error_type="rate_limit_error", code="server_busy")
            self._send_json(error.status, error.payload(), {"Retry-After": "1"})
            return
        except RunnerProtocolError as exc:
            error = APIError(
                HTTPStatus.BAD_GATEWAY, str(exc),
                error_type="server_error", code="runner_protocol_error")
            self._send_json(error.status, error.payload())
            return

        if stream:
            body = _stream_completion(result)
            self._send_bytes(
                HTTPStatus.OK, body, "text/event-stream; charset=utf-8",
                {"Cache-Control": "no-cache"})
        else:
            self._send_json(HTTPStatus.OK, _normal_completion(result))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--context-window", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--enable-tools", action="store_true",
        help="enable strict MiniCPM5 XML-to-OpenAI structured tool calling")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--runner-socket", type=Path)
    transport.add_argument(
        "--runner-command", nargs=argparse.REMAINDER,
        help="runner executable and arguments; must be the final service option")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    tokenizer = MiniCPM5Tokenizer(args.tokenizer_json, args.chat_template)
    backend: RunnerBackend
    process_backend: SubprocessRunnerBackend | None = None
    if args.runner_socket is not None:
        backend = UnixSocketRunnerBackend(args.runner_socket)
    else:
        if not args.runner_command:
            raise SystemExit("--runner-command requires an executable")
        process_backend = SubprocessRunnerBackend(args.runner_command)
        backend = process_backend
    service = MiniCPM5Service(
        tokenizer, backend, args.context_window, args.max_tokens,
        enable_tools=args.enable_tools)
    server = MiniCPM5HTTPServer((args.host, args.port), service)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual board lifecycle
        pass
    finally:
        server.server_close()
        if process_backend is not None:
            process_backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
