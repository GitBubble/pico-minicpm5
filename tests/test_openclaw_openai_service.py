# SPDX-License-Identifier: Apache-2.0
"""Service-level suite for the OpenClaw OpenAI-compatible HTTP facade.

The official MiniCPM5 tokenizer assets are release-external (see
MODEL_PROVENANCE.md).  Tokenizer-dependent tests locate them through the
deployment convention ``$PICO_HOME/assets`` and skip with a precise reason
when the assets or the optional ``tokenizers``/``jinja2`` dependencies are
absent; protocol, transport and tool-XML tests always run.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
from typing import Iterator, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


PROJECT = Path(__file__).resolve().parents[1]
OPENCLAW_SRC = PROJECT / "app" / "openclaw" / "src"


def _service_module():
    sys.path.insert(0, str(OPENCLAW_SRC))
    spec = importlib.util.spec_from_file_location(
        "openclaw_openai_service_test", OPENCLAW_SRC / "openai_service.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service_mod = _service_module()

ASSETS = Path(os.environ["PICO_HOME"]) / "assets" \
    if "PICO_HOME" in os.environ else None
TOKENIZER_JSON = ASSETS / "tokenizer.json" if ASSETS else None
CHAT_TEMPLATE = ASSETS / "chat_template.jinja" if ASSETS else None
WEATHER_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
# Greedy output from the pinned official BF16 checkpoint for a Shenzhen
# weather request with WEATHER_TOOL.  The four XML structural tokens are
# tokenizer special IDs 18..21 and must survive structured decoding.
WEATHER_XML_IDS = [
    18, 2546, 943, 1135, 84, 29996, 1822, 20, 2546, 943, 35605,
    1822, 110100, 74359, 21, 19, 130073,
]


def _tokenizer_skip_reason() -> str | None:
    if TOKENIZER_JSON is None or CHAT_TEMPLATE is None:
        return "PICO_HOME with assets/tokenizer.json is not configured"
    if not TOKENIZER_JSON.is_file() or not CHAT_TEMPLATE.is_file():
        return "official tokenizer assets are missing from PICO_HOME/assets"
    for dependency in ("tokenizers", "jinja2"):
        if importlib.util.find_spec(dependency) is None:
            return f"optional service dependency {dependency} is not installed"
    return None


class FakeBackend:
    def __init__(
            self, output_ids: list[int] | None = None, *,
            finish_reason: str = "stop") -> None:
        self.output_ids = output_ids if output_ids is not None else [101, 632]
        self.finish_reason = finish_reason
        self.requests: list[dict[str, object]] = []

    def generate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        copied = json.loads(json.dumps(request))
        self.requests.append(copied)
        return {
            "protocol": service_mod.RUNNER_PROTOCOL,
            "request_id": request["request_id"],
            "ok": True,
            "output_ids": list(self.output_ids),
            "finish_reason": self.finish_reason,
        }


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test failed to release blocking backend")
        return super().generate(request)


@pytest.fixture(scope="module")
def tokenizer() -> service_mod.MiniCPM5Tokenizer:
    reason = _tokenizer_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    return service_mod.MiniCPM5Tokenizer(TOKENIZER_JSON, CHAT_TEMPLATE)


def make_service(
        tokenizer: service_mod.MiniCPM5Tokenizer,
        backend: service_mod.RunnerBackend | None = None,
        *, context_window: int = 256,
        max_tokens: int = 32,
        enable_tools: bool = False) -> service_mod.MiniCPM5Service:
    return service_mod.MiniCPM5Service(
        tokenizer, backend or FakeBackend(), context_window, max_tokens,
        enable_tools=enable_tools)


@contextmanager
def running_server(
        service: service_mod.MiniCPM5Service) -> Iterator[str]:
    server = service_mod.MiniCPM5HTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def http_get(url: str) -> tuple[int, dict[str, object]]:
    with urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read())


def http_post(
        url: str, payload: object) -> tuple[int, bytes, Mapping[str, str]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, response.read(), response.headers
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers


def chat_request(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": service_mod.MODEL_ID,
        "messages": [{"role": "user", "content": "Reply exactly: pong"}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
    }
    payload.update(overrides)
    return payload


def test_tokenizer_is_pinned_to_official_minicpm5_assets_and_template(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    assert tokenizer.tokenizer_json == TOKENIZER_JSON.resolve()
    assert tokenizer.chat_template == CHAT_TEMPLATE.resolve()
    assert tokenizer.vocab_size == 130560
    prompt, token_ids = tokenizer.encode_messages([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hello"},
    ])
    assert prompt.startswith("<s><|im_start|>system\nBe terse.<|im_end|>\n")
    assert prompt.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert token_ids[0] == 0
    # The service must not add a second BOS through the tokenizer post-processor.
    assert token_ids[:2] != [0, 0]


def test_official_template_accepts_tools_and_structural_tokens_survive_decode(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    prompt, token_ids = tokenizer.encode_messages(
        [{"role": "user", "content": "深圳天气如何？"}], [WEATHER_TOOL])
    assert "# Tools" in prompt
    assert '"name": "get_weather"' in prompt
    assert "查询城市天气" in prompt
    assert "\\u67e5" not in prompt
    assert token_ids[0] == 0
    assert tokenizer.decode_structured(WEATHER_XML_IDS) == (
        '<function name="get_weather"><param name="city">Shenzhen</param>'
        '</function>')
    # Text-only decode intentionally strips IDs 18..21 and therefore must not
    # be used by the structured parser.
    assert tokenizer.decode(WEATHER_XML_IDS) != \
        tokenizer.decode_structured(WEATHER_XML_IDS)


@pytest.mark.parametrize(
    ("asset_name", "error"),
    [
        ("tokenizer.json", "tokenizer.json hash drift"),
        ("chat_template.jinja", "chat_template.jinja hash drift"),
    ],
)
def test_tokenizer_rejects_unofficial_or_legacy_asset_drift(
        tmp_path: Path, asset_name: str, error: str) -> None:
    reason = _tokenizer_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    tokenizer_path = tmp_path / "tokenizer.json"
    template_path = tmp_path / "chat_template.jinja"
    tokenizer_path.write_bytes(TOKENIZER_JSON.read_bytes())
    template_path.write_bytes(CHAT_TEMPLATE.read_bytes())
    path = tokenizer_path if asset_name == "tokenizer.json" else template_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=error):
        service_mod.MiniCPM5Tokenizer(tokenizer_path, template_path)


def test_health_models_and_nonstream_completion_are_openai_compatible(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend()
    service = make_service(tokenizer, backend)
    with running_server(service) as base_url:
        status, health = http_get(f"{base_url}/healthz")
        assert status == 200
        assert health == {
            "status": "ok",
            "model": service_mod.MODEL_ID,
            "busy": False,
            "supportsTools": False,
            "context_window": 256,
        }
        status, models = http_get(f"{base_url}/v1/models")
        assert status == 200
        assert models["object"] == "list"
        assert [entry["id"] for entry in models["data"]] == [
            service_mod.MODEL_ID]

        status, body, headers = http_post(
            f"{base_url}/v1/chat/completions", chat_request())
        assert status == 200
        assert headers.get_content_type() == "application/json"
        completion = json.loads(body)
        assert completion["object"] == "chat.completion"
        assert completion["model"] == service_mod.MODEL_ID
        assert completion["choices"] == [{
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop",
        }]
        assert completion["usage"]["completion_tokens"] == 2
        assert completion["usage"]["total_tokens"] == (
            completion["usage"]["prompt_tokens"] + 2)

    assert len(backend.requests) == 1
    runner_request = backend.requests[0]
    assert set(runner_request) == {
        "protocol", "request_id", "op", "model", "input_ids",
        "max_new_tokens", "temperature", "eos_token_ids", "reset_kv",
    }
    assert runner_request["protocol"] == "pico.minicpm5.runner.v1"
    assert runner_request["op"] == "generate"
    assert runner_request["model"] == service_mod.MODEL_ID
    assert runner_request["max_new_tokens"] == 8
    assert runner_request["temperature"] == 0.0
    assert runner_request["eos_token_ids"] == [1, 130073]
    assert runner_request["reset_kv"] is True
    assert isinstance(runner_request["input_ids"], list)
    assert runner_request["input_ids"][0] == 0
    # Packed KV is private to the runner; no per-layer KV arrays cross JSON.
    assert all("past" not in key and "kv_" not in key for key in runner_request)


def test_stream_true_emits_one_data_chunk_then_done(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    service = make_service(tokenizer)
    with running_server(service) as base_url:
        status, body, headers = http_post(
            f"{base_url}/v1/chat/completions",
            chat_request(stream=True))
    assert status == 200
    assert headers.get_content_type() == "text/event-stream"
    events = [item for item in body.decode("utf-8").split("\n\n") if item]
    assert len(events) == 2
    assert events[1] == "data: [DONE]"
    chunk = json.loads(events[0].removeprefix("data: "))
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"] == [{
        "index": 0,
        "delta": {"role": "assistant", "content": "pong"},
        "finish_reason": "stop",
    }]


def tool_chat_request(**overrides: object) -> dict[str, object]:
    payload = chat_request(
        messages=[{"role": "user", "content": "深圳天气如何？"}],
        tools=[WEATHER_TOOL], max_tokens=32)
    payload.update(overrides)
    return payload


def completed_weather_tool_messages() -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "深圳天气如何？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_weather_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Shenzhen"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_1",
            "name": "get_weather",
            "content": '{"condition":"sunny","temperature_c":30}',
        },
    ]


def test_tool_completion_maps_official_bf16_xml_to_openai_normal_response(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend(WEATHER_XML_IDS)
    service = make_service(
        tokenizer, backend, context_window=512, max_tokens=32,
        enable_tools=True)
    with running_server(service) as base_url:
        status, health = http_get(f"{base_url}/healthz")
        assert status == 200
        assert health["supportsTools"] is True
        status, body, _ = http_post(
            f"{base_url}/v1/chat/completions", tool_chat_request())
    assert status == 200
    completion = json.loads(body)
    assert set(completion) == {
        "id", "object", "created", "model", "choices", "usage"}
    assert completion["choices"] == [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": completion["choices"][0]["message"]["tool_calls"][0]["id"],
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Shenzhen"}',
                },
            }],
        },
        "finish_reason": "tool_calls",
    }]
    call_id = completion["choices"][0]["message"]["tool_calls"][0]["id"]
    assert call_id.startswith("call_")
    assert service_mod.TOOL_CALL_ID_RE.fullmatch(call_id)
    assert completion["usage"]["completion_tokens"] == len(WEATHER_XML_IDS)
    assert backend.requests[0]["input_ids"][0] == 0


def test_tool_completion_stream_uses_delta_tool_calls_then_done(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    service = make_service(
        tokenizer, FakeBackend(WEATHER_XML_IDS), context_window=512,
        max_tokens=32, enable_tools=True)
    with running_server(service) as base_url:
        status, body, headers = http_post(
            f"{base_url}/v1/chat/completions",
            tool_chat_request(stream=True))
    assert status == 200
    assert headers.get_content_type() == "text/event-stream"
    events = [item for item in body.decode().split("\n\n") if item]
    assert events[-1] == "data: [DONE]"
    chunk = json.loads(events[0].removeprefix("data: "))
    assert set(chunk) == {"id", "object", "created", "model", "choices"}
    choice = chunk["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert set(choice["delta"]) == {"role", "content", "tool_calls"}
    assert choice["delta"]["content"] is None
    assert choice["delta"]["tool_calls"][0]["index"] == 0
    assert choice["delta"]["tool_calls"][0]["function"] == {
        "name": "get_weather", "arguments": '{"city":"Shenzhen"}'}


@pytest.mark.parametrize("stream", [False, True])
def test_narration_around_complete_xml_is_suppressed_for_tool_history(
        tokenizer: service_mod.MiniCPM5Tokenizer, stream: bool) -> None:
    raw = (
        "I will execute the weather lookup now.\n"
        '<function name="get_weather"><param name="city">Shenzhen</param>'
        "</function>\nThe result will follow.")
    output_ids = tokenizer._tokenizer.encode(
        raw, add_special_tokens=False).ids + [130073]
    service = make_service(
        tokenizer, FakeBackend(output_ids), context_window=512,
        max_tokens=64, enable_tools=True)
    with running_server(service) as base_url:
        status, body, _ = http_post(
            f"{base_url}/v1/chat/completions",
            tool_chat_request(stream=stream, max_tokens=64))
    assert status == 200
    if stream:
        first_event = body.decode().split("\n\n", 1)[0]
        choice = json.loads(first_event.removeprefix("data: "))["choices"][0]
        message = choice["delta"]
    else:
        choice = json.loads(body)["choices"][0]
        message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"] == {
        "name": "get_weather", "arguments": '{"city":"Shenzhen"}'}


def test_openai_tool_history_is_normalized_for_official_template_and_feedback(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    service = make_service(
        tokenizer, context_window=1024, enable_tools=True)
    payload = tool_chat_request(messages=[
        {"role": "user", "content": "深圳天气如何？"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_weather_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Shenzhen"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_1",
            "name": "get_weather",
            "content": '{"condition":"sunny","temperature_c":30}',
        },
        {"role": "assistant", "content": "深圳天气晴朗，气温 30°C。"},
        {"role": "user", "content": "用一句话总结。"},
    ])
    request = service._validate_request(payload)
    arguments = request.messages[1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"city": "Shenzhen"}
    assert request.final_only is False
    prompt = tokenizer.render(request.messages, request.tools)
    assert prompt.count(
        '<function name="get_weather"><param name="city">Shenzhen</param>'
        '</function>') == 1
    assert (
        '<tool_response>\n{"condition":"sunny","temperature_c":30}\n'
        '</tool_response>') in prompt


def test_tool_call_block_is_limited_to_one_call_before_runner(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend()
    service = make_service(tokenizer, backend, enable_tools=True)
    messages = completed_weather_tool_messages()
    first_call = messages[1]["tool_calls"][0]
    assert isinstance(first_call, dict)
    second_call = json.loads(json.dumps(first_call))
    second_call["id"] = "call_weather_2"
    messages[1]["tool_calls"].append(second_call)

    with pytest.raises(service_mod.APIError) as caught:
        service.complete(tool_chat_request(messages=messages))

    assert caught.value.code == "invalid_tool_history"
    assert backend.requests == []


def test_compacted_suatatu_history_cannot_bypass_one_block_budget(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend()
    service = make_service(tokenizer, backend, enable_tools=True)
    messages = completed_weather_tool_messages()
    messages.insert(0, {"role": "system", "content": "Be concise."})
    messages.extend([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_weather_2",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Shenzhen"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_2",
            "name": "get_weather",
            "content": '{"condition":"sunny"}',
        },
        {"role": "user", "content": "Continue after compaction."},
    ])

    with pytest.raises(service_mod.APIError) as caught:
        service.complete(tool_chat_request(messages=messages))

    assert caught.value.code == "tool_call_budget_exceeded"
    assert backend.requests == []


def test_new_user_message_opens_a_fresh_one_call_budget(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend(WEATHER_XML_IDS)
    service = make_service(
        tokenizer, backend, context_window=1024, max_tokens=32,
        enable_tools=True)
    messages = completed_weather_tool_messages()
    messages.extend([
        {"role": "assistant", "content": "深圳天气晴朗。"},
        {"role": "user", "content": "再查一次深圳天气。"},
    ])

    result, _ = service.complete(tool_chat_request(messages=messages))

    assert result.finish_reason == "tool_calls"
    assert len(result.tool_calls) == 1
    prompt = tokenizer._tokenizer.decode(
        backend.requests[0]["input_ids"], skip_special_tokens=False)
    assert "# Tools" in prompt


def test_user_inserted_after_tool_result_does_not_reset_active_turn_budget(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend(WEATHER_XML_IDS)
    service = make_service(
        tokenizer, backend, context_window=1024, max_tokens=32,
        enable_tools=True)
    messages = completed_weather_tool_messages()
    # Models/runtimes may inject a synthetic user message while compacting.
    # Without an intervening plain assistant final it is still the same
    # active tool turn and must not receive another call budget.
    messages.append({
        "role": "user", "content": "Continue after compaction."})
    payload = tool_chat_request(messages=messages)

    request = service._validate_request(payload)
    result, _ = service.complete(payload)

    assert request.final_only is True
    assert result.tool_calls == ()
    assert result.text == ""
    assert result.finish_reason == "stop"
    prompt = tokenizer._tokenizer.decode(
        backend.requests[0]["input_ids"], skip_special_tokens=False)
    assert "# Tools" not in prompt
    assert service_mod.FINAL_ONLY_SYSTEM_DIRECTIVE in prompt


def test_completed_tool_round_hides_catalog_and_requires_a_final_response(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend()
    service = make_service(
        tokenizer, backend, context_window=1024, enable_tools=True)
    payload = tool_chat_request(messages=completed_weather_tool_messages())

    request = service._validate_request(payload)
    result, _ = service.complete(payload)

    assert request.final_only is True
    assert request.tool_choice == service_mod.ToolChoice("none")
    assert result.text == "pong"
    assert result.tool_calls == ()
    assert result.finish_reason == "stop"
    prompt = tokenizer._tokenizer.decode(
        backend.requests[0]["input_ids"], skip_special_tokens=False)
    assert "# Tools" not in prompt
    assert "<tool_response>" in prompt
    assert service_mod.FINAL_ONLY_SYSTEM_DIRECTIVE in prompt


@pytest.mark.parametrize("mode", ["required", "function"])
def test_required_or_named_choice_after_tool_budget_is_a_pre_runner_400(
        tokenizer: service_mod.MiniCPM5Tokenizer, mode: str) -> None:
    backend = FakeBackend()
    service = make_service(tokenizer, backend, enable_tools=True)
    tool_choice: object = "required"
    if mode == "function":
        tool_choice = {
            "type": "function", "function": {"name": "get_weather"}}

    with pytest.raises(service_mod.APIError) as caught:
        service.complete(tool_chat_request(
            messages=completed_weather_tool_messages(),
            tool_choice=tool_choice))

    assert caught.value.status == 400
    assert caught.value.param == "tool_choice"
    assert caught.value.code == "tool_call_budget_exceeded"
    assert backend.requests == []


@pytest.mark.parametrize("case", ["repeat_xml", "empty", "length", "bad_xml"])
def test_final_only_round_safely_stops_without_another_tool_call(
        tokenizer: service_mod.MiniCPM5Tokenizer, case: str) -> None:
    finish_reason = "stop"
    if case == "repeat_xml":
        output_ids = WEATHER_XML_IDS
    elif case == "empty":
        output_ids = []
    elif case == "length":
        output_ids = tokenizer._tokenizer.encode(
            "partial final", add_special_tokens=False).ids
        finish_reason = "length"
    else:
        output_ids = tokenizer._tokenizer.encode(
            '<function name="unknown">', add_special_tokens=False).ids
    backend = FakeBackend(output_ids, finish_reason=finish_reason)
    service = make_service(
        tokenizer, backend, context_window=1024, max_tokens=32,
        enable_tools=True)

    result, _ = service.complete(tool_chat_request(
        messages=completed_weather_tool_messages()))

    assert result.tool_calls == ()
    assert result.finish_reason == "stop"
    if case in {"repeat_xml", "empty", "bad_xml"}:
        assert result.text == ""
    else:
        assert result.text == "partial final"


@pytest.mark.parametrize(
    ("tool_choice", "output_ids", "status"),
    [
        ("none", WEATHER_XML_IDS, 502),
        ("required", [101, 632], 502),
        ("required", WEATHER_XML_IDS, 200),
        ("auto", [101, 632], 200),
    ],
)
def test_tool_choice_modes_fail_closed_at_model_output_boundary(
        tokenizer: service_mod.MiniCPM5Tokenizer, tool_choice: str,
        output_ids: list[int], status: int) -> None:
    service = make_service(
        tokenizer, FakeBackend(output_ids), context_window=512,
        max_tokens=32, enable_tools=True)
    with running_server(service) as base_url:
        actual, body, _ = http_post(
            f"{base_url}/v1/chat/completions",
            tool_chat_request(tool_choice=tool_choice))
    assert actual == status
    if status == 502:
        assert json.loads(body)["error"]["code"] == "runner_protocol_error"


def test_named_tool_choice_restricts_rendering_and_rejects_other_calls(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    calculator = {
        "type": "function",
        "function": {
            "name": "calculate",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
    backend = FakeBackend(WEATHER_XML_IDS)
    service = make_service(
        tokenizer, backend, context_window=512, max_tokens=32,
        enable_tools=True)
    payload = tool_chat_request(
        tools=[WEATHER_TOOL, calculator],
        tool_choice={"type": "function", "function": {"name": "calculate"}})
    with running_server(service) as base_url:
        status, body, _ = http_post(
            f"{base_url}/v1/chat/completions", payload)
    assert status == 502
    assert "named tool_choice" in json.loads(body)["error"]["message"]
    prompt = tokenizer._tokenizer.decode(
        backend.requests[0]["input_ids"], skip_special_tokens=False)
    assert '"name": "calculate"' in prompt
    assert '"name": "get_weather"' not in prompt


@pytest.mark.parametrize(
    "tool_choice",
    [
        "sometimes",
        "required",
        {"type": "function", "function": {"name": "unknown"}},
        {"type": "function", "function": {"name": "get_weather", "x": 1}},
    ],
)
def test_invalid_or_unfulfillable_tool_choice_is_a_400(
        tokenizer: service_mod.MiniCPM5Tokenizer, tool_choice: object) -> None:
    service = make_service(tokenizer, enable_tools=True)
    payload = tool_chat_request(tool_choice=tool_choice)
    if tool_choice == "required":
        payload["tools"] = []
    with pytest.raises(service_mod.APIError) as caught:
        service._validate_request(payload)
    assert caught.value.param == "tool_choice"
    assert caught.value.code == "invalid_tool_choice"


@pytest.mark.parametrize(
    ("override", "status", "param", "code"),
    [
        ({"model": "wrong"}, 404, "model", "model_not_found"),
        ({"temperature": 0.1}, 400, "temperature", "unsupported_temperature"),
        ({"temperature": True}, 400, "temperature", "unsupported_temperature"),
        ({"stream": "false"}, 400, "stream", "invalid_stream"),
        ({"max_tokens": 0}, 400, "max_tokens", "invalid_max_tokens"),
        ({"max_tokens": 33}, 400, "max_tokens", "invalid_max_tokens"),
        ({"tools": [{"type": "function"}]}, 400, "tools",
         "tools_not_supported"),
        ({"max_completion_tokens": 8}, 400, "max_completion_tokens",
         "conflicting_parameters"),
        ({"messages": [{"role": "tool", "content": "x"}]}, 400,
         "messages", "invalid_messages"),
    ],
)
def test_strict_request_validation(
        tokenizer: service_mod.MiniCPM5Tokenizer,
        override: dict[str, object], status: int, param: str, code: str) -> None:
    service = make_service(tokenizer)
    with running_server(service) as base_url:
        actual_status, body, _ = http_post(
            f"{base_url}/v1/chat/completions", chat_request(**override))
    assert actual_status == status
    error = json.loads(body)["error"]
    assert error["param"] == param
    assert error["code"] == code


def test_openclaw_max_completion_tokens_alias_is_supported(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = FakeBackend()
    service = make_service(tokenizer, backend)
    payload = chat_request()
    del payload["max_tokens"]
    payload["max_completion_tokens"] = 8

    result, stream = service.complete(payload)

    assert stream is False
    assert result.text == "pong"
    assert backend.requests[-1]["max_new_tokens"] == 8


def test_context_validation_counts_official_template_tokens_plus_generation(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    messages = [{"role": "user", "content": "hello"}]
    _, input_ids = tokenizer.encode_messages(messages)
    service = make_service(
        tokenizer, context_window=len(input_ids), max_tokens=1)
    with pytest.raises(service_mod.APIError) as caught:
        service.complete(chat_request(messages=messages, max_tokens=1))
    assert caught.value.status == 400
    assert caught.value.code == "context_length_exceeded"


def test_busy_generation_returns_429_without_queueing(
        tokenizer: service_mod.MiniCPM5Tokenizer) -> None:
    backend = BlockingBackend()
    service = make_service(tokenizer, backend)
    first: list[tuple[int, bytes, Mapping[str, str]]] = []
    with running_server(service) as base_url:
        endpoint = f"{base_url}/v1/chat/completions"
        thread = threading.Thread(
            target=lambda: first.append(http_post(endpoint, chat_request())))
        thread.start()
        assert backend.entered.wait(timeout=5)
        status, body, headers = http_post(endpoint, chat_request())
        assert status == 429
        assert headers["Retry-After"] == "1"
        error = json.loads(body)["error"]
        assert error["type"] == "rate_limit_error"
        assert error["code"] == "server_busy"
        backend.release.set()
        thread.join(timeout=5)
    assert first and first[0][0] == 200
    assert len(backend.requests) == 1


def test_stdio_json_line_transport_is_exact() -> None:
    response = {
        "protocol": service_mod.RUNNER_PROTOCOL,
        "request_id": "req-1",
        "ok": True,
        "output_ids": [101, 632],
        "finish_reason": "stop",
    }
    reader = io.StringIO(json.dumps(response) + "\n")
    writer = io.StringIO()
    backend = service_mod.JsonLineStreamBackend(reader, writer)
    request = {
        "protocol": service_mod.RUNNER_PROTOCOL,
        "request_id": "req-1",
        "op": "generate",
    }
    assert backend.generate(request) == response
    assert writer.getvalue() == (
        '{"protocol":"pico.minicpm5.runner.v1","request_id":"req-1",'
        '"op":"generate"}\n')


def test_subprocess_backend_closes_stdin_for_graceful_runner_shutdown() -> None:
    backend = service_mod.SubprocessRunnerBackend([
        sys.executable, "-u", "-c",
        "import sys; sys.stdin.read(); raise SystemExit(0)",
    ])
    backend.close()
    assert backend._process.returncode == 0


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="AF_UNIX unavailable")
def test_unix_socket_transport_uses_the_same_one_line_protocol() -> None:
    # Darwin caps sockaddr_un paths at 104 bytes; pytest's standard temporary
    # directory can exceed that before the socket basename is appended.
    with tempfile.TemporaryDirectory(prefix="pico-runner-", dir="/tmp") as root:
        socket_path = Path(root) / "r.sock"
        _exercise_unix_socket_transport(socket_path)


def _exercise_unix_socket_transport(socket_path: Path) -> None:
    received: list[dict[str, object]] = []
    ready = threading.Event()

    def fake_c_runner() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            listener.listen(1)
            ready.set()
            connection, _ = listener.accept()
            with connection:
                with connection.makefile("rwb") as stream:
                    request = json.loads(stream.readline())
                    received.append(request)
                    response = {
                        "protocol": service_mod.RUNNER_PROTOCOL,
                        "request_id": request["request_id"],
                        "ok": True,
                        "output_ids": [59800],
                        "finish_reason": "stop",
                    }
                    stream.write((json.dumps(response) + "\n").encode("utf-8"))
                    stream.flush()

    thread = threading.Thread(target=fake_c_runner, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    backend = service_mod.UnixSocketRunnerBackend(socket_path, timeout_seconds=5)
    request = {
        "protocol": service_mod.RUNNER_PROTOCOL,
        "request_id": "unix-1",
        "op": "generate",
    }
    response = backend.generate(request)
    thread.join(timeout=5)
    assert received == [request]
    assert response["output_ids"] == [59800]


@pytest.mark.parametrize(
    "response",
    [
        {"protocol": "wrong", "request_id": "unused", "ok": True,
         "output_ids": [], "finish_reason": "stop"},
        {"protocol": service_mod.RUNNER_PROTOCOL, "request_id": "unused",
         "ok": True, "output_ids": [-1], "finish_reason": "stop"},
        {"protocol": service_mod.RUNNER_PROTOCOL, "request_id": "unused",
         "ok": True, "output_ids": [], "finish_reason": "unknown"},
    ],
)
def test_runner_protocol_validation_fails_closed(
        tokenizer: service_mod.MiniCPM5Tokenizer,
        response: dict[str, object]) -> None:
    class BadBackend:
        def generate(
                self, request: Mapping[str, object]) -> Mapping[str, object]:
            result = dict(response)
            if result["protocol"] == service_mod.RUNNER_PROTOCOL:
                result["request_id"] = request["request_id"]
            return result

    service = make_service(tokenizer, BadBackend())
    with running_server(service) as base_url:
        status, body, _ = http_post(
            f"{base_url}/v1/chat/completions", chat_request())
    assert status == 502
    assert json.loads(body)["error"]["code"] == "runner_protocol_error"


@pytest.mark.parametrize(
    "xml",
    [
        '<function name="get_weather"><param name="city">Shenzhen</param>',
        '<function name="unknown"><param name="city">Shenzhen</param>'
        '</function>',
        '<function name="get_weather"><param name="city">A</param>'
        '<param name="city">B</param></function>',
        '<function name="get_weather"><param name="unknown">A</param>'
        '</function>',
        '<function name="get_weather"><param name="city"><b>A</b></param>'
        '</function>',
        '<function name="get_weather"><function name="get_weather" />'
        '</function>',
        '<function name="get_weather"><param name="city">A</param>'
        '</function><function name="get_weather"><param name="city">A</param>'
        '</function>',
        '<!DOCTYPE x><function name="get_weather"><param name="city">A</param>'
        '</function>',
        '<!--x--><function name="get_weather"><param name="city">A</param>'
        '</function>',
    ],
)
def test_illegal_unknown_duplicate_or_nested_tool_xml_is_rejected(
        xml: str) -> None:
    schemas = {
        "get_weather": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    with pytest.raises(service_mod.RunnerProtocolError):
        service_mod.MiniCPM5Service._parse_tool_xml(
            xml, schemas, "chatcmpl-deadbeef")


@pytest.mark.parametrize(
    "raw",
    [
        'prefix </function><function name="get_weather">'
        '<param name="city">A</param></function>',
        '<function name="get_weather"><param name="city">A</param>'
        '</function> suffix <param name="x">',
        '<function name="get_weather"><param name="city">A</param>'
        '</function><![CDATA[stray]]>',
        '<function name="get_weather"><param name="city">A</param>'
        '</function><function name="get_weather">',
    ],
)
def test_tool_xml_markers_outside_complete_functions_are_rejected(
        raw: str) -> None:
    schemas = {
        "get_weather": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    with pytest.raises(service_mod.RunnerProtocolError):
        service_mod.MiniCPM5Service._parse_tool_xml(
            raw, schemas, "chatcmpl-deadbeef")


def test_tool_xml_parser_uses_schema_types_and_preserves_cdata() -> None:
    schemas = {
        "typed": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
                "tags": {"type": "array"},
                "config": {"type": "object"},
            },
            "required": ["text", "count", "enabled", "tags", "config"],
        },
    }
    calls = service_mod.MiniCPM5Service._parse_tool_xml(
        '<function name="typed">'
        '<param name="text"><![CDATA[a < b & c\nnext]]></param>'
        '<param name="count">3</param>'
        '<param name="enabled">true</param>'
        '<param name="tags">["a","b"]</param>'
        '<param name="config">{"x":1}</param>'
        '</function>',
        schemas, "chatcmpl-deadbeef")
    assert len(calls) == 1
    assert json.loads(calls[0].arguments) == {
        "text": "a < b & c\nnext",
        "count": 3,
        "enabled": True,
        "tags": ["a", "b"],
        "config": {"x": 1},
    }


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "content": "sunny", "tool_call_id": "call_1"}],
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Shenzhen"},
                    },
                }],
            },
        ],
        [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Shenzhen"}',
                    },
                }],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "wrong"},
        ],
    ],
)
def test_invalid_tool_result_history_is_rejected_before_runner(
        tokenizer: service_mod.MiniCPM5Tokenizer,
        messages: list[dict[str, object]]) -> None:
    service = make_service(tokenizer, enable_tools=True)
    with pytest.raises(service_mod.APIError) as caught:
        service._validate_request(tool_chat_request(messages=messages))
    assert caught.value.code == "invalid_tool_history"
