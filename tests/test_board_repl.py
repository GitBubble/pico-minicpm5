from __future__ import annotations

import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
APP_SRC = PROJECT / "app" / "src"


def _server_module():
    sys.path.insert(0, str(APP_SRC))
    spec = importlib.util.spec_from_file_location(
        "pico_minicpm5_board_repl_test", APP_SRC / "merged_board_server.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repl_reuses_one_session_and_handles_commands(monkeypatch, capsys) -> None:
    server = _server_module()
    calls: list[list[int]] = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return SimpleNamespace(ids=[len(text)])

        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens
            return f"reply-{ids[0]}"

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = Tokenizer()
        last_phase_steps = []
        closed = False

        def __init__(self, **_kwargs):
            pass

        def generate(self, ids, *_args, **_kwargs):
            calls.append(ids)
            self.last_phase_steps = [{"position": 0}]
            if _kwargs.get("on_token") is not None:
                _kwargs["on_token"]((100 + len(calls),))
            return "max", [100 + len(calls)], [1.0, 2.0]

        def close(self):
            self.closed = True

    prompts = iter(["hello", "/reset", "/max 64", "world", "/help", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--interactive",
    ])

    assert server.main() == 0

    assert calls == [[0, 5], [0, 5]]
    output = capsys.readouterr().out
    assert output.count("loaded 3 handles") == 1
    assert "MiniCPM 5" in output
    assert "ctx1024 · resident KV · streaming" in output
    assert "MiniCPM ✦ reply-101" in output
    assert "Context reset." in output
    assert "Commands: /help · /max N · /reset · /quit" in output
    assert "max-new=64" in output
    assert "allowed range is 1..1023" in output


def test_agent_repl_executes_native_tool_call(monkeypatch, capsys, tmp_path) -> None:
    server = _server_module()
    (tmp_path / "note.txt").write_text("hello from tool\n", encoding="utf-8")
    rendered_prompts = []
    generated = [
        '<function name="read_file"><param name="path">note.txt</param>'
        '</function><|im_end|>',
        "The note says hello from tool.<|im_end|>",
    ]

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            rendered_prompts.append(text)
            return SimpleNamespace(ids=[0, 130072, 42, 130073])

        def decode(self, ids, skip_special_tokens=True):
            assert not skip_special_tokens
            return generated[int(ids[0]) - 100]

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = Tokenizer()
        last_phase_steps = []

        def __init__(self, **_kwargs):
            pass

        def generate(self, ids, *_args, **kwargs):
            index = 100 + (len(rendered_prompts) - 1)
            self.last_phase_steps = [{"position": 0}]
            if kwargs.get("on_token") is not None:
                kwargs["on_token"]((index,))
            return "eos", [index], [1.0] * (len(ids) + 1)

        def close(self):
            pass

    prompts = iter([
        "/help", "/help max", "/help nonsense", "/think", "/think on",
        "Please read note.txt", "/think off", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--agent",
        "--workspace", str(tmp_path),
    ])

    assert server.main() == 0

    output = capsys.readouterr().out
    assert "⚙ read_file(path='note.txt')" in output
    assert "✓ read_file: 1: hello from tool" in output
    assert "MiniCPM ✦ The note says hello from tool." in output
    assert "thinking=off" in output
    assert "thinking=on" in output
    assert "MiniCPM Agent 内置命令" in output
    assert "/help [COMMAND]" in output
    assert "配置范围 1..1023，硬件范围 1..1023；当前为 128" in output
    assert "未知帮助主题: nonsense" in output
    assert "<tool_response>" in rendered_prompts[1]
    assert "hello from tool" in rendered_prompts[1]
    assert "Use path='.' for that root" in rendered_prompts[0]
    assert str(tmp_path) in rendered_prompts[0]
    assert rendered_prompts[0].endswith(
        "<|im_start|>assistant\n<think>\n")


def test_agent_help_describes_command_scope_and_ranges(tmp_path) -> None:
    server = _server_module()

    overview = server.agent_command_help(
        "", context=1024, max_new=128, thinking=False, max_tool_steps=4,
        workspace=tmp_path)
    permissions = server.agent_command_help(
        "/permissions", context=1024, max_new=128, thinking=False,
        max_tool_steps=4, workspace=tmp_path)
    clear = server.agent_command_help(
        "reset", context=1024, max_new=128, thinking=False,
        max_tool_steps=4, workspace=tmp_path)

    for command in (
            "/help", "/profile", "/tools", "/permissions", "/think", "/context",
            "/clear", "/max", "/quit"):
        assert command in overview
    assert "N=1..1023" in overview
    assert "每个用户请求最多 4 轮工具调用" in overview
    assert str(tmp_path) in overview
    assert "write_file/run_shell 每次询问" in permissions
    assert "不会退出进程或重新加载三个模型句柄" in clear


def test_agent_direct_directory_route_skips_model(
        monkeypatch, capsys, tmp_path) -> None:
    server = _server_module()
    (tmp_path / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    report = tmp_path / "report.json"

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = object()

        def __init__(self, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            raise AssertionError("direct tool route must not invoke MiniCPM5")

        def close(self):
            pass

    prompts = iter(["列出目录文件", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--agent",
        "--workspace", str(tmp_path), "--report", str(report),
    ])

    assert server.main() == 0

    output = capsys.readouterr().out
    assert "model skipped" in output
    assert "目录内容（.）：" in output
    assert "alpha.txt" in output and "folder/" in output
    evidence = json.loads(report.read_text())["runs"][0]
    assert evidence["reason"] == "tool_direct"
    assert evidence["route_mode"] == "DIRECT_TOOL"
    assert evidence["model_called"] is False
    assert evidence["ids"] == [] and evidence["phase_ms"] == []
    assert evidence["route_ms"] >= 0 and evidence["tool_ms"] >= 0


def test_ctx128_profile_rejects_agent_before_loading_models(
        monkeypatch, capsys, tmp_path) -> None:
    server = _server_module()

    class MustNotLoad:
        def __init__(self, **_kwargs):
            raise AssertionError("ctx128 capability gate must run before model load")

    monkeypatch.setattr(server, "Merged", MustNotLoad)
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--profile", "ctx128", "--deployment-root", str(tmp_path),
        "--embedding", "embedding.bin", "--tokenizer", "tokenizer.json",
        "--agent", "--allow-unqualified-profile",
    ])

    with pytest.raises(SystemExit) as raised:
        server.main()

    assert raised.value.code == 2
    assert "ctx128 is chat-only" in capsys.readouterr().err


def test_chat_repl_uses_template_and_streams_split_cjk_once(
        monkeypatch, capsys) -> None:
    server = _server_module()
    rendered_prompts = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            rendered_prompts.append(text)
            return SimpleNamespace(ids=[0, 42])

        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens
            values = tuple(ids)
            if values == (100,):
                return "秦汉的兵马\ufffd"
            return "秦汉的兵马俑"

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = Tokenizer()
        last_phase_steps = []

        def __init__(self, **_kwargs):
            pass

        def generate(self, ids, *_args, **kwargs):
            callback = kwargs["on_token"]
            callback((100,))
            callback((100, 101))
            callback((100, 101, 130073))
            self.last_phase_steps = [{"position": 0}]
            return "eos", [100, 101, 130073], [1.0] * (len(ids) + 3)

        def close(self):
            pass

    prompts = iter(["我爱中国。", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--chat",
    ])

    assert server.main() == 0

    output = capsys.readouterr().out
    assert output.count("秦汉的兵马俑") == 1
    assert "\ufffd" not in output
    assert "Chat ready" in output
    assert "<|im_start|>user\n我爱中国。<|im_end|>" in rendered_prompts[0]
    assert "<tools>" not in rendered_prompts[0]
    assert rendered_prompts[0].endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_terminal_ui_color_can_be_forced(monkeypatch, capsys) -> None:
    server = _server_module()
    monkeypatch.delenv("NO_COLOR", raising=False)
    ui = server.TerminalUI(
        active=True, context=1024, color="always", spinner=False)

    ui.banner()
    ui.ready(3, 10.25)

    output = capsys.readouterr().out
    assert "\033[" in output
    assert "MiniCPM 5" in output
    assert "ctx1024" in output
    assert "loaded" in output

    prompt = ui.prompt()
    assert "\033[" in prompt
    if server._readline is not None:
        assert "\001\033[" in prompt
        assert "m\002You\001\033[0m\002" in prompt

    plain = server.TerminalUI(
        active=True, context=1024, color="never", spinner=False).prompt()
    assert plain == "You ❯ "
    assert "\001" not in plain and "\002" not in plain


def test_interactive_executor_can_hide_low_level_stderr(monkeypatch) -> None:
    server = _server_module()
    captured = []

    def fake_popen(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(server.probe.subprocess, "Popen", fake_popen)
    server.probe._start(
        Path("executor"), [Path("decode.om")], [Path("lib")], 0,
        quiet=True)

    assert captured[0][1]["stderr"] is subprocess.DEVNULL


def test_merged_accepts_a_legacy_executor_launcher(monkeypatch, tmp_path) -> None:
    server = _server_module()
    calls = []

    class FakeProcess:
        stdout = object()

    def legacy_start(*args, **kwargs):
        calls.append(kwargs)
        if kwargs:
            raise TypeError("_start() got an unexpected keyword argument 'quiet'")
        return FakeProcess()

    monkeypatch.setattr(server.probe, "_start", legacy_start)
    cache_bytes = 48 * 1023 * 128 * 2
    transformer = (
        [1536 * 16, 1024 * 4, 128 * 128 * 4,
         cache_bytes, cache_bytes],
        [48 * 128 * 4, 48 * 128 * 4, 1536 * 16],
    )
    monkeypatch.setattr(
        server.probe, "_read_ready",
        lambda *_args: [transformer, ([], [])])
    monkeypatch.setattr(server.Merged, "_identify_kv", lambda _self: {})
    embedding = tmp_path / "embedding.bin"
    embedding.write_bytes(b"")

    session = server.Merged(
        executable=Path("executor"), decode=Path("decode.om"), prefill=None,
        head=Path("head.om"), library_paths=[], embedding=embedding,
        context=1024, timeout=1.0, quiet_executor=True)
    session.embed.close()

    assert calls == [{"quiet": True}, {}]


def test_merged_rejects_om_context_descriptor_mismatch() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.context = 4096
    session.cache_bytes = 48 * 4095 * 128 * 2
    session.decode_index = 0
    session.prefill_index = 0
    # This is a valid ctx1024 transformer's public input geometry.
    ctx1024_cache = 48 * 1023 * 128 * 2
    session.descriptors = [(
        [1536 * 16, 1024 * 4, 128 * 128 * 4,
         ctx1024_cache, ctx1024_cache],
        [48 * 128 * 4, 48 * 128 * 4, 1536 * 16],
    )]

    with pytest.raises(RuntimeError, match="descriptor/context mismatch"):
        session._validate_context_descriptors()


def _fake_python(tmp_path: Path) -> Path:
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_chat_and_agent_launchers_are_separate(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.update({
        "PICO_MINICPM5_ROOT": str(tmp_path / "deploy"),
        "PICO_RUNTIME_LIB": str(tmp_path / "lib"),
        "PYTHON": str(_fake_python(tmp_path)),
    })
    environment.pop("PROMPT", None)
    script = PROJECT / "app" / "chat.sh"

    repl = subprocess.run(
        ["sh", str(script)], env=environment, text=True,
        capture_output=True, check=True).stdout.splitlines()
    assert "--agent" not in repl
    assert "--chat" in repl
    assert "--interactive" not in repl
    assert "--prompt" not in repl
    assert repl[repl.index("--profile") + 1] == "ctx1024"
    assert repl[repl.index("--deployment-root") + 1] == str(tmp_path / "deploy")

    chat_with_options = subprocess.run(
        ["sh", str(script), "--no-spinner", "--color", "never"],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert "--chat" in chat_with_options
    assert "--no-spinner" in chat_with_options
    assert "--interactive" not in chat_with_options

    agent = subprocess.run(
        ["sh", str(PROJECT / "app" / "agent.sh")], env=environment,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert "--agent" in agent
    assert "--chat" not in agent
    assert "--interactive" not in agent
    assert "--prompt" not in agent
    assert agent[agent.index("--profile") + 1] == "ctx1024"
    assert "--max-new" not in agent

    ctx128 = environment.copy()
    ctx128["PICO_PROFILE"] = "ctx128"
    chat128 = subprocess.run(
        ["sh", str(script)], env=ctx128, text=True,
        capture_output=True, check=True).stdout.splitlines()
    assert chat128[chat128.index("--profile") + 1] == "ctx128"

    thinking_environment = environment.copy()
    thinking_environment["THINKING"] = "on"
    thinking_agent = subprocess.run(
        ["sh", str(PROJECT / "app" / "agent.sh")],
        env=thinking_environment, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert "--thinking" in thinking_agent

    one_shot = subprocess.run(
        ["sh", str(script), "--prompt", "hello", "--max-new", "7"],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert one_shot.count("--prompt") == 1
    assert "--agent" not in one_shot
    assert "--interactive" not in one_shot
    assert one_shot[-4:] == ["--prompt", "hello", "--max-new", "7"]
