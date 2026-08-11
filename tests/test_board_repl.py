from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


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

    prompts = iter(["Please read note.txt", "/quit"])
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
    assert "<tool_response>" in rendered_prompts[1]
    assert "hello from tool" in rendered_prompts[1]


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
    monkeypatch.setattr(server.probe, "_read_ready", lambda *_args: [([], [])] * 2)
    monkeypatch.setattr(server.Merged, "_identify_kv", lambda _self: {})
    embedding = tmp_path / "embedding.bin"
    embedding.write_bytes(b"")

    session = server.Merged(
        executable=Path("executor"), decode=Path("decode.om"), prefill=None,
        head=Path("head.om"), library_paths=[], embedding=embedding,
        context=1024, timeout=1.0, quiet_executor=True)
    session.embed.close()

    assert calls == [{"quiet": True}, {}]


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
    assert "--interactive" in repl
    assert "--prompt" not in repl
    assert repl[repl.index("--context") + 1] == "1024"

    agent = subprocess.run(
        ["sh", str(PROJECT / "app" / "agent.sh")], env=environment,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert "--agent" in agent
    assert "--interactive" not in agent
    assert "--prompt" not in agent
    assert agent[agent.index("--max-new") + 1] == "128"

    one_shot = subprocess.run(
        ["sh", str(script), "--prompt", "hello", "--max-new", "7"],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert one_shot.count("--prompt") == 1
    assert "--agent" not in one_shot
    assert "--interactive" not in one_shot
    assert one_shot[-4:] == ["--prompt", "hello", "--max-new", "7"]
