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
            return "max", [100 + len(calls)], [1.0, 2.0]

        def close(self):
            self.closed = True

    prompts = iter(["hello", "/reset", "world", "/help", "/quit"])
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
    assert "MiniCPM> reply-101" in output
    assert "Context reset." in output
    assert "Commands: /help, /reset, /quit" in output


def _fake_python(tmp_path: Path) -> Path:
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_chat_sh_defaults_to_repl_and_forwards_explicit_prompt(tmp_path: Path) -> None:
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
    assert "--interactive" in repl
    assert "--prompt" not in repl

    one_shot = subprocess.run(
        ["sh", str(script), "--prompt", "hello", "--max-new", "7"],
        env=environment, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert one_shot.count("--prompt") == 1
    assert "--interactive" not in one_shot
    assert one_shot[-4:] == ["--prompt", "hello", "--max-new", "7"]
