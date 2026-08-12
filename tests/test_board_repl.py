from __future__ import annotations

import argparse
import builtins
import importlib.util
import io
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
    assert "Hi3403 端侧 AI" in output
    assert "本地运行 · 隐私安全 · 实时响应" in output
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
        "What does note.txt say?", "/think off", "/quit"])
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
    assert "HiAgent" in output
    assert "( o.o )    MiniCPM 5" not in output
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
    assert '"name":"read_file"' in rendered_prompts[0]
    assert '"name":"write_file"' in rendered_prompts[0]
    assert '"name":"run_shell"' in rendered_prompts[0]
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


def test_agent_direct_file_route_skips_model(
        monkeypatch, capsys, tmp_path) -> None:
    server = _server_module()
    (tmp_path / "note.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = object()

        def __init__(self, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            raise AssertionError("direct file route must not invoke MiniCPM5")

        def close(self):
            pass

    prompts = iter(["读取 note.txt 前 2 行", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--agent",
        "--reuse-session-kv", "--fixed-prefix-snapshots",
        "--workspace", str(tmp_path),
    ])

    assert server.main() == 0

    output = capsys.readouterr().out
    assert "model skipped" in output
    assert "文件内容（note.txt）：" in output
    assert "1: alpha" in output and "2: beta" in output


def test_agent_tool_then_model_routes_known_read_before_summary(
        monkeypatch, capsys, tmp_path) -> None:
    server = _server_module()
    (tmp_path / "note.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    rendered = []
    generate_calls = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            rendered.append(text)
            return SimpleNamespace(ids=[0, 42, 43])

        def decode(self, _ids, skip_special_tokens=False):
            return "文件包含 alpha 和 beta。<|im_end|>"

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = Tokenizer()
        last_phase_steps = [{"position": 0}]

        def __init__(self, **_kwargs):
            pass

        def generate(self, ids, *_args, **kwargs):
            generate_calls.append(kwargs)
            if kwargs.get("on_token"):
                kwargs["on_token"]((100,))
            return "eos", [100], [1.0] * (len(ids) + 1)

        def close(self):
            pass

    prompts = iter(["读取 note.txt 前 2 行并总结", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--agent",
        "--reuse-session-kv", "--fixed-prefix-snapshots",
        "--workspace", str(tmp_path),
    ])

    assert server.main() == 0
    output = capsys.readouterr().out
    assert len(generate_calls) == 1
    assert generate_calls[0]["reuse_prefix"] is True
    assert generate_calls[0]["prefix_snapshot_key"] == "none"
    assert generate_calls[0]["prefix_snapshot_tokens"] == (0, 42, 43)
    assert "✓ read_file: 1: alpha" in output
    assert "MiniCPM ✦ 文件包含 alpha 和 beta。" in output
    assert "本地工具已经成功执行" in rendered[0]
    assert "请只用中文直接、简洁地回答" in rendered[0]
    assert "已验证结果：" in rendered[0]
    assert "1: alpha" in rendered[0]
    assert "<tools>" not in rendered[0]
    assert '"name":"read_file"' not in rendered[0]


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


def test_terminal_ui_agent_banner_uses_hiagent(capsys) -> None:
    server = _server_module()
    ui = server.TerminalUI(
        active=True, context=1024, brand="HiAgent",
        color="never", spinner=False)

    ui.banner()

    output = capsys.readouterr().out
    assert "( o.o )    HiAgent" in output
    assert "( o.o )    MiniCPM 5" not in output


def test_terminal_ui_pet_animation_has_distinct_safe_frames(
        monkeypatch, capsys) -> None:
    server = _server_module()
    agent_ui = server.TerminalUI(
        active=True, context=1024, brand="HiAgent",
        color="never", spinner=True)
    chat_ui = server.TerminalUI(
        active=True, context=1024, brand="MiniCPM 5",
        color="never", spinner=True)

    assert agent_ui.pet_faces == (
        "( o.o )", "( o.O )", "( O.o )", "( -.- )")
    assert chat_ui.pet_faces == (
        "( o.o )", "( -.- )", "( o.o )", "( ^.^ )")

    # Simulate a real terminal without sleeping: repainting is restricted to
    # the first wait directly below the banner and restores a friendly face.
    agent_ui.is_tty = True
    agent_ui.animate = True
    monkeypatch.setattr(agent_ui, "_start_pet_process", lambda: False)
    agent_ui.banner()
    agent_ui.start_wait("Loading")
    agent_ui.stop_wait()
    output = capsys.readouterr().out
    assert "\033[4A" in output
    assert "\033[?25l" in output and "\033[?25h" in output
    assert "( o.O )    HiAgent" in output
    assert "Loading  0.0s" in output
    assert "( ^.^ )    HiAgent" in output
    assert agent_ui._pet_can_animate is False

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


def test_merged_overlaps_tokenizer_after_executor_spawn(
        monkeypatch, tmp_path) -> None:
    server = _server_module()
    events = []

    class FakeProcess:
        stdout = object()

    class FakeTokenizer:
        @staticmethod
        def from_file(path):
            events.append(("tokenizer", path))
            return object()

    cache_bytes = 48 * 1023 * 128 * 2
    transformer = (
        [1536 * 16, 1024 * 4, 128 * 128 * 4,
         cache_bytes, cache_bytes],
        [48 * 128 * 4, 48 * 128 * 4, 1536 * 16],
    )

    def start(*_args, **_kwargs):
        events.append(("spawn", None))
        return FakeProcess()

    def ready(*_args):
        events.append(("ready", None))
        return [transformer, ([], [])]

    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))
    monkeypatch.setattr(server.probe, "_start", start)
    monkeypatch.setattr(server.probe, "_read_ready", ready)
    monkeypatch.setattr(
        server.Merged, "_identify_kv",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("static slots must skip the four-execute probe")))
    embedding = tmp_path / "embedding.bin"
    embedding.write_bytes(b"")

    session = server.Merged(
        executable=Path("executor"), decode=Path("decode.om"), prefill=None,
        head=Path("head.om"), library_paths=[], embedding=embedding,
        tokenizer=tmp_path / "tokenizer.json", context=1024, timeout=1.0,
        transformer_output_slots=(0, 1, 2))
    session.embed.close()

    assert [event[0] for event in events] == ["spawn", "tokenizer", "ready"]
    assert set(session.startup_ms) == {
        "executor_spawn", "tokenizer", "executor_ready", "kv_identify",
        "kv_contract", "total"}
    assert session.startup_ms["kv_contract"] == "static"
    assert session.startup_ms["kv_identify"] < 10
    assert session.kv_slots == {0: (0, 1)}
    assert session.hidden_slots == {0: 2}


def test_merged_rehashes_imported_protocol_runner_and_models_immediately_before_spawn(
        monkeypatch, tmp_path: Path) -> None:
    server = _server_module()
    events = []

    class FakeProcess:
        stdout = object()

    class Registry:
        context = 1024
        handlers = ()
        enabled_widths = (1,)

        def to_dict(self):
            return {"schema": "fake.runtime", "enabled_widths": [1]}

        def plan(self, *_args):
            return SimpleNamespace(to_dict=lambda: {"segments": []})

        def validate_live_startup_identity(self, **kwargs):
            events.append(("validate", kwargs))

        def validate_loaded_handlers(self, *_args):
            pass

    cache_bytes = 48 * 1023 * 128 * 2
    transformer = (
        [1536 * 16, 1024 * 4, 128 * 128 * 4,
         cache_bytes, cache_bytes],
        [48 * 128 * 4, 48 * 128 * 4, 1536 * 16],
    )

    def start(*_args, **_kwargs):
        events.append(("spawn", None))
        return FakeProcess()

    monkeypatch.setattr(server.probe, "_start", start)
    monkeypatch.setattr(
        server.probe, "_read_ready", lambda *_args: [transformer, ([], [])])
    embedding = tmp_path / "embedding.bin"
    embedding.write_bytes(b"")
    executor = tmp_path / "executor"
    decode = tmp_path / "decode.om"
    head = tmp_path / "head.om"

    session = server.Merged(
        executable=executor, decode=decode, prefill=None, head=head,
        library_paths=[], embedding=embedding, context=1024, timeout=1.0,
        transformer_output_slots=(0, 1, 2), prefill_runtime=Registry())
    session.embed.close()

    assert [event[0] for event in events[:2]] == ["validate", "spawn"]
    live = events[0][1]
    assert live == {
        "executable": executor,
        "decode": decode,
        "prefill": None,
        "head": head,
        "embedding": embedding,
        "runner": Path(server.runner.__file__).resolve(),
    }


def test_merged_cleans_executor_when_tokenizer_startup_fails(
        monkeypatch, tmp_path) -> None:
    server = _server_module()

    class FakeProcess:
        stdout = object()
        terminated = False
        waited = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 30
            self.waited = True

    process = FakeProcess()

    class BrokenTokenizer:
        @staticmethod
        def from_file(_path):
            raise RuntimeError("tokenizer fixture failed")

    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=BrokenTokenizer))
    monkeypatch.setattr(
        server.probe, "_start", lambda *_args, **_kwargs: process)
    embedding = tmp_path / "embedding.bin"
    embedding.write_bytes(b"")

    with pytest.raises(RuntimeError, match="tokenizer fixture failed"):
        server.Merged(
            executable=Path("executor"), decode=Path("decode.om"),
            prefill=None, head=Path("head.om"), library_paths=[],
            embedding=embedding, tokenizer=tmp_path / "bad-tokenizer.json",
            context=1024, timeout=1.0)

    assert process.terminated and process.waited


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


def test_resident_prefix_plan_is_token_exact_and_reexecutes_full_hit() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session._resident_tokens = [1, 2, 3, 4]

    assert session._prefix_plan([1, 2, 9, 10], True) == 2
    assert session.last_prefix_metrics == {
        "prompt_tokens_new": 2,
        "prompt_tokens_replayed": 0,
        "prefix_cache_hit": 2,
        "prefix_snapshot_hit": 0,
        "prefix_snapshot_created": 0,
        "prefix_snapshot_restore_ms": 0.0,
    }

    assert session._prefix_plan([1, 2, 9, 10], False) == 0
    assert session.last_prefix_metrics == {
        "prompt_tokens_new": 2,
        "prompt_tokens_replayed": 2,
        "prefix_cache_hit": 0,
        "prefix_snapshot_hit": 0,
        "prefix_snapshot_created": 0,
        "prefix_snapshot_restore_ms": 0.0,
    }

    assert session._prefix_plan([1, 2, 3, 4], True) == 3
    assert session.last_prefix_metrics["prefix_cache_hit"] == 3
    session.reset_prefix_cache()
    assert session._resident_tokens == []


def test_fixed_prefix_snapshot_ranges_and_protocol_frames() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.past = 15
    session.row_f16 = 8
    session.decode_index = 2
    session.process = SimpleNamespace(stdin=io.BytesIO())
    session._respond = lambda sizes, expected_model=None: []

    ranges = session._snapshot_ranges(3)
    assert len(ranges) == 96
    assert ranges[0] == (3, 0, 24)
    assert ranges[1] == (3, 120, 24)
    assert ranges[48] == (4, 0, 24)

    session._save_input_snapshot(4, 3)
    payload = session.process.stdin.getvalue()
    header = server.runner._PERSISTENT_REQUEST.unpack(
        payload[:server.runner._PERSISTENT_REQUEST.size])
    assert header == (
        server.runner.PERSISTENT_REQUEST_MAGIC,
        server.runner.PERSISTENT_PROTOCOL_VERSION,
        server.runner.PERSISTENT_OP_SNAPSHOT_INPUTS,
        2, 4, 96, 0)
    first = server.runner._PERSISTENT_SNAPSHOT_RANGE.unpack(
        payload[server.runner._PERSISTENT_REQUEST.size:
                server.runner._PERSISTENT_REQUEST.size +
                server.runner._PERSISTENT_SNAPSHOT_RANGE.size])
    assert first == (3, 0, 0, 24)

    session.process.stdin = io.BytesIO()
    session._restore_input_snapshot(4)
    assert server.runner._PERSISTENT_REQUEST.unpack(
        session.process.stdin.getvalue()) == (
            server.runner.PERSISTENT_REQUEST_MAGIC,
            server.runner.PERSISTENT_PROTOCOL_VERSION,
            server.runner.PERSISTENT_OP_RESTORE_INPUTS,
            2, 4, 0, 0)


def test_native_prefill_scatter_publishes_complete_block_to_canonical_cache() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.context = 1024
    session.past = 1023
    session.row_f16 = 256
    session.cache_bytes = 48 * 1023 * 128 * 2
    session.models = [object(), object(), object()]
    session.decode_index = 0
    block_bytes = 48 * 16 * 128 * 4
    cache_inputs = [1536 * 16, 1024 * 4, 128 * 128 * 4,
                    session.cache_bytes, session.cache_bytes]
    session.descriptors = [
        (cache_inputs, [48 * 128 * 4, 48 * 128 * 4, 1536 * 16]),
        (cache_inputs, [block_bytes, block_bytes, 1536 * 16]),
        (cache_inputs, [48 * 128 * 4, 48 * 128 * 4, 1536 * 16]),
    ]
    session.kv_slots = {1: (0, 1)}
    session.process = SimpleNamespace(stdin=io.BytesIO())
    responses = []
    session._respond = lambda sizes, expected_model=None: \
        responses.append((sizes, expected_model)) or []

    session._scatter_kv_rows(1, 643, 16)

    blob = session.process.stdin.getvalue()
    header_size = server.runner._PERSISTENT_REQUEST.size
    record_size = server.runner._PERSISTENT_SCATTER_F32_TO_F16.size
    frame_size = header_size + 2 * record_size
    assert len(blob) == frame_size
    for frame_index, destination_model in enumerate((0,)):
        offset = frame_index * frame_size
        assert server.runner._PERSISTENT_REQUEST.unpack(
            blob[offset:offset + header_size]) == (
                server.runner.PERSISTENT_REQUEST_MAGIC,
                server.runner.PERSISTENT_PROTOCOL_VERSION,
                server.runner.PERSISTENT_OP_SCATTER_F32_TO_F16,
                destination_model, 2, 0, 0)
        first = server.runner._PERSISTENT_SCATTER_F32_TO_F16.unpack(
            blob[offset + header_size:offset + header_size + record_size])
        second = server.runner._PERSISTENT_SCATTER_F32_TO_F16.unpack(
            blob[offset + header_size + record_size:offset + frame_size])
        expected_common = (
            1, 0, 0, 643 * 256, 1023 * 256, 48, 16 * 128, 0)
        assert first == (3,) + expected_common
        assert second == (4, 1, 1, 0, 643 * 256, 1023 * 256,
                          48, 16 * 128, 0)
    assert responses == [((), 0)]


def test_native_prefill_scatter_rejects_terminal_partial_or_abi_drift() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.context = 1024
    session.past = 1023
    session.row_f16 = 256
    session.cache_bytes = 48 * 1023 * 128 * 2
    session.models = [object(), object()]
    inputs = [1536 * 16, 1024 * 4, 128 * 128 * 4,
              session.cache_bytes, session.cache_bytes]
    block_bytes = 48 * 16 * 128 * 4
    session.descriptors = [
        (inputs, [48 * 128 * 4, 48 * 128 * 4, 1536 * 16]),
        (inputs, [block_bytes, block_bytes - 4, 1536 * 16]),
    ]
    session.decode_index = 0
    session.kv_slots = {1: (0, 1)}
    session.process = SimpleNamespace(stdin=io.BytesIO())
    session._respond = lambda _sizes, expected_model=None: []

    with pytest.raises(ValueError, match="context-1 cache"):
        session._scatter_kv_rows(1, 1008, 16)
    with pytest.raises(ValueError, match="contiguous FP32 block ABI"):
        session._scatter_kv_rows(1, 1, 16)


def test_native_prefill_copies_canonical_prefix_to_wide_handle() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.past = 1023
    session.row_f16 = 256
    session.cache_bytes = 48 * 1023 * 256
    session.models = [object(), object()]
    cache_inputs = [24576, 4096, 65536,
                    session.cache_bytes, session.cache_bytes]
    session.descriptors = [
        (cache_inputs, [24576, 24576, 24576]),
        (cache_inputs, [24576, 24576, 24576]),
    ]
    session.process = SimpleNamespace(stdin=io.BytesIO())
    responses = []
    session._respond = lambda sizes, expected_model=None: \
        responses.append((sizes, expected_model)) or []

    session._copy_resident_prefix(0, 1, 643)

    payload = session.process.stdin.getvalue()
    header_size = server.runner._PERSISTENT_REQUEST.size
    record_size = server.runner._PERSISTENT_INPUT_COPY.size
    assert len(payload) == header_size + 96 * record_size
    assert server.runner._PERSISTENT_REQUEST.unpack(payload[:header_size]) == (
        server.runner.PERSISTENT_REQUEST_MAGIC,
        server.runner.PERSISTENT_PROTOCOL_VERSION,
        server.runner.PERSISTENT_OP_COPY_INPUTS,
        0, 96, 0, 0)
    first = server.runner._PERSISTENT_INPUT_COPY.unpack(
        payload[header_size:header_size + record_size])
    last = server.runner._PERSISTENT_INPUT_COPY.unpack(payload[-record_size:])
    assert first == (1, 3, 0, 0, 3, 0, 643 * 256, 0)
    assert last == (
        1, 4, 47 * 1023 * 256, 0, 4, 47 * 1023 * 256,
        643 * 256, 0)
    assert responses == [((), 0)]

    session.process.stdin = io.BytesIO()
    session._copy_resident_prefix(0, 1, 0)
    assert session.process.stdin.getvalue() == b""


def test_merged_response_rejects_model_count_or_size_drift() -> None:
    server = _server_module()

    def session_for(frame: bytes):
        session = object.__new__(server.Merged)
        stream = io.BytesIO(frame)
        session._read = lambda size: stream.read(size)
        return session

    header = server.runner._PERSISTENT_RESPONSE.pack(
        server.runner.PERSISTENT_RESPONSE_MAGIC,
        server.runner.PERSISTENT_PROTOCOL_VERSION,
        0, 2, 1, 0, 0)
    payload = header + server.runner._PERSISTENT_U64.pack(4) + b"data"
    assert session_for(payload)._respond((4,), 2) == [b"data"]

    with pytest.raises(RuntimeError, match="frame is invalid"):
        session_for(payload)._respond((4,), 1)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        session_for(payload)._respond((), 2)
    wrong_size = header + server.runner._PERSISTENT_U64.pack(3) + b"bad"
    with pytest.raises(RuntimeError, match="sizes mismatch"):
        session_for(wrong_size)._respond((4,), 2)


def test_merged_close_waits_for_shutdown_ack_before_native_cleanup() -> None:
    server = _server_module()

    class CaptureBytesIO(io.BytesIO):
        def close(self):
            self.was_closed = True

    class Process:
        def __init__(self):
            self.stdin = CaptureBytesIO()
            self.stdout = CaptureBytesIO()
            self.returncode = None
            self.waits = []
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.waits.append(timeout)
            self.returncode = 0
            return 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    session = object.__new__(server.Merged)
    session.process = Process()
    session.embed = io.BytesIO(b"embedding")
    responses = []
    session._respond = lambda sizes, expected_model=None: \
        responses.append((sizes, expected_model)) or []
    process = session.process

    session.close()
    session.close()

    assert server.runner._PERSISTENT_REQUEST.unpack(
        process.stdin.getvalue()) == (
            server.runner.PERSISTENT_REQUEST_MAGIC,
            server.runner.PERSISTENT_PROTOCOL_VERSION,
            server.runner.PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0)
    assert responses == [((), 0)]
    assert process.waits == [30]
    assert not process.terminated and not process.killed
    assert session.process is None and session.embed.closed


def test_fixed_prefix_snapshot_restore_and_key_drift_are_fail_closed() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session._resident_tokens = [9, 9]
    session._prefix_snapshots = {"read_only": (3, (1, 2, 3))}
    session.timeout = 1.0
    restored = []
    session._restore_input_snapshot = restored.append

    hit, elapsed = session._prepare_prefix_snapshot(
        "read_only", (1, 2, 3), (1, 2, 3, 4))
    assert restored == [3]
    assert session._resident_tokens == [1, 2, 3]
    assert hit == 3 and elapsed >= 0.0

    with pytest.raises(RuntimeError, match="changed token content"):
        session._prepare_prefix_snapshot(
            "read_only", (1, 2, 8), (1, 2, 8, 4))
    with pytest.raises(RuntimeError, match="not a prefix"):
        session._prepare_prefix_snapshot(
            "none", (5, 6), (5, 7, 8))


@pytest.mark.parametrize("operation", ("save", "restore"))
def test_snapshot_protocol_failure_discards_resident_session(
        operation: str) -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.context = 16
    session.past = 15
    session.row_f16 = 8
    session.decode_index = 0
    session._resident_tokens = [1, 2, 3]
    session._prefix_snapshots = {"fixed": (1, (1, 2, 3))}
    session._next_prefix_snapshot_id = 2
    session._wide_session_discarded = False
    session.last_prefix_metrics = session._empty_prefix_metrics()
    session.process = SimpleNamespace(
        stdin=io.BytesIO(), poll=lambda: None)
    terminated = []
    session._terminate_process = terminated.append
    session._respond = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected snapshot response failure"))

    method = (lambda: session._save_input_snapshot(1, 3)) \
        if operation == "save" else \
        (lambda: session._restore_input_snapshot(1))
    with pytest.raises(RuntimeError, match="session discarded"):
        method()

    assert session._wide_session_discarded is True
    assert session._resident_tokens == []
    assert session._prefix_snapshots == {}
    assert session._next_prefix_snapshot_id == 1
    assert terminated == [session.process]
    with pytest.raises(RuntimeError, match="wide session was discarded"):
        session._require_live_wide_session()


def test_fixed_prefix_snapshot_is_created_once_and_survives_clear() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session._resident_tokens = [1, 2, 3]
    session._prefix_snapshots = {}
    session._next_prefix_snapshot_id = 1
    session.timeout = 1.0
    saved = []
    session._save_input_snapshot = lambda snapshot_id, count: saved.append(
        (snapshot_id, count))

    assert session._maybe_create_prefix_snapshot(
        "none", (1, 2, 3)) == 3
    assert saved == [(1, 3)]
    assert session._maybe_create_prefix_snapshot("none", (1, 2, 3)) == 0
    session.reset_prefix_cache()
    assert session._resident_tokens == []
    assert session._prefix_snapshots == {"none": (1, (1, 2, 3))}


def test_agent_context_rebase_runs_before_ctx_overflow(
        monkeypatch, capsys, tmp_path) -> None:
    server = _server_module()
    report = tmp_path / "rebase.json"

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            turns = text.count("<|im_start|>user\nhello")
            return SimpleNamespace(ids=list(range(250 + turns * 180)))

        def decode(self, _ids, skip_special_tokens=False):
            assert not skip_special_tokens
            return "ok<|im_end|>"

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        tokenizer = Tokenizer()
        last_phase_steps = [{"position": 0}]

        def __init__(self, **_kwargs):
            pass

        def generate(self, ids, *_args, **kwargs):
            if kwargs.get("on_token"):
                kwargs["on_token"]((100,))
            return "eos", [100], [1.0] * (len(ids) + 1)

        def close(self):
            pass

    prompts = iter(["hello1", "hello2", "hello3", "hello4", "/quit"])
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(prompts))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--tokenizer", "tokenizer.json", "--agent", "--context", "1024",
        "--max-new", "8", "--max-tool-steps", "4",
        "--report", str(report),
    ])

    assert server.main() == 0
    runs = json.loads(report.read_text())["runs"]
    rebased = [run for run in runs if run.get("context_rebased")]
    assert len(rebased) == 1
    assert rebased[0]["context_tokens_before"] == 970
    assert rebased[0]["context_tokens_after"] == 610
    assert rebased[0]["context_turns_compacted"] == 2
    assert "Context rebased: 970→610 tokens" in capsys.readouterr().out


def test_final_context_position_does_not_scatter_past_cache_end() -> None:
    server = _server_module()
    session = object.__new__(server.Merged)
    session.context = 2
    session.past = 1
    session.timeout = 1.0
    session.resident_kv = True
    session._resident_tokens = []
    session.last_prefix_metrics = session._empty_prefix_metrics()
    session._prefix_snapshots = {}
    session._next_prefix_snapshot_id = 1
    session.models = [object(), object()]
    session.decode_index = 0
    session.prefill_index = 0
    session.head_index = 1
    session.descriptors = [
        ([4, 8, 8, 8, 8], [4, 4, 4]),
        ([4, 4, 4], [8]),
    ]
    session.kv_slots = {0: (0, 1)}
    session.hidden_slots = {0: 2}
    session.process = SimpleNamespace(stdin=io.BytesIO())
    session._hidden_input = lambda _token, want: bytes(want)
    session._rope_matrix_bytes = lambda _position: bytes(8)
    session._run = lambda *_args, **_kwargs: []
    scattered = []
    session._scatter_kv = lambda _model, position: scattered.append(position)
    responses = []

    def respond(sizes, expected_model=None):
        responses.append(sizes)
        return [server.struct.pack("<If", 99, 1.0)] if sizes == (8,) else []

    session._respond = respond

    reason, ids, _steps = session.generate(
        [11, 12], 1, {1, 130073}, reuse_prefix=True)

    assert reason == "max" and ids == [99]
    assert scattered == [0]
    assert session._resident_tokens == [11, 12]
    assert responses == [(), (8,)]
    assert [phase["head_skipped"] for phase in session.last_phase_steps] == [
        True, False]
    assert session.last_phase_steps[0]["head_execute_ms"] == 0.0
    assert session.last_phase_steps[0]["argmax_ms"] == 0.0
    assert session.last_phase_steps[0]["prefill_width"] == 1
    assert session.last_phase_steps[1]["prefill_width"] == 1
    assert session.last_prefill_schedule["policy"] == \
        "largest_first_strict_s1"
    assert session.last_prefill_schedule["counts"] == {
        "S128": 0, "S32": 0, "S16": 0, "S1": 2}


def test_cli_plumbs_profile_context_and_activation_before_model_start(
        monkeypatch, tmp_path: Path) -> None:
    server = _server_module()
    manifest = tmp_path / "activation.json"
    report = tmp_path / "report.json"
    loaded = []
    started = []

    class Registry:
        enabled_widths = (1,)

        def to_dict(self):
            return {"schema": "fake.runtime", "enabled_widths": [1]}

    registry = Registry()

    def load_runtime_registry(**kwargs):
        loaded.append(kwargs)
        return registry

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        last_phase_steps = [{"position": 0, "head_skipped": False}]

        def __init__(self, **kwargs):
            started.append(kwargs)

        def generate(self, *_args, **_kwargs):
            return "max", [7], [1.0]

        def close(self):
            pass

    monkeypatch.setattr(
        server.prefill_runtime_contract,
        "load_runtime_registry", load_runtime_registry)
    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--profile", "ctx1024", "--deployment-root", str(tmp_path),
        "--embedding", "embedding.bin", "--prompt-ids", "11",
        "--prefill-activation-manifest", str(manifest),
        "--available-bytes", "900", "--base-resident-bytes", "400",
        "--reserve-bytes", "200", "--report", str(report),
    ])

    assert server.main() == 0
    assert loaded == [{
        "activation_manifest": manifest,
        "deployment_root": tmp_path,
        "context": 1024,
        "available_bytes": 900,
        "base_resident_bytes": 400,
        "reserve_bytes": 200,
    }]
    assert started[0]["prefill_runtime"] is registry
    assert started[0]["decode"] == tmp_path / "models/decode.om"
    assert json.loads(report.read_text())["prefill_runtime"] == {
        "schema": "fake.runtime", "enabled_widths": [1]}


def test_cli_plumbs_mixed_prefill_window_profile(
        monkeypatch, tmp_path: Path) -> None:
    server = _server_module()
    report = tmp_path / "report.json"
    started = []

    class Registry:
        enabled_widths = (1,)

        def to_dict(self):
            return {"schema": "fake.runtime", "enabled_widths": [1]}

    monkeypatch.setattr(
        server.prefill_runtime_contract,
        "load_runtime_registry", lambda **kwargs: Registry())

    class Session:
        models = [object(), object(), object()]
        kv_slots = {0: (0, 1), 1: (0, 1)}
        last_phase_steps = [{"position": 0, "head_skipped": False}]

        def __init__(self, **kwargs):
            started.append(kwargs)

        def generate(self, *_args, **_kwargs):
            return "max", [7], [1.0]

        def close(self):
            pass

    monkeypatch.setattr(server, "Merged", Session)
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--profile", "ctx4096", "--deployment-root", str(tmp_path),
        "--embedding", "embedding.bin", "--prompt-ids", "11",
        "--report", str(report),
    ])

    assert server.main() == 0
    assert started[0]["context"] == 4096
    assert started[0]["prefill_context"] == 1024
    assert started[0]["prefill"] == tmp_path / "models/prefill.om"
    assert started[0]["decode"] == tmp_path / "models/ctx4096/decode.om"


def test_cli_rejects_host_kv_under_mixed_profile(
        monkeypatch, capsys, tmp_path: Path) -> None:
    server = _server_module()
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--profile", "ctx4096", "--deployment-root", str(tmp_path),
        "--embedding", "embedding.bin", "--prompt-ids", "11",
        "--host-kv",
    ])
    with pytest.raises(SystemExit):
        server.main()
    assert "mixed prefill window" in capsys.readouterr().err


def test_mixed_descriptor_validation_uses_per_model_windows() -> None:
    server = _server_module()
    merged = server.Merged.__new__(server.Merged)
    merged.context = 4096
    merged.cache_bytes = server.CHANNELS * 4095 * server.HEAD_DIM * 2
    merged.prefill_context = 1024
    merged.prefill_cache_bytes = server.CHANNELS * 1023 * server.HEAD_DIM * 2
    merged.decode_index = 0
    merged.prefill_index = 1
    rope = server.HEAD_DIM * server.HEAD_DIM * 4
    outputs = (24576, 24576, 24576)
    decode_inputs = (
        24576, 4096 * 4, rope, merged.cache_bytes, merged.cache_bytes)
    prefill_inputs = (
        24576, 1024 * 4, rope,
        merged.prefill_cache_bytes, merged.prefill_cache_bytes)
    merged.descriptors = {
        0: (decode_inputs, outputs),
        1: (prefill_inputs, outputs),
    }
    merged._validate_context_descriptors()

    merged.descriptors[1] = (decode_inputs, outputs)
    with pytest.raises(RuntimeError, match="ctx1024"):
        merged._validate_context_descriptors()


def test_mixed_init_guards_fail_closed(tmp_path: Path) -> None:
    server = _server_module()
    with pytest.raises(RuntimeError, match="dedicated prefill handle"):
        server.Merged(
            executable="x", decode="d", prefill=None, head="h",
            library_paths=[], embedding=tmp_path / "embedding.bin",
            context=4096, timeout=1.0,
            transformer_output_slots=(0, 1, 2), prefill_context=1024)
    with pytest.raises(RuntimeError, match="dynamic probing"):
        server.Merged(
            executable="x", decode="d", prefill="p", head="h",
            library_paths=[], embedding=tmp_path / "embedding.bin",
            context=4096, timeout=1.0, prefill_context=1024)


def test_cli_rejects_invalid_strict_s1_before_model_start(
        monkeypatch, capsys, tmp_path: Path) -> None:
    server = _server_module()
    started = []

    def reject(**_kwargs):
        raise server.prefill_runtime_contract.activation_contract.\
            PrefillActivationError("live strict-S1 anchor mismatch")

    monkeypatch.setattr(
        server.prefill_runtime_contract, "load_runtime_registry", reject)
    monkeypatch.setattr(server, "Merged", lambda **kwargs: started.append(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "merged_board_server.py", "--persistent-executor", "executor",
        "--decode-model", "decode.om", "--prefill-model", "prefill.om",
        "--head-model", "head.om", "--embedding", "embedding.bin",
        "--prompt-ids", "11", "--prefill-activation-manifest",
        str(tmp_path / "activation.json"), "--available-bytes", "900",
        "--base-resident-bytes", "400", "--reserve-bytes", "200",
    ])

    with pytest.raises(SystemExit):
        server.main()
    assert started == []
    stderr = capsys.readouterr().err
    assert "native prefill activation failed" in stderr
    assert "strict-S1 anchor mismatch" in stderr


def test_merged_rejects_untyped_wide_registry_before_open_or_spawn(
        monkeypatch, tmp_path: Path) -> None:
    server = _server_module()
    spawned = []
    monkeypatch.setattr(
        server.probe, "_start", lambda *_args, **_kwargs: spawned.append(True))
    unsafe = SimpleNamespace(context=1024, enabled_widths=(16, 1))

    with pytest.raises(RuntimeError, match="without an exact typed"):
        server.Merged(
            executable=tmp_path / "executor",
            decode=tmp_path / "decode.om",
            prefill=tmp_path / "prefill.om",
            head=tmp_path / "head.om",
            library_paths=[],
            embedding=tmp_path / "missing-embedding.bin",
            context=1024,
            timeout=1.0,
            prefill_runtime=unsafe,
        )
    assert spawned == []


def test_transformer_output_slot_parser_is_fail_closed() -> None:
    server = _server_module()

    assert server.parse_transformer_output_slots("0,1,2") == (0, 1, 2)
    with pytest.raises(argparse.ArgumentTypeError, match="distinct"):
        server.parse_transformer_output_slots("0,0,2")
    with pytest.raises(argparse.ArgumentTypeError, match="integers"):
        server.parse_transformer_output_slots("k,v,h")


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
    assert "--reuse-session-kv" in repl
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
    assert "--reuse-session-kv" in agent
    assert "--fixed-prefix-snapshots" in agent
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

    no_reuse = environment.copy()
    no_reuse["REUSE_SESSION_KV"] = "0"
    replay = subprocess.run(
        ["sh", str(script)], env=no_reuse, text=True,
        capture_output=True, check=True).stdout.splitlines()
    assert "--reuse-session-kv" not in replay

    no_snapshots = environment.copy()
    no_snapshots["FIXED_PREFIX_SNAPSHOTS"] = "0"
    no_snapshot_agent = subprocess.run(
        ["sh", str(PROJECT / "app" / "agent.sh")], env=no_snapshots,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert "--fixed-prefix-snapshots" not in no_snapshot_agent

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
