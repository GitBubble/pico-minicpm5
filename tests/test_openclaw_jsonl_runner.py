# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
OPENCLAW_SRC = PROJECT / "app" / "openclaw" / "src"


def _runner_module():
    sys.path.insert(0, str(OPENCLAW_SRC))
    spec = importlib.util.spec_from_file_location(
        "openclaw_merged_jsonl_runner_test",
        OPENCLAW_SRC / "merged_jsonl_runner.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merged_jsonl = _runner_module()


def _request(**updates: object) -> dict[str, object]:
    request: dict[str, object] = {
        "protocol": merged_jsonl.RUNNER_PROTOCOL,
        "request_id": "req-1",
        "op": "generate",
        "model": merged_jsonl.MODEL_ID,
        "input_ids": [0, 42],
        "max_new_tokens": 2,
        "temperature": 0.0,
        "eos_token_ids": [1, 130073],
        "reset_kv": True,
    }
    request.update(updates)
    return request


class FakeSession:
    def __init__(
        self, reason: str = "max", output_ids: list[int] | None = None,
        *, fail: bool = False,
    ) -> None:
        self.reason = reason
        self.output_ids = [17, 18] if output_ids is None else output_ids
        self.fail = fail
        self.calls: list[tuple[tuple[int, ...], int, set[int], int]] = []
        self.extra_kwargs: list[dict[str, object]] = []
        self.closed = False

    def generate(self, prompt_ids, max_new, eos, *, start=0, **kwargs):
        self.calls.append((tuple(prompt_ids), max_new, set(eos), start))
        self.extra_kwargs.append(dict(kwargs))
        print("native diagnostic must not enter JSONL stdout")
        if self.fail:
            raise RuntimeError("private board detail")
        return self.reason, list(self.output_ids), [1.0]

    def close(self):
        self.closed = True


def test_success_maps_native_max_to_length(capsys):
    session = FakeSession()
    runner = merged_jsonl.MergedJsonlRunner(
        session, merged_jsonl.RequestLimits(8192, 256))
    response = runner.generate(_request())
    assert response == {
        "protocol": merged_jsonl.RUNNER_PROTOCOL,
        "request_id": "req-1",
        "ok": True,
        "output_ids": [17, 18],
        "finish_reason": "length",
    }
    assert session.calls == [((0, 42), 2, {1, 130073}, 0)]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "native diagnostic" in captured.err


def test_success_maps_eos_to_stop():
    session = FakeSession("eos", [17, 130073])
    runner = merged_jsonl.MergedJsonlRunner(
        session, merged_jsonl.RequestLimits(8192, 256))
    assert runner.generate(_request())["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"extra": 1}, "fields"),
        ({"protocol": "wrong"}, "contract"),
        ({"request_id": "../bad"}, "request_id"),
        ({"temperature": True}, "temperature"),
        ({"temperature": 0.1}, "temperature"),
        ({"input_ids": []}, "non-empty"),
        ({"input_ids": [True]}, "vocabulary"),
        ({"input_ids": [130560]}, "vocabulary"),
        ({"eos_token_ids": list(range(9))}, "at most 8"),
        ({"max_new_tokens": True}, "configured limit"),
        ({"max_new_tokens": 257}, "configured limit"),
        ({"reset_kv": False}, "contract"),
    ],
)
def test_request_validation_is_fail_closed(updates, message):
    with pytest.raises(ValueError, match=message):
        merged_jsonl.validate_request(
            _request(**updates), merged_jsonl.RequestLimits(8192, 256))


def test_context_preflight_happens_before_native_call():
    session = FakeSession("max", [17, 18])
    runner = merged_jsonl.MergedJsonlRunner(
        session, merged_jsonl.RequestLimits(4, 4))
    with pytest.raises(ValueError, match="context"):
        runner.generate(_request(input_ids=[0, 1, 2, 3], max_new_tokens=2))
    assert session.calls == []


def test_context_budget_accepts_equality_and_rejects_one_token_over():
    limits = merged_jsonl.RequestLimits(4, 4)
    accepted = merged_jsonl.validate_request(
        _request(input_ids=[0, 1, 2], max_new_tokens=1), limits)
    assert accepted["max_new_tokens"] == 1
    with pytest.raises(ValueError, match="context"):
        merged_jsonl.validate_request(
            _request(input_ids=[0, 1, 2, 3], max_new_tokens=1), limits)


def test_ctx16384_request_limits_enforce_the_extended_window():
    limits = merged_jsonl.RequestLimits(16384, 2048)
    prompt = list(range(2))
    accepted = merged_jsonl.validate_request(
        _request(input_ids=prompt, max_new_tokens=2048), limits)
    assert accepted["max_new_tokens"] == 2048
    full_window = merged_jsonl.validate_request(
        _request(input_ids=[7] * 16383, max_new_tokens=1), limits)
    assert len(full_window["input_ids"]) == 16383
    with pytest.raises(ValueError, match="context"):
        merged_jsonl.validate_request(
            _request(input_ids=[7] * 16384, max_new_tokens=1), limits)
    with pytest.raises(ValueError, match="within the model context"):
        merged_jsonl.RequestLimits(16384, 16385)


@pytest.mark.parametrize(
    ("reason", "output", "message"),
    [
        ("eos", [17, 18], "without an EOS"),
        ("max", [17], "before max_new_tokens"),
        ("context", [], "context after request preflight"),
        ("stop", [17, 18], "unsupported reason"),
        ("max", [17, True], "invalid output token"),
    ],
)
def test_impossible_native_results_are_rejected(reason, output, message):
    runner = merged_jsonl.MergedJsonlRunner(
        FakeSession(reason, output), merged_jsonl.RequestLimits(8192, 256))
    with pytest.raises(merged_jsonl.NativeExecutionError, match=message):
        runner.generate(_request())


def test_jsonl_loop_recovers_after_invalid_and_execution_errors():
    class AlternatingSession(FakeSession):
        def generate(self, prompt_ids, max_new, eos, *, start=0, **kwargs):
            if prompt_ids[-1] == 9:
                raise RuntimeError("do not disclose this detail")
            return super().generate(prompt_ids, max_new, eos, start=start)

    runner = merged_jsonl.MergedJsonlRunner(
        AlternatingSession(), merged_jsonl.RequestLimits(8192, 256))
    source = (
        '{"request_id":"dup","request_id":"dup2"}\n'
        + json.dumps(_request(request_id="native-fail", input_ids=[0, 9])) + "\n"
        + json.dumps(_request(request_id="good")) + "\n"
    )
    writer, diagnostics = io.StringIO(), io.StringIO()
    merged_jsonl.serve_jsonl(
        runner, io.StringIO(source), writer, diagnostics=diagnostics)
    responses = [json.loads(line) for line in writer.getvalue().splitlines()]
    assert [response["request_id"] for response in responses] == [
        "unknown", "native-fail", "good"]
    assert [response["ok"] for response in responses] == [False, False, True]
    assert responses[0]["error"]["code"] == "invalid_request"
    assert responses[1]["error"] == {
        "code": "execution_error", "message": "native generation failed"}
    assert "do not disclose this detail" in diagnostics.getvalue()


def test_native_type_error_is_not_misclassified_as_bad_client_input():
    class BrokenSession(FakeSession):
        def generate(self, prompt_ids, max_new, eos, *, start=0, **kwargs):
            raise TypeError("runtime API drift")

    runner = merged_jsonl.MergedJsonlRunner(
        BrokenSession(), merged_jsonl.RequestLimits(8192, 256))
    writer = io.StringIO()
    merged_jsonl.serve_jsonl(
        runner, io.StringIO(json.dumps(_request()) + "\n"), writer,
        diagnostics=io.StringIO())
    response = json.loads(writer.getvalue())
    assert response["error"]["code"] == "execution_error"


def test_jsonl_requires_newline_and_preserves_next_record():
    runner = merged_jsonl.MergedJsonlRunner(
        FakeSession(), merged_jsonl.RequestLimits(8192, 256))
    writer = io.StringIO()
    merged_jsonl.serve_jsonl(
        runner, io.StringIO(json.dumps(_request())), writer,
        diagnostics=io.StringIO())
    response = json.loads(writer.getvalue())
    assert response["ok"] is False
    assert "newline-terminated" in response["error"]["message"]


# ------------------------------------------------ reuse_prefix plumbing
def test_reuse_prefix_is_forwarded_only_when_enabled():
    session = FakeSession()
    runner = merged_jsonl.MergedJsonlRunner(
        session, merged_jsonl.RequestLimits(8192, 256), reuse_prefix=True)
    runner.generate(_request())
    assert session.extra_kwargs == [{"reuse_prefix": True}]


def test_reuse_prefix_is_absent_by_default():
    session = FakeSession()
    runner = merged_jsonl.MergedJsonlRunner(
        session, merged_jsonl.RequestLimits(8192, 256))
    runner.generate(_request())
    assert session.extra_kwargs == [{}]


# ------------------------------------------------ build_runner lineages
class OpenclawLineageMerged:
    """Signature twin of the app/openclaw/src Merged (short-decode lineage)."""

    last: "OpenclawLineageMerged | None" = None

    def __init__(self, *, executable, decode, prefill, head, library_paths,
                 embedding, context, timeout, tokenizer=None,
                 resident_kv=True, decode_short=None, short_context=128,
                 short_kv_slots=None, allow_unsafe_short_context=False,
                 allow_c8192_short_characterization=False,
                 executor_uncached=False, decode_no_cache=False,
                 characterize_decode_workspace_zero_once=False):
        observed = dict(locals())
        observed.pop("self")
        self.observed = observed
        self.closed = False
        type(self).last = self

    def generate(self, prompt_ids, max_new, eos, *, start=0):
        return "max", [17] * max_new, [1.0]

    def close(self):
        self.closed = True


class AppSrcLineageMerged:
    """Signature twin of the app/src Merged (mixed prefill-window lineage)."""

    last: "AppSrcLineageMerged | None" = None

    def __init__(self, *, executable, decode, prefill, head, library_paths,
                 embedding, context, timeout, tokenizer=None,
                 resident_kv=True, quiet_executor=False,
                 transformer_output_slots=None, prefill_runtime=None,
                 prefill_context=None, retain_decode_workspace=False):
        observed = dict(locals())
        observed.pop("self")
        self.observed = observed
        self.closed = False
        type(self).last = self

    def generate(self, prompt_ids, max_new, eos, *, start=0,
                 reuse_prefix=False):
        return "max", [17] * max_new, [1.0]

    def close(self):
        self.closed = True


def _touch(path: Path) -> Path:
    path.write_bytes(b"fixture")
    return path


def _args(tmp_path: Path, **updates: object) -> argparse.Namespace:
    libs = tmp_path / "lib"
    libs.mkdir(exist_ok=True)
    runtime = tmp_path / "runtime.py"
    runtime.write_text("class Merged: pass\n", encoding="utf-8")
    namespace = argparse.Namespace(
        context=8192,
        max_new_limit=256,
        persistent_executor=_touch(tmp_path / "executor"),
        decode_model=_touch(tmp_path / "decode.om"),
        prefill_model=_touch(tmp_path / "prefill.om"),
        head_model=_touch(tmp_path / "head.om"),
        embedding=_touch(tmp_path / "embedding.bin"),
        decode_short_model=None,
        library_path=[libs],
        short_kv_slots=None,
        runtime_module=runtime,
        python_path=[],
        short_context=None,
        allow_unsafe_short_context=False,
        allow_c8192_short_characterization=False,
        executor_uncached=False,
        decode_no_cache=False,
        characterize_decode_workspace_zero_once=False,
        prefill_context=None,
        transformer_output_slots=None,
        reuse_session_kv=False,
        timeout=3600.0,
    )
    for key, value in updates.items():
        setattr(namespace, key, value)
    return namespace


def _use_lineage(monkeypatch, lineage: type) -> None:
    module = type("Runtime", (), {"Merged": lineage})
    monkeypatch.setattr(
        merged_jsonl, "_load_runtime_module",
        lambda runtime_module, python_paths: module)


CORE_KWARGS = {
    "executable", "decode", "prefill", "head", "library_paths", "embedding",
    "context", "timeout", "tokenizer", "resident_kv",
}


@pytest.mark.parametrize(
    "lineage", [OpenclawLineageMerged, AppSrcLineageMerged])
def test_core_contract_reaches_both_lineages_without_optional_kwargs(
    tmp_path, monkeypatch, lineage,
):
    # Both signature twins reject unknown keywords, so a successful
    # construction with default flags proves the runner sent only the shared
    # core contract; a single unconditional lineage-specific kwarg would be a
    # TypeError on the opposite lineage.
    _use_lineage(monkeypatch, lineage)
    runner = merged_jsonl.build_runner(_args(tmp_path))
    session = lineage.last
    assert session is not None
    assert CORE_KWARGS <= set(session.observed)
    assert session.observed["context"] == 8192
    assert session.observed["resident_kv"] is True
    assert session.observed["tokenizer"] is None
    assert session.observed["decode"].name == "decode.om"
    assert runner.limits == merged_jsonl.RequestLimits(8192, 256)
    runner.close()
    assert session.closed is True


def test_openclaw_lineage_still_accepts_the_short_decode_flags(
    tmp_path, monkeypatch,
):
    _use_lineage(monkeypatch, OpenclawLineageMerged)
    args = _args(
        tmp_path,
        decode_short_model=_touch(tmp_path / "decode.short.om"),
        short_context=128,
        short_kv_slots="0,1",
        decode_no_cache=True,
        characterize_decode_workspace_zero_once=True,
    )
    runner = merged_jsonl.build_runner(args)
    observed = OpenclawLineageMerged.last.observed
    assert observed["decode_short"].name == "decode.short.om"
    assert observed["short_context"] == 128
    assert observed["short_kv_slots"] == (0, 1)
    assert observed["decode_no_cache"] is True
    assert observed["characterize_decode_workspace_zero_once"] is True
    runner.close()


def test_appsrc_lineage_receives_the_mixed_prefill_window_contract(
    tmp_path, monkeypatch,
):
    _use_lineage(monkeypatch, AppSrcLineageMerged)
    args = _args(
        tmp_path,
        context=16384,
        prefill_context=1024,
        transformer_output_slots="0,1,2",
    )
    runner = merged_jsonl.build_runner(args)
    observed = AppSrcLineageMerged.last.observed
    assert observed["context"] == 16384
    assert observed["prefill_context"] == 1024
    assert observed["transformer_output_slots"] == (0, 1, 2)
    assert runner.limits == merged_jsonl.RequestLimits(16384, 256)
    runner.close()


def test_prefill_context_fails_closed_on_the_short_decode_lineage(
    tmp_path, monkeypatch,
):
    _use_lineage(monkeypatch, OpenclawLineageMerged)
    args = _args(tmp_path, prefill_context=1024)
    OpenclawLineageMerged.last = None
    with pytest.raises(ValueError) as caught:
        merged_jsonl.build_runner(args)
    assert "--prefill-context" in str(caught.value)
    assert str(args.runtime_module) in str(caught.value)
    assert OpenclawLineageMerged.last is None


def test_transformer_output_slots_fail_closed_on_the_short_decode_lineage(
    tmp_path, monkeypatch,
):
    _use_lineage(monkeypatch, OpenclawLineageMerged)
    args = _args(tmp_path, transformer_output_slots="0,1,2")
    with pytest.raises(ValueError, match="--transformer-output-slots"):
        merged_jsonl.build_runner(args)


def test_short_decode_flags_fail_closed_on_the_appsrc_lineage(
    tmp_path, monkeypatch,
):
    _use_lineage(monkeypatch, AppSrcLineageMerged)
    args = _args(
        tmp_path, decode_short_model=_touch(tmp_path / "decode.short.om"))
    AppSrcLineageMerged.last = None
    with pytest.raises(ValueError, match="--decode-short-model"):
        merged_jsonl.build_runner(args)
    assert AppSrcLineageMerged.last is None


@pytest.mark.parametrize("raw", ["0,1", "0,1,1", "1,2,3", "a,b,c"])
def test_malformed_transformer_output_slots_are_rejected(
    tmp_path, monkeypatch, raw,
):
    _use_lineage(monkeypatch, AppSrcLineageMerged)
    args = _args(tmp_path, transformer_output_slots=raw)
    with pytest.raises(ValueError, match="output slots"):
        merged_jsonl.build_runner(args)


def test_reuse_session_kv_builds_and_arms_the_runner(tmp_path, monkeypatch):
    _use_lineage(monkeypatch, AppSrcLineageMerged)
    runner = merged_jsonl.build_runner(_args(tmp_path, reuse_session_kv=True))
    assert runner.reuse_prefix is True
    runner.close()


def test_reuse_session_kv_fails_closed_without_generate_support(
    tmp_path, monkeypatch,
):
    _use_lineage(monkeypatch, OpenclawLineageMerged)
    args = _args(tmp_path, reuse_session_kv=True)
    OpenclawLineageMerged.last = None
    with pytest.raises(ValueError, match="--reuse-session-kv"):
        merged_jsonl.build_runner(args)
    assert OpenclawLineageMerged.last is None


def test_build_runner_requires_short_slot_model(tmp_path):
    args = _args(tmp_path, short_kv_slots="0,1")
    with pytest.raises(ValueError, match="requires --decode-short-model"):
        merged_jsonl.build_runner(args)


def test_runtime_loader_rejects_module_without_merged(tmp_path):
    runtime = tmp_path / "runtime.py"
    runtime.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ImportError, match="does not export Merged"):
        merged_jsonl._load_runtime_module(runtime, ())


def test_cli_help_does_not_load_native_runtime(capsys):
    with pytest.raises(SystemExit) as raised:
        merged_jsonl.build_parser().parse_args(["--help"])
    assert raised.value.code == 0
    assert "--runtime-module" in capsys.readouterr().out


def test_real_cli_process_keeps_native_logs_out_of_jsonl(tmp_path):
    runtime = tmp_path / "runtime.py"
    runtime.write_text(
        "class Merged:\n"
        "    def __init__(self, **kwargs):\n"
        "        print('fake native init')\n"
        "    def generate(self, prompt_ids, max_new, eos, *, start=0):\n"
        "        print('fake native generate')\n"
        "        return 'max', [42] * max_new, []\n"
        "    def close(self):\n"
        "        print('fake native close')\n",
        encoding="utf-8")
    executor = _touch(tmp_path / "executor")
    decode = _touch(tmp_path / "decode.om")
    prefill = _touch(tmp_path / "prefill.om")
    head = _touch(tmp_path / "head.om")
    embedding = _touch(tmp_path / "embedding.bin")
    libs = tmp_path / "lib"
    libs.mkdir()
    requests = "".join(
        json.dumps(_request(request_id=request_id)) + "\n"
        for request_id in ("first", "second"))
    completed = subprocess.run(
        [
            sys.executable,
            str(OPENCLAW_SRC / "merged_jsonl_runner.py"),
            "--serve-jsonl",
            "--runtime-module", str(runtime),
            "--persistent-executor", str(executor),
            "--decode-model", str(decode),
            "--prefill-model", str(prefill),
            "--head-model", str(head),
            "--library-path", str(libs),
            "--embedding", str(embedding),
            "--context", "8192",
            "--max-new-limit", "256",
        ],
        input=requests, text=True, capture_output=True, timeout=10,
        check=False)
    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["request_id"] for response in responses] == [
        "first", "second"]
    assert all(response["output_ids"] == [42, 42] for response in responses)
    assert "fake native init" in completed.stderr
    assert completed.stderr.count("fake native generate") == 2
    assert "fake native close" in completed.stderr
