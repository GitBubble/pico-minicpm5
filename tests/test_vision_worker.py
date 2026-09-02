"""The vision worker: wire framing, the generation loop, and serving.

None of this needs a board. The executor is a pipe with a documented frame
format, so a fake process exercises the part that actually broke during
bring-up -- reading the response -- and a fake session exercises the loop
above it.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "app" / "src"


def _module(name: str, filename: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def worker():
    return _module("vision_worker_test", "vision_worker.py")


@pytest.fixture()
def vlm(worker):
    return sys.modules["minicpm4v_vision"]


@pytest.fixture()
def runner(worker):
    return sys.modules["pico_minicpm5_split_board_runner"]


# --------------------------------------------------------------- description

def test_a_finished_sentence_is_reported_as_itself(worker) -> None:
    answer = worker.describe({
        "text": "这张图片展示了一个界面。", "tokens": 12,
        "stopped": "end token", "alternatives": [], "vocab_size": 8})

    assert answer == "这张图片展示了一个界面。"


def test_hitting_the_cap_is_marked_rather_than_hidden(worker) -> None:
    """A truncated description must not read as a complete one."""
    answer = worker.describe({
        "text": "这张图片展示了", "tokens": 40,
        "stopped": "length", "alternatives": [], "vocab_size": 8})

    assert answer.startswith("这张图片展示了")
    assert "40" in answer and "上限" in answer


def test_generating_nothing_reports_the_distribution_not_an_empty_string(
        worker) -> None:
    """Silence would look like a crash; the candidates show the model ran."""
    answer = worker.describe({
        "text": "", "tokens": 0, "stopped": "end token",
        "alternatives": ["这张", "这是一", "图片"], "vocab_size": 8})

    assert "这张" in answer and "这是一" in answer


# -------------------------------------------------------------- wire framing

class FakeProcess:
    """Stands in for the persistent executor.

    stdout is a real file rather than a BytesIO: the protocol reader selects
    on it, which needs a file descriptor. A regular file always selects as
    readable, so the whole response is available immediately.
    """

    def __init__(self, response: bytes, tmp_path) -> None:
        path = tmp_path / "executor.stdout"
        path.write_bytes(response)
        self.stdin = io.BytesIO()
        self.stdout = open(path, "rb")   # noqa: SIM115 -- closed by the test
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def _response(runner, tensors) -> bytes:
    """A well-formed response: header, every size, then every payload."""
    head = runner._PERSISTENT_RESPONSE.pack(
        runner.PERSISTENT_RESPONSE_MAGIC, runner.PERSISTENT_PROTOCOL_VERSION,
        0, 0, len(tensors), 0, 0)
    sizes = b"".join(runner._PERSISTENT_U64.pack(len(t)) for t in tensors)
    return head + sizes + b"".join(tensors)


def _session(worker, runner, response: bytes, outputs, tmp_path, inputs=2):
    session = worker.VisionSession.__new__(worker.VisionSession)
    session.timeout = 30.0
    session.process = FakeProcess(response, tmp_path)
    session.inputs = {0: inputs}
    session.outputs = {0: outputs}
    return session


def test_every_size_is_read_before_any_payload(
        worker, runner, tmp_path) -> None:
    """The executor writes all sizes, then all payloads.

    Reading them interleaved parses the second size out of the first
    tensor's bytes. That produced a length of 4.6 quintillion during
    bring-up, which surfaced as a MemoryError far from its cause.
    """
    tensors = [b"A" * 8, b"B" * 16, b"C" * 4]
    session = _session(worker, runner, _response(runner, tensors),
                       [8, 16, 4], tmp_path)

    out = session.execute(0, [(0, b"payload")])

    assert [bytes(t) for t in out] == tensors


def test_a_large_tensor_is_read_in_chunks(worker, runner, tmp_path) -> None:
    """The logits alone are 58 MB; one read of that is not attempted."""
    big = bytes(range(256)) * (worker.CHUNK_BYTES // 256 + 3)
    session = _session(worker, runner, _response(runner, [big]),
                       [len(big)], tmp_path)

    out = session.execute(0, [])

    assert bytes(out[0]) == big


def test_the_request_declares_the_models_input_ports_not_the_write_count(
        worker, runner, tmp_path) -> None:
    """public_inputs is the model's port count. Sending the number of writes
    instead deadlocks: the executor waits for inputs that never arrive."""
    session = _session(worker, runner, _response(runner, [b"z" * 4]), [4],
                       tmp_path, inputs=5)

    session.execute(0, [(0, b"one"), (1, b"two")])

    frame = session.process.stdin.getvalue()
    magic, _v, op, model, public_inputs, out_count, writes = \
        runner._PERSISTENT_REQUEST.unpack(
            frame[:runner._PERSISTENT_REQUEST.size])
    assert magic == runner.PERSISTENT_REQUEST_MAGIC
    assert op == runner.PERSISTENT_OP_EXECUTE_RESIDENT
    assert model == 0
    assert public_inputs == 5, "the model has five ports; two were written"
    assert out_count == 1 and writes == 2


def test_a_failed_execute_raises_with_the_executors_message(
        worker, runner, tmp_path) -> None:
    message = b"model refused the image"
    head = runner._PERSISTENT_RESPONSE.pack(
        runner.PERSISTENT_RESPONSE_MAGIC, runner.PERSISTENT_PROTOCOL_VERSION,
        1, 0, 0, 0, len(message))
    session = _session(worker, runner, head + message, [4], tmp_path)

    with pytest.raises(worker.WorkerError, match="refused the image"):
        session.execute(0, [])


def test_a_missing_handle_is_refused_before_a_process_is_started(
        worker, tmp_path) -> None:
    with pytest.raises(worker.WorkerError, match="missing handle"):
        worker.VisionSession(tmp_path, "/bin/true", [])


# ----------------------------------------------------------- generation loop

class FakeSession:
    """Returns canned tensors and records what was asked of it."""

    def __init__(self, vocab_size, script) -> None:
        self.vocab_size = vocab_size
        self.script = list(script)
        self.calls = []

    def execute(self, model, writes):
        self.calls.append(model)
        if model == 0:                                   # vision
            return [bytearray(16)]
        if model == 1:                                   # resample
            tokens = np.zeros((64, 1024), dtype=np.float32)
            return [bytearray(tokens.tobytes())]
        # prefill: K, V, then logits for the whole window
        wanted = self.script.pop(0) if self.script else 0
        logits = np.zeros((200, self.vocab_size), dtype=np.float32)
        for row in range(200):
            logits[row, wanted] = 10.0
        return [bytearray(4), bytearray(4), bytearray(logits.tobytes())]


@pytest.fixture()
def model_dir(tmp_path, vlm):
    """A miniature 4v model: small vocabulary, small embedding table."""
    directory = tmp_path / "vlm"
    directory.mkdir()
    vocab = {"这张": 1, "图片": 2, "是": 3, "描述": 4, "<end>": 5}
    (directory / "tokenizer.json").write_text(
        '{"model": {"vocab": %s}}' % str(vocab).replace("'", '"'),
        encoding="utf-8")
    rows = max(max(vlm.PRE_TEMPLATE), max(vlm.POST_TEMPLATE), 8) + 1
    table = np.arange(rows * vlm.EMB_DIM, dtype=np.float32)
    (directory / "token_emb.bin").write_bytes(table.tobytes())
    return directory, vocab


@pytest.fixture()
def image(tmp_path):
    from PIL import Image
    path = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), (20, 90, 180)).save(path)
    return path


def _model(worker, vlm, model_dir, script, vocab_size=8):
    directory, _vocab = model_dir
    instance = worker.VisionModel.__new__(worker.VisionModel)
    instance.session = FakeSession(vocab_size, script)
    instance.vocab = vlm.VocabTable.from_tokenizer_json(
        directory / "tokenizer.json")
    instance.embeddings = vlm.RowTable(directory / "token_emb.bin",
                                       vlm.EMB_DIM)
    return instance


def test_generation_stops_at_the_end_token(
        worker, vlm, model_dir, image, monkeypatch) -> None:
    monkeypatch.setattr(vlm, "END_TOKEN", 5)
    model = _model(worker, vlm, model_dir, [1, 2, 5])

    result = model.look(image, "描述", max_new=10)

    assert result["stopped"] == "end token"
    assert result["tokens"] == 2, "the end token is not part of the answer"
    assert result["text"] == "这张图片"
    model.close()


def test_generation_stops_at_the_cap_and_says_so(
        worker, vlm, model_dir, image, monkeypatch) -> None:
    monkeypatch.setattr(vlm, "END_TOKEN", 5)
    model = _model(worker, vlm, model_dir, [1, 1, 1, 1])

    result = model.look(image, "描述", max_new=3)

    assert result["stopped"] == "length"
    assert result["tokens"] == 3
    model.close()


def test_the_image_is_preprocessed_once_not_once_per_token(
        worker, vlm, model_dir, image, monkeypatch) -> None:
    """Vision and resample are paid per image; only prefill repeats."""
    monkeypatch.setattr(vlm, "END_TOKEN", 5)
    model = _model(worker, vlm, model_dir, [1, 2, 3, 5])

    model.look(image, "描述", max_new=10)

    calls = model.session.calls
    assert calls.count(worker.VISION) == 1
    assert calls.count(worker.RESAMPLE) == 1
    assert calls.count(worker.PREFILL) == 4, "one prefill per token"
    model.close()


def test_each_token_is_published_as_it_lands(
        worker, vlm, model_dir, image, monkeypatch) -> None:
    monkeypatch.setattr(vlm, "END_TOKEN", 5)
    model = _model(worker, vlm, model_dir, [1, 2, 5])
    seen = []

    model.look(image, "描述", max_new=10,
               on_token=lambda text, count: seen.append((count, text)))

    assert seen == [(1, "这张"), (2, "这张图片")]
    model.close()


def test_a_publisher_that_raises_does_not_lose_the_answer(
        worker, vlm, model_dir, image, monkeypatch) -> None:
    """Progress is a convenience; the description is the deliverable."""
    monkeypatch.setattr(vlm, "END_TOKEN", 5)
    model = _model(worker, vlm, model_dir, [1, 2, 5])

    def explode(text, count):
        raise OSError("queue vanished")

    result = model.look(image, "描述", max_new=10, on_token=explode)

    assert result["text"] == "这张图片"
    model.close()


def test_a_full_window_ends_generation_rather_than_looping(
        worker, vlm, model_dir, image, monkeypatch) -> None:
    """build_prefill_inputs refuses to overflow; the loop must stop, not spin."""
    monkeypatch.setattr(vlm, "END_TOKEN", 5)
    monkeypatch.setattr(vlm, "TOTAL_PREFILL_LEN", 200)
    model = _model(worker, vlm, model_dir, [1] * 400)
    # A question long enough that the window fills after a few tokens.
    monkeypatch.setattr(vlm, "MAX_TEXT_LEN", 118)
    long_question = "描述" * 200

    result = model.look(image, long_question, max_new=300)

    assert result["stopped"] == "window full"
    model.close()


# ------------------------------------------------------------------- serving

def test_serve_claims_finishes_and_returns(
        worker, tmp_path, monkeypatch, image) -> None:
    jobs = _module("vision_jobs_worker_test", "vision_jobs.py")
    queue = jobs.VisionQueue(tmp_path / "q")
    queue.submit(image, "描述这张图片。")

    class OneShotModel:
        def look(self, path, question, max_new, on_token=None):
            on_token("这张", 1)
            return {"text": "这张图片是一张柱状图。", "tokens": 8,
                    "stopped": "end token", "alternatives": [],
                    "vocab_size": 8}

        def close(self):
            pass

    monkeypatch.setattr(worker, "VisionSession",
                        lambda *a, **k: type("S", (), {"close": lambda s: None})())
    monkeypatch.setattr(worker, "VisionModel", lambda *a, **k: OneShotModel())
    monkeypatch.setattr(worker, "vision_jobs", jobs)

    assert worker.serve(tmp_path / "q", tmp_path, "/bin/true", [],
                        poll=0, once=True) == 0

    done = queue.collect()
    assert [job.state for job in done] == ["done"]
    assert done[0].answer == "这张图片是一张柱状图。"
    assert done[0].elapsed_seconds is not None


def test_a_model_that_raises_fails_the_job_instead_of_the_worker(
        worker, tmp_path, monkeypatch, image) -> None:
    jobs = _module("vision_jobs_worker_test", "vision_jobs.py")
    queue = jobs.VisionQueue(tmp_path / "q")
    queue.submit(image, "描述")

    class Broken:
        def look(self, *a, **k):
            raise RuntimeError("handle refused the image")

        def close(self):
            pass

    monkeypatch.setattr(worker, "VisionSession",
                        lambda *a, **k: type("S", (), {"close": lambda s: None})())
    monkeypatch.setattr(worker, "VisionModel", lambda *a, **k: Broken())
    monkeypatch.setattr(worker, "vision_jobs", jobs)

    assert worker.serve(tmp_path / "q", tmp_path, "/bin/true", [],
                        poll=0, once=True) == 0

    failed = queue.collect()
    assert [job.state for job in failed] == ["failed"]
    assert "refused the image" in failed[0].error
    assert failed[0].answer is None


def test_serve_returns_on_an_empty_queue_when_asked_for_one_job(
        worker, tmp_path, monkeypatch) -> None:
    jobs = _module("vision_jobs_worker_test", "vision_jobs.py")
    monkeypatch.setattr(worker, "VisionSession",
                        lambda *a, **k: type("S", (), {"close": lambda s: None})())
    monkeypatch.setattr(worker, "VisionModel",
                        lambda *a, **k: type("M", (), {"close": lambda s: None})())
    monkeypatch.setattr(worker, "vision_jobs", jobs)

    assert worker.serve(tmp_path / "q", tmp_path, "/bin/true", [],
                        poll=0, once=True) == 0


def test_the_token_cap_is_a_documented_default(worker) -> None:
    """Every token is a whole prefill, so this is a latency budget."""
    assert worker.MAX_NEW == 40
    assert worker.MODELS == ("vision.om", "resample.om", "prefill_decode.om")
    assert "decode.om" not in worker.MODELS, "53+49 ports exceeds the cap of 32"


# ------------------------------------------------------------------ shutdown

def test_close_sends_a_shutdown_frame(worker, runner, tmp_path) -> None:
    """The executor holds 543 MB of handles; it is asked to leave, not killed."""
    session = _session(worker, runner, b"", [4], tmp_path)

    session.close()

    frame = session.process.stdin.getvalue()
    magic, _v, op, _m, _i, _o, _w = runner._PERSISTENT_REQUEST.unpack(
        frame[:runner._PERSISTENT_REQUEST.size])
    assert magic == runner.PERSISTENT_REQUEST_MAGIC
    assert op == runner.PERSISTENT_OP_SHUTDOWN
    assert not session.process.killed


def test_close_kills_an_executor_that_will_not_take_the_frame(
        worker, runner, tmp_path) -> None:
    """A wedged executor must not hold the worker open."""
    session = _session(worker, runner, b"", [4], tmp_path)

    def refuse(*_args, **_kwargs):
        raise OSError("broken pipe")

    session.process.stdin.write = refuse

    session.close()

    assert session.process.killed


def test_main_wires_the_command_line_to_serve(worker, monkeypatch) -> None:
    seen = {}

    def fake_serve(queue, model_dir, executable, library_paths, poll, once,
                   max_new):
        seen.update(queue=str(queue), model_dir=str(model_dir),
                    executable=str(executable),
                    library_paths=[str(p) for p in library_paths],
                    poll=poll, once=once, max_new=max_new)
        return 0

    monkeypatch.setattr(worker, "serve", fake_serve)

    rc = worker.main([
        "--queue", "/tmp/q", "--model-dir", "/tmp/vlm",
        "--executable", "/tmp/exe",
        "--library-path", "/opt/lib", "--library-path", "/opt/lib/npu",
        "--poll-seconds", "0.5", "--once", "--max-new", "12"])

    assert rc == 0
    assert seen["queue"] == "/tmp/q"
    assert seen["library_paths"] == ["/opt/lib", "/opt/lib/npu"]
    assert seen["poll"] == 0.5
    assert seen["once"] is True
    assert seen["max_new"] == 12


def test_the_token_cap_defaults_to_the_documented_budget(
        worker, monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(
        worker, "serve",
        lambda *a: seen.setdefault("max_new", a[6]) and 0 or 0)

    worker.main(["--queue", "/tmp/q", "--model-dir", "/tmp/v",
                 "--executable", "/tmp/e"])

    assert seen["max_new"] == worker.MAX_NEW
