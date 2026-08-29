"""The agent half of the multimodal path.

The vision model runs in its own process behind a job queue, so the contract
under test here is that the agent never waits on it: ``describe_image``
returns a job id, and the description arrives on a later turn through
``report_vision``. These tests also pin the disclosure rule, because the tool
schema is a prefill cost paid on every turn that names it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "app" / "src"


def _module(name: str, filename: str):
    """Import a source module once and cache it under a test-only name."""
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
def agent():
    return _module("minicpm_agent_vision_test", "minicpm_agent.py")


@pytest.fixture()
def jobs():
    return _module("vision_jobs_agent_test", "vision_jobs.py")


@pytest.fixture()
def server():
    return _module("merged_board_server_vision_test", "merged_board_server.py")


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    return root


@pytest.fixture()
def queue(jobs, tmp_path):
    return jobs.VisionQueue(tmp_path / "vision")


@pytest.fixture()
def tools(agent, workspace, queue):
    return agent.WorkspaceTools(workspace, vision_queue=queue)


class RecordingUI:
    """Stands in for the terminal; keeps what would have been printed."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, text: str) -> None:
        self.lines.append(str(text))


# ---------------------------------------------------------------- disclosure

@pytest.mark.parametrize("query", [
    "描述一下 photo.png",
    "看看这张图 photo.png",
    "这张图片里有什么 photo.png",
    "describe photo.png",
])
def test_an_image_question_discloses_the_vision_tool(agent, query) -> None:
    assert agent.WorkspaceTools.select_schema_profile(
        query, has_vision=True) == "vision"


@pytest.mark.parametrize("query,profile", [
    ("看看这个目录", "read_only"),      # a look verb with no image
    ("看看这张图", "read_only"),        # an image with no file to look at
    ("你好，喵", "none"),
    ("列出当前目录", "read_only"),
])
def test_a_turn_without_an_image_referent_pays_nothing_for_vision(
        agent, query, profile) -> None:
    assert agent.WorkspaceTools.select_schema_profile(
        query, has_vision=True) == profile


def test_vision_is_not_disclosed_where_no_worker_can_serve_it(agent) -> None:
    """The schema is charged in prefill tokens whether or not it can work.

    A deployment with no vision worker falls back to the read set, which can
    at least report what the file is, instead of paying for a tool whose only
    possible outcome is a refusal.
    """
    assert agent.WorkspaceTools.select_schema_profile(
        "描述一下 photo.png", has_vision=False) == "read_only"


def test_the_default_is_no_vision_so_disclosure_is_opt_in(agent) -> None:
    assert agent.WorkspaceTools.select_schema_profile(
        "描述一下 photo.png") == "read_only"


def test_the_vision_profile_is_exactly_one_tool(agent) -> None:
    """Widening it would charge every image turn for the whole read set."""
    assert agent.WorkspaceTools._PROFILES["vision"] == ("describe_image",)


# -------------------------------------------------------------------- submit

def test_describe_image_returns_a_job_id_without_waiting(
        tools, queue) -> None:
    """The agent must not block on the other model's NPU handles."""
    call = _call(tools, "describe_image",
                 {"path": "photo.png", "question": "这是什么？"})
    payload = json.loads(tools.execute(call))

    assert payload["ok"] is True
    assert payload["type"] == "vision_job"
    assert queue.pending() == 1
    submitted = queue.list("queued")[0]
    assert submitted.question == "这是什么？"
    assert Path(submitted.image_path).name == "photo.png"


def test_the_job_id_is_reported_so_a_later_answer_can_be_matched(
        tools, queue) -> None:
    payload = json.loads(tools.execute(
        _call(tools, "describe_image", {"path": "photo.png"})))

    job_id = queue.list("queued")[0].job_id
    assert job_id in payload["output"]


def test_describe_image_refuses_a_path_outside_the_workspace(tools) -> None:
    payload = json.loads(tools.execute(
        _call(tools, "describe_image", {"path": "../../etc/hosts"})))
    assert payload["ok"] is False


def test_describe_image_refuses_a_file_that_is_not_there(tools) -> None:
    payload = json.loads(tools.execute(
        _call(tools, "describe_image", {"path": "absent.png"})))
    assert payload["ok"] is False


def test_describe_image_says_so_when_no_worker_is_configured(
        agent, workspace) -> None:
    bare = agent.WorkspaceTools(workspace)
    payload = json.loads(bare.execute(
        _call(bare, "describe_image", {"path": "photo.png"})))

    assert payload["ok"] is False
    assert "no vision worker" in payload["output"]


def _call(tools, name, arguments):
    module = sys.modules[type(tools).__module__]
    return module.ToolCall(name, arguments)


# -------------------------------------------------------------------- report

def test_a_finished_description_reaches_the_user_and_the_transcript(
        server, queue, workspace) -> None:
    """This is the point of the queue: the answer arrives on a later turn."""
    queue.submit(workspace / "photo.png", "这是什么？")
    queue.finish(queue.claim(), "一只猫坐在窗台上。", elapsed=41.7)

    ui = RecordingUI()
    messages = [{"role": "system", "content": "..."}]
    reported = server.report_vision(queue, ui, messages)

    assert [job.state for job in reported] == ["done"]
    assert "一只猫坐在窗台上。" in ui.lines[0]
    assert "photo.png" in ui.lines[0]
    assert "41.7s" in ui.lines[0]
    assert messages[-1]["role"] == "system"
    assert "一只猫坐在窗台上。" in messages[-1]["content"]


def test_an_answer_is_reported_exactly_once(server, queue, workspace) -> None:
    queue.submit(workspace / "photo.png", "描述")
    queue.finish(queue.claim(), "答案", elapsed=1.0)

    messages = []
    first = server.report_vision(queue, RecordingUI(), messages)
    second = server.report_vision(queue, RecordingUI(), messages)

    assert len(first) == 1 and second == []
    assert len(messages) == 1, "a redelivered answer would be charged twice"


def test_a_failed_job_is_reported_without_inventing_a_description(
        server, queue, workspace) -> None:
    queue.submit(workspace / "photo.png", "描述")
    queue.fail(queue.claim(), "vision handle refused the image")

    ui = RecordingUI()
    messages = []
    server.report_vision(queue, ui, messages)

    assert "failed" in ui.lines[0] and "refused" in ui.lines[0]
    assert messages == [], "a failure must not enter the transcript as fact"


def test_reporting_is_a_no_op_without_a_queue(server) -> None:
    messages = []
    assert server.report_vision(None, RecordingUI(), messages) == []
    assert messages == []


def test_a_broken_queue_does_not_take_the_conversation_down(
        server, queue) -> None:
    """Accounting for the other model must never end a turn."""
    class Exploding:
        def collect(self):
            raise OSError("queue directory vanished")

    ui = RecordingUI()
    assert server.report_vision(Exploding(), ui, []) == []
    assert "unreadable" in ui.lines[0]


def test_nothing_is_reported_while_the_job_is_still_running(
        server, queue, workspace) -> None:
    queue.submit(workspace / "photo.png", "描述")
    queue.claim()

    assert server.report_vision(queue, RecordingUI(), []) == []
    assert queue.pending() == 1


# ------------------------------------------------------------------ progress

def test_progress_publishes_partial_text_without_finishing(
        queue, workspace) -> None:
    """Half a second per token is unavoidable; hiding it is not."""
    queue.submit(workspace / "photo.png", "描述")
    claimed = queue.claim()

    queue.progress(claimed, "这张图片", 4)

    live = queue.watching()
    assert [job.partial for job in live] == ["这张图片"]
    assert live[0].tokens == 4
    assert live[0].state == "claimed"
    assert queue.collect() == [], "partial work is not a delivery"


def test_progress_is_overwritten_not_appended(queue, workspace) -> None:
    queue.submit(workspace / "photo.png", "描述")
    claimed = queue.claim()

    queue.progress(claimed, "这张", 2)
    queue.progress(claimed, "这张图片展示", 6)

    assert queue.watching()[0].partial == "这张图片展示"


def test_finishing_clears_the_partial_so_it_cannot_be_shown_twice(
        queue, workspace) -> None:
    queue.submit(workspace / "photo.png", "描述")
    claimed = queue.claim()
    moving = queue.progress(claimed, "这张图片", 4)

    done = queue.finish(moving, "这张图片展示了一个界面。", elapsed=13.2)

    assert done.partial is None
    assert done.answer == "这张图片展示了一个界面。"
    assert queue.watching() == []


def test_watching_is_empty_before_a_worker_claims(queue, workspace) -> None:
    queue.submit(workspace / "photo.png", "描述")
    assert queue.watching() == []


# --------------------------------------------------------------------- input

def test_the_prompt_polls_the_queue_instead_of_blocking(
        server, queue, workspace, monkeypatch, capsys) -> None:
    """The answer becomes ready while the user sits at the prompt.

    A blocking read would show nothing until they typed again, so the wait
    has to be a poll that can draw in the meantime.
    """
    queue.submit(workspace / "photo.png", "描述")
    claimed = queue.claim()
    queue.progress(claimed, "这张图片展示", 6)

    calls = {"select": 0}

    class FakeStdin:
        def isatty(self):
            return True

        def readline(self):
            return "你好\n"

    def fake_select(rlist, _w, _x, _timeout):
        calls["select"] += 1
        # Idle for two wakeups, then the user types.
        return (rlist, [], []) if calls["select"] > 2 else ([], [], [])

    monkeypatch.setattr(server.sys, "stdin", FakeStdin())
    monkeypatch.setattr(server.select, "select", fake_select)
    ui = RecordingUI()
    ui.paint = lambda text, _code: text

    line = server.watch_vision(queue, ui, "You > ", timeout=0)

    assert line == "你好\n"
    assert calls["select"] == 3
    drawn = capsys.readouterr().out
    assert "这张图片展示" in drawn, "progress must reach the screen while idle"
    assert "6 词" in drawn


def test_a_plain_read_is_used_when_there_is_no_queue(
        server, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
    assert server.watch_vision(None, RecordingUI(), "You > ") == "hello"


def test_a_pipe_does_not_get_an_animation(server, queue, monkeypatch) -> None:
    """Without a tty there is nothing to redraw; recordings must stay clean."""
    class PipedStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(server.sys, "stdin", PipedStdin())
    monkeypatch.setattr("builtins.input", lambda prompt="": "piped")

    assert server.watch_vision(queue, RecordingUI(), "You > ") == "piped"
