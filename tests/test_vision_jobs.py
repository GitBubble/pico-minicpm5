from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "app" / "src" / "vision_jobs.py"


def _jobs():
    """Load once and cache.

    A second load defines a second ``JobError`` class, and a fixture that
    raised the first one would never match ``pytest.raises`` against the
    second -- the failure looks like the guard did not fire when in fact it
    fired with the wrong identity.
    """
    cached = sys.modules.get("vision_jobs_test")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("vision_jobs_test",
                                                  MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def queue(tmp_path):
    return _jobs().VisionQueue(tmp_path / "vision")


@pytest.fixture()
def image(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return path


def test_submitting_returns_immediately_and_leaves_a_queued_record(
        queue, image) -> None:
    """The agent posts and carries on; the worker does the looking."""
    job = queue.submit(image, "描述这张图片。")

    assert job.state == "queued"
    assert len(job.job_id) == 12
    assert queue.pending() == 1
    assert queue.collect() == [], "nothing is ready to report yet"

    stored = json.loads((queue.root / f"queued.{job.job_id}.json").read_text())
    assert stored["schema"] == _jobs().SCHEMA
    assert stored["question"] == "描述这张图片。"


def test_submit_fails_closed_on_a_missing_image_or_empty_question(
        queue, image, tmp_path) -> None:
    module = _jobs()
    with pytest.raises(module.JobError, match="image not found"):
        queue.submit(tmp_path / "absent.png", "描述")
    with pytest.raises(module.JobError, match="must not be empty"):
        queue.submit(image, "   ")


def test_a_job_is_claimed_once_even_with_two_workers(queue, image) -> None:
    """Claiming is a rename, so the loser sees an empty queue."""
    queue.submit(image, "one")

    first = queue.claim()
    second = queue.claim()

    assert first is not None and first.state == "claimed"
    assert first.claimed is not None
    assert second is None
    assert not (queue.root / f"queued.{first.job_id}.json").exists()


def test_claiming_takes_the_oldest_first(queue, image) -> None:
    older = queue.submit(image, "first")
    newer = queue.submit(image, "second")
    # Same-second timestamps must still order deterministically.
    assert {older.job_id, newer.job_id} == {
        job.job_id for job in queue.list("queued")}

    claimed = queue.claim()
    assert claimed.question in ("first", "second")
    assert queue.pending() == 2, "one claimed and one still queued"


def test_finishing_moves_the_job_and_carries_the_answer(queue, image) -> None:
    queue.submit(image, "描述")
    claimed = queue.claim()

    done = queue.finish(claimed, "一只猫坐在窗台上。", elapsed=41.7)

    assert done.state == "done"
    assert done.answer == "一只猫坐在窗台上。"
    assert done.elapsed_seconds == 41.7
    assert not (queue.root / f"claimed.{done.job_id}.json").exists()
    assert queue.pending() == 0


def test_a_failed_job_records_why_and_does_not_pretend_to_have_an_answer(
        queue, image) -> None:
    queue.submit(image, "描述")
    claimed = queue.claim()

    failed = queue.fail(claimed, "vision handle refused the image")

    assert failed.state == "failed"
    assert failed.answer is None
    assert "refused" in failed.error
    assert queue.pending() == 0


def test_collect_surfaces_each_finished_job_exactly_once(queue, image) -> None:
    """A restart between finishing and reading must not replay an answer."""
    queue.submit(image, "描述")
    queue.finish(queue.claim(), "答案", elapsed=1.0)

    first = queue.collect()
    second = queue.collect()

    assert [job.answer for job in first] == ["答案"]
    assert second == []


def test_collect_reports_failures_alongside_answers(queue, image) -> None:
    queue.submit(image, "a")
    queue.fail(queue.claim(), "boom")
    queue.submit(image, "b")
    queue.finish(queue.claim(), "ok", elapsed=2.0)

    collected = queue.collect()

    assert {job.state for job in collected} == {"done", "failed"}


def test_a_long_answer_is_clipped_to_the_context_budget(queue, image) -> None:
    module = _jobs()
    queue.submit(image, "描述")

    done = queue.finish(queue.claim(), "x" * 9000, elapsed=1.0)

    assert len(done.answer) == module.MAX_ANSWER_CHARS


def test_get_finds_a_job_in_any_state_and_rejects_a_malformed_id(
        queue, image) -> None:
    module = _jobs()
    job = queue.submit(image, "描述")

    assert queue.get(job.job_id).state == "queued"
    queue.claim()
    assert queue.get(job.job_id).state == "claimed"
    assert queue.get("0" * 12) is None
    with pytest.raises(module.JobError, match="malformed"):
        queue.get("../etc/passwd")


def test_a_foreign_json_file_in_the_queue_is_ignored(queue, image) -> None:
    queue.submit(image, "描述")
    (queue.root / "queued.deadbeefcafe.json").write_text(
        json.dumps({"schema": "something.else.v1"}), encoding="utf-8")

    # The stray file is skipped rather than crashing the listing.
    assert len(queue.list("queued")) == 1


def test_a_crashed_worker_leaves_the_job_visible_not_lost(queue, image) -> None:
    """A claimed job that never finishes is still on disk to be requeued."""
    queue.submit(image, "描述")
    claimed = queue.claim()

    # The worker dies here: no finish, no fail.
    assert queue.pending() == 1
    assert queue.get(claimed.job_id).state == "claimed"
    assert queue.collect() == []
