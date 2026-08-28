#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A filesystem job queue between the agent and the vision model.

The two models cannot share the NPU turn by turn: the language model holds
three resident handles and the vision model holds three of its own, and a
board-side agent that blocked on the vision pipeline would stop answering for
as long as an image takes. So the agent posts a job and returns immediately,
a separate worker process claims it, and the answer is collected on a later
turn. The agent keeps talking while the image is being read.

The queue is a directory of JSON files and nothing else: no daemon to keep
alive, no socket to bind, and a job survives either process restarting. A job
moves ``queued -> claimed -> done|failed`` by atomic rename, so a crashed
worker leaves a claimed job that can be seen and requeued rather than a lost
one. Every transition is a rename within one directory, which is atomic on
the board's ext4 root.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

SCHEMA = "pico.minicpm5.vision-job.v1"
STATES = ("queued", "claimed", "done", "failed")
_JOB_ID = re.compile(r"[0-9a-f]{12}")
#: A description the agent will read back into a 1024-token context.
MAX_ANSWER_CHARS = 2000


class JobError(ValueError):
    """A job record or transition is not valid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    job_id: str
    state: str
    image_path: str
    question: str
    created: str
    claimed: str | None = None
    finished: str | None = None
    answer: str | None = None
    error: str | None = None
    elapsed_seconds: float | None = None

    def record(self) -> dict:
        payload = {
            "schema": SCHEMA,
            "job_id": self.job_id,
            "state": self.state,
            "image_path": self.image_path,
            "question": self.question,
            "created": self.created,
        }
        for name in ("claimed", "finished", "answer", "error",
                     "elapsed_seconds"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


class VisionQueue:
    """Directory-backed queue; one JSON file per job, state in the name."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str, state: str) -> Path:
        if state not in STATES:
            raise JobError(f"unknown job state {state!r}")
        return self.root / f"{state}.{job_id}.json"

    def _write(self, job: Job) -> Path:
        path = self._path(job.job_id, job.state)
        staging = path.with_suffix(".partial")
        staging.write_text(
            json.dumps(job.record(), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        os.replace(staging, path)
        return path

    def submit(self, image_path, question: str) -> Job:
        """Record a job and return at once; the worker does the looking."""
        image = Path(image_path)
        if not image.is_file():
            raise JobError(f"image not found: {image}")
        question = str(question).strip()
        if not question:
            raise JobError("question must not be empty")
        job = Job(job_id=uuid.uuid4().hex[:12], state="queued",
                  image_path=str(image.resolve()), question=question,
                  created=_now())
        self._write(job)
        return job

    def _load(self, path: Path) -> Job:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise JobError(f"unreadable job {path.name}") from error
        if payload.get("schema") != SCHEMA:
            raise JobError(f"{path.name} is not a {SCHEMA} record")
        known = {field for field in Job.__dataclass_fields__}
        return Job(**{key: value for key, value in payload.items()
                      if key in known})

    def get(self, job_id: str) -> Job | None:
        if not _JOB_ID.fullmatch(str(job_id)):
            raise JobError(f"malformed job id {job_id!r}")
        for state in STATES:
            path = self._path(job_id, state)
            if path.is_file():
                return self._load(path)
        return None

    def list(self, state: str | None = None) -> list[Job]:
        pattern = f"{state}.*.json" if state else "*.json"
        jobs = []
        for path in sorted(self.root.glob(pattern)):
            if path.suffix == ".partial":
                continue
            try:
                jobs.append(self._load(path))
            except JobError:
                continue
        return sorted(jobs, key=lambda job: job.created)

    def claim(self) -> Job | None:
        """Take the oldest queued job, if any. Claiming is a rename, so two
        workers cannot take the same job."""
        for job in self.list("queued"):
            source = self._path(job.job_id, "queued")
            claimed = Job(**{**job.__dict__, "state": "claimed",
                             "claimed": _now()})
            target = self._path(job.job_id, "claimed")
            try:
                os.rename(source, target)
            except OSError:
                continue          # another worker won the race
            # The rename put the record at the claimed path; rewrite it so
            # the file's contents carry the new state and timestamp.
            self._write(claimed)
            return claimed
        return None

    def finish(self, job: Job, answer: str, elapsed: float) -> Job:
        done = Job(**{**job.__dict__, "state": "done", "finished": _now(),
                      "answer": str(answer)[:MAX_ANSWER_CHARS],
                      "elapsed_seconds": round(float(elapsed), 2)})
        self._write(done)
        self._path(job.job_id, "claimed").unlink(missing_ok=True)
        return done

    def fail(self, job: Job, error: str) -> Job:
        failed = Job(**{**job.__dict__, "state": "failed", "finished": _now(),
                        "error": str(error)[:MAX_ANSWER_CHARS]})
        self._write(failed)
        self._path(job.job_id, "claimed").unlink(missing_ok=True)
        return failed

    def collect(self) -> list[Job]:
        """Finished jobs the agent has not been shown yet.

        Reporting is a rename into ``.seen``, so a job is surfaced once even
        if the agent restarts between finishing and reading.
        """
        ready = []
        for state in ("done", "failed"):
            for job in self.list(state):
                path = self._path(job.job_id, state)
                seen = path.with_name(path.name + ".seen")
                try:
                    os.rename(path, seen)
                except OSError:
                    continue
                ready.append(job)
        return sorted(ready, key=lambda job: job.finished or job.created)

    def pending(self) -> int:
        return len(self.list("queued")) + len(self.list("claimed"))
