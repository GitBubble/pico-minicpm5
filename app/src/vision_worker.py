#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The vision half of the multimodal agent: claim a job, look, answer.

This process owns the MiniCPM-4v-0.5B handles and nothing else. It polls the
job queue, runs one image through vision -> resample -> prefill, writes the
answer back and returns to polling, so the language model on the other side of
the queue is never blocked by an image.

Only three of the four published handles load. ``decode.om`` declares 53
inputs and 49 outputs -- five plus a port per layer for each of K and V --
and this SDK caps a model at 32 ports, so it is refused at load time. The
prefill handle emits logits for its whole window, so the answer available
here is the distribution at the last prompt position rather than a generated
sentence. That is a real limit of this deployment and the worker reports it
as such rather than pretending to a full decode.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import minicpm4v_vision as vlm  # noqa: E402
import pico_minicpm5_split_board_runner as runner  # noqa: E402
import probe_om_execute_latency as probe  # noqa: E402
import vision_jobs  # noqa: E402

MODELS = ("vision.om", "resample.om", "prefill_decode.om")
VISION, RESAMPLE, PREFILL = 0, 1, 2
#: Read a large tensor in pieces; the logits alone are 58 MB.
CHUNK_BYTES = 1 << 20
#: How many alternatives to report alongside the leading token.
TOP_K = 5


class WorkerError(RuntimeError):
    """The worker cannot serve jobs."""


class VisionSession:
    """One executor process holding the three loadable 4v handles."""

    def __init__(self, model_dir, executable, library_paths, timeout=1800.0):
        self.model_dir = Path(model_dir)
        self.timeout = float(timeout)
        argv = [str(Path(executable)), "--device", "0"]
        for name in MODELS:
            path = self.model_dir / name
            if not path.is_file():
                raise WorkerError(f"missing handle: {path}")
            argv += ["--model", str(path)]
        env = {"LD_LIBRARY_PATH": ":".join(str(p) for p in library_paths),
               "PATH": "/usr/bin:/bin"}
        self.process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, env=env)
        deadline = time.monotonic() + self.timeout
        descriptors = probe._read_ready(self.process.stdout, len(MODELS),
                                        deadline)
        self.inputs = {index: len(desc[0])
                       for index, desc in enumerate(descriptors)}
        self.outputs = {index: list(desc[1])
                        for index, desc in enumerate(descriptors)}

    def execute(self, model: int, writes) -> list[bytearray]:
        """One OP_EXECUTE_RESIDENT against a resident handle."""
        deadline = time.monotonic() + self.timeout
        out_sizes = self.outputs[model]
        trailer = bytearray(
            b"".join(runner._PERSISTENT_U64.pack(size) for size in out_sizes))
        for slot, blob in writes:
            trailer += runner._PERSISTENT_WRITE.pack(
                slot, runner.PERSISTENT_WRITE_FLAG_PAYLOAD, 0, len(blob))
            trailer += blob
        runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
            runner.PERSISTENT_REQUEST_MAGIC, runner.PERSISTENT_PROTOCOL_VERSION,
            runner.PERSISTENT_OP_EXECUTE_RESIDENT, model, self.inputs[model],
            len(out_sizes), len(writes)) + bytes(trailer))
        self.process.stdin.flush()

        read = lambda size: runner._read_exact_until(
            self.process.stdout, size, deadline)
        magic, _version, status, _model, count, _reserved, error_bytes = \
            runner._PERSISTENT_RESPONSE.unpack(
                read(runner._PERSISTENT_RESPONSE.size))
        if magic != runner.PERSISTENT_RESPONSE_MAGIC or status:
            message = read(error_bytes).decode("utf-8", "replace") \
                if error_bytes else ""
            raise WorkerError(f"model[{model}] failed: {message}")
        # Every size arrives before any payload; reading them interleaved
        # parses the second size out of the first tensor's bytes.
        sizes = [runner._PERSISTENT_U64.unpack(
            read(runner._PERSISTENT_U64.size))[0] for _ in range(count)]
        tensors = []
        for size in sizes:
            buffer = bytearray(size)
            view = memoryview(buffer)
            filled = 0
            while filled < size:
                chunk = read(min(CHUNK_BYTES, size - filled))
                view[filled:filled + len(chunk)] = chunk
                filled += len(chunk)
            tensors.append(buffer)
        return tensors

    def close(self) -> None:
        try:
            runner._write_all(self.process.stdin,
                              runner._PERSISTENT_REQUEST.pack(
                                  runner.PERSISTENT_REQUEST_MAGIC,
                                  runner.PERSISTENT_PROTOCOL_VERSION,
                                  runner.PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0))
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.process.kill()


class VisionModel:
    """Image plus question in, a described answer out."""

    def __init__(self, session: VisionSession, model_dir) -> None:
        self.session = session
        directory = Path(model_dir)
        self.vocab = vlm.VocabTable.from_tokenizer_json(
            directory / "tokenizer.json")
        self.embeddings = vlm.RowTable(directory / "token_emb.bin",
                                       vlm.EMB_DIM)

    def look(self, image_path, question: str) -> dict:
        staged = vlm.preprocess_image(image_path)
        hidden = self.session.execute(VISION, [(0, bytes(staged.data))])[0]
        resampled = self.session.execute(RESAMPLE, [(0, bytes(hidden))])[0]
        tokens = np.frombuffer(bytes(resampled), dtype=np.float32).reshape(
            vlm.VISION_TOKEN_LEN, vlm.EMB_DIM)

        built = vlm.build_prefill_inputs(tokens, question, self.vocab,
                                         self.embeddings)
        outputs = self.session.execute(PREFILL, [
            (0, built.inputs_embeds.tobytes()),
            (1, built.attention_mask.tobytes()),
        ])
        # The K/V tensors are not needed without the decode handle; drop them
        # before parsing 58 MB of logits so both are never resident at once.
        logits_bytes = bytes(outputs[2])
        outputs.clear()
        logits = np.frombuffer(logits_bytes, dtype=np.float32)
        vocab_size = logits.size // vlm.TOTAL_PREFILL_LEN
        row = logits.reshape(vlm.TOTAL_PREFILL_LEN, vocab_size)[
            built.prefill_len - 1]
        ranked = np.argsort(row)[-TOP_K:][::-1]
        return {
            "leading_token": int(ranked[0]),
            "leading_text": self.vocab.decode([int(ranked[0])]),
            "alternatives": [self.vocab.decode([int(index)])
                             for index in ranked],
            "prefill_len": built.prefill_len,
            "vocab_size": int(vocab_size),
        }

    def close(self) -> None:
        self.embeddings.close()


def describe(result: dict) -> str:
    """Render what this deployment can actually claim to have seen."""
    alternatives = " / ".join(result["alternatives"])
    return (f"{result['leading_text'].strip()}"
            f"（视觉模型的首词分布：{alternatives}）")


def serve(queue_root, model_dir, executable, library_paths, poll=2.0,
          once=False) -> int:
    queue = vision_jobs.VisionQueue(queue_root)
    session = VisionSession(model_dir, executable, library_paths)
    model = VisionModel(session, model_dir)
    print(f"vision_worker=ready handles={len(MODELS)} queue={queue.root}",
          flush=True)
    try:
        while True:
            job = queue.claim()
            if job is None:
                if once:
                    return 0
                time.sleep(poll)
                continue
            started = time.time()
            try:
                result = model.look(job.image_path, job.question)
            except Exception as error:                       # noqa: BLE001
                queue.fail(job, f"{type(error).__name__}: {error}")
                print(f"vision_worker job={job.job_id} failed: {error}",
                      flush=True)
            else:
                elapsed = time.time() - started
                queue.finish(job, describe(result), elapsed)
                print(f"vision_worker job={job.job_id} done in "
                      f"{elapsed:.2f}s: {result['leading_text']!r}", flush=True)
            if once:
                return 0
    finally:
        model.close()
        session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--library-path", action="append", default=[],
                        type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true",
                        help="serve at most one job, then exit")
    args = parser.parse_args(argv)
    return serve(args.queue, args.model_dir, args.executable,
                 args.library_path, args.poll_seconds, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
