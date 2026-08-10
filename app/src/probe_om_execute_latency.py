#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time raw svp_acl_mdl_execute for arbitrary OMs via the persistent executor.

Unlike ``profile_minicpm5_segment_latency.py`` this takes a bare list of OM
paths and needs no runner config, so it can compare candidate schedules (for
example a fused single-layer OM against the A/R/B/C split) directly.

Inputs are zero-filled to the descriptor's public prefix: only the
write/wait/read split is meaningful, not the numerics.  Run it ON THE BOARD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pico_minicpm5_split_board_runner as runner  # noqa: E402


def _start(executable: Path, models: Sequence[Path], library_paths: Sequence[Path],
           device: int, *, quiet: bool = False) -> subprocess.Popen:
    command = [str(executable), "--device", str(device)]
    for model in models:
        command.extend(("--model", str(model)))
    environment = os.environ.copy()
    configured = ":".join(str(path) for path in library_paths)
    inherited = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(
        value for value in (configured, inherited) if value)
    return subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else None, env=environment)


def _read_ready(stream, model_count: int, deadline: float):
    read = lambda size: runner._read_exact_until(stream, size, deadline)
    magic, version, status, count, error_bytes = runner._PERSISTENT_READY.unpack(
        read(runner._PERSISTENT_READY.size))
    if magic != runner.PERSISTENT_READY_MAGIC or status:
        message = read(error_bytes).decode("utf-8", "replace") if error_bytes else ""
        raise RuntimeError(f"executor startup failed: {message}")
    if count != model_count:
        raise RuntimeError(f"model table {count} != requested {model_count}")
    descriptors = []
    for _ in range(count):
        n_in, n_out = runner._PERSISTENT_COUNTS.unpack(
            read(runner._PERSISTENT_COUNTS.size))
        inputs = tuple(runner._PERSISTENT_U64.unpack(
            read(runner._PERSISTENT_U64.size))[0] for _ in range(n_in))
        outputs = tuple(runner._PERSISTENT_U64.unpack(
            read(runner._PERSISTENT_U64.size))[0] for _ in range(n_out))
        descriptors.append((inputs, outputs))
    return descriptors


def run(*, executable: Path, models: Sequence[Path], library_paths: Sequence[Path],
        repeats: int, public_inputs: int | None, public_outputs: int | None,
        device: int, timeout: float, output: Path | None,
        input_dir: Path | None = None,
        raw_output_dir: Path | None = None) -> dict[str, Any]:
    if (input_dir is not None or raw_output_dir is not None) and len(models) != 1:
        raise ValueError("raw input/output capture requires exactly one model")
    process = _start(executable, models, library_paths, device)
    assert process.stdin is not None and process.stdout is not None
    deadline = time.monotonic() + timeout
    descriptors = _read_ready(process.stdout, len(models), deadline)
    records = []
    try:
        for index, model in enumerate(models):
            desc_in, desc_out = descriptors[index]
            n_in = len(desc_in) if public_inputs is None else min(
                public_inputs, len(desc_in))
            n_out = len(desc_out) if public_outputs is None else min(
                public_outputs, len(desc_out))
            if input_dir is None:
                payloads = tuple(bytes(size) for size in desc_in[:n_in])
            else:
                payloads = tuple(
                    (input_dir / f"input.{slot}.bin").read_bytes()
                    for slot in range(n_in))
                for slot, (payload, size) in enumerate(
                        zip(payloads, desc_in[:n_in])):
                    if len(payload) != size:
                        raise ValueError(
                            f"input {slot} has {len(payload)} bytes, expected {size}")
            out_sizes = tuple(desc_out[:n_out])
            samples = []
            for _ in range(repeats):
                header = runner._PERSISTENT_REQUEST.pack(
                    runner.PERSISTENT_REQUEST_MAGIC,
                    runner.PERSISTENT_PROTOCOL_VERSION,
                    runner.PERSISTENT_OP_EXECUTE, index, n_in, n_out, 0)
                start = time.perf_counter()
                runner._write_all(process.stdin, header)
                for size in (*[len(p) for p in payloads], *out_sizes):
                    runner._write_all(
                        process.stdin, runner._PERSISTENT_U64.pack(size))
                for payload in payloads:
                    runner._write_all(process.stdin, payload)
                process.stdin.flush()
                wrote = time.perf_counter()
                limit = time.monotonic() + timeout
                read = lambda size: runner._read_exact_until(
                    process.stdout, size, limit)
                frame = read(runner._PERSISTENT_RESPONSE.size)
                (magic, _v, status, midx, out_count,
                 error_bytes, _r) = runner._PERSISTENT_RESPONSE.unpack(frame)
                if status:
                    message = read(error_bytes).decode("utf-8", "replace")
                    raise RuntimeError(f"{model.name} failed: {message}")
                waited = time.perf_counter()
                for _ in range(out_count):
                    read(runner._PERSISTENT_U64.size)
                raw_outputs = tuple(read(size) for size in out_sizes)
                done = time.perf_counter()
                samples.append((wrote - start, waited - wrote, done - waited))
            if raw_output_dir is not None:
                raw_output_dir.mkdir(parents=True, exist_ok=True)
                for slot, payload in enumerate(raw_outputs):
                    (raw_output_dir / f"out.{slot}.bin").write_bytes(payload)
            write_ms = statistics.median(s[0] * 1000 for s in samples)
            wait_ms = statistics.median(s[1] * 1000 for s in samples)
            read_ms = statistics.median(s[2] * 1000 for s in samples)
            records.append({
                "name": model.name, "bytes": model.stat().st_size,
                "public_inputs": n_in, "public_outputs": n_out,
                "input_bytes": sum(desc_in[:n_in]),
                "output_bytes": sum(desc_out[:n_out]),
                "write_ms": write_ms, "wait_ms": wait_ms, "read_ms": read_ms,
            })
            print(f"{model.name:<52} in={sum(desc_in[:n_in]):>9} "
                  f"write={write_ms:7.3f} wait={wait_ms:8.3f} "
                  f"read={read_ms:6.3f} ms", flush=True)
    finally:
        try:
            runner._write_all(process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0))
            process.stdin.flush()
        except Exception:
            pass
        process.terminate()
        process.wait(timeout=30)

    report = {"schema": "pico.om_execute_latency.v1", "repeats": repeats,
              "segments": records,
              "totals": {
                  "write_ms": sum(r["write_ms"] for r in records),
                  "wait_ms": sum(r["wait_ms"] for r in records),
                  "read_ms": sum(r["read_ms"] for r in records)}}
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-executor", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--library-path", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--public-inputs", type=int)
    parser.add_argument("--public-outputs", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--raw-output-dir", type=Path)
    args = parser.parse_args(argv)
    report = run(executable=args.persistent_executor, models=args.model,
                 library_paths=args.library_path, repeats=args.repeats,
                 public_inputs=args.public_inputs,
                 public_outputs=args.public_outputs, device=args.device,
                 timeout=args.timeout, output=args.output,
                 input_dir=args.input_dir,
                 raw_output_dir=args.raw_output_dir)
    totals = report["totals"]
    print()
    print(f"write {totals['write_ms']:.2f} ms | "
          f"wait {totals['wait_ms']:.2f} ms | read {totals['read_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
