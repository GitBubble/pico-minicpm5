#!/usr/bin/env python3
"""Drive an interactive agent session on the board and record it with timings.

Writes a cast file: one JSON line per output chunk, [elapsed_seconds, text].
Input is fed after the prompt marker appears, so the recording tracks the real
board latency rather than a scripted delay.
"""
from __future__ import annotations

import json
import os
import pty
import re
import select
import subprocess
import sys
import time

BOARD = "root@192.168.137.100"
REMOTE = (
    "cd /root/minicpm5_opt_resident_scatter && "
    "TERM=xterm-256color PICO_MINICPM5_COLOR=always MAX_NEW=96 ./agent.sh"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
PROMPT_RE = re.compile(r"You\s*❯")


def _plain(text: str) -> str:
    return ANSI_RE.sub("", text)


def record(turns: list[str], out_path: str, idle_quit: float = 240.0) -> None:
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    proc = subprocess.Popen(
        ["ssh", "-tt", "-o", "ConnectTimeout=10", BOARD, REMOTE],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)

    started = time.time()
    frames: list[tuple[float, str]] = []
    pending = list(turns)
    buffer = ""
    last_out = time.time()
    sent_at = 0.0

    try:
        while True:
            ready, _, _ = select.select([master], [], [], 0.25)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                text = chunk.decode("utf-8", "replace")
                frames.append((round(time.time() - started, 3), text))
                buffer += text
                last_out = time.time()
                sys.stdout.write(text)
                sys.stdout.flush()

            # Feed the next turn once the REPL is idle at its prompt.
            if pending and PROMPT_RE.search(_plain(buffer)[-200:]) \
                    and time.time() - last_out > 1.2 \
                    and time.time() - sent_at > 3.0:
                turn = pending.pop(0)
                # Type it in, one character at a time: keeps multi-byte input
                # intact for the remote readline and reads as real typing.
                for char in turn:
                    os.write(master, char.encode("utf-8"))
                    time.sleep(0.055)
                    while select.select([master], [], [], 0)[0]:
                        echo = os.read(master, 65536)
                        if not echo:
                            break
                        text = echo.decode("utf-8", "replace")
                        frames.append((round(time.time() - started, 3), text))
                        sys.stdout.write(text)
                        sys.stdout.flush()
                time.sleep(0.35)
                os.write(master, b"\r")
                sent_at = last_out = time.time()
                buffer = ""

            if not pending and time.time() - last_out > 20.0:
                break
            if time.time() - started > idle_quit:
                break
            if proc.poll() is not None:
                break
    finally:
        try:
            os.close(master)
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    with open(out_path, "w", encoding="utf-8") as handle:
        for at, text in frames:
            handle.write(json.dumps([at, text], ensure_ascii=False) + "\n")
    print(f"\n\n[recorded {len(frames)} frames, "
          f"{frames[-1][0] if frames else 0:.1f}s -> {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    out = sys.argv[1]
    record(sys.argv[2:], out)
