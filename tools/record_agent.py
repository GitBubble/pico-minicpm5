#!/usr/bin/env python3
"""Drive an interactive agent session on the board and record it with timings.

Writes a cast file: one JSON line per output chunk, [elapsed_seconds, text].
Input is fed after the prompt marker appears, so the recording tracks the real
board latency rather than a scripted delay.
"""
from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time

BOARD = os.environ.get("PICO_DEMO_BOARD", "root@192.168.137.100")
REMOTE = os.environ.get("PICO_DEMO_REMOTE") or (
    "cd /root/minicpm5_opt_resident_scatter && "
    "TERM=xterm-256color PICO_MINICPM5_COLOR=always MAX_NEW=96 ./agent.sh"
)
COLS = int(os.environ.get("PICO_DEMO_COLS", "100"))
LINES = int(os.environ.get("PICO_DEMO_LINES", "28"))
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
PROMPT_RE = re.compile(r"You\s*❯")
# write_file and run_shell ask before every call. A recording that
# cannot answer would stall at the first one.
APPROVE_RE = re.compile(r"Allow once\?\s*\[y/N\]")


def _plain(text: str) -> str:
    return ANSI_RE.sub("", text)


def record(turns: list[str], out_path: str, idle_quit: float = 240.0) -> None:
    master, slave = pty.openpty()
    # Size the pty before the remote starts, so the board's UI wraps at the
    # same width the renderer will draw. Without this the remote assumes 80
    # columns and every long line breaks in the wrong place.
    fcntl.ioctl(slave, termios.TIOCSWINSZ,
                struct.pack("HHHH", LINES, COLS, 0, 0))
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

            tail = _plain(buffer)[-200:]
            if APPROVE_RE.search(tail) and time.time() - last_out > 0.4:
                os.write(master, b"y\r")
                buffer, last_out = "", time.time()
                continue

            # Feed the next turn once the REPL is idle at its prompt.
            if pending and PROMPT_RE.search(tail) \
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
