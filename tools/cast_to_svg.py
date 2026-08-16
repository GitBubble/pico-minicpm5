#!/usr/bin/env python3
"""Render a timestamped terminal cast into a self-contained animated SVG.

No external assets, no script: a CSS keyframe per snapshot toggles visibility,
so the result renders inside a GitHub README via a plain <img>.

Idle stretches are compressed by a stated factor and the compression is drawn
on screen, so the animation never implies the board was faster than it was.
"""
from __future__ import annotations

import html
import json
import re
import sys

COLS, ROWS = 96, 22
CELL_W, CELL_H = 8.4, 19.0
PAD_X, PAD_Y = 14.0, 34.0

# Terminal palette (dark). 256-colour entries used by the app are mapped here.
PALETTE = {
    "fg": "#d5dbe5", "bg": "#12151c", "dim": "#7c8797",
    45: "#22d3ee", 75: "#6aa9f4", 114: "#7ddba0", 141: "#b39cf5",
    244: "#7c8797", 250: "#aab4c2", 203: "#f2707a", 179: "#e0b568",
}
SGR_RE = re.compile(r"\x1b\[([0-9;?]*)([a-zA-Z])")


class Screen:
    def __init__(self) -> None:
        self.buf = [[(" ", "fg", False) for _ in range(COLS)] for _ in range(ROWS)]
        self.row = self.col = 0
        self.colour = "fg"
        self.bold = False

    def _scroll(self) -> None:
        self.buf.pop(0)
        self.buf.append([(" ", "fg", False) for _ in range(COLS)])
        self.row = ROWS - 1

    def write(self, text: str) -> None:
        index = 0
        while index < len(text):
            match = SGR_RE.match(text, index)
            if match:
                self._escape(match.group(1), match.group(2))
                index = match.end()
                continue
            char = text[index]
            index += 1
            if char == "\r":
                self.col = 0
            elif char == "\n":
                self.row += 1
                if self.row >= ROWS:
                    self._scroll()
            elif char == "\b":
                self.col = max(0, self.col - 1)
            elif char == "\x1b":
                continue
            elif char >= " ":
                if self.col >= COLS:
                    self.col, self.row = 0, self.row + 1
                    if self.row >= ROWS:
                        self._scroll()
                self.buf[self.row][self.col] = (char, self.colour, self.bold)
                self.col += 1

    def _escape(self, params: str, final: str) -> None:
        args = [int(p) for p in params.split(";") if p.isdigit()]
        if final == "m":
            if not args:
                self.colour, self.bold = "fg", False
            skip = 0
            for position, value in enumerate(args):
                if skip:
                    skip -= 1
                    continue
                if value == 0:
                    self.colour, self.bold = "fg", False
                elif value == 1:
                    self.bold = True
                elif value == 2:
                    self.colour = "dim"
                elif value == 38 and args[position + 1:position + 2] == [5]:
                    self.colour = args[position + 2] if position + 2 < len(args) else "fg"
                    skip = 2
        elif final == "A":
            self.row = max(0, self.row - max(1, args[0] if args else 1))
        elif final == "B":
            self.row = min(ROWS - 1, self.row + max(1, args[0] if args else 1))
        elif final == "C":
            self.col = min(COLS - 1, self.col + max(1, args[0] if args else 1))
        elif final == "D":
            self.col = max(0, self.col - max(1, args[0] if args else 1))
        elif final == "K":
            mode = args[0] if args else 0
            start = 0 if mode in (1, 2) else self.col
            stop = COLS if mode in (0, 2) else self.col + 1
            for column in range(start, min(stop, COLS)):
                self.buf[self.row][column] = (" ", "fg", False)
        elif final == "H":
            self.row = min(ROWS - 1, (args[0] if args else 1) - 1)
            self.col = min(COLS - 1, (args[1] if len(args) > 1 else 1) - 1)

    def snapshot(self) -> tuple:
        return tuple(tuple(row) for row in self.buf)


def colour_of(key) -> str:
    return PALETTE.get(key, PALETTE["fg"]) if not isinstance(key, str) \
        else PALETTE.get(key, PALETTE["fg"])


def render(cast_path: str, out_path: str, *, idle_cap: float = 0.9,
           idle_speedup: float = 14.0, max_frames: int = 96,
           title: str = "") -> None:
    frames = [json.loads(line) for line in open(cast_path, encoding="utf-8") if line.strip()]
    screen = Screen()

    # Build (real_time, snapshot) pairs, one per visible change.
    steps: list[tuple[float, tuple]] = []
    last = None
    for at, text in frames:
        screen.write(text)
        snap = screen.snapshot()
        if snap != last:
            steps.append((at, snap))
            last = snap
    if not steps:
        raise SystemExit("cast produced no frames")

    real_total = steps[-1][0]

    def changed_cells(a: tuple, b: tuple) -> int:
        return sum(1 for ra, rb in zip(a, b)
                   for ca, cb in zip(ra, rb) if ca != cb)

    # A frame that moves only a handful of cells is a spinner tick, not
    # content: play those at a reduced rate so the wait stays visible but
    # watchable. Real elapsed board time is drawn on every frame.
    play, clock = [], 0.0
    for index, (at, snap) in enumerate(steps):
        if index == 0:
            play.append((0.0, at, snap))
            continue
        gap = at - steps[index - 1][0]
        moved = changed_cells(steps[index - 1][1], snap)
        if moved <= 6:                      # spinner / clock tick
            gap /= idle_speedup
        elif gap > idle_cap:                # long silence before output
            gap = idle_cap
        clock += gap
        play.append((clock, at, snap))

    # Thin to max_frames, always keeping the last.
    if len(play) > max_frames:
        stride = len(play) / max_frames
        keep = {int(i * stride) for i in range(max_frames)}
        keep.add(len(play) - 1)
        play = [f for i, f in enumerate(play) if i in keep]

    duration = max(play[-1][0], 0.1) + 2.2
    width = PAD_X * 2 + COLS * CELL_W
    height = PAD_Y + ROWS * CELL_H + 20

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="ui-monospace,SFMono-Regular,'
        f'Menlo,Consolas,monospace" font-size="13">',
        "<style>",
        f"  .f{{opacity:0}}",
        *[f"  .f{i}{{animation:s{i} {duration:.2f}s steps(1,end) infinite}}"
          for i in range(len(play))],
        *[f"  @keyframes s{i}{{0%,{100 * play[i][0] / duration:.3f}%{{opacity:0}}"
          f"{max(100 * play[i][0] / duration, 0.001):.3f}%,"
          f"{100 * (play[i + 1][0] if i + 1 < len(play) else duration) / duration:.3f}%"
          f"{{opacity:1}}100%{{opacity:0}}}}"
          for i in range(len(play))],
        "</style>",
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="9" fill="{PALETTE["bg"]}"/>',
        f'<circle cx="24" cy="19" r="5.5" fill="#f2707a"/>'
        f'<circle cx="42" cy="19" r="5.5" fill="#e0b568"/>'
        f'<circle cx="60" cy="19" r="5.5" fill="#7ddba0"/>',
    ]
    if title:
        out.append(f'<text x="{width/2:.0f}" y="24" fill="{PALETTE["dim"]}" '
                   f'text-anchor="middle" font-size="12">{html.escape(title)}</text>')

    for index, (_start, real_at, snap) in enumerate(play):
        out.append(f'<g class="f f{index}">')
        for row_index, row in enumerate(snap):
            runs, current = [], None
            for char, key, bold in row:
                if current and current[1] == key and current[2] == bold:
                    current[0].append(char)
                else:
                    current = [[char], key, bold]
                    runs.append(current)
            column = 0
            for chars, key, bold in runs:
                text = "".join(chars)
                if text.strip():
                    out.append(
                        f'<text x="{PAD_X + column * CELL_W:.1f}" '
                        f'y="{PAD_Y + row_index * CELL_H:.1f}" fill="{colour_of(key)}"'
                        + (' font-weight="bold"' if bold else "")
                        + f' xml:space="preserve">{html.escape(text)}</text>')
                column += len(chars)
        out.append(f'<text x="{width - PAD_X:.0f}" y="{height - 8:.0f}" '
                   f'fill="{PALETTE["dim"]}" text-anchor="end" font-size="11">'
                   f'board t = {real_at:6.1f}s</text>')
        out.append("</g>")

    note = (f"real board session {real_total:.0f}s · waits shown at "
            f"{idle_speedup:.0f}x · live board clock at right")
    out.append(f'<text x="{PAD_X:.0f}" y="{height - 8:.0f}" fill="{PALETTE["dim"]}" '
               f'font-size="11">{html.escape(note)}</text>')
    out.append("</svg>")

    document = "\n".join(out)
    open(out_path, "w", encoding="utf-8").write(document)
    print(f"{out_path}: {len(play)} frames, real {real_total:.1f}s -> "
          f"play {play[-1][0]:.1f}s, {len(document) / 1024:.0f} KB")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2],
           title=sys.argv[3] if len(sys.argv) > 3 else "")
