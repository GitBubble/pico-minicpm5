#!/usr/bin/env python3
"""Render a timestamped terminal cast into a self-contained animated SVG.

No external asset and no script: a CSS keyframe per snapshot toggles
visibility, so the result animates from a plain <img> in a README.

Two things make the output align on every renderer. Character width follows
the East Asian Width property, so a CJK glyph occupies two terminal cells the
way it does in a real terminal. Each coloured run is then drawn with an
explicit ``textLength``, which forces it into exactly the width those cells
occupy no matter what the viewer's monospace font actually measures.

Idle stretches are compressed by a stated factor and the compression is drawn
on screen, so the animation never implies the board was faster than it was.
"""
from __future__ import annotations

import html
import json
import re
import sys
import unicodedata

COLS, ROWS = 96, 24
CELL_W, CELL_H = 8.6, 19.0
PAD_X, PAD_Y = 16.0, 40.0

PALETTE = {
    "fg": "#d5dbe5", "bg": "#11141b", "dim": "#7c8797",
    45: "#22d3ee", 75: "#6aa9f4", 114: "#7ddba0", 141: "#b39cf5",
    244: "#7c8797", 250: "#aab4c2", 203: "#f2707a", 179: "#e0b568",
}
FONT = ("ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono',"
        "'Noto Sans Mono CJK SC','Microsoft YaHei Mono',monospace")
SGR_RE = re.compile(r"\x1b\[([0-9;?]*)([a-zA-Z])")
BLANK = (" ", "fg", False, 1)


def char_width(char: str) -> int:
    """Terminal cells a character occupies."""
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


class Screen:
    """Enough of a terminal to replay a recorded session faithfully.

    A wide character owns two cells: the glyph sits in the first and the
    second is a continuation placeholder, so column arithmetic downstream
    matches what the real terminal did.
    """

    def __init__(self) -> None:
        self.buf = [[BLANK] * COLS for _ in range(ROWS)]
        self.row = self.col = 0
        self.colour = "fg"
        self.bold = False
        # A recording is a stream of read() chunks, and an escape sequence can
        # straddle two of them. Whatever is left mid-sequence waits here for
        # the next chunk instead of leaking to the screen as literal text.
        self.pending = ""

    def _scroll(self) -> None:
        self.buf.pop(0)
        self.buf.append([BLANK] * COLS)
        self.row = ROWS - 1

    def _newline(self) -> None:
        self.row += 1
        if self.row >= ROWS:
            self._scroll()

    def write(self, text: str) -> None:
        text, self.pending = self.pending + text, ""
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\x1b":
                match = SGR_RE.match(text, index)
                if match:
                    self._escape(match.group(1), match.group(2))
                    index = match.end()
                    continue
                tail = text[index:]
                if len(tail) < 16 and re.fullmatch(r"\x1b\[?[0-9;?]*", tail):
                    self.pending = tail        # incomplete: wait for more
                    return
                index += 1                     # not a CSI we model; drop it
                continue
            index += 1
            if char == "\r":
                self.col = 0
            elif char == "\n":
                self._newline()
            elif char == "\b":
                self.col = max(0, self.col - 1)
            elif char in ("\x1b", "\x07"):
                continue
            elif char >= " ":
                width = char_width(char)
                if width == 0:
                    continue
                if self.col + width > COLS:
                    self.col = 0
                    self._newline()
                self.buf[self.row][self.col] = (char, self.colour, self.bold, width)
                for offset in range(1, width):
                    self.buf[self.row][self.col + offset] = \
                        ("", self.colour, self.bold, 0)
                self.col += width

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
                self.buf[self.row][column] = BLANK
        elif final == "J" and args and args[0] == 2:
            self.buf = [[BLANK] * COLS for _ in range(ROWS)]
            self.row = self.col = 0
        elif final == "H":
            self.row = min(ROWS - 1, (args[0] if args else 1) - 1)
            self.col = min(COLS - 1, (args[1] if len(args) > 1 else 1) - 1)

    def snapshot(self) -> tuple:
        return tuple(tuple(row) for row in self.buf)


def _runs(row: tuple) -> list[tuple[int, int, str, bool, str]]:
    """Group a row into (start_col, width_cells, colour, bold, text) runs."""
    runs: list[list] = []
    for column, (char, colour, bold, width) in enumerate(row):
        if width == 0:
            # Continuation cell of a wide glyph. The glyph already claimed
            # both cells when it was appended, so this one adds nothing.
            continue
        if runs and runs[-1][2] == colour and runs[-1][3] == bold \
                and runs[-1][0] + runs[-1][1] == column:
            runs[-1][1] += width
            runs[-1][4].append(char)
        else:
            runs.append([column, width, colour, bold, [char]])
    return [(start, width, colour, bold, "".join(chars))
            for start, width, colour, bold, chars in runs]


def render(cast_path: str, out_path: str, *, idle_cap: float = 0.9,
           idle_speedup: float = 4.5, max_frames: int = 80,
           title: str = "") -> None:
    frames = [json.loads(line) for line in open(cast_path, encoding="utf-8")
              if line.strip()]
    screen = Screen()

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

    def moved(a: tuple, b: tuple) -> int:
        return sum(1 for ra, rb in zip(a, b)
                   for ca, cb in zip(ra, rb) if ca != cb)

    play, clock = [], 0.0
    for index, (at, snap) in enumerate(steps):
        if index == 0:
            play.append((0.0, at, snap))
            continue
        gap = at - steps[index - 1][0]
        if moved(steps[index - 1][1], snap) <= 6:
            gap /= idle_speedup                 # spinner or clock tick
        elif gap > idle_cap:
            gap = idle_cap
        clock += gap
        play.append((clock, at, snap))

    if len(play) > max_frames:
        stride = len(play) / max_frames
        keep = {int(i * stride) for i in range(max_frames)}
        keep.add(len(play) - 1)
        play = [f for i, f in enumerate(play) if i in keep]

    duration = max(play[-1][0], 0.1) + 2.4
    width = PAD_X * 2 + COLS * CELL_W
    height = PAD_Y + ROWS * CELL_H + 22

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="{FONT}" font-size="13">',
        "<style>",
        "  .f{opacity:0}",
        *[f"  .f{i}{{animation:s{i} {duration:.2f}s steps(1,end) infinite}}"
          for i in range(len(play))],
        *[f"  @keyframes s{i}{{0%,{100 * play[i][0] / duration:.3f}%{{opacity:0}}"
          f"{max(100 * play[i][0] / duration, 0.001):.3f}%,"
          f"{100 * (play[i + 1][0] if i + 1 < len(play) else duration) / duration:.3f}%"
          f"{{opacity:1}}100%{{opacity:0}}}}"
          for i in range(len(play))],
        "</style>",
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="10" fill="{PALETTE["bg"]}"/>',
        '<circle cx="26" cy="21" r="5.5" fill="#f2707a"/>'
        '<circle cx="44" cy="21" r="5.5" fill="#e0b568"/>'
        '<circle cx="62" cy="21" r="5.5" fill="#7ddba0"/>',
    ]
    if title:
        out.append(f'<text x="{width / 2:.0f}" y="26" fill="{PALETTE["dim"]}" '
                   f'text-anchor="middle" font-size="12">{html.escape(title)}</text>')

    for index, (_start, real_at, snap) in enumerate(play):
        out.append(f'<g class="f f{index}">')
        for row_index, row in enumerate(snap):
            for start, cells, colour, bold, text in _runs(row):
                if not text.strip():
                    continue
                out.append(
                    f'<text x="{PAD_X + start * CELL_W:.1f}" '
                    f'y="{PAD_Y + row_index * CELL_H:.1f}" '
                    f'textLength="{cells * CELL_W:.1f}" lengthAdjust="spacingAndGlyphs" '
                    f'fill="{PALETTE.get(colour, PALETTE["fg"])}"'
                    + (' font-weight="bold"' if bold else "")
                    + f' xml:space="preserve">{html.escape(text)}</text>')
        out.append(f'<text x="{width - PAD_X:.0f}" y="{height - 9:.0f}" '
                   f'fill="{PALETTE["dim"]}" text-anchor="end" font-size="11">'
                   f'board clock {real_at:6.1f}s</text>')
        out.append("</g>")

    note = (f"real session {real_total:.0f}s · waits played at {idle_speedup:.1f}x "
            f"· output in real time")
    out.append(f'<text x="{PAD_X:.0f}" y="{height - 9:.0f}" fill="{PALETTE["dim"]}" '
               f'font-size="11">{html.escape(note)}</text>')
    out.append("</svg>")

    document = "\n".join(out)
    open(out_path, "w", encoding="utf-8").write(document)
    print(f"{out_path}: {len(play)} frames, real {real_total:.1f}s -> "
          f"play {play[-1][0]:.1f}s, {len(document) / 1024:.0f} KB")


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2],
           title=sys.argv[3] if len(sys.argv) > 3 else "")
