#!/usr/bin/env python3
"""Render a terminal cast to an animated GIF that looks like the real terminal.

The cast is replayed through a VT emulator, so what is drawn is the screen the
board actually produced rather than a re-interpretation of the byte stream.
Each cell is then drawn at its own grid position with a font that has the glyph:
a monospace face for narrow cells and a CJK face, scaled to exactly two cells,
for wide ones. Nothing is measured against a nominal advance width, so no glyph
can drift into its neighbour.

Timing is taken from the cast unmodified. Frames are emitted when the screen
changes and are rate-limited, which drops duplicate spinner ticks without
altering when anything happens.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

# Terminal.app's Pro theme, sampled from a screen recording of the board.
BACKGROUND = (25, 27, 32)
FOREGROUND = (222, 226, 233)
TITLEBAR = (58, 58, 62)
TITLEBAR_TEXT = (206, 206, 210)
WINDOW_EDGE = (12, 13, 16)
TRAFFIC = ((255, 95, 86), (255, 189, 46), (39, 201, 63))

NAMED = {
    "black": (0, 0, 0), "red": (204, 68, 61), "green": (109, 179, 90),
    "brown": (176, 134, 47), "yellow": (176, 134, 47), "blue": (73, 129, 209),
    "magenta": (170, 90, 190), "cyan": (60, 176, 190), "white": (222, 226, 233),
    "brightblack": (117, 121, 130), "brightred": (255, 111, 100),
    "brightgreen": (125, 219, 160), "brightbrown": (255, 213, 91),
    "brightyellow": (255, 213, 91), "brightblue": (106, 169, 244),
    "brightmagenta": (208, 136, 240), "brightcyan": (94, 216, 232),
    "brightwhite": (255, 255, 255), "default": None,
}

# Candidate faces, first match wins. A terminal recording needs three kinds of
# glyph and no single common face carries all of them: Menlo has no braille,
# which is what the spinner is drawn from, and Apple Symbols has no check mark.
FONT_MONO = ("/System/Library/Fonts/Menlo.ttc",
             "/System/Library/Fonts/Monaco.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_CJK = ("/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_SYMBOL = ("/System/Library/Fonts/Apple Symbols.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def _first_present(paths, label):
    for path in paths:
        if Path(path).exists():
            return path
    raise SystemExit(
        f"no {label} font found; tried:\n  " + "\n  ".join(paths))


def wide(char: str) -> bool:
    return unicodedata.east_asian_width(char) in ("W", "F")


def xterm256(index: int) -> tuple[int, int, int]:
    if index < 16:
        base = ("black", "red", "green", "brown", "blue", "magenta", "cyan",
                "white")
        name = base[index % 8]
        return NAMED["bright" + name if index >= 8 else name]
    if index < 232:
        index -= 16
        level = (0, 95, 135, 175, 215, 255)
        return (level[index // 36], level[index // 6 % 6], level[index % 6])
    grey = 8 + (index - 232) * 10
    return (grey, grey, grey)


def colour(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if value in NAMED:
        return NAMED[value] or fallback
    if len(value) == 6:
        try:
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return fallback
    if value.isdigit():
        return xterm256(int(value))
    return fallback


class Face:
    """A monospace grid whose wide cells are drawn by a second, scaled face."""

    def __init__(self, size: int) -> None:
        mono_path = _first_present(FONT_MONO, "monospace")
        cjk_path = _first_present(FONT_CJK, "CJK")
        self.mono = ImageFont.truetype(mono_path, size)
        try:
            self.mono_bold = ImageFont.truetype(mono_path, size, index=1)
        except OSError:
            self.mono_bold = self.mono
        self.cell_w = round(self.mono.getlength("M"))
        self.cell_h = round(size * 1.36)
        # Scale the CJK face until one glyph is exactly two cells wide, which
        # is what a terminal does when it falls back for a wide character.
        probe = ImageFont.truetype(cjk_path, size)
        advance = probe.getlength("中") or size
        self.cjk = ImageFont.truetype(
            cjk_path, max(1, round(size * (2 * self.cell_w) / advance)))
        self.symbol = ImageFont.truetype(_first_present(FONT_SYMBOL, "symbol"),
                                         size)
        self.baseline = round(size * 1.06)
        self._notdef = {}
        self._resolved = {}

    def _draws(self, font: ImageFont.FreeTypeFont, char: str) -> bool:
        """Whether the face has this glyph rather than the .notdef box."""
        key = id(font)
        if key not in self._notdef:
            self._notdef[key] = self._bitmap(font, "\ue000")
        return self._bitmap(font, char) != self._notdef[key]

    @staticmethod
    def _bitmap(font: ImageFont.FreeTypeFont, char: str) -> bytes:
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).text((8, 8), char, font=font, fill=255)
        return image.tobytes()

    def pick(self, char: str, bold: bool) -> ImageFont.FreeTypeFont:
        if wide(char):
            return self.cjk
        chosen = self.mono_bold if bold else self.mono
        cached = self._resolved.get((char, bold))
        if cached is not None:
            return cached
        if not self._draws(chosen, char) and self._draws(self.symbol, char):
            # The spinner is drawn from braille, which Menlo does not carry.
            chosen = self.symbol
        self._resolved[(char, bold)] = chosen
        return chosen


def replay(rows: list[tuple[float, str]], cols: int, lines: int,
           min_interval: float) -> list[tuple[float, list[list]]]:
    """Feed the cast to a VT emulator and snapshot the screen as it changes."""
    screen = pyte.Screen(cols, lines)
    stream = pyte.Stream(screen)
    frames: list[tuple[float, list[list]]] = []
    last_emit = -1.0
    last_state: list[list] | None = None

    def snapshot() -> list[list]:
        out = []
        for y in range(lines):
            row = screen.buffer[y]
            out.append([(row[x].data, row[x].fg, row[x].bg, row[x].bold,
                         row[x].reverse) for x in range(cols)])
        return out

    for at, text in rows:
        stream.feed(text)
        if at - last_emit < min_interval:
            continue
        state = snapshot()
        if state == last_state:
            continue
        frames.append((at, state))
        last_state = state
        last_emit = at

    state = snapshot()
    if state != last_state:
        frames.append((rows[-1][0], state))
    return frames


def draw(state: list[list], face: Face, cols: int, lines: int,
         chrome: str | None) -> Image.Image:
    pad = face.cell_w
    grid_w = cols * face.cell_w + pad * 2
    grid_h = lines * face.cell_h + pad
    bar = round(face.cell_h * 1.7) if chrome else 0
    image = Image.new("RGB", (grid_w, grid_h + bar), BACKGROUND)
    pen = ImageDraw.Draw(image)

    if chrome:
        pen.rectangle([0, 0, grid_w, bar], fill=TITLEBAR)
        radius = round(bar * 0.17)
        for index, shade in enumerate(TRAFFIC):
            cx = pad + index * radius * 3 + radius
            cy = bar // 2
            pen.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                        fill=shade)
        width = face.mono.getlength(chrome)
        pen.text(((grid_w - width) / 2, (bar - face.cell_h) / 2 + 1),
                 chrome, font=face.mono, fill=TITLEBAR_TEXT)
        pen.line([(0, bar), (grid_w, bar)], fill=WINDOW_EDGE)

    skip = 0
    for y, row in enumerate(state):
        top = bar + pad // 2 + y * face.cell_h
        for x, (char, fg, bg, bold, reverse) in enumerate(row):
            if skip:
                skip -= 1
                continue
            ink = colour(fg, FOREGROUND)
            paper = colour(bg, BACKGROUND) if bg != "default" else BACKGROUND
            if reverse:
                ink, paper = paper, ink
            span = 2 if wide(char) else 1
            if wide(char):
                skip = 1
            left = pad + x * face.cell_w
            if paper != BACKGROUND:
                pen.rectangle(
                    [left, top, left + span * face.cell_w, top + face.cell_h],
                    fill=paper)
            if char and char != " ":
                font = face.pick(char, bold)
                offset = 0
                if wide(char):
                    offset = (span * face.cell_w - font.getlength(char)) / 2
                pen.text((left + offset, top + face.baseline), char,
                         font=font, fill=ink, anchor="ls")
    return image


def build(cast: Path, out: Path, cols: int, lines: int, size: int,
          min_interval: float, chrome: str | None, scale: float,
          tail: float) -> None:
    rows = [tuple(json.loads(line)) for line in
            cast.read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = replay(rows, cols, lines, min_interval)
    face = Face(size)

    images: list[Image.Image] = []
    durations: list[int] = []
    for index, (at, state) in enumerate(frames):
        nxt = frames[index + 1][0] if index + 1 < len(frames) else at + tail
        image = draw(state, face, cols, lines, chrome)
        if scale != 1.0:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.LANCZOS)
        images.append(image.quantize(colors=128, method=Image.MEDIANCUT))
        durations.append(max(20, round((nxt - at) * 1000)))

    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    total = sum(durations) / 1000
    print(f"{out}: {len(images)} frames, {total:.1f}s, "
          f"{images[0].size[0]}x{images[0].size[1]}, "
          f"{out.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cast", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--lines", type=int, default=26)
    parser.add_argument("--font-size", type=int, default=26)
    parser.add_argument("--min-interval", type=float, default=0.1,
                        help="seconds; drops duplicate ticks, keeps timing")
    parser.add_argument("--title", default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--tail", type=float, default=2.5,
                        help="seconds to hold the last frame")
    args = parser.parse_args()
    build(args.cast, args.out, args.cols, args.lines, args.font_size,
          args.min_interval, args.title, args.scale, args.tail)


if __name__ == "__main__":
    main()
