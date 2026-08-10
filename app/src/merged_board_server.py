#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode MiniCPM5-1B on the board from ONE merged 24-layer OM.

Runs ON THE SS928.  Same protocol, same executor, same head and same embedding
table as scripts/minicpm5_board_server.py -- the only difference is that the 24
per-layer handles collapse into one, so a token step is 2 execute frames
instead of 25 and the mask and the RoPE matrix are written once instead of 24
times each.

Merged contract, read off the board's own model_desc:

    input[0] hidden        24576 B [1,1536,1,1]   f32 on the C4 carrier
    input[1] attention_mask 4096 B [1,1,1,1024]   f32
    input[2] rope_r        65536 B [1,1,128,128]  f32
    input[3] k_cache_all 12570624 B [1,48,1023,128] f16   channel = layer*2+head
    input[4] v_cache_all 12570624 B [1,48,1023,128] f16
    input[5],[6]                                   ATC-synthesised, left zero
    output   two [1,48,1,128] f32   k_cur_all / v_cur_all, order identified
             one [1,1536,1,1] f32 on the C4 carrier -- next_hidden of layer 23

Standard library only; the board has Python 3.10 and no numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qualify_minicpm_greedy_chain as gc  # noqa: E402
import pico_minicpm5_split_board_runner as runner  # noqa: E402
import probe_om_execute_latency as probe  # noqa: E402

HIDDEN, KV_HEADS, HEAD_DIM, LAYERS = gc.HIDDEN, gc.KV_HEADS, gc.HEAD_DIM, gc.LAYERS
CHANNELS = LAYERS * KV_HEADS


class TerminalUI:
    """Small dependency-free terminal UI for the resident board REPL.

    Animation and ANSI colour are intentionally restricted to a real TTY.
    Redirected output remains stable plain text for scripts and board logs.
    """

    _SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, *, active, context, color="auto", spinner=True):
        self.active = bool(active)
        self.context = int(context)
        self.is_tty = self.active and bool(getattr(sys.stdout, "isatty", lambda: False)())
        env_color = os.environ.get("PICO_MINICPM5_COLOR")
        if color == "auto" and env_color in {"auto", "always", "never"}:
            color = env_color
        self.color = (
            color == "always"
            or (color == "auto" and self.is_tty
                and "NO_COLOR" not in os.environ
                and os.environ.get("TERM") != "dumb")
        )
        self.animate = bool(spinner and self.is_tty)
        self._wait_stop = threading.Event()
        self._wait_thread = None
        self._wait_label = ""
        self._wait_started = 0.0

    def paint(self, text, code):
        if not self.color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def banner(self):
        if not self.active:
            return
        lines = (
            "        /\\_/\\",
            "       ( o.o )    MiniCPM 5",
            "        > ^ <     SS928 local AI",
        )
        shades = ("38;5;141", "1;38;5;45", "38;5;75")
        print("", flush=True)
        for line, shade in zip(lines, shades):
            print(self.paint(line, shade), flush=True)
        status = f"     ctx{self.context} · resident KV · streaming"
        print(self.paint(status, "2;38;5;114"), flush=True)
        print("", flush=True)

    def start_wait(self, label):
        if not self.active:
            return
        self.stop_wait()
        self._wait_label = str(label)
        self._wait_started = time.perf_counter()
        if not self.animate:
            print(self.paint(f"... {self._wait_label}", "2;38;5;75"), flush=True)
            return
        self._wait_stop.clear()
        self._wait_thread = threading.Thread(
            target=self._spin, name="minicpm-terminal-spinner", daemon=True)
        self._wait_thread.start()

    def _spin(self):
        index = 0
        while not self._wait_stop.is_set():
            elapsed = time.perf_counter() - self._wait_started
            frame = self.paint(self._SPINNER[index % len(self._SPINNER)],
                               "1;38;5;45")
            label = self.paint(self._wait_label, "38;5;75")
            sys.stdout.write(f"\r\033[2K{frame} {label} {elapsed:4.1f}s")
            sys.stdout.flush()
            index += 1
            self._wait_stop.wait(0.08)

    def stop_wait(self):
        thread = self._wait_thread
        if thread is None:
            return
        self._wait_stop.set()
        thread.join(timeout=1.0)
        self._wait_thread = None
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def ready(self, handles, seconds):
        mark = self.paint("✓", "1;38;5;114")
        label = self.paint("ready", "1;38;5;45")
        detail = self.paint(
            f"{handles} handles · ctx{self.context} · {seconds:.1f}s",
            "2;38;5;250")
        print(f"{mark} {label} · loaded {detail}", flush=True)

    def prompt(self):
        return f"{self.paint('You', '1;38;5;141')} {self.paint('❯', '1;38;5;45')} "

    def model_prefix(self):
        brand = self.paint("MiniCPM", "1;38;5;45")
        sparkle = self.paint("✦", "1;38;5;141")
        print(f"{brand} {sparkle} ", end="", flush=True)

    def info(self, text):
        print(self.paint(text, "38;5;75"), flush=True)

    def turn_summary(self, tokens, step_ms, reason):
        if not self.is_tty:
            return
        total_s = sum(step_ms) / 1000.0
        rate = tokens / total_s if total_s > 0 else 0.0
        text = f"  {tokens} tokens · {rate:.2f} tok/s · {reason}"
        print(self.paint(text, "2;38;5;244"), flush=True)


class Merged:
    def __init__(self, *, executable, decode, prefill, head, library_paths,
                 embedding, context, timeout, tokenizer=None,
                 resident_kv=True, quiet_executor=False):
        self.context = context
        self.past = context - 1
        self.timeout = timeout
        self.row_f16 = HEAD_DIM * 2
        self.cache_bytes = CHANNELS * self.past * HEAD_DIM * 2
        self.embed = open(embedding, "rb")
        self._embedding_half = struct.Struct(f"<{HIDDEN}e")
        self._embedding_f32 = struct.Struct(f"<{HIDDEN}f")
        self._embedding_c4 = struct.Struct("<" + "f12x" * HIDDEN)
        self._rope_inverse = tuple(
            1.0 / (gc.ROPE_THETA ** ((2 * i) / HEAD_DIM))
            for i in range(HEAD_DIM // 2))
        self.tokenizer = None
        self.resident_kv = bool(resident_kv)
        if tokenizer:
            from tokenizers import Tokenizer  # noqa: PLC0415
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.models = [decode] + ([prefill] if prefill else []) + [head]
        self.decode_index = 0
        self.prefill_index = 1 if prefill else 0
        self.head_index = len(self.models) - 1
        self.process = probe._start(
            executable, self.models, library_paths, 0, quiet=quiet_executor)
        self._deadline = time.monotonic() + timeout
        self.descriptors = probe._read_ready(
            self.process.stdout, len(self.models), self._deadline)
        self.kv_slots = self._identify_kv()

    # -- transport -------------------------------------------------------
    def _read(self, n):
        return runner._read_exact_until(self.process.stdout, n, self._deadline)

    def _respond(self, out_sizes):
        (_m, _v, status, _i, count, error_bytes, _r) = \
            runner._PERSISTENT_RESPONSE.unpack(
                self._read(runner._PERSISTENT_RESPONSE.size))
        if status:
            raise RuntimeError(self._read(error_bytes).decode("utf-8", "replace"))
        for _ in range(count):
            self._read(runner._PERSISTENT_U64.size)
        return [self._read(size) for size in out_sizes]

    def _hidden_values(self, token):
        self.embed.seek(token * HIDDEN * 2)
        payload = self.embed.read(HIDDEN * 2)
        if len(payload) != HIDDEN * 2:
            raise RuntimeError(f"embedding row {token} is truncated")
        return self._embedding_half.unpack(payload)

    def _hidden_input(self, token, want):
        row = self._hidden_values(token)
        if want == HIDDEN * 4:
            return self._embedding_f32.pack(*row)
        if want == HIDDEN * 16:
            return self._embedding_c4.pack(*row)
        raise RuntimeError(f"unsupported hidden carrier size {want}")

    def _rope_matrix_bytes(self, position):
        matrix = bytearray(HEAD_DIM * HEAD_DIM * 4)
        half = HEAD_DIM // 2
        for i, inverse in enumerate(self._rope_inverse):
            angle = position * inverse
            cosine = math.cos(angle)
            sine = math.sin(angle)
            struct.pack_into("<f", matrix, (i * HEAD_DIM + i) * 4, cosine)
            struct.pack_into(
                "<f", matrix,
                ((i + half) * HEAD_DIM + i + half) * 4, cosine)
            struct.pack_into(
                "<f", matrix, ((i + half) * HEAD_DIM + i) * 4, -sine)
            struct.pack_into(
                "<f", matrix, (i * HEAD_DIM + i + half) * 4, sine)
        return bytes(matrix)

    def _hidden_output(self, index):
        """Which published slot is next_hidden.

        Size alone cannot tell: the merged model publishes k_cur_all and
        v_cur_all as [1,48,1,128] f32 = 24576 B, and next_hidden as
        [1,1536,1,1] f32 on the 16-byte C4 carrier = 24576 B as well.  All
        three are the same size.  The KV pair is identified structurally in
        _identify_kv (RoPE moves k_cur and not v_cur); next_hidden is whatever
        is left over."""
        published = len(self.descriptors[index][1])
        if index in self.kv_slots:
            taken = set(self.kv_slots[index])
            rest = [s for s in range(published) if s not in taken]
            if len(rest) != 1:
                raise RuntimeError(f"model {index}: {len(rest)} non-KV outputs")
            return rest[0]
        for slot, size in enumerate(self.descriptors[index][1]):
            if size in (HIDDEN * 4, HIDDEN * 16):
                return slot
        raise RuntimeError(f"model {index} publishes no hidden")

    def _run(self, model, writes, chains=(), *, publish=True,
             output_count=None):
        desc_in, desc_out = self.descriptors[model]
        if output_count is None:
            output_count = 2 if publish else 0
        if output_count < 0 or output_count > len(desc_out):
            raise ValueError(f"invalid output_count {output_count}")
        output_sizes = tuple(desc_out[:output_count])
        runner._write_all(self.process.stdin, gc._frame(
            model, min(5, len(desc_in)), output_sizes, writes, chains))
        self.process.stdin.flush()
        return self._respond(output_sizes)

    def _scatter_kv(self, source_model, position):
        """Keep packed current K/V rows inside the resident executor.

        The transformer publishes contiguous channel-major FP32 rows, while
        its cache inputs are FP16 with one context-stride between channels.
        Opcode 6 performs the same RNE conversion as struct.pack('e') and
        writes both rows directly into the decode handle's retained inputs.
        """
        k_slot, v_slot = self.kv_slots[source_model]
        header = runner._PERSISTENT_REQUEST.pack(
            runner.PERSISTENT_REQUEST_MAGIC,
            runner.PERSISTENT_PROTOCOL_VERSION,
            runner.PERSISTENT_OP_SCATTER_F32_TO_F16,
            self.decode_index, 2, 0, 0)
        base = position * self.row_f16
        stride = self.past * self.row_f16
        records = (
            runner._PERSISTENT_SCATTER_F32_TO_F16.pack(
                3, source_model, k_slot, 0, base, stride,
                CHANNELS, HEAD_DIM, 0),
            runner._PERSISTENT_SCATTER_F32_TO_F16.pack(
                4, source_model, v_slot, 0, base, stride,
                CHANNELS, HEAD_DIM, 0),
        )
        runner._write_all(self.process.stdin, header)
        for record in records:
            runner._write_all(self.process.stdin, record)
        self.process.stdin.flush()
        self._respond(())

    def _identify_kv(self):
        """RoPE moves k_cur and not v_cur.  Same structural test the split
        runner uses, applied once to the merged model."""
        zero_k = bytes(self.cache_bytes)
        mask = bytearray(struct.pack("<f", gc.MASK_NEG) * self.context)
        struct.pack_into("<f", mask, (self.context - 1) * 4, 0.0)
        hidden = self._hidden_input(
            0, self.descriptors[self.decode_index][0][0])
        out = {}
        for model in {self.decode_index, self.prefill_index}:
            seen = []
            for position in (0, 1):
                writes = [(0, 0, hidden), (1, 0, bytes(mask)),
                          (2, 0, self._rope_matrix_bytes(position)),
                          (3, 0, zero_k), (4, 0, zero_k)]
                seen.append(self._run(model, writes))
            moved = [i for i in range(2) if seen[0][i] != seen[1][i]]
            if len(moved) != 1:
                raise RuntimeError(
                    f"model {model}: RoPE moved {len(moved)} of 2 KV outputs; "
                    "the k/v identification is not safe")
            out[model] = (moved[0], 1 - moved[0])
        return out

    # -- decode ----------------------------------------------------------
    def generate(self, prompt_ids, max_new, eos, *, start=0,
                 kv_in=None, kv_out=None, stop_after=None,
                 capture_dir=None, capture_position=None, on_token=None):
        self.last_phase_steps = []
        host_kv = not self.resident_kv or kv_in is not None or kv_out is not None
        k_cache = bytearray(self.cache_bytes) if host_kv else None
        v_cache = bytearray(self.cache_bytes) if host_kv else None
        if kv_in is not None:
            blob = Path(kv_in).read_bytes()
            k_cache[:] = blob[:self.cache_bytes]
            v_cache[:] = blob[self.cache_bytes:2 * self.cache_bytes]
        token = prompt_ids[start]
        produced, produced_ids, steps = 0, [], []
        total = len(prompt_ids) + max_new
        for position in range(start, total):
            if position >= self.context:
                return "context", produced_ids, steps
            began = time.perf_counter()
            self._deadline = time.monotonic() + self.timeout
            model = (self.prefill_index if position == 0 else self.decode_index)
            desc_in = self.descriptors[model][0]

            mask = bytearray(struct.pack("<f", gc.MASK_NEG) * self.context)
            for slot in range(position):
                struct.pack_into("<f", mask, slot * 4, 0.0)
            struct.pack_into("<f", mask, (self.context - 1) * 4, 0.0)

            writes = [(0, 0, self._hidden_input(token, desc_in[0])),
                      (1, 0, bytes(mask)),
                      (2, 0, self._rope_matrix_bytes(position))]
            if host_kv and position <= 1:
                # position 0's caches are genuinely zero; position 1 is the
                # first time the DECODE model's resident buffers are touched
                # and they already have to carry position 0's row.
                writes.append((3, 0, bytes(k_cache)))
                writes.append((4, 0, bytes(v_cache)))
            elif host_kv:
                prior = position - 1
                for slot, mirror in ((3, k_cache), (4, v_cache)):
                    for c in range(CHANNELS):
                        at = (c * self.past + prior) * self.row_f16
                        writes.append(
                            (slot, at, bytes(mirror[at:at + self.row_f16])))
            prepared_at = time.perf_counter()
            capture = capture_dir is not None and position == capture_position
            published = self._run(
                model, writes, publish=host_kv or capture,
                output_count=3 if capture else None)
            transformer_at = time.perf_counter()
            if capture:
                capture_dir.mkdir(parents=True, exist_ok=True)
                for slot, blob in enumerate(published):
                    (capture_dir / f"out.{slot}.bin").write_bytes(blob)
            if host_kv:
                k_slot, v_slot = self.kv_slots[model]
                for mirror, blob in ((k_cache, published[k_slot]),
                                     (v_cache, published[v_slot])):
                    values = struct.unpack(f"<{CHANNELS * HEAD_DIM}f", blob)
                    for c in range(CHANNELS):
                        at = (c * self.past + position) * self.row_f16
                        mirror[at:at + self.row_f16] = struct.pack(
                            f"<{HEAD_DIM}e",
                            *values[c * HEAD_DIM:(c + 1) * HEAD_DIM])
            else:
                self._scatter_kv(model, position)
            kv_at = time.perf_counter()

            head_in = self.descriptors[self.head_index][0]
            src = self._hidden_output(model)
            runner._write_all(self.process.stdin, gc._frame(
                self.head_index, min(3, len(head_in)), (), [],
                [(0, 0, min(head_in[0], self.descriptors[model][1][src]),
                  model, src)]))
            self.process.stdin.flush()
            self._respond(())
            head_at = time.perf_counter()
            runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_ARGMAX, self.head_index, 0, 0, 0))
            self.process.stdin.flush()
            (payload,) = self._respond((8,))
            argmax_at = time.perf_counter()
            predicted, _value = struct.unpack("<If", payload)
            steps.append((argmax_at - began) * 1000.0)
            self.last_phase_steps.append({
                "position": position,
                "prepare_ms": (prepared_at - began) * 1000.0,
                "transformer_ms": (transformer_at - prepared_at) * 1000.0,
                "kv_ms": (kv_at - transformer_at) * 1000.0,
                "head_execute_ms": (head_at - kv_at) * 1000.0,
                "argmax_ms": (argmax_at - head_at) * 1000.0,
                "total_ms": (argmax_at - began) * 1000.0,
            })
            if kv_out is not None and stop_after is not None \
                    and position == stop_after:
                Path(kv_out).write_bytes(bytes(k_cache) + bytes(v_cache))
                return "stop", [int(predicted)], steps

            if position + 1 < len(prompt_ids):
                token = prompt_ids[position + 1]
                continue
            produced_ids.append(int(predicted))
            produced += 1
            if on_token is not None:
                on_token(tuple(produced_ids))
            if int(predicted) in eos:
                return "eos", produced_ids, steps
            if produced >= max_new:
                return "max", produced_ids, steps
            token = int(predicted)
        return "max", produced_ids, steps

    def close(self):
        try:
            runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0))
            self.process.stdin.flush()
        except Exception:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent-executor", type=Path, required=True)
    ap.add_argument("--decode-model", type=Path, required=True)
    ap.add_argument("--prefill-model", type=Path)
    ap.add_argument("--head-model", type=Path, required=True)
    ap.add_argument("--library-path", type=Path, action="append", default=[])
    ap.add_argument("--embedding", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument(
        "--host-kv", action="store_true",
        help="compatibility path: return packed K/V to Python and write it back")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--prompt-ids", action="append", default=[])
    ap.add_argument(
        "--interactive", action="store_true",
        help="keep the three model handles loaded and read prompts from stdin")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto",
        help="terminal colour policy (default: auto; PICO_MINICPM5_COLOR also applies)")
    ap.add_argument(
        "--no-spinner", action="store_true",
        help="disable the animated loading/thinking indicator")
    ap.add_argument(
        "--verbose-executor", action="store_true",
        help="show low-level executor startup diagnostics in interactive mode")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--start-position", type=int, default=0)
    ap.add_argument("--kv-in", type=Path)
    ap.add_argument("--kv-out", type=Path)
    ap.add_argument("--stop-after", type=int)
    ap.add_argument("--capture-dir", type=Path)
    ap.add_argument("--capture-position", type=int)
    args = ap.parse_args()
    if (args.capture_dir is None) != (args.capture_position is None):
        ap.error("--capture-dir and --capture-position must be provided together")
    if args.capture_position is not None and args.capture_position < 0:
        ap.error("--capture-position must be nonnegative")
    if args.interactive and args.tokenizer is None:
        ap.error("--interactive requires --tokenizer")
    if args.interactive and (
            args.prompt_ids or args.start_position != 0 or args.kv_in is not None
            or args.kv_out is not None or args.stop_after is not None
            or args.capture_dir is not None):
        ap.error("--interactive is incompatible with prompt-id/KV/capture controls")
    if not args.interactive and not args.prompt and not args.prompt_ids:
        ap.error("provide --prompt, --prompt-ids or --interactive")

    ui = TerminalUI(active=args.interactive, context=args.context,
                    color=args.color, spinner=not args.no_spinner)
    ui.banner()
    ui.start_wait("Loading three resident model handles")
    began = time.perf_counter()
    try:
        session = Merged(executable=args.persistent_executor,
                         decode=args.decode_model, prefill=args.prefill_model,
                         head=args.head_model, library_paths=args.library_path,
                         embedding=args.embedding, context=args.context,
                         timeout=args.timeout, tokenizer=args.tokenizer,
                         resident_kv=not args.host_kv,
                         quiet_executor=args.interactive
                         and not args.verbose_executor)
    finally:
        ui.stop_wait()
    load_ms = (time.perf_counter() - began) * 1000.0
    if args.interactive:
        ui.ready(len(session.models), load_ms / 1000.0)
    else:
        print(f"loaded {len(session.models)} handles in {load_ms / 1000:.1f} s; "
              f"kv slots {session.kv_slots}", flush=True)
    results = []

    def run_text_prompt(spec, *, repl=False, repl_epoch=0, max_new=None,
                        on_token=None):
        limit = args.max_new if max_new is None else int(max_new)
        ids = [0] + list(session.tokenizer.encode(
            spec, add_special_tokens=False).ids)
        reason, out, steps = session.generate(
            ids, limit, {1, 130073}, start=args.start_position,
            kv_in=args.kv_in, kv_out=args.kv_out,
            stop_after=args.stop_after, capture_dir=args.capture_dir,
            capture_position=args.capture_position, on_token=on_token)
        text = session.tokenizer.decode(out, skip_special_tokens=True)
        record = {"prompt": spec, "ids": out, "text": text,
                  "reason": reason, "step_ms": steps,
                  "phase_ms": session.last_phase_steps}
        if repl:
            record.update(mode="repl", repl_epoch=repl_epoch,
                          max_new=limit)
        return record, steps[len(ids) - 1:]

    try:
        for spec in args.prompt:
            record, gen = run_text_prompt(spec)
            print(f"prompt={spec!r}\n  ids={record['ids']}\n"
                  f"  text={record['text']!r}\n  reason={record['reason']} "
                  f"steps_ms={[round(s,1) for s in gen]}",
                  flush=True)
            results.append(record)
        for spec in args.prompt_ids:
            ids = [int(v) for v in spec.split(",")]
            reason, out, steps = session.generate(
                ids, args.max_new, {1, 130073}, start=args.start_position,
                kv_in=args.kv_in, kv_out=args.kv_out,
                stop_after=args.stop_after, capture_dir=args.capture_dir,
                capture_position=args.capture_position)
            print(f"prompt_ids={ids}\n  ids={out}\n  reason={reason}\n"
                  f"  steps_ms={[round(s,1) for s in steps]}", flush=True)
            results.append({"prompt_ids": ids, "ids": out, "reason": reason,
                            "step_ms": steps,
                            "phase_ms": session.last_phase_steps})
        if args.interactive:
            ui.info("Commands: /help · /max N · /reset · /quit")
            epoch = 0
            repl_max_new = args.max_new
            while True:
                try:
                    spec = input(ui.prompt()).strip()
                except EOFError:
                    print("", flush=True)
                    break
                except KeyboardInterrupt:
                    print("\nUse /quit or Ctrl-D to exit.", flush=True)
                    continue
                if not spec:
                    continue
                if spec in {"/quit", "/exit"}:
                    break
                if spec == "/help":
                    ui.info(
                        "Each prompt starts a fresh logical context while "
                        "the three model handles remain loaded. "
                        f"Current context is ctx{args.context}; max-new is "
                        f"{repl_max_new}; allowed range is "
                        f"1..{args.context - 1}. Effective output is also "
                        "limited by prompt tokens. Use /max N to change it, "
                        "/reset to mark a new transcript and /quit to exit.")
                    continue
                if spec == "/max":
                    print(f"max-new={repl_max_new}; allowed=1..{args.context - 1}; "
                          "effective output also depends on prompt length",
                          flush=True)
                    continue
                if spec.startswith("/max "):
                    try:
                        requested = int(spec.split(None, 1)[1])
                    except ValueError:
                        print("Usage: /max N", flush=True)
                        continue
                    if requested < 1 or requested >= args.context:
                        print(f"N must be in [1, {args.context - 1}]", flush=True)
                        continue
                    repl_max_new = requested
                    print(f"max-new={repl_max_new}", flush=True)
                    continue
                if spec == "/reset":
                    epoch += 1
                    ui.info("Context reset.")
                    continue
                shown = ""
                prefix_shown = False

                def stream(token_ids):
                    nonlocal shown, prefix_shown
                    if not prefix_shown:
                        ui.stop_wait()
                        ui.model_prefix()
                        prefix_shown = True
                    rendered = session.tokenizer.decode(
                        token_ids, skip_special_tokens=True)
                    if rendered.startswith(shown):
                        print(rendered[len(shown):], end="", flush=True)
                        shown = rendered

                ui.start_wait("MiniCPM is thinking")
                try:
                    record, _gen = run_text_prompt(
                        spec, repl=True, repl_epoch=epoch,
                        max_new=repl_max_new, on_token=stream)
                finally:
                    ui.stop_wait()
                results.append(record)
                if not prefix_shown:
                    ui.model_prefix()
                    prefix_shown = True
                if record["text"].startswith(shown):
                    print(record["text"][len(shown):], flush=True)
                else:
                    print("", flush=True)
                    ui.model_prefix()
                    print(record["text"], flush=True)
                ui.turn_summary(len(record["ids"]), _gen, record["reason"])
                if record["reason"] == "max":
                    print(f"[reached max-new={repl_max_new}; "
                          "use /max N to increase it]", flush=True)
                elif record["reason"] == "context":
                    print(f"[reached ctx{args.context} limit]", flush=True)
    finally:
        session.close()
    if args.report:
        args.report.write_text(json.dumps(
            {"schema": "pico.minicpm5.merged_board.v1",
             "load_seconds": load_ms / 1000.0, "runs": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
