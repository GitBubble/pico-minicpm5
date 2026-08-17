#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode MiniCPM5-1B on the board from ONE merged 24-layer OM.

Runs ON THE Hi3403.  Same protocol, same executor, same head and same embedding
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
import subprocess
import sys
import threading
import time

try:
    import readline as _readline  # noqa: F401
except ImportError:  # pragma: no cover - optional outside POSIX board runtimes
    _readline = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qualify_minicpm_greedy_chain as gc  # noqa: E402
import pico_minicpm5_split_board_runner as runner  # noqa: E402
import probe_om_execute_latency as probe  # noqa: E402
import minicpm_agent as agent  # noqa: E402
import minicpm_profile as profile_contract  # noqa: E402
import context_ledger as ledger  # noqa: E402
import minicpm_prefill_schedule as prefill_schedule  # noqa: E402
import minicpm_prefill_runtime as prefill_runtime_contract  # noqa: E402

HIDDEN, KV_HEADS, HEAD_DIM, LAYERS = gc.HIDDEN, gc.KV_HEADS, gc.HEAD_DIM, gc.LAYERS
#: Input snapshots the persistent executor can hold at once.
PREFIX_SNAPSHOT_SLOTS = 8
CHANNELS = LAYERS * KV_HEADS


def observe_prompt_token_ms(phase_steps, observed):
    """Accumulate this board's measured cost of ingesting one prompt token.

    Prompt positions are the ones that skipped the vocabulary head, so they
    time the ingest path alone. Measuring instead of assuming keeps the ledger
    honest across profiles: the same code reports `79.49 ms` on ctx1024 and
    `144.02 ms` on ctx8192 without being told which it is running.

    :param phase_steps: `Merged.last_phase_steps` from one generate call.
    :param observed: running totals, mutated in place.
    :returns: the mean millisecond cost, or `None` before any observation.
    """
    for step in phase_steps or ():
        if not step.get("head_skipped"):
            continue
        # A step without an explicit width covers one position: the wide
        # prefill families are the only producers of `token_count`.
        tokens = int(step.get("token_count") or 1)
        if tokens <= 0:
            continue
        observed["tokens"] += tokens
        observed["total_ms"] += float(step.get("total_ms", 0.0))
    if not observed["tokens"]:
        return None
    observed["ms"] = observed["total_ms"] / observed["tokens"]
    return observed["ms"]


def parse_transformer_output_slots(value):
    try:
        slots = tuple(int(part) for part in str(value).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "output slots must be comma-separated K,V,H integers") from error
    if len(slots) != 3 or set(slots) != {0, 1, 2}:
        raise argparse.ArgumentTypeError(
            "output slots must be distinct K,V,H indices covering 0,1,2")
    return slots


def agent_command_help(topic, *, context, max_new, thinking, max_tool_steps,
                       workspace, max_new_limit=None, profile_name="legacy"):
    """Return Linux-style help for the agent's local REPL commands."""
    canonical = str(topic or "").strip().lower().lstrip("/")
    aliases = {"exit": "quit", "reset": "clear"}
    canonical = aliases.get(canonical, canonical)
    hardware_maximum = int(context) - 1
    maximum = hardware_maximum if max_new_limit is None else min(
        int(max_new_limit), hardware_maximum)
    state = "on" if thinking else "off"
    command_names = "help|profile|tools|permissions|think|context|clear|max|quit"

    details = {
        "help": (
            "NAME\n"
            "  /help - 显示 Agent 内置命令帮助\n\n"
            "SYNOPSIS\n"
            "  /help\n"
            "  /help COMMAND\n\n"
            "SCOPE\n"
            "  COMMAND 可写为 max 或 /max；不带参数时列出全部命令。"),
        "profile": (
            "NAME\n"
            "  /profile - 显示当前运行时 profile 与能力边界\n\n"
            "SYNOPSIS\n"
            "  /profile\n\n"
            "SCOPE\n"
            f"  当前为 {profile_name} / ctx{context}；profile 只能在启动时选择。"),
        "tools": (
            "NAME\n"
            "  /tools - 列出当前 Agent 注册的原生工具\n\n"
            "SYNOPSIS\n"
            "  /tools\n\n"
            "SCOPE\n"
            f"  仅查看工具名，不执行工具；文件工具限制在 workspace {workspace}。"),
        "permissions": (
            "NAME\n"
            "  /permissions - 显示工具授权策略\n\n"
            "SYNOPSIS\n"
            "  /permissions\n\n"
            "SCOPE\n"
            "  list/read/search/git/calculate 自动执行；write_file/run_shell "
            "每次询问，默认拒绝。"),
        "think": (
            "NAME\n"
            "  /think - 查看或切换模型 thinking\n\n"
            "SYNOPSIS\n"
            "  /think\n"
            "  /think on|off\n\n"
            "RANGE\n"
            f"  on 或 off；当前为 {state}。切换只影响后续生成并占用同一 ctx{context}。"),
        "context": (
            "NAME\n"
            "  /context - 查看当前 prompt 的上下文 token 用量\n\n"
            "SYNOPSIS\n"
            "  /context\n\n"
            "SCOPE\n"
            f"  总容量固定为 ctx{context}，统计包含系统提示、工具定义和会话历史。"),
        "clear": (
            "NAME\n"
            "  /clear - 清空对话及工具历史\n\n"
            "SYNOPSIS\n"
            "  /clear\n"
            "  /reset\n\n"
            "SCOPE\n"
            "  仅重置当前会话上下文；不会退出进程或重新加载三个模型句柄。"),
        "max": (
            "NAME\n"
            "  /max - 查看或设置单次回答的生成 token 上限\n\n"
            "SYNOPSIS\n"
            "  /max\n"
            "  /max N\n\n"
            "RANGE\n"
            f"  配置范围 1..{maximum}，硬件范围 1..{hardware_maximum}；"
            f"当前为 {max_new}。实际输出还受剩余上下文限制。"),
        "quit": (
            "NAME\n"
            "  /quit - 退出 Agent REPL\n\n"
            "SYNOPSIS\n"
            "  /quit\n"
            "  /exit\n"
            "  Ctrl-D\n\n"
            "SCOPE\n"
            "  关闭当前常驻会话及模型句柄；不会修改 workspace 文件。"),
    }
    if canonical:
        if canonical in details:
            return details[canonical]
        return (
            f"未知帮助主题: {topic}\n"
            f"用法: /help [{command_names}]")

    return (
        "MiniCPM Agent 内置命令\n\n"
        "用法:\n"
        "  /help [COMMAND]\n\n"
        "命令:\n"
        "  /help [COMMAND]  显示全部帮助，或查看一个命令的详细说明\n"
        f"  /profile         显示运行 profile（当前 {profile_name}）\n"
        "  /tools           列出原生工具；只查看，不执行\n"
        "  /permissions     显示自动执行与逐次授权范围\n"
        f"  /think [on|off]  查看或切换 thinking（当前 {state}）\n"
        f"  /context         显示 ctx{context} prompt token 用量\n"
        "  /clear           清空对话和工具历史；别名 /reset\n"
        f"  /max [N]         查看或设置生成上限，N=1..{maximum}（当前 {max_new}）\n"
        "  /quit            退出；别名 /exit，也可按 Ctrl-D\n\n"
        "范围:\n"
        f"  workspace: {workspace}\n"
        f"  每个用户请求最多 {max_tool_steps} 轮工具调用。\n"
        "  上述命令由本地 REPL 处理，不发送给模型。\n\n"
        "详细帮助:\n"
        "  /help profile   查看 profile 的选择范围\n"
        "  /help max       查看 /max 的参数范围\n"
        "  /help think     查看 thinking 的作用域\n"
        "  /help permissions")


class TerminalUI:
    """Small dependency-free terminal UI for the resident board REPL.

    Animation and ANSI colour are intentionally restricted to a real TTY.
    Redirected output remains stable plain text for scripts and board logs.
    """

    _SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _AGENT_FACES = ("( o.o )", "( o.O )", "( O.o )", "( -.- )")
    _CHAT_FACES = ("( o.o )", "( -.- )", "( o.o )", "( ^.^ )")

    def __init__(self, *, active, context, brand="MiniCPM 5",
                 color="auto", spinner=True):
        self.active = bool(active)
        self.context = int(context)
        self.brand = str(brand)
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
        self._wait_index = 0
        self._wait_started = 0.0
        self._pet_can_animate = False
        self._pet_process_pid = None
        self._pet_stop_fd = None

    @property
    def pet_faces(self):
        return self._AGENT_FACES if self.brand == "HiAgent" else self._CHAT_FACES

    def _pet_line(self, face):
        return f"       {face}    {self.brand}"

    def _draw_pet_face(self, face):
        """Update the banner face while the cursor is directly below it."""
        line = self.paint(self._pet_line(face), "1;38;5;45")
        sys.stdout.write(f"\033[4A\r\033[2K{line}\033[4B\r")

    def paint(self, text, code):
        if not self.color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def banner(self):
        if not self.active:
            return
        lines = (
            "        /\\_/\\",
            self._pet_line("( o.o )"),
            "        > ^ <     Hi3403 端侧 AI",
        )
        shades = ("38;5;141", "1;38;5;45", "38;5;75")
        print("", flush=True)
        for line, shade in zip(lines, shades):
            print(self.paint(line, shade), flush=True)
        # Keep the pet as application branding. Runtime details belong to the
        # ready/profile status printed after the selected models are loaded.
        status = "     本地运行 · 隐私安全 · 实时响应"
        print(self.paint(status, "2;38;5;114"), flush=True)
        print("", flush=True)
        # The first wait begins immediately below this five-line banner, so
        # its worker can safely repaint only the face. Later waits may occur
        # anywhere in the transcript and must never move back into old text.
        self._pet_can_animate = True

    def start_wait(self, label):
        if not self.active:
            return
        self.stop_wait()
        self._wait_label = str(label)
        self._wait_started = time.perf_counter()
        if not self.animate:
            self._pet_can_animate = False
            print(self.paint(f"... {self._wait_label}", "2;38;5;75"), flush=True)
            return
        self._wait_stop.clear()
        # Render the first *changed* face synchronously. Tokenizer.from_file()
        # holds Python execution for about 2.5 s on Hi3403, so a worker-only
        # first frame otherwise appears to lag even though loading has begun.
        sys.stdout.write("\033[?25l")
        if self._pet_can_animate:
            self._draw_pet_face(self.pet_faces[1])
        frame = self.paint(self._SPINNER[0], "1;38;5;45")
        label = self.paint(self._wait_label, "38;5;75")
        sys.stdout.write(f"\r\033[2K{frame} {label}  0.0s")
        sys.stdout.flush()
        # The board tokenizer parser holds the GIL for roughly 2.5 seconds.
        # Keep only this first, pre-runtime animation in a tiny forked helper
        # so the pet remains smooth while the parent is inside native parsing.
        # Later waits use a thread: by then the tokenizer/runtime are live and
        # forking would be both unnecessary and unsafe.
        if self._pet_can_animate and self._start_pet_process():
            return
        self._wait_thread = threading.Thread(
            target=self._spin, name="minicpm-terminal-spinner", daemon=True)
        self._wait_thread.start()

    def update_wait(self, label):
        """Retitle a running wait without restarting it.

        The board ingests a prompt one token at a time, so a request can sit
        for tens of seconds behind a spinner that says nothing. The caller
        knows exactly how many tokens are left, and this turns that into the
        line the spinner is already redrawing.
        """
        running = self._wait_thread is not None
        if not self.active or not running:
            return
        # Set the label only. The spinner thread owns the cursor; a second
        # writer on the same line interleaves with its carriage return and
        # leaves the previous label's tail on screen.
        self._wait_label = str(label)

    def _render_wait_frame(self, index):
        self._wait_index = index
        elapsed = time.perf_counter() - self._wait_started
        if self._pet_can_animate and index % 4 == 0:
            face = self.pet_faces[
                ((index // 4) + 1) % len(self.pet_faces)]
            self._draw_pet_face(face)
        frame = self.paint(self._SPINNER[index % len(self._SPINNER)],
                           "1;38;5;45")
        label = self.paint(self._wait_label, "38;5;75")
        sys.stdout.write(f"\r\033[2K{frame} {label} {elapsed:4.1f}s")
        sys.stdout.flush()

    def _spin(self):
        index = 1
        while not self._wait_stop.is_set():
            if self._wait_stop.wait(0.08):
                break
            self._render_wait_frame(index)
            index += 1

    def _start_pet_process(self):
        if not hasattr(os, "fork"):
            return False
        read_fd, write_fd = os.pipe()
        try:
            pid = os.fork()
        except OSError:
            os.close(read_fd)
            os.close(write_fd)
            return False
        if pid == 0:  # pragma: no cover - exercised on the Hi3403 board
            os.close(write_fd)
            try:
                import select  # noqa: PLC0415
                index = 1
                while not select.select([read_fd], [], [], 0.08)[0]:
                    self._render_wait_frame(index)
                    index += 1
            except BaseException:
                pass
            finally:
                os.close(read_fd)
                os._exit(0)
        os.close(read_fd)
        self._pet_process_pid = pid
        self._pet_stop_fd = write_fd
        return True

    def stop_wait(self):
        process_pid = self._pet_process_pid
        if process_pid is not None:
            try:
                os.write(self._pet_stop_fd, b"x")
            except OSError:
                pass
            finally:
                os.close(self._pet_stop_fd)
                self._pet_stop_fd = None
            try:
                os.waitpid(process_pid, 0)
            except ChildProcessError:
                pass
            self._pet_process_pid = None
        thread = self._wait_thread
        if thread is None and process_pid is None:
            return
        if thread is not None:
            self._wait_stop.set()
            thread.join(timeout=1.0)
            self._wait_thread = None
        if self._pet_can_animate:
            self._draw_pet_face("( ^.^ )")
        self._pet_can_animate = False
        sys.stdout.write("\r\033[2K\033[?25h")
        sys.stdout.flush()

    def ready(self, handles, seconds):
        mark = self.paint("✓", "1;38;5;114")
        label = self.paint("ready", "1;38;5;45")
        detail = self.paint(
            f"{handles} handles · ctx{self.context} · {seconds:.1f}s",
            "2;38;5;250")
        print(f"{mark} {label} · loaded {detail}", flush=True)

    def prompt(self):
        if not self.color:
            return "You ❯ "
        if _readline is None:
            return f"{self.paint('You', '1;38;5;141')} {self.paint('❯', '1;38;5;45')} "

        # GNU readline must be told which prompt bytes are zero-width. Without
        # SOH/STX markers it counts ANSI colour sequences as visible columns,
        # leaving stale characters behind when users erase back to the prompt.
        def prompt_paint(text, code):
            return f"\001\033[{code}m\002{text}\001\033[0m\002"

        return f"{prompt_paint('You', '1;38;5;141')} " \
               f"{prompt_paint('❯', '1;38;5;45')} "

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

    def context_summary(self, record):
        """One line naming the largest consumer of the next prompt.

        The turn summary above reports generation. This reports the other half,
        which is the expensive one: a prompt token costs an ingest, so the
        segment holding most of the context is also holding most of the wait.
        """
        if not self.is_tty or not record["segments"]:
            return
        largest = max(record["segments"], key=lambda row: row["tokens"])
        used = record["total_tokens"]
        budget = record.get("budget_tokens")
        text = f"  context {used}"
        if budget:
            text += f"/{budget}"
        text += (f" tok · {largest['label'][:28]} {largest['share'] * 100:.0f}%"
                 f" · {record['new_tokens']} new")
        if "ingest_ms" in record and record["new_tokens"]:
            text += f" ({record['ingest_ms'] / 1000:.1f}s)"
        print(self.paint(text, "2;38;5;244"), flush=True)


class Merged:
    def __init__(self, *, executable, decode, prefill, head, library_paths,
                 embedding, context, timeout, tokenizer=None,
                 resident_kv=True, quiet_executor=False,
                 transformer_output_slots=None, prefill_runtime=None,
                 prefill_context=None):
        if prefill_context is None:
            prefill_context = context
        if prefill_context > context or prefill_context < 2:
            raise RuntimeError(
                "prefill context window must be in [2, context]")
        if prefill_context != context and prefill is None:
            raise RuntimeError(
                "a mixed prefill window requires a dedicated prefill handle")
        if prefill_context != context and transformer_output_slots is None:
            raise RuntimeError(
                "a mixed prefill window requires trusted static "
                "transformer output slots; dynamic probing assumes one "
                "uniform context")
        if prefill_runtime is None:
            prefill_runtime = prefill_runtime_contract.strict_s1_runtime(context)
        if prefill_runtime.context != context:
            raise RuntimeError(
                "prefill runtime registry context does not match model context")
        handlers = tuple(getattr(prefill_runtime, "handlers", ()))
        enabled_wide = {
            width for width in getattr(prefill_runtime, "enabled_widths", ())
            if width > 1
        }
        registered_wide = {
            handler.spec.width for handler in handlers
            if isinstance(
                handler, prefill_runtime_contract.WidePrefillHandler)
        }
        if enabled_wide != registered_wide or len(registered_wide) != len(handlers):
            raise RuntimeError(
                "prefill runtime enables wide blocks without an exact typed "
                "runtime handler")
        base_models = [decode] + ([prefill] if prefill else [])
        wide_models = []
        for offset, handler in enumerate(handlers):
            expected_index = len(base_models) + offset
            if handler.spec.model_index != expected_index:
                raise RuntimeError(
                    f"S{handler.spec.width} model index must be "
                    f"{expected_index} in this resident table")
            wide_models.append(handler.spec.model_path)
        self.context = context
        self.past = context - 1
        self.prefill_context = prefill_context
        self.timeout = timeout
        self.row_f16 = HEAD_DIM * 2
        self.cache_bytes = CHANNELS * self.past * HEAD_DIM * 2
        self.prefill_cache_bytes = (
            CHANNELS * (prefill_context - 1) * HEAD_DIM * 2)
        self.embed = open(embedding, "rb")
        self._embedding_half = struct.Struct(f"<{HIDDEN}e")
        self._embedding_f32 = struct.Struct(f"<{HIDDEN}f")
        self._embedding_c4 = struct.Struct("<" + "f12x" * HIDDEN)
        self._rope_inverse = tuple(
            1.0 / (gc.ROPE_THETA ** ((2 * i) / HEAD_DIM))
            for i in range(HEAD_DIM // 2))
        self.tokenizer = None
        self.resident_kv = bool(resident_kv)
        self._resident_tokens = []
        self._prefix_snapshots = {}
        self._next_prefix_snapshot_id = 1
        self._wide_session_discarded = False
        self.last_prefix_metrics = self._empty_prefix_metrics()
        self.prefill_runtime = prefill_runtime
        self.last_prefill_runtime = prefill_runtime.to_dict()
        self.last_prefill_schedule = prefill_runtime.plan(0, 0).to_dict()
        self.models = base_models + wide_models + [head]
        self.decode_index = 0
        self.prefill_index = 1 if prefill else 0
        self.head_index = len(self.models) - 1
        startup_started = time.perf_counter()
        self.startup_ms = {}
        try:
            # Revalidate at the last host-side boundary before the executor
            # reopens the route artifacts.  ``runner`` is the protocol module
            # used by this orchestrator, not this wrapper file itself.
            prefill_runtime.validate_live_startup_identity(
                executable=Path(executable),
                decode=Path(decode),
                prefill=Path(prefill) if prefill is not None else None,
                head=Path(head),
                embedding=Path(embedding),
                runner=Path(runner.__file__).resolve(),
            )
            try:
                self.process = probe._start(
                    executable, self.models, library_paths, 0,
                    quiet=quiet_executor)
            except TypeError as error:
                # A board may retain the pre-UI helper while only the REPL
                # server is refreshed. Keep that partial upgrade runnable.
                if "unexpected keyword argument 'quiet'" not in str(error):
                    raise
                if quiet_executor:
                    saved_stderr = os.dup(2)
                    try:
                        with open(os.devnull, "wb") as sink:
                            os.dup2(sink.fileno(), 2)
                            self.process = probe._start(
                                executable, self.models, library_paths, 0)
                    finally:
                        os.dup2(saved_stderr, 2)
                        os.close(saved_stderr)
                else:
                    self.process = probe._start(
                        executable, self.models, library_paths, 0)
            spawned_at = time.perf_counter()
            self.startup_ms["executor_spawn"] = (
                spawned_at - startup_started) * 1000.0
            self._deadline = time.monotonic() + timeout

            # The executor loads 1.58 GB of OMs in its own process. Parse the
            # tokenizer only after that work has started, overlapping the
            # board's ~2.5 s tokenizer cost with model loading.
            tokenizer_started = time.perf_counter()
            if tokenizer:
                from tokenizers import Tokenizer  # noqa: PLC0415
                self.tokenizer = Tokenizer.from_file(str(tokenizer))
            tokenizer_done = time.perf_counter()
            self.startup_ms["tokenizer"] = (
                tokenizer_done - tokenizer_started) * 1000.0

            self.descriptors = probe._read_ready(
                self.process.stdout, len(self.models), self._deadline)
            ready_at = time.perf_counter()
            self.startup_ms["executor_ready"] = (
                ready_at - spawned_at) * 1000.0
            self._validate_context_descriptors()
            prefill_runtime.validate_loaded_handlers(
                self.descriptors, self.models)
            probe_started = time.perf_counter()
            if transformer_output_slots is None:
                self.kv_slots = self._identify_kv()
                self.startup_ms["kv_contract"] = "dynamic-probe"
            else:
                self.kv_slots = self._bind_output_slots(
                    transformer_output_slots)
                self.startup_ms["kv_contract"] = "static"
            for handler in handlers:
                publisher = handler.spec.publisher
                self.kv_slots[handler.spec.model_index] = (
                    publisher.k_output_slot, publisher.v_output_slot)
            self.hidden_slots = {
                model: next(iter(
                    {0, 1, 2} - set(self.kv_slots[model])))
                for model in self.kv_slots
            }
            self.startup_ms["kv_identify"] = (
                time.perf_counter() - probe_started) * 1000.0
            self.startup_ms["total"] = (
                time.perf_counter() - startup_started) * 1000.0
        except BaseException:
            process = getattr(self, "process", None)
            if process is not None:
                self._terminate_process(process)
            self.embed.close()
            raise

    @staticmethod
    def _terminate_process(process):
        try:
            process.terminate()
        except Exception:
            return
        try:
            process.wait(timeout=30)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    # -- transport -------------------------------------------------------
    def _read(self, n):
        return runner._read_exact_until(self.process.stdout, n, self._deadline)

    def _respond(self, out_sizes, expected_model=None):
        (magic, version, status, model_index, count, error_bytes, reserved) = \
            runner._PERSISTENT_RESPONSE.unpack(
                self._read(runner._PERSISTENT_RESPONSE.size))
        expected_sizes = tuple(int(size) for size in out_sizes)
        if magic != runner.PERSISTENT_RESPONSE_MAGIC \
                or version != runner.PERSISTENT_PROTOCOL_VERSION \
                or reserved != 0 \
                or (expected_model is not None and model_index != expected_model) \
                or count > runner.PERSISTENT_MAX_MODEL_IO \
                or error_bytes > runner.PERSISTENT_MAX_ERROR_BYTES:
            raise RuntimeError("persistent executor response frame is invalid")
        if status:
            if count != 0:
                raise RuntimeError("failed executor response carries outputs")
            raise RuntimeError(self._read(error_bytes).decode("utf-8", "replace"))
        if error_bytes != 0 or count != len(expected_sizes):
            raise RuntimeError("persistent executor response contract mismatch")
        sizes = tuple(runner._PERSISTENT_U64.unpack(
            self._read(runner._PERSISTENT_U64.size))[0]
            for _ in range(count))
        if sizes != expected_sizes or any(
                size <= 0 or size > runner.PERSISTENT_MAX_TENSOR_BYTES
                for size in sizes):
            raise RuntimeError("persistent executor response sizes mismatch")
        return [self._read(size) for size in sizes]

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
        hidden_slots = getattr(self, "hidden_slots", {})
        if index in hidden_slots:
            return hidden_slots[index]
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

    def _validate_context_descriptors(self):
        """Fail before execution when an OM does not match runtime context.

        Under the mixed prefill contract the position-zero handle keeps
        the compiled ABI of its own (smaller) qualified context window,
        while the decode handle owns the full runtime context."""
        windows = {self.decode_index: (self.context, self.cache_bytes)}
        if self.prefill_index != self.decode_index:
            windows[self.prefill_index] = (
                self.prefill_context, self.prefill_cache_bytes)
        for model, (window, cache_bytes) in windows.items():
            expected = (
                window * 4,
                HEAD_DIM * HEAD_DIM * 4,
                cache_bytes,
                cache_bytes,
            )
            inputs, outputs = self.descriptors[model]
            if len(inputs) < 5 or len(outputs) != 3:
                raise RuntimeError(
                    f"model {model}: expected >=5 inputs/3 outputs for "
                    f"ctx{window}, got {len(inputs)}/{len(outputs)}")
            observed = tuple(inputs[1:5])
            if observed != expected:
                raise RuntimeError(
                    f"model {model}: descriptor/context mismatch for "
                    f"ctx{window}; mask/rope/k/v={observed}, "
                    f"expected={expected}")

    def _bind_output_slots(self, slots):
        values = tuple(slots)
        if len(values) != 3 or any(type(value) is not int for value in values) \
                or set(values) != {0, 1, 2}:
            raise RuntimeError(
                "transformer output slots must be distinct K,V,H indices "
                "covering 0,1,2")
        k_slot, v_slot, _hidden_slot = values
        bound = {}
        for model in {self.decode_index, self.prefill_index}:
            if len(self.descriptors[model][1]) != 3:
                raise RuntimeError(
                    f"model {model}: static output-slot contract requires "
                    "exactly 3 outputs")
            bound[model] = (k_slot, v_slot)
        return bound

    @staticmethod
    def _common_prefix(left, right):
        count = 0
        for old, new in zip(left, right):
            if old != new:
                break
            count += 1
        return count

    @staticmethod
    def _empty_prefix_metrics():
        return {
            "prompt_tokens_new": 0,
            "prompt_tokens_replayed": 0,
            "prefix_cache_hit": 0,
            "prefix_snapshot_hit": 0,
            "prefix_snapshot_created": 0,
            "prefix_snapshot_restore_ms": 0.0,
            "prefix_snapshot_evicted": 0,
        }

    def _prefix_plan(self, prompt_ids, reuse_prefix):
        previous = tuple(self._resident_tokens)
        common = self._common_prefix(previous, prompt_ids)
        if reuse_prefix and previous:
            # The cache contains hidden/KV for input tokens, not reusable
            # logits. Re-execute the final token on a complete-prefix hit.
            start = min(common, max(0, len(prompt_ids) - 1))
        else:
            start = 0
        hit = start if reuse_prefix else 0
        self.last_prefix_metrics = {
            "prompt_tokens_new": len(prompt_ids) - common,
            "prompt_tokens_replayed": common if previous and not reuse_prefix else 0,
            "prefix_cache_hit": hit,
            "prefix_snapshot_hit": 0,
            "prefix_snapshot_created": 0,
            "prefix_snapshot_restore_ms": 0.0,
        }
        return start

    def reset_prefix_cache(self):
        """Invalidate host metadata; stale device rows remain causally masked."""
        self._resident_tokens = []
        self.last_prefix_metrics = self._empty_prefix_metrics()

    def _snapshot_ranges(self, token_count):
        if type(token_count) is not int or token_count < 1 \
                or token_count > self.past:
            raise ValueError(
                f"snapshot token count must be in [1, {self.past}]")
        length = token_count * self.row_f16
        stride = self.past * self.row_f16
        return tuple(
            (slot, channel * stride, length)
            for slot in (3, 4)
            for channel in range(CHANNELS))

    def _save_input_snapshot(self, snapshot_id, token_count):
        ranges = self._snapshot_ranges(token_count)
        try:
            header = runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_SNAPSHOT_INPUTS,
                self.decode_index, snapshot_id, len(ranges), 0)
            runner._write_all(self.process.stdin, header)
            for slot, offset, length in ranges:
                runner._write_all(
                    self.process.stdin,
                    runner._PERSISTENT_SNAPSHOT_RANGE.pack(
                        slot, 0, offset, length))
            self.process.stdin.flush()
            self._respond((), self.decode_index)
        except Exception as error:
            self._discard_resident_session("snapshot-save", error)
            raise RuntimeError(
                f"resident snapshot save failed; session discarded: {error}"
            ) from error

    def _restore_input_snapshot(self, snapshot_id):
        try:
            runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_RESTORE_INPUTS,
                self.decode_index, snapshot_id, 0, 0))
            self.process.stdin.flush()
            self._respond((), self.decode_index)
        except Exception as error:
            self._discard_resident_session("snapshot-restore", error)
            raise RuntimeError(
                f"resident snapshot restore failed; session discarded: {error}"
            ) from error

    def _prepare_prefix_snapshot(self, key, tokens, prompt_ids):
        if key is None:
            return 0, 0.0
        fixed = tuple(int(token) for token in tokens)
        prompt = tuple(int(token) for token in prompt_ids)
        if not fixed or prompt[:len(fixed)] != fixed:
            raise RuntimeError(
                "fixed-prefix snapshot is not a prefix of the model prompt")
        existing = self._prefix_snapshots.get(str(key))
        if existing is None:
            return 0, 0.0
        # Touch for the sliding window: dict order is insertion order, so
        # re-inserting makes the first entry the least recently used one.
        self._prefix_snapshots[str(key)] = \
            self._prefix_snapshots.pop(str(key))
        snapshot_id, saved_tokens = existing
        if saved_tokens != fixed:
            raise RuntimeError(
                f"fixed-prefix snapshot key {key!r} changed token content")
        if self._common_prefix(self._resident_tokens, prompt) >= len(fixed):
            return 0, 0.0
        began = time.perf_counter()
        self._deadline = time.monotonic() + self.timeout
        self._restore_input_snapshot(snapshot_id)
        self._resident_tokens = list(fixed)
        return len(fixed), (time.perf_counter() - began) * 1000.0

    def _maybe_create_prefix_snapshot(self, key, tokens):
        if key is None or str(key) in self._prefix_snapshots:
            return 0
        fixed = tuple(int(token) for token in tokens)
        if tuple(self._resident_tokens) != fixed:
            return 0
        if self._next_prefix_snapshot_id <= PREFIX_SNAPSHOT_SLOTS:
            snapshot_id = self._next_prefix_snapshot_id
            self._next_prefix_snapshot_id += 1
        else:
            # The executor holds a fixed number of input snapshots, and the
            # key is the profile name, so a full cache used to cap how finely
            # tool disclosure could be split -- and it raised, ending a live
            # board session over a cache condition. Slide the window instead:
            # the least recently used slot is overwritten in place, and the
            # profile that lost it simply re-ingests its prefix the next time
            # it is selected, which is what every profile did before snapshots
            # existed.
            evicted = next(iter(self._prefix_snapshots))
            snapshot_id = self._prefix_snapshots.pop(evicted)[0]
            self.last_prefix_metrics["prefix_snapshot_evicted"] = 1
        self._deadline = time.monotonic() + self.timeout
        self._save_input_snapshot(snapshot_id, len(fixed))
        self._prefix_snapshots[str(key)] = (snapshot_id, fixed)
        return len(fixed)

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
        return self._respond(output_sizes, model)

    def _require_live_wide_session(self):
        if getattr(self, "_wide_session_discarded", False):
            raise RuntimeError(
                "wide session was discarded; rebuild the resident model table")

    def wide_copy_prefix(self, *, source_model, destination_model,
                         token_count):
        self._require_live_wide_session()
        self._copy_resident_prefix(
            source_model, destination_model, token_count)

    def wide_execute(self, *, spec, writes):
        self._require_live_wide_session()
        spec.verify_loaded_descriptor(self.descriptors[spec.model_index])
        payload_writes = tuple(
            (write.input_slot, write.offset, write.payload)
            for write in writes)
        outputs = self._run(
            spec.model_index, payload_writes, publish=False, output_count=0)
        if outputs:
            raise RuntimeError("wide execute unexpectedly returned host outputs")

    def wide_publish_kv(self, *, spec, start):
        self._require_live_wide_session()
        self._scatter_kv_rows(spec.model_index, start, spec.width)

    def wide_discard_session(self, *, spec, stage, error):
        """Poison the complete resident table after any wide-path failure.

        The current executor cannot unload only a wide handle. Killing the
        process is therefore the only safe rollback after an uncertain execute
        or publication. Host resident-token metadata is invalidated before a
        caller is told to rebuild; this method never resumes S1 in-place.
        """
        self._discard_resident_session(stage, error)

    def _discard_resident_session(self, stage, error):
        """Invalidate all resident host/device state after protocol failure."""
        self._wide_session_discarded = True
        self.reset_prefix_cache()
        self._prefix_snapshots = {}
        self._next_prefix_snapshot_id = 1
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    def _copy_resident_prefix(self, source_model, destination_model,
                              token_count):
        """Copy canonical packed FP16 K/V prefix into one wide handle.

        Cache ownership stays with the decode handle.  Before a lazy or shared
        wide model executes, this opcode-9 frame mirrors exactly
        ``[0, token_count)`` for every K/V channel.  The executor validates all
        96 records before copying any byte, so a descriptor or range mismatch
        cannot leave a partially updated destination.
        """
        values = (source_model, destination_model, token_count)
        if any(type(value) is not int for value in values):
            raise ValueError("resident prefix copy fields must be integers")
        if not 0 <= source_model < len(self.models) \
                or not 0 <= destination_model < len(self.models):
            raise ValueError("resident prefix copy model is out of range")
        if not 0 <= token_count <= self.past:
            raise ValueError("resident prefix copy escapes context-1 cache")
        if token_count == 0:
            return
        for model in (source_model, destination_model):
            inputs = self.descriptors[model][0]
            if len(inputs) < 5 or any(
                    inputs[slot] != self.cache_bytes for slot in (3, 4)):
                raise ValueError(
                    "resident prefix copy requires exact packed K/V inputs")

        stride = self.past * self.row_f16
        length = token_count * self.row_f16
        records = tuple(
            runner.ResidentInputCopy(
                destination_model=destination_model,
                destination_input=slot,
                destination_offset=channel * stride,
                source_model=source_model,
                source_input=slot,
                source_offset=channel * stride,
                length=length)
            for slot in (3, 4)
            for channel in range(CHANNELS))
        runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
            runner.PERSISTENT_REQUEST_MAGIC,
            runner.PERSISTENT_PROTOCOL_VERSION,
            runner.PERSISTENT_OP_COPY_INPUTS, 0, len(records), 0, 0))
        for record in records:
            runner._write_all(
                self.process.stdin, runner._PERSISTENT_INPUT_COPY.pack(
                    record.destination_model, record.destination_input,
                    record.destination_offset, record.source_model,
                    record.source_input, record.source_offset,
                    record.length, 0))
        self.process.stdin.flush()
        self._respond((), 0)

    def _scatter_kv_rows(self, source_model, start, width):
        """Publish ``width`` contiguous FP32 K/V rows into resident caches.

        A native prefill block publishes channel-major ``[48, width, 128]``
        FP32 tensors.  Executor opcode 6 converts each channel's complete
        block to FP16 RNE and writes it at the same absolute row in the
        canonical decode handle. Requiring the source output to be exact-sized is
        deliberate: a different pitch or hidden prefix must never be mistaken
        for this physical publisher ABI.

        The resident cache has ``context - 1`` rows.  A block ending at the
        final context position therefore cannot be partially scattered with
        opcode 6 (the source has no row-stride field); callers must omit that
        terminal scatter rather than silently shifting channel boundaries.
        """
        if type(source_model) is not int or not 0 <= source_model < len(self.models):
            raise ValueError("scatter source model is out of range")
        if type(start) is not int or type(width) is not int \
                or start < 0 or width < 1 or start + width > self.past:
            raise ValueError(
                "scatter range must fit entirely inside context-1 cache rows")
        if source_model not in self.kv_slots:
            raise ValueError("scatter source model has no bound K/V outputs")
        k_slot, v_slot = self.kv_slots[source_model]
        expected_source_bytes = CHANNELS * width * HEAD_DIM * 4
        source_outputs = self.descriptors[source_model][1]
        for slot in (k_slot, v_slot):
            if slot >= len(source_outputs) \
                    or source_outputs[slot] != expected_source_bytes:
                raise ValueError(
                    "scatter source does not match contiguous FP32 block ABI")

        base = start * self.row_f16
        stride = self.past * self.row_f16
        elements = width * HEAD_DIM
        destination_model = self.decode_index
        if not 0 <= destination_model < len(self.models):
            raise ValueError("scatter destination model is out of range")
        destination_inputs = self.descriptors[destination_model][0]
        if len(destination_inputs) < 5 or any(
                destination_inputs[slot] != self.cache_bytes
                for slot in (3, 4)):
            raise ValueError(
                "scatter destination does not expose the resident cache ABI")
        header = runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_SCATTER_F32_TO_F16,
                destination_model, 2, 0, 0)
        records = (
            runner._PERSISTENT_SCATTER_F32_TO_F16.pack(
                3, source_model, k_slot, 0, base, stride,
                CHANNELS, elements, 0),
            runner._PERSISTENT_SCATTER_F32_TO_F16.pack(
                4, source_model, v_slot, 0, base, stride,
                CHANNELS, elements, 0),
        )
        runner._write_all(self.process.stdin, header)
        for record in records:
            runner._write_all(self.process.stdin, record)
        self.process.stdin.flush()
        self._respond((), destination_model)

    def _scatter_kv(self, source_model, position):
        """Backward-compatible strict-S1 resident K/V publication."""
        self._scatter_kv_rows(source_model, position, 1)

    def _predict_from_resident_hidden(self, hidden):
        """Run the head once from an exact resident hidden tensor reference."""
        if not isinstance(hidden, prefill_runtime_contract.ResidentHidden):
            raise RuntimeError("head handoff requires a resident hidden reference")
        if not 0 <= hidden.model_index < len(self.models):
            raise RuntimeError("head handoff model index is out of range")
        source_outputs = self.descriptors[hidden.model_index][1]
        if not 0 <= hidden.output_slot < len(source_outputs) \
                or source_outputs[hidden.output_slot] != hidden.byte_size:
            raise RuntimeError("head handoff hidden descriptor drift")
        head_in = self.descriptors[self.head_index][0]
        if not head_in or head_in[0] != hidden.byte_size:
            raise RuntimeError(
                "head input descriptor does not exactly match resident hidden")
        began = time.perf_counter()
        runner._write_all(self.process.stdin, gc._frame(
            self.head_index, min(3, len(head_in)), (), [],
            [(0, 0, hidden.byte_size,
              hidden.model_index, hidden.output_slot)]))
        self.process.stdin.flush()
        self._respond((), self.head_index)
        head_at = time.perf_counter()
        runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
            runner.PERSISTENT_REQUEST_MAGIC,
            runner.PERSISTENT_PROTOCOL_VERSION,
            runner.PERSISTENT_OP_ARGMAX, self.head_index, 0, 0, 0))
        self.process.stdin.flush()
        (payload,) = self._respond((8,), self.head_index)
        argmax_at = time.perf_counter()
        predicted, _value = struct.unpack("<If", payload)
        return (
            int(predicted),
            (head_at - began) * 1000.0,
            (argmax_at - head_at) * 1000.0,
        )

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
                 capture_dir=None, capture_position=None, on_token=None,
                 on_prompt=None, reuse_prefix=False, prefix_snapshot_key=None,
                 prefix_snapshot_tokens=()):
        self.last_phase_steps = []
        # A failed wide transaction invalidates the shared resident process,
        # including the canonical decode cache.  Never reuse that process as
        # an apparent S1 fallback; the owner must construct a fresh Merged
        # session first.
        self._require_live_wide_session()
        host_kv = not self.resident_kv or kv_in is not None or kv_out is not None
        # Object-level diagnostic tests construct sessions without __init__;
        # they always run the uniform single-context contract.
        prefill_window = getattr(self, "prefill_context", self.context)
        if host_kv and prefill_window != self.context:
            raise RuntimeError(
                "host KV mirroring assumes one uniform context; the mixed "
                "prefill-window contract requires resident KV")
        if reuse_prefix and (host_kv or start != 0):
            raise ValueError(
                "prefix reuse requires resident KV and start position zero")
        if prefix_snapshot_key is not None and not reuse_prefix:
            raise ValueError("fixed-prefix snapshots require prefix reuse")
        fixed_tokens = tuple(prefix_snapshot_tokens or ())
        if self.resident_kv and start == 0:
            snapshot_hit, snapshot_restore_ms = self._prepare_prefix_snapshot(
                prefix_snapshot_key, fixed_tokens, prompt_ids)
            start = self._prefix_plan(prompt_ids, bool(reuse_prefix))
            self.last_prefix_metrics["prefix_snapshot_hit"] = snapshot_hit
            self.last_prefix_metrics["prefix_snapshot_restore_ms"] = \
                snapshot_restore_ms
            self._resident_tokens = self._resident_tokens[:start]
        else:
            self.reset_prefix_cache()
            self.last_prefix_metrics["prompt_tokens_new"] = len(prompt_ids)
        runtime_registry = getattr(self, "prefill_runtime", None)
        if runtime_registry is None:
            # Object-level diagnostic tests and older direct callers retain the
            # exact strict-S1 default. Runtime CLI paths always set a registry.
            runtime_registry = prefill_runtime_contract.strict_s1_runtime(
                self.context)
            self.prefill_runtime = runtime_registry
            self.last_prefill_runtime = runtime_registry.to_dict()
        hard_boundaries = ()
        if prefix_snapshot_key is not None \
                and start < len(fixed_tokens) < len(prompt_ids):
            hard_boundaries = (len(fixed_tokens),)
        schedule = runtime_registry.plan(
            start, len(prompt_ids), hard_boundaries=hard_boundaries)
        self.last_prefill_schedule = schedule.to_dict()

        wide_starts = {}
        for segment in schedule.segments:
            if segment.width == 1:
                continue
            handler = runtime_registry.handler_for(segment.width)
            for block_start in range(
                    segment.start, segment.stop, segment.width):
                block_stop = handler.validate_range(block_start)
                if block_stop > len(prompt_ids):
                    raise RuntimeError("wide block escaped the prompt range")
                if block_start < len(fixed_tokens) < block_stop:
                    raise RuntimeError(
                        "fixed-prefix snapshot boundary splits a wide block")
                wide_starts[block_start] = handler
        if wide_starts and (host_kv or capture_dir is not None
                            or kv_out is not None or stop_after is not None):
            raise RuntimeError(
                "wide prefill requires resident KV and has no legacy "
                "KV/capture control path")

        def scheduled_prefill_width(position):
            for segment in schedule.segments:
                if segment.start <= position < segment.stop:
                    return segment.width
            return None
        k_cache = bytearray(self.cache_bytes) if host_kv else None
        v_cache = bytearray(self.cache_bytes) if host_kv else None
        if kv_in is not None:
            blob = Path(kv_in).read_bytes()
            k_cache[:] = blob[:self.cache_bytes]
            v_cache[:] = blob[self.cache_bytes:2 * self.cache_bytes]
        token = prompt_ids[start]
        produced, produced_ids, steps = 0, [], []
        total = len(prompt_ids) + max_new
        position = start
        while position < total:
            if position >= self.context:
                return "context", produced_ids, steps
            began = time.perf_counter()
            self._deadline = time.monotonic() + self.timeout

            handler = wide_starts.get(position)
            if handler is not None:
                if len(self._resident_tokens) != position:
                    raise RuntimeError(
                        "resident prefix metadata diverged before wide execute")
                result = handler.execute(
                    transport=self,
                    canonical_model=self.decode_index,
                    start=position,
                    token_ids=prompt_ids[position:position + handler.spec.width],
                    embedding_row=self._hidden_input,
                    rope_row=self._rope_matrix_bytes,
                )
                if result.stop > len(prompt_ids):
                    raise RuntimeError("wide block escaped the prompt range")
                self._resident_tokens.extend(
                    int(value) for value in prompt_ids[result.start:result.stop])
                created = self._maybe_create_prefix_snapshot(
                    prefix_snapshot_key, fixed_tokens)
                if created:
                    self.last_prefix_metrics["prefix_snapshot_created"] = created
                kv_at = time.perf_counter()
                needs_prediction = result.stop >= len(prompt_ids)
                if not needs_prediction:
                    total_ms = (kv_at - began) * 1000.0
                    steps.extend([0.0] * (result.width - 1) + [total_ms])
                    self.last_phase_steps.append({
                        "position": result.start,
                        "stop": result.stop,
                        "token_count": result.width,
                        "prefill_width": result.width,
                        "prepare_ms": result.prepare_ms,
                        "prefix_copy_ms": result.prefix_copy_ms,
                        "transformer_ms": result.execute_ms,
                        "kv_ms": result.publish_ms,
                        "head_execute_ms": 0.0,
                        "argmax_ms": 0.0,
                        "head_skipped": True,
                        "total_ms": total_ms,
                    })
                    position = result.stop
                    token = prompt_ids[position]
                    continue

                predicted, head_ms, argmax_ms = \
                    self._predict_from_resident_hidden(result.hidden)
                finished_at = time.perf_counter()
                total_ms = (finished_at - began) * 1000.0
                steps.extend([0.0] * (result.width - 1) + [total_ms])
                self.last_phase_steps.append({
                    "position": result.start,
                    "stop": result.stop,
                    "token_count": result.width,
                    "prefill_width": result.width,
                    "prepare_ms": result.prepare_ms,
                    "prefix_copy_ms": result.prefix_copy_ms,
                    "transformer_ms": result.execute_ms,
                    "kv_ms": result.publish_ms,
                    "head_execute_ms": head_ms,
                    "argmax_ms": argmax_ms,
                    "head_skipped": False,
                    "hidden_handoff": {
                        "model_index": result.hidden.model_index,
                        "output_slot": result.hidden.output_slot,
                        "position": result.hidden.position,
                    },
                    "total_ms": total_ms,
                })
                produced_ids.append(predicted)
                produced += 1
                if on_token is not None:
                    on_token(tuple(produced_ids))
                if predicted in eos:
                    return "eos", produced_ids, steps
                if produced >= max_new:
                    return "max", produced_ids, steps
                position = result.stop
                token = predicted
                continue

            model = (self.prefill_index if position == 0 else self.decode_index)
            desc_in = self.descriptors[model][0]

            # Position 0 runs on the prefill handle, whose mask carries the
            # width of its own compiled window (== context except under the
            # mixed prefill contract, where a smaller qualified window
            # bootstraps a longer decode context).
            mask_width = (
                prefill_window if position == 0 else self.context)
            mask = bytearray(struct.pack("<f", gc.MASK_NEG) * mask_width)
            for slot in range(position):
                struct.pack_into("<f", mask, slot * 4, 0.0)
            struct.pack_into("<f", mask, (mask_width - 1) * 4, 0.0)

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
                if position < self.past:
                    for mirror, blob in ((k_cache, published[k_slot]),
                                         (v_cache, published[v_slot])):
                        values = struct.unpack(f"<{CHANNELS * HEAD_DIM}f", blob)
                        for c in range(CHANNELS):
                            at = (c * self.past + position) * self.row_f16
                            mirror[at:at + self.row_f16] = struct.pack(
                                f"<{HEAD_DIM}e",
                                *values[c * HEAD_DIM:(c + 1) * HEAD_DIM])
            else:
                # Cache inputs contain context-1 past rows. At the final legal
                # position there is no future token that can consume a current
                # row, so scattering it would exceed the descriptor by one.
                if position < self.past:
                    self._scatter_kv(model, position)
                if len(self._resident_tokens) != position:
                    raise RuntimeError(
                        "resident prefix metadata diverged from device position")
                self._resident_tokens.append(int(token))
                created = self._maybe_create_prefix_snapshot(
                    prefix_snapshot_key, fixed_tokens)
                if created:
                    self.last_prefix_metrics["prefix_snapshot_created"] = created
            kv_at = time.perf_counter()

            # Every token before the end of the supplied prompt already has a
            # known successor. Running the 200 MB vocabulary head and argmax
            # at those positions computes a prediction that is immediately
            # discarded. Preserve the legacy diagnostic contract when a
            # caller explicitly asks to stop and export KV at this position;
            # normal prompt ingestion can proceed directly to the next token.
            needs_prediction = position + 1 >= len(prompt_ids) or (
                kv_out is not None and stop_after == position)
            if not needs_prediction:
                finished_at = kv_at
                steps.append((finished_at - began) * 1000.0)
                self.last_phase_steps.append({
                    "position": position,
                    "stop": position + 1,
                    "token_count": 1,
                    "prefill_width": scheduled_prefill_width(position),
                    "prepare_ms": (prepared_at - began) * 1000.0,
                    "prefix_copy_ms": 0.0,
                    "transformer_ms": (transformer_at - prepared_at) * 1000.0,
                    "kv_ms": (kv_at - transformer_at) * 1000.0,
                    "head_execute_ms": 0.0,
                    "argmax_ms": 0.0,
                    "head_skipped": True,
                    "total_ms": (finished_at - began) * 1000.0,
                })
                token = prompt_ids[position + 1]
                position += 1
                if on_prompt is not None:
                    on_prompt(position, len(prompt_ids))
                continue

            src = self._hidden_output(model)
            hidden = prefill_runtime_contract.ResidentHidden(
                model_index=model,
                output_slot=src,
                byte_size=self.descriptors[model][1][src],
                position=position,
            )
            predicted, head_ms, argmax_ms = \
                self._predict_from_resident_hidden(hidden)
            finished_at = time.perf_counter()
            steps.append((finished_at - began) * 1000.0)
            self.last_phase_steps.append({
                "position": position,
                "stop": position + 1,
                "token_count": 1,
                "prefill_width": scheduled_prefill_width(position),
                "prepare_ms": (prepared_at - began) * 1000.0,
                "prefix_copy_ms": 0.0,
                "transformer_ms": (transformer_at - prepared_at) * 1000.0,
                "kv_ms": (kv_at - transformer_at) * 1000.0,
                "head_execute_ms": head_ms,
                "argmax_ms": argmax_ms,
                "head_skipped": False,
                "total_ms": (finished_at - began) * 1000.0,
            })
            if kv_out is not None and stop_after is not None \
                    and position == stop_after:
                Path(kv_out).write_bytes(bytes(k_cache) + bytes(v_cache))
                return "stop", [int(predicted)], steps

            if position + 1 < len(prompt_ids):
                token = prompt_ids[position + 1]
                position += 1
                continue
            produced_ids.append(predicted)
            produced += 1
            if on_token is not None:
                on_token(tuple(produced_ids))
            if predicted in eos:
                return "eos", produced_ids, steps
            if produced >= max_new:
                return "max", produced_ids, steps
            token = predicted
            position += 1
        return "max", produced_ids, steps

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.poll() is None:
                self._deadline = time.monotonic() + 30.0
                runner._write_all(
                    process.stdin, runner._PERSISTENT_REQUEST.pack(
                        runner.PERSISTENT_REQUEST_MAGIC,
                        runner.PERSISTENT_PROTOCOL_VERSION,
                        runner.PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0))
                process.stdin.flush()
                self._respond((), 0)
                # Shutdown ACK precedes native model/device teardown. Wait for
                # natural exit before lifecycle code reports MMZ as released.
                process.wait(timeout=30)
        except (BrokenPipeError, OSError, RuntimeError,
                subprocess.TimeoutExpired):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            self.process = None
            self.embed.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent-executor", type=Path, required=True)
    ap.add_argument("--decode-model", type=Path)
    ap.add_argument("--prefill-model", type=Path)
    ap.add_argument("--head-model", type=Path)
    ap.add_argument(
        "--profile",
        help="runtime profile name or JSON path (ctx128/1024/4096/8192)")
    ap.add_argument(
        "--deployment-root", type=Path, default=HERE.parent.parent,
        help="root used to resolve model paths in a runtime profile")
    ap.add_argument(
        "--allow-unqualified-profile", action="store_true",
        help="allow a pending profile for controlled development only")
    ap.add_argument(
        "--prefill-activation-manifest", type=Path,
        help="optional release-v4 native-prefill activation manifest")
    ap.add_argument(
        "--available-bytes", type=int,
        help="live MMZ bytes available before optional prefill activation")
    ap.add_argument(
        "--base-resident-bytes", type=int,
        help="live bytes occupied by strict-S1/base resident handles")
    ap.add_argument(
        "--reserve-bytes", type=int,
        help="MMZ safety reserve retained after optional prefill activation")
    ap.add_argument("--library-path", type=Path, action="append", default=[])
    ap.add_argument("--embedding", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument(
        "--host-kv", action="store_true",
        help="compatibility path: return packed K/V to Python and write it back")
    ap.add_argument(
        "--reuse-session-kv", action="store_true",
        help="reuse the token-exact resident prefix across chat/agent turns")
    ap.add_argument(
        "--fixed-prefix-snapshots", action="store_true",
        help="snapshot/restore immutable Agent system/tool K/V prefixes")
    ap.add_argument(
        "--transformer-output-slots", type=parse_transformer_output_slots,
        metavar="K,V,H",
        help="trusted transformer K,V,hidden slots; omit for dynamic probing")
    ap.add_argument("--context", type=int)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--prompt-ids", action="append", default=[])
    ap.add_argument(
        "--interactive", action="store_true",
        help="legacy raw-completion REPL")
    ap.add_argument(
        "--chat", action="store_true",
        help="run the official MiniCPM5 no-tools chat REPL")
    ap.add_argument(
        "--agent", action="store_true",
        help="run the native MiniCPM5 chat/tool-calling agent REPL")
    ap.add_argument(
        "--thinking", action="store_true",
        help="start agent mode with model thinking enabled")
    ap.add_argument(
        "--workspace", type=Path, default=Path.cwd(),
        help="filesystem boundary for agent tools (default: current directory)")
    ap.add_argument(
        "--max-tool-steps", type=int,
        help="maximum model/tool rounds per user turn")
    ap.add_argument("--max-new", type=int)
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
    runtime_profile = None
    mode = "agent" if args.agent else "chat" if args.chat else "completion"
    if args.profile:
        try:
            runtime_profile = profile_contract.load_runtime_profile(
                args.profile, HERE.parent / "profiles")
            runtime_profile.require_mode(
                mode, allow_unqualified=args.allow_unqualified_profile)
        except profile_contract.ProfileError as error:
            ap.error(str(error))
        configured = runtime_profile.model_paths(args.deployment_root)
        for attribute, role in (
                ("decode_model", "decode"), ("prefill_model", "prefill"),
                ("head_model", "head")):
            supplied = getattr(args, attribute)
            expected = configured[role]
            if supplied is not None and supplied.expanduser().resolve() != expected:
                ap.error(
                    f"--{attribute.replace('_', '-')} conflicts with profile "
                    f"{runtime_profile.name}: expected {expected}")
            setattr(args, attribute, expected)
        if args.context is not None and args.context != runtime_profile.context:
            ap.error(
                f"--context {args.context} conflicts with profile "
                f"{runtime_profile.name} ctx{runtime_profile.context}")
        args.context = runtime_profile.context
        args.prefill_context = runtime_profile.prefill_window
        if args.host_kv and args.prefill_context != args.context:
            ap.error(
                f"--host-kv is incompatible with profile "
                f"{runtime_profile.name}: its mixed prefill window "
                f"({args.prefill_context} < {args.context}) requires "
                "resident KV")
        if args.max_new is None:
            args.max_new = runtime_profile.default_max_new
        if args.max_tool_steps is None:
            args.max_tool_steps = runtime_profile.max_tool_steps
        args.max_new_limit = runtime_profile.max_new_limit
        args.profile_name = runtime_profile.name
        profile_slots = (
            *runtime_profile.kv_output_slots,
            runtime_profile.hidden_output_slot)
        if args.transformer_output_slots is not None \
                and args.transformer_output_slots != profile_slots:
            ap.error(
                "--transformer-output-slots conflicts with profile "
                f"{runtime_profile.name}: expected "
                f"{','.join(map(str, profile_slots))}")
        args.transformer_output_slots = profile_slots
    else:
        args.context = 1024 if args.context is None else args.context
        args.prefill_context = args.context
        args.max_new = 128 if args.max_new is None else args.max_new
        args.max_tool_steps = 4 if args.max_tool_steps is None else args.max_tool_steps
        args.max_new_limit = args.context - 1
        args.profile_name = "legacy"
        if args.decode_model is None or args.head_model is None:
            ap.error("--decode-model and --head-model are required without --profile")
    if args.max_new < 1 or args.max_new > args.max_new_limit:
        ap.error(f"--max-new must be in [1, {args.max_new_limit}]")
    if args.host_kv and args.reuse_session_kv:
        ap.error("--reuse-session-kv requires resident KV")
    if args.fixed_prefix_snapshots and not args.agent:
        ap.error("--fixed-prefix-snapshots requires --agent")
    if args.fixed_prefix_snapshots and not args.reuse_session_kv:
        ap.error("--fixed-prefix-snapshots requires --reuse-session-kv")
    if (args.capture_dir is None) != (args.capture_position is None):
        ap.error("--capture-dir and --capture-position must be provided together")
    if args.capture_position is not None and args.capture_position < 0:
        ap.error("--capture-position must be nonnegative")
    if sum((args.interactive, args.chat, args.agent)) > 1:
        ap.error("--interactive, --chat and --agent are mutually exclusive")
    if (args.interactive or args.chat or args.agent) and args.tokenizer is None:
        ap.error("--interactive/--chat/--agent requires --tokenizer")
    if (args.interactive or args.chat or args.agent) and (
            args.prompt_ids or args.start_position != 0 or args.kv_in is not None
            or args.kv_out is not None or args.stop_after is not None
            or args.capture_dir is not None):
        ap.error("REPL modes are incompatible with prompt-id/KV/capture controls")
    if (args.chat or args.agent) and args.prompt:
        ap.error("--chat/--agent are interactive REPLs; omit --prompt")
    if args.thinking and not args.agent:
        ap.error("--thinking requires --agent")
    if args.agent and (args.max_tool_steps < 1 or args.max_tool_steps > 16):
        ap.error("--max-tool-steps must be in [1, 16] for agent mode")
    if not args.agent and (args.max_tool_steps < 0 or args.max_tool_steps > 16):
        ap.error("--max-tool-steps must be in [0, 16]")
    if not args.interactive and not args.chat and not args.agent \
            and not args.prompt and not args.prompt_ids:
        ap.error("provide --prompt, --prompt-ids or a REPL mode")

    try:
        prefill_runtime = prefill_runtime_contract.load_runtime_registry(
            activation_manifest=args.prefill_activation_manifest,
            deployment_root=args.deployment_root,
            context=args.context,
            available_bytes=args.available_bytes,
            base_resident_bytes=args.base_resident_bytes,
            reserve_bytes=args.reserve_bytes,
        )
    except (prefill_runtime_contract.PrefillRuntimeError,
            prefill_runtime_contract.activation_contract.PrefillActivationError,
            OSError, UnicodeError) as error:
        ap.error(f"native prefill activation failed: {error}")
    prefill_runtime_report = prefill_runtime.to_dict()
    prefill_label = "/".join(
        f"S{width}" for width in prefill_runtime.enabled_widths)

    ui = TerminalUI(active=args.interactive or args.chat or args.agent,
                    context=args.context,
                    brand="HiAgent" if args.agent else "MiniCPM 5",
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
                         transformer_output_slots=args.transformer_output_slots,
                         prefill_runtime=prefill_runtime,
                         prefill_context=args.prefill_context,
                         quiet_executor=(args.interactive or args.chat or args.agent)
                         and not args.verbose_executor)
    finally:
        ui.stop_wait()
    load_ms = (time.perf_counter() - began) * 1000.0
    if args.interactive or args.chat or args.agent:
        ui.ready(len(session.models), load_ms / 1000.0)
        startup = getattr(session, "startup_ms", None)
        if startup:
            kv_status = (
                "kv-slots=static" if startup["kv_contract"] == "static"
                else f"kv-probe={startup['kv_identify'] / 1000.0:.1f}s")
            ui.info(
                "startup: "
                f"executor-ready={startup['executor_ready'] / 1000.0:.1f}s · "
                f"tokenizer={startup['tokenizer'] / 1000.0:.1f}s overlapped · "
                f"{kv_status}")
        window_label = (
            "" if args.prefill_context == args.context
            else f" · prefill-window={args.prefill_context}")
        ui.info(
            f"profile={args.profile_name} · ctx{args.context} · "
            f"max-new={args.max_new} · configured-limit={args.max_new_limit} · "
            f"prefill={prefill_label}{window_label}")
    else:
        print(f"loaded {len(session.models)} handles in {load_ms / 1000:.1f} s; "
              f"kv slots {session.kv_slots}", flush=True)
    results = []

    def current_prefill_schedule():
        """Return runtime telemetry without requiring legacy test sessions to opt in."""
        schedule = getattr(session, "last_prefill_schedule", None)
        if schedule is not None:
            return schedule
        return prefill_schedule.plan_prefill(
            0, 0, context=args.context).to_dict()

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
                  "phase_ms": session.last_phase_steps,
                  "prefill_schedule": current_prefill_schedule()}
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
                            "phase_ms": session.last_phase_steps,
                            "prefill_schedule": current_prefill_schedule()})
        if args.agent:
            tool_output_limit = runtime_profile.max_tool_output_chars \
                if runtime_profile else agent.MAX_TOOL_OUTPUT_CHARS
            workspace_tools = agent.WorkspaceTools(
                args.workspace, max_output_chars=tool_output_limit)
            # The system prompt is disclosed with the tools, not beside them.
            # Tool discipline is unreadable advice on a turn that discloses no
            # tools, and it is not free: the paragraph below is 124 of the 136
            # tokens a greeting used to spend, or 9.9 s of ingest on ctx1024.
            SYSTEM_BASE = ("You are MiniCPM Agent running locally on a "
                           "Hi3403. Keep answers concise.")
            SYSTEM_TOOLS = (
                "You are MiniCPM Agent running locally on a Hi3403. "
                f"The configured workspace root is {workspace_tools.root}. "
                "Use path='.' for that root. Never ask the user for a path "
                "that a filesystem tool can inspect. When filesystem facts "
                "are requested, call the appropriate tool immediately; do "
                "not merely say that you will call it. Inspect before changing, "
                "and never claim a tool succeeded unless its response says ok. "
                "When a tool response is already present, answer from that "
                "evidence instead of announcing or repeating the same call. "
                "Summarize tool results and do not repeat long output verbatim. "
                "Keep final answers concise.")

            def system_for(profile):
                return {"role": "system",
                        "content": SYSTEM_TOOLS
                        if workspace_tools.names_for_profile(profile)
                        else SYSTEM_BASE}

            system_message = system_for("all")
            messages = [system_message]
            repl_max_new = args.max_new
            thinking_enabled = bool(args.thinking)
            active_schema_profile = "all"
            compact_threshold = runtime_profile.compact_at_tokens \
                if runtime_profile else max(1, args.context - 128)
            compact_reserve = runtime_profile.reserve_tokens \
                if runtime_profile else min(128, args.context // 4)
            compact_target = max(
                64, compact_threshold - compact_reserve)
            ui.info(
                "Agent ready · /help · /tools · /think on|off · "
                "/context · /clear · /quit")

            def agent_prompt(prompt_messages=None):
                """Render the prompt and encode it once, keeping the offsets.

                The offsets are what lets the context ledger say which part of
                the prompt each token belongs to, and they come free with the
                encoding the runtime performs anyway.
                """
                turn = list(messages if prompt_messages is None
                            else prompt_messages)
                if turn and turn[0].get("role") == "system":
                    turn[0] = system_for(active_schema_profile)
                rendered = agent.render_chat(
                    turn,
                    workspace_tools.definitions_for_profile(
                        active_schema_profile),
                    add_generation_prompt=True,
                    enable_thinking=thinking_enabled)
                return rendered, session.tokenizer.encode(
                    rendered, add_special_tokens=False)

            def agent_ids(prompt_messages=None):
                return list(agent_prompt(prompt_messages)[1].ids)

            # Measured on this board, by this process, from the prompt
            # positions of every generate call so far.
            prompt_rate = {"ms": None, "tokens": 0, "total_ms": 0.0}

            def agent_ledger():
                rendered, encoding = agent_prompt()
                ids = list(encoding.ids)
                # What is resident is the COMMON PREFIX of the cache and this
                # prompt, not the cache length: the cache also holds the
                # positions generated after the last prompt, and those are not
                # a prefix of the next one. Using the length would report work
                # as already paid for that the board is about to redo.
                resident = session._common_prefix(
                    getattr(session, "_resident_tokens", ()), ids)
                return ledger.measure(
                    rendered, ledger.segment_prompt(rendered),
                    encoding.offsets, resident_tokens=resident,
                    prompt_token_ms=prompt_rate["ms"], capacity=args.context,
                    reserve_tokens=compact_reserve)

            def agent_fixed_prefix_ids(profile):
                rendered = agent.render_chat(
                    [system_for(profile)],
                    workspace_tools.definitions_for_profile(profile),
                    add_generation_prompt=False,
                    enable_thinking=False)
                return tuple(session.tokenizer.encode(
                    rendered, add_special_tokens=False).ids)

            def approve_tool(preview):
                ui.stop_wait()
                label = ui.paint("Permission required", "1;38;5;214")
                print(f"{label}: {preview}", flush=True)
                try:
                    answer = input(ui.paint("Allow once? [y/N] ", "38;5;214"))
                except (EOFError, KeyboardInterrupt):
                    print("", flush=True)
                    return False
                return answer.strip().lower() in {"y", "yes"}

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
                if spec == "/help" or spec.startswith("/help "):
                    topic = spec[5:].strip()
                    ui.info(agent_command_help(
                        topic, context=args.context, max_new=repl_max_new,
                        thinking=thinking_enabled,
                        max_tool_steps=args.max_tool_steps,
                        workspace=workspace_tools.root,
                        max_new_limit=args.max_new_limit,
                        profile_name=args.profile_name))
                    continue
                if spec == "/profile":
                    capability = "agent" if runtime_profile is None \
                        or runtime_profile.agent else "chat-only"
                    status = runtime_profile.status if runtime_profile else "legacy"
                    ui.info(
                        f"profile={args.profile_name}; ctx={args.context}; "
                        f"capability={capability}; status={status}; "
                        f"prefill={prefill_label}; "
                        f"prefill-status={prefill_runtime_report['status']}; "
                        f"max-new=1..{args.max_new_limit}")
                    continue
                if spec == "/tools":
                    ui.info("Tools: " + ", ".join(workspace_tools.names))
                    continue
                if spec == "/permissions":
                    ui.info(
                        "list/read/search/git/calculate run automatically; "
                        "write_file and run_shell require approval every time.")
                    continue
                if spec == "/think":
                    ui.info(f"thinking={'on' if thinking_enabled else 'off'}")
                    continue
                if spec in {"/think on", "/think off"}:
                    thinking_enabled = spec.endswith("on")
                    ui.info(f"thinking={'on' if thinking_enabled else 'off'}")
                    continue
                if spec.startswith("/think "):
                    ui.info("Usage: /think on|off")
                    continue
                if spec == "/context":
                    try:
                        record = agent_ledger()
                    except Exception as error:
                        ui.info(f"context inspection failed: {error}")
                    else:
                        used = record["total_tokens"]
                        ui.info(
                            f"ctx{args.context}: {used} prompt tokens used, "
                            f"{max(0, args.context - used)} available")
                        for line in ledger.render(record).splitlines():
                            ui.info(line)
                    continue
                if spec in {"/clear", "/reset"}:
                    messages[:] = [system_message]
                    getattr(session, "reset_prefix_cache", lambda: None)()
                    ui.info("Conversation and tool history cleared.")
                    continue
                if spec == "/max":
                    ui.info(
                        f"max-new={repl_max_new}; configured=1..{args.max_new_limit}; "
                        f"hardware=1..{args.context - 1}")
                    continue
                if spec.startswith("/max "):
                    try:
                        requested = int(spec.split(None, 1)[1])
                    except ValueError:
                        ui.info("Usage: /max N")
                        continue
                    if requested < 1 or requested > args.max_new_limit:
                        ui.info(f"N must be in [1, {args.max_new_limit}]")
                        continue
                    repl_max_new = requested
                    ui.info(f"max-new={repl_max_new}")
                    continue

                turn_started = time.perf_counter()
                previous_assistant = next((
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "assistant"), "")
                active_schema_profile = workspace_tools.select_schema_profile(
                    spec, has_context=bool(previous_assistant))
                messages.append({"role": "user", "content": spec})
                route_started = time.perf_counter()
                route_decision = agent.route_obvious_read_only(
                    spec, previous_assistant)
                route_ms = (time.perf_counter() - route_started) * 1000.0
                tool_total_ms = 0.0
                initial_tool_rounds = 0
                turn_prompt_messages = None
                turn_prefix_metrics = {
                    "prompt_tokens_new": 0, "prompt_tokens_replayed": 0,
                    "prefix_cache_hit": 0,
                    "prefix_snapshot_hit": 0,
                    "prefix_snapshot_created": 0,
                    "prefix_snapshot_restore_ms": 0.0}
                turn_rebase_metrics = {
                    "context_rebased": False,
                    "context_tokens_before": 0,
                    "context_tokens_after": 0,
                    "context_turns_compacted": 0}
                if route_decision is not None:
                    if len(route_decision.tool_calls) != 1:
                        raise RuntimeError(
                            "deterministic route must contain exactly one tool call")
                    routed_call = route_decision.tool_calls[0]
                    routed_xml = agent.format_tool_call(routed_call)
                    messages.append({"role": "assistant", "content": routed_xml})
                    ui.info(f"⚙ {workspace_tools.preview(routed_call)}")
                    tool_started = time.perf_counter()
                    tool_result = workspace_tools.execute(routed_call)
                    tool_ms = (time.perf_counter() - tool_started) * 1000.0
                    tool_total_ms += tool_ms
                    messages.append({
                        "role": "tool",
                        "content": workspace_tools.for_model(tool_result)})
                    decoded = json.loads(tool_result)
                    mark = "✓" if decoded["ok"] else "✗"
                    if route_decision.mode == "DIRECT_TOOL":
                        final_text = workspace_tools.for_direct(
                            routed_call, tool_result)
                        messages.append({
                            "role": "assistant", "content": final_text})
                        ui.info(
                            f"{mark} {routed_call.name} · {tool_ms:.1f} ms · "
                            "model skipped")
                        print(final_text, flush=True)
                        total_ms = (time.perf_counter() - turn_started) * 1000.0
                        results.append({
                            "mode": "agent", "prompt": spec,
                            "text": final_text, "ids": [],
                            "reason": "tool_direct", "thinking": False,
                            "tool_rounds": 1, "step_ms": [], "phase_ms": [],
                            "route_mode": route_decision.mode,
                            "route_reason": route_decision.reason,
                            "route_confidence": route_decision.confidence,
                            "response_policy": route_decision.response_policy,
                            "schema_profile": route_decision.schema_profile,
                            "route_ms": route_ms, "tool_ms": tool_total_ms,
                            "model_called": False, "total_ms": total_ms,
                            **turn_prefix_metrics,
                        })
                        continue
                    active_schema_profile = (
                        route_decision.schema_profile
                        if decoded.get("next_offset") is not None else "none")
                    verified_result = workspace_tools.for_model(tool_result)
                    if any("\u3400" <= ch <= "\u9fff" for ch in spec):
                        evidence_prompt = (
                            "本地工具已经成功执行。不要调用工具，也不要描述调用过程。"
                            "请只用中文直接、简洁地回答原始请求，且只能依据以下"
                            "已验证证据。\n\n"
                            f"原始请求：{spec}\n\n"
                            f"已验证结果：\n{verified_result}")
                    else:
                        evidence_prompt = (
                            "A local tool has already completed successfully. "
                            "Do not call or announce a tool. Answer the original "
                            "request directly and concisely using only this "
                            "verified evidence.\n\n"
                            f"Original request: {spec}\n\n"
                            f"Verified result:\n{verified_result}")
                    turn_prompt_messages = [system_message, {
                        "role": "user", "content": evidence_prompt,
                    }]
                    summary = decoded["output"].splitlines()[0][:160]
                    ui.info(f"{mark} {routed_call.name}: {summary}")
                    initial_tool_rounds = 1
                turn_finished = False
                # A model that cannot satisfy a request tends to repeat the
                # attempt verbatim rather than change it. Every repeat costs a
                # full prompt ingest and appends its own failure to the
                # transcript, so the prompt grows while the answer does not:
                # one measured turn produced the same non-existent tool four
                # times, 801 to 1185 tokens, 231 s, no result. The host stops
                # that; spending model rounds to notice it is the expensive
                # way to learn nothing.
                attempted_calls = set()
                for tool_round in range(
                        initial_tool_rounds, args.max_tool_steps + 1):
                    ids = agent_ids(turn_prompt_messages)
                    if turn_prompt_messages is None and tool_round == 0 \
                            and len(ids) >= compact_threshold:
                        original_messages = list(messages)
                        before = len(ids)
                        best_messages = original_messages
                        best_ids = ids
                        best_info = {"changed": False, "turns_compacted": 0}
                        for keep_turns, memory_chars in (
                                (2, 1200), (1, 800), (0, 400)):
                            candidate, info = agent.compact_agent_messages(
                                original_messages,
                                keep_recent_turns=keep_turns,
                                memory_chars=memory_chars)
                            if not info["changed"]:
                                continue
                            candidate_ids = agent_ids(candidate)
                            if len(candidate_ids) < len(best_ids):
                                best_messages = candidate
                                best_ids = candidate_ids
                                best_info = info
                            if len(candidate_ids) <= compact_target:
                                break
                        if best_info["changed"]:
                            messages[:] = best_messages
                            ids = best_ids
                            turn_rebase_metrics = {
                                "context_rebased": True,
                                "context_tokens_before": before,
                                "context_tokens_after": len(ids),
                                "context_turns_compacted": int(
                                    best_info["turns_compacted"]),
                            }
                            ui.info(
                                f"Context rebased: {before}→{len(ids)} tokens; "
                                f"{best_info['turns_compacted']} older turns compacted.")
                    if len(ids) >= args.context - 8:
                        ui.info(
                            f"Context full ({len(ids)}/{args.context}); "
                            "use /clear or shorten the request/tool output.")
                        break

                    limit = min(repl_max_new, args.context - len(ids))
                    stream_decoder = agent.StableTextStream(
                        session.tokenizer, skip_special_tokens=False)
                    shown = ""
                    stream_mode = None
                    prefix_shown = False

                    def agent_stream(token_ids):
                        nonlocal shown, stream_mode, prefix_shown
                        stream_decoder.update(token_ids)
                        partial = agent.clean_stream_generated(
                            stream_decoder.text)
                        candidate = partial.lstrip()
                        if stream_mode is None:
                            if not candidate or "<function".startswith(candidate):
                                return
                            if candidate.startswith("<function"):
                                stream_mode = "tool"
                                return
                            stream_mode = "answer"
                            ui.stop_wait()
                            ui.model_prefix()
                            prefix_shown = True
                        if stream_mode == "answer":
                            # Thinking may precede a tool call. Show the
                            # thinking text, but never leak raw function XML.
                            visible_partial = partial.split("<function", 1)[0]
                            if visible_partial.startswith(shown):
                                print(visible_partial[len(shown):], end="",
                                      flush=True)
                                shown = visible_partial

                    stage = ("Planning" if tool_round == 0
                             else "Using tool result")

                    def ingest_progress(done, total, _stage=stage):
                        """Turn the silent prompt ingest into a moving line.

                        The board reads a prompt one token at a time, so the
                        remaining count is the wait, exactly. Reporting the
                        rate from this ingest alone keeps the estimate right
                        on any profile without being told which one it is.
                        """
                        left = total - done
                        if left <= 0:
                            ui.update_wait(f"{_stage} · reading the prompt")
                            return
                        elapsed = time.perf_counter() - ingest_started
                        rate = elapsed / done if done else 0.0
                        ui.update_wait(
                            f"{_stage} · read {done}/{total} prompt tokens"
                            + (f" · {left * rate:.0f}s left" if rate else ""))

                    ui.start_wait(stage)
                    ingest_started = time.perf_counter()
                    try:
                        reason, out, steps = session.generate(
                            ids, limit, {1, 130073}, start=0,
                            on_token=agent_stream, on_prompt=ingest_progress,
                            reuse_prefix=args.reuse_session_kv,
                            prefix_snapshot_key=(
                                active_schema_profile
                                if args.fixed_prefix_snapshots else None),
                            prefix_snapshot_tokens=(
                                agent_fixed_prefix_ids(active_schema_profile)
                                if args.fixed_prefix_snapshots else ()))
                    finally:
                        ui.stop_wait()
                    raw = agent.clean_generated(session.tokenizer.decode(
                        out, skip_special_tokens=False))
                    turn_prompt_messages = None
                    stream_decoder.finish(out)
                    prefix_metrics = getattr(session, "last_prefix_metrics", {
                        "prompt_tokens_new": len(ids),
                        "prompt_tokens_replayed": 0,
                        "prefix_cache_hit": 0})
                    observe_prompt_token_ms(
                        getattr(session, "last_phase_steps", ()), prompt_rate)
                    turn_prefix_metrics["prompt_tokens_new"] += \
                        prefix_metrics["prompt_tokens_new"]
                    turn_prefix_metrics["prompt_tokens_replayed"] += \
                        prefix_metrics["prompt_tokens_replayed"]
                    turn_prefix_metrics["prefix_cache_hit"] = max(
                        turn_prefix_metrics["prefix_cache_hit"],
                        prefix_metrics["prefix_cache_hit"])
                    turn_prefix_metrics["prefix_snapshot_hit"] = max(
                        turn_prefix_metrics["prefix_snapshot_hit"],
                        prefix_metrics.get("prefix_snapshot_hit", 0))
                    turn_prefix_metrics["prefix_snapshot_created"] += \
                        prefix_metrics.get("prefix_snapshot_created", 0)
                    turn_prefix_metrics["prefix_snapshot_restore_ms"] += \
                        prefix_metrics.get("prefix_snapshot_restore_ms", 0.0)
                    generated_steps = steps[max(
                        0, len(ids) - 1 -
                        prefix_metrics["prefix_cache_hit"]):]
                    try:
                        calls, visible = agent.parse_tool_calls(raw)
                    except agent.ToolProtocolError as error:
                        if prefix_shown:
                            print("", flush=True)
                        if not workspace_tools.names_for_profile(
                                active_schema_profile):
                            # A turn planned without a schema has no calling
                            # template to copy, so a malformed call is evidence
                            # that the plan needed one.
                            active_schema_profile = "read_only"
                            ui.info(
                                "Tool call attempted without a disclosed "
                                "schema; disclosing read-only tools.")
                            continue
                        ui.info(f"Invalid tool call: {error}")
                        break

                    if not calls:
                        final_text = visible or raw
                        if not prefix_shown:
                            ui.model_prefix()
                            print(final_text, flush=True)
                        elif final_text.startswith(shown):
                            print(final_text[len(shown):], flush=True)
                        else:
                            print("", flush=True)
                            ui.info(
                                "Streaming decoder could not reconcile the "
                                "final text.")
                        ui.turn_summary(len(out), generated_steps, reason)
                        messages.append({"role": "assistant", "content": final_text})
                        try:
                            ui.context_summary(agent_ledger())
                        except Exception:
                            # Accounting must never take a turn down with it.
                            pass
                        results.append({
                            "mode": "agent", "prompt": spec,
                            "text": final_text, "ids": out, "reason": reason,
                            "thinking": thinking_enabled,
                            "tool_rounds": tool_round,
                            "step_ms": steps,
                            "phase_ms": session.last_phase_steps,
                            "prefill_schedule": current_prefill_schedule(),
                            "route_mode": route_decision.mode
                            if route_decision else (
                                "MODEL_NATIVE" if tool_round else "MODEL_ONLY"),
                            "route_reason": route_decision.reason
                            if route_decision else "model-native fallback",
                            "route_confidence": route_decision.confidence
                            if route_decision else None,
                            "response_policy": route_decision.response_policy
                            if route_decision else "MODEL_REASON",
                            "schema_profile": active_schema_profile,
                            "route_ms": route_ms, "tool_ms": tool_total_ms,
                            "model_called": True,
                            **turn_prefix_metrics,
                            **turn_rebase_metrics,
                            "total_ms": (
                                time.perf_counter() - turn_started) * 1000.0,
                        })
                        turn_finished = True
                        break

                    if tool_round >= args.max_tool_steps:
                        if prefix_shown:
                            print("", flush=True)
                        ui.info("Maximum tool rounds reached before a final answer.")
                        break
                    if prefix_shown:
                        print("", flush=True)
                    if visible and not prefix_shown:
                        ui.info(visible)
                    widened = workspace_tools.escalate_schema_profile(
                        active_schema_profile,
                        tuple(call.name for call in calls))
                    if widened is not None:
                        # The model asking for a tool is better evidence of
                        # intent than any keyword. Re-plan with it disclosed;
                        # the rejected plan never enters the transcript.
                        active_schema_profile = widened
                        ui.info(
                            f"Disclosing the {widened} tool schema and "
                            "re-planning; approval is still required per call.")
                        continue
                    # Repeat detection belongs after widening: re-planning
                    # the same call with the tool now disclosed is the
                    # designed recovery, not a loop. What is a loop is the
                    # same call being executed twice with the same arguments
                    # and the same schema.
                    signature = tuple(
                        (call.name, tuple(sorted(call.arguments.items())))
                        for call in calls)
                    if signature in attempted_calls:
                        ui.info(
                            "The same tool call was produced again; stopping "
                            "this turn rather than repeating it.")
                        final_text = visible or raw
                        turn_finished = True
                        break
                    attempted_calls.add(signature)
                    messages.append({"role": "assistant", "content": raw})
                    for call in calls:
                        preview = workspace_tools.preview(call)
                        ui.info(f"⚙ {preview}")
                        tool_started = time.perf_counter()
                        tool_result = workspace_tools.execute(
                            call, approve=approve_tool,
                            allowed_names=workspace_tools.names_for_profile(
                                active_schema_profile))
                        tool_total_ms += (
                            time.perf_counter() - tool_started) * 1000.0
                        messages.append({
                            "role": "tool",
                            "content": workspace_tools.for_model(tool_result)})
                        decoded = json.loads(tool_result)
                        mark = "✓" if decoded["ok"] else "✗"
                        summary = decoded["output"].splitlines()[0][:160]
                        ui.info(f"{mark} {call.name}: {summary}")
                if not turn_finished:
                    results.append({"mode": "agent", "prompt": spec,
                                    "reason": "agent_incomplete",
                                    "schema_profile": active_schema_profile})
        if args.chat:
            messages = []
            repl_max_new = args.max_new
            ui.info(
                "Chat ready · /help · /profile · /context · /clear · /max N · /quit")

            def chat_ids(*, generation_prompt=True):
                rendered = agent.render_chat(
                    messages, tools=[], add_generation_prompt=generation_prompt,
                    enable_thinking=False)
                return list(session.tokenizer.encode(
                    rendered, add_special_tokens=False).ids)

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
                        "Official MiniCPM5 chat template; no tools. "
                        f"profile={args.profile_name}; ctx{args.context}; "
                        f"max-new={repl_max_new}. Commands: /profile, /context, "
                        "/clear, /max N, /quit.")
                    continue
                if spec == "/profile":
                    status = runtime_profile.status if runtime_profile else "legacy"
                    ui.info(
                        f"profile={args.profile_name}; ctx={args.context}; "
                        f"capability=chat; status={status}; "
                        f"prefill={prefill_label}; "
                        f"prefill-status={prefill_runtime_report['status']}; "
                        f"max-new=1..{args.max_new_limit}")
                    continue
                if spec == "/context":
                    used = len(chat_ids(generation_prompt=False))
                    ui.info(
                        f"ctx{args.context}: {used} history tokens used, "
                        f"{max(0, args.context - used)} available")
                    continue
                if spec in {"/clear", "/reset"}:
                    messages.clear()
                    getattr(session, "reset_prefix_cache", lambda: None)()
                    ui.info("Conversation cleared.")
                    continue
                if spec == "/max":
                    ui.info(
                        f"max-new={repl_max_new}; configured=1..{args.max_new_limit}; "
                        f"hardware=1..{args.context - 1}")
                    continue
                if spec.startswith("/max "):
                    try:
                        requested = int(spec.split(None, 1)[1])
                    except ValueError:
                        ui.info("Usage: /max N")
                        continue
                    if requested < 1 or requested > args.max_new_limit:
                        ui.info(f"N must be in [1, {args.max_new_limit}]")
                        continue
                    repl_max_new = requested
                    ui.info(f"max-new={repl_max_new}")
                    continue

                messages.append({"role": "user", "content": spec})
                ids = chat_ids()
                cleared = False
                while len(ids) >= args.context - 8 and len(messages) > 1:
                    # Remove the oldest complete user/assistant turn while
                    # retaining the current user request.
                    del messages[:2]
                    cleared = True
                    ids = chat_ids()
                if cleared:
                    ui.info(
                        f"Older conversation was cleared to fit ctx{args.context}.")
                if len(ids) >= args.context - 8:
                    messages.pop()
                    ui.info(
                        f"Context full ({len(ids)}/{args.context}); shorten the "
                        "request or use /clear.")
                    continue

                limit = min(repl_max_new, args.context - len(ids))
                decoder = agent.StableTextStream(
                    session.tokenizer, skip_special_tokens=True)
                prefix_shown = False

                def chat_stream(token_ids):
                    nonlocal prefix_shown
                    suffix = decoder.update(token_ids)
                    if not suffix:
                        return
                    if not prefix_shown:
                        ui.stop_wait()
                        ui.model_prefix()
                        prefix_shown = True
                    print(suffix, end="", flush=True)

                ui.start_wait("MiniCPM is thinking")
                try:
                    reason, out, steps = session.generate(
                        ids, limit, {1, 130073}, start=0,
                        on_token=chat_stream,
                        reuse_prefix=args.reuse_session_kv)
                finally:
                    ui.stop_wait()
                suffix, reconciled = decoder.finish(out)
                if not prefix_shown:
                    ui.model_prefix()
                    prefix_shown = True
                print(suffix, flush=True)
                final_text = session.tokenizer.decode(
                    out, skip_special_tokens=True).strip()
                if not reconciled:
                    ui.info("Streaming decoder could not reconcile the final text.")
                messages.append({"role": "assistant", "content": final_text})
                prefix_metrics = getattr(session, "last_prefix_metrics", {
                    "prompt_tokens_new": len(ids),
                    "prompt_tokens_replayed": 0,
                    "prefix_cache_hit": 0})
                generated_steps = steps[max(
                    0, len(ids) - 1 - prefix_metrics["prefix_cache_hit"]):]
                ui.turn_summary(len(out), generated_steps, reason)
                results.append({
                    "mode": "chat", "prompt": spec, "text": final_text,
                    "ids": out, "reason": reason, "max_new": repl_max_new,
                    "step_ms": steps, "phase_ms": session.last_phase_steps,
                    "prefill_schedule": current_prefill_schedule(),
                    **prefix_metrics,
                })
                if reason == "max":
                    print(f"[reached max-new={repl_max_new}; "
                          "use /max N to increase it]", flush=True)
                elif reason == "context":
                    print(f"[reached ctx{args.context} limit]", flush=True)
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
                decoder = agent.StableTextStream(
                    session.tokenizer, skip_special_tokens=True)
                prefix_shown = False

                def stream(token_ids):
                    nonlocal prefix_shown
                    suffix = decoder.update(token_ids)
                    if not suffix:
                        return
                    if not prefix_shown:
                        ui.stop_wait()
                        ui.model_prefix()
                        prefix_shown = True
                    print(suffix, end="", flush=True)

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
                suffix, reconciled = decoder.finish(record["ids"])
                print(suffix, flush=True)
                if not reconciled:
                    ui.info("Streaming decoder could not reconcile the final text.")
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
             "profile": args.profile_name, "context": args.context,
             "load_seconds": load_ms / 1000.0,
             "prefill_runtime": prefill_runtime_report,
             "runs": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
