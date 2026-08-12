#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expose the qualified merged MiniCPM5 runtime over the service JSONL ABI.

This adapter deliberately lives beside, rather than inside, the board runtime.
The merged runtime remains responsible for model admission, descriptor checks,
resident KV, transformer execution and the vocabulary head.  This module owns
only the narrow process protocol consumed by ``pico_minicpm5_openai_service``.

Standard output is reserved for one compact JSON response per input line.
Native runtime diagnostics are redirected to standard error so they cannot
corrupt the protocol stream.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Mapping, Protocol, Sequence, TextIO


RUNNER_PROTOCOL = "pico.minicpm5.runner.v1"
MODEL_ID = "minicpm5-1b"
VOCAB_SIZE = 130_560
MAX_JSONL_LINE_BYTES = 16 << 20
MAX_EOS_IDS = 8
REQUEST_FIELDS = frozenset({
    "protocol", "request_id", "op", "model", "input_ids",
    "max_new_tokens", "temperature", "eos_token_ids", "reset_kv",
})
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,159}\Z")


class NativeSession(Protocol):
    """Subset of the merged board runtime used by this adapter."""

    def generate(
        self, prompt_ids: Sequence[int], max_new: int, eos: set[int],
        *, start: int = 0,
    ) -> tuple[str, Sequence[int], object]:
        """Generate tokens from a fresh logical position-zero request."""

    def close(self) -> None:
        """Release resident native resources."""


class NativeExecutionError(RuntimeError):
    """The merged runtime returned an impossible result or failed."""


@dataclass(frozen=True)
class RequestLimits:
    context: int
    max_new_tokens: int
    vocab_size: int = VOCAB_SIZE

    def __post_init__(self) -> None:
        if type(self.context) is not int or self.context < 2:
            raise ValueError("context must be an integer of at least 2")
        if type(self.max_new_tokens) is not int \
                or not 1 <= self.max_new_tokens <= self.context:
            raise ValueError(
                "max_new_tokens limit must be within the model context")
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError("MiniCPM5 vocabulary contract drift")


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_request(
    request: Mapping[str, object], limits: RequestLimits,
) -> dict[str, object]:
    """Validate one service request without coercing attacker-controlled data."""
    if set(request) != REQUEST_FIELDS:
        raise ValueError("runner request fields mismatch")
    if request["protocol"] != RUNNER_PROTOCOL \
            or request["op"] != "generate" \
            or request["model"] != MODEL_ID \
            or request["reset_kv"] is not True:
        raise ValueError("runner request contract mismatch")

    request_id = request["request_id"]
    if not isinstance(request_id, str) \
            or not SAFE_REQUEST_ID.fullmatch(request_id):
        raise ValueError("request_id contains unsupported characters")

    temperature = request["temperature"]
    if isinstance(temperature, bool) \
            or not isinstance(temperature, (int, float)) \
            or not math.isfinite(float(temperature)) \
            or float(temperature) != 0.0:
        raise ValueError("only temperature=0 is supported")

    input_ids = request["input_ids"]
    eos_ids = request["eos_token_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("input_ids must be a non-empty array")
    if not isinstance(eos_ids, list) or len(eos_ids) > MAX_EOS_IDS:
        raise ValueError("eos_token_ids must be an array of at most 8 IDs")
    if any(type(value) is not int or not 0 <= value < limits.vocab_size
           for value in (*input_ids, *eos_ids)):
        raise ValueError("token id escapes MiniCPM5 vocabulary")

    max_new = request["max_new_tokens"]
    if type(max_new) is not int or max_new <= 0 \
            or max_new > limits.max_new_tokens:
        raise ValueError("max_new_tokens escapes configured limit")
    # Match the OpenAI service/OpenClaw context budget exactly: prompt tokens
    # plus the requested completion budget must fit the compiled window.
    if len(input_ids) + max_new > limits.context:
        raise ValueError("request exceeds native decode context")

    return {
        "request_id": request_id,
        "input_ids": tuple(input_ids),
        "eos_token_ids": tuple(eos_ids),
        "max_new_tokens": max_new,
    }


class MergedJsonlRunner:
    """Strict protocol facade over one resident merged MiniCPM5 session."""

    def __init__(self, session: NativeSession, limits: RequestLimits) -> None:
        self.session = session
        self.limits = limits

    def generate(self, request: Mapping[str, object]) -> dict[str, object]:
        validated = validate_request(request, self.limits)
        eos = set(validated["eos_token_ids"])
        # Replay from position zero for every request.  Every visible resident
        # cache row is overwritten before it becomes unmasked, which is the
        # merged runtime's reset_kv=true contract.
        with redirect_stdout(sys.stderr):
            reason, output_ids, _steps = self.session.generate(
                validated["input_ids"], validated["max_new_tokens"], eos,
                start=0)
        output_ids = list(output_ids)
        if len(output_ids) > validated["max_new_tokens"] \
                or any(type(token_id) is not int
                       or not 0 <= token_id < self.limits.vocab_size
                       for token_id in output_ids):
            raise NativeExecutionError(
                "native runtime returned invalid output token IDs")

        if reason == "eos":
            if not output_ids or output_ids[-1] not in eos:
                raise NativeExecutionError(
                    "native runtime reported EOS without an EOS token")
            finish_reason = "stop"
        elif reason == "max":
            if len(output_ids) != validated["max_new_tokens"]:
                raise NativeExecutionError(
                    "native runtime stopped before max_new_tokens")
            finish_reason = "length"
        elif reason == "context":
            raise NativeExecutionError(
                "native runtime reached context after request preflight")
        else:
            raise NativeExecutionError(
                f"native runtime returned unsupported reason {reason!r}")

        return {
            "protocol": RUNNER_PROTOCOL,
            "request_id": validated["request_id"],
            "ok": True,
            "output_ids": output_ids,
            "finish_reason": finish_reason,
        }

    def close(self) -> None:
        with redirect_stdout(sys.stderr):
            self.session.close()


def _error_response(
    request_id: str, code: str, message: str,
) -> dict[str, object]:
    return {
        "protocol": RUNNER_PROTOCOL,
        "request_id": request_id,
        "ok": False,
        "error": {"code": code, "message": message},
    }


def _drain_unterminated_line(reader: TextIO) -> None:
    """Consume the rest of an oversized JSONL record before continuing."""
    while True:
        tail = reader.readline(MAX_JSONL_LINE_BYTES + 1)
        if not tail or tail.endswith("\n"):
            return


def serve_jsonl(
    runner: MergedJsonlRunner, reader: TextIO, writer: TextIO,
    *, diagnostics: TextIO = sys.stderr,
) -> None:
    """Serve service-compatible requests until EOF, preserving stream sync."""
    while True:
        line = reader.readline(MAX_JSONL_LINE_BYTES + 1)
        if not line:
            break
        request_id = "unknown"
        try:
            encoded_size = len(line.encode("utf-8"))
            if not line.endswith("\n"):
                _drain_unterminated_line(reader)
                raise ValueError(
                    "runner request must be one newline-terminated JSON object")
            if encoded_size > MAX_JSONL_LINE_BYTES:
                raise ValueError("runner request line exceeds size limit")
            request = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(request, dict):
                raise ValueError("runner request root must be an object")
            candidate = request.get("request_id")
            if isinstance(candidate, str) \
                    and SAFE_REQUEST_ID.fullmatch(candidate):
                request_id = candidate
            response = runner.generate(request)
        except ValueError as exc:
            response = _error_response(
                request_id, "invalid_request", str(exc))
        except NativeExecutionError as exc:
            response = _error_response(
                request_id, "execution_error", str(exc))
        except Exception as exc:  # keep the long-lived service available
            print(
                f"native request {request_id!r} failed: "
                f"{type(exc).__name__}: {exc}",
                file=diagnostics, flush=True)
            response = _error_response(
                request_id, "execution_error", "native generation failed")
        writer.write(json.dumps(
            response, ensure_ascii=False, separators=(",", ":")) + "\n")
        writer.flush()


def _load_runtime_module(
    runtime_module: Path, python_paths: Sequence[Path],
) -> ModuleType:
    runtime_module = Path(runtime_module).resolve()
    if not runtime_module.is_file():
        raise FileNotFoundError(f"runtime module not found: {runtime_module}")
    search_paths = [runtime_module.parent.resolve()]
    for path in python_paths:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"python path not found: {resolved}")
        search_paths.append(resolved)
    for path in reversed(search_paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    identity = hashlib.sha256(str(runtime_module).encode("utf-8")).hexdigest()[:12]
    name = f"_pico_minicpm5_merged_runtime_{identity}"
    spec = importlib.util.spec_from_file_location(name, runtime_module)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runtime module: {runtime_module}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    if not callable(getattr(module, "Merged", None)):
        raise ImportError("runtime module does not export Merged")
    return module


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _parse_short_kv_slots(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    try:
        slots = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("short K/V slots must be integer K,V indices") from exc
    if len(slots) != 2 or len(set(slots)) != 2 or min(slots) < 0:
        raise ValueError("short K/V slots must be two distinct nonnegative indices")
    return slots


def build_runner(args: argparse.Namespace) -> MergedJsonlRunner:
    """Load the explicit runtime module and admit one resident model session."""
    limits = RequestLimits(args.context, args.max_new_limit)
    executable = _require_file(args.persistent_executor, "persistent executor")
    decode = _require_file(args.decode_model, "decode model")
    prefill = _require_file(args.prefill_model, "prefill model")
    head = _require_file(args.head_model, "head model")
    embedding = _require_file(args.embedding, "embedding")
    decode_short = (
        _require_file(args.decode_short_model, "short decode model")
        if args.decode_short_model is not None else None)
    library_paths = []
    for path in args.library_path:
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"library path not found: {resolved}")
        library_paths.append(resolved)

    short_kv_slots = _parse_short_kv_slots(args.short_kv_slots)
    if short_kv_slots is not None and decode_short is None:
        raise ValueError("--short-kv-slots requires --decode-short-model")
    module = _load_runtime_module(args.runtime_module, args.python_path)
    with redirect_stdout(sys.stderr):
        session = module.Merged(
            executable=executable,
            decode=decode,
            prefill=prefill,
            head=head,
            library_paths=library_paths,
            embedding=embedding,
            context=args.context,
            timeout=args.timeout,
            tokenizer=None,
            resident_kv=True,
            decode_short=decode_short,
            short_context=args.short_context,
            short_kv_slots=short_kv_slots,
            allow_unsafe_short_context=args.allow_unsafe_short_context,
            allow_c8192_short_characterization=
            args.allow_c8192_short_characterization,
            executor_uncached=args.executor_uncached,
            decode_no_cache=args.decode_no_cache,
            characterize_decode_workspace_zero_once=
            args.characterize_decode_workspace_zero_once,
        )
    return MergedJsonlRunner(session, limits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-jsonl", action="store_true", required=True)
    parser.add_argument("--runtime-module", type=Path, required=True)
    parser.add_argument("--python-path", type=Path, action="append", default=[])
    parser.add_argument("--persistent-executor", type=Path, required=True)
    parser.add_argument("--decode-model", type=Path, required=True)
    parser.add_argument("--decode-short-model", type=Path)
    parser.add_argument("--short-context", type=int, default=128)
    parser.add_argument("--short-kv-slots")
    parser.add_argument("--allow-unsafe-short-context", action="store_true")
    parser.add_argument(
        "--allow-c8192-short-characterization", action="store_true")
    parser.add_argument("--prefill-model", type=Path, required=True)
    parser.add_argument("--head-model", type=Path, required=True)
    parser.add_argument("--library-path", type=Path, action="append", default=[])
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--max-new-limit", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=3600.0)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--executor-uncached", action="store_true")
    cache.add_argument("--decode-no-cache", action="store_true")
    parser.add_argument(
        "--characterize-decode-workspace-zero-once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner: MergedJsonlRunner | None = None
    try:
        runner = build_runner(args)
        print(
            f"merged_jsonl_runner=ready protocol={RUNNER_PROTOCOL} "
            f"model={MODEL_ID} context={args.context} "
            f"max_new={args.max_new_limit}",
            file=sys.stderr, flush=True)
        serve_jsonl(runner, sys.stdin, sys.stdout)
        return 0
    except (OSError, ImportError, ValueError, RuntimeError) as exc:
        print(
            f"merged_jsonl_runner startup failed: {type(exc).__name__}: {exc}",
            file=sys.stderr, flush=True)
        return 2
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
