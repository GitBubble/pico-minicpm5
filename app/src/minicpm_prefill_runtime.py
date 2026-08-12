#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runtime registry between native-prefill activation and request scheduling.

Release qualification says that a block *may* be loaded.  A runtime handler
is a separate requirement: it must own a real wide model handle and implement
input preparation, resident-prefix copy, execute, K/V publication and hidden
handoff.  Keeping those two facts separate prevents an activated S16/S32/S128
artifact from being reported by the scheduler while the process still runs the
request token-by-token through S1.

The currently shipped board runtime registers no wide handler.  Consequently
    an optional v4 activation manifest is fully verified and reported, but the
executable scheduling set remains strict S1 until such a handler is added.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct
import time
from typing import Callable, Protocol, Sequence

import minicpm_prefill_activation as activation_contract
import minicpm_prefill_schedule as schedule_contract


SCHEMA = "pico.minicpm5.prefill-runtime-registry.v1"
WIDE_WIDTHS = (128, 32, 16)
HIDDEN = 1536
CHANNELS = 48
HEAD_DIM = 128
ROW_F16_BYTES = HEAD_DIM * 2
MASK_NEGATIVE = -64.0
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PrefillRuntimeError(ValueError):
    """Native-prefill startup/runtime state is incomplete or unsafe."""


class WideBlockExecutionError(RuntimeError):
    """A wide session was discarded; the caller must rebuild before retry."""

    def __init__(self, *, width: int, start: int, stage: str, cause: Exception):
        super().__init__(
            f"S{width} failed during {stage}; wide session was discarded and "
            f"must be rebuilt: {cause}")
        self.width = width
        self.start = start
        self.stage = stage
        self.requires_rebuild = True


@dataclass(frozen=True)
class WidePublisherABI:
    """Exact resident-output contract for one qualified wide handle."""

    width: int
    k_output_slot: int
    v_output_slot: int
    hidden_output_slot: int
    hidden_output_bytes: int
    dtype: str = "FP32"
    layout: str = "contiguous-channel-major"
    resident_dtype: str = "FP16"
    rounding: str = "RNE"
    channels: int = CHANNELS
    head_dim: int = HEAD_DIM

    def __post_init__(self) -> None:
        if self.width not in WIDE_WIDTHS:
            raise PrefillRuntimeError("publisher width is unsupported")
        if (self.k_output_slot, self.v_output_slot,
                self.hidden_output_slot) != (0, 1, 2):
            raise PrefillRuntimeError(
                "publisher slots must be exactly K=0,V=1,hidden=2")
        if self.dtype != "FP32" \
                or self.layout != "contiguous-channel-major" \
                or self.resident_dtype != "FP16" or self.rounding != "RNE" \
                or self.channels != CHANNELS or self.head_dim != HEAD_DIM:
            raise PrefillRuntimeError("wide publisher ABI is not canonical")
        if type(self.hidden_output_bytes) is not int \
                or self.hidden_output_bytes <= 0:
            raise PrefillRuntimeError("hidden output byte size is invalid")

    @property
    def kv_output_bytes(self) -> int:
        return self.channels * self.width * self.head_dim * 4


@dataclass(frozen=True)
class WideHandleDescriptor:
    """Ready-descriptor sizes and identity observed for one model index."""

    input_bytes: tuple[int, ...]
    output_bytes: tuple[int, ...]
    ready_descriptor_sha256: str

    def __post_init__(self) -> None:
        if len(self.input_bytes) < 5 or len(self.output_bytes) != 3 \
                or any(type(size) is not int or size <= 0
                       for size in self.input_bytes + self.output_bytes):
            raise PrefillRuntimeError(
                "wide descriptor requires >=5 positive inputs and 3 outputs")
        if not isinstance(self.ready_descriptor_sha256, str) \
                or _SHA256.fullmatch(self.ready_descriptor_sha256) is None:
            raise PrefillRuntimeError(
                "wide ready descriptor needs a lowercase SHA-256")


@dataclass(frozen=True)
class WideHandleSpec:
    """Qualified model-table binding used by a real or fake transport."""

    width: int
    context: int
    model_index: int
    model_path: Path
    descriptor: WideHandleDescriptor
    publisher: WidePublisherABI
    embedding_row_bytes: int
    embedding_input_slot: int = 0
    mask_input_slot: int = 1
    rope_input_slot: int = 2
    k_cache_input_slot: int = 3
    v_cache_input_slot: int = 4
    mask_negative: float = MASK_NEGATIVE

    def __post_init__(self) -> None:
        if self.width not in WIDE_WIDTHS or self.publisher.width != self.width:
            raise PrefillRuntimeError("wide handle width/publisher mismatch")
        if type(self.context) is not int or self.context < 128:
            raise PrefillRuntimeError("wide handle context must be >= 128")
        if type(self.model_index) is not int or not 0 <= self.model_index < 128:
            raise PrefillRuntimeError("wide model index is invalid")
        if not isinstance(self.model_path, Path) or not self.model_path.name:
            raise PrefillRuntimeError("wide model path is invalid")
        if type(self.embedding_row_bytes) is not int \
                or self.embedding_row_bytes <= 0 \
                or self.embedding_row_bytes % 4:
            raise PrefillRuntimeError("wide embedding row bytes are invalid")
        if (self.embedding_input_slot, self.mask_input_slot,
                self.rope_input_slot, self.k_cache_input_slot,
                self.v_cache_input_slot) != (0, 1, 2, 3, 4):
            raise PrefillRuntimeError(
                "wide input slots must be exactly embedding=0,mask=1,"
                "rope=2,K=3,V=4")
        if isinstance(self.mask_negative, bool) \
                or not isinstance(self.mask_negative, (int, float)) \
                or not math.isfinite(float(self.mask_negative)) \
                or float(self.mask_negative) != MASK_NEGATIVE:
            raise PrefillRuntimeError(
                "wide mask negative value does not match qualification")
        cache_bytes = CHANNELS * (self.context - 1) * ROW_F16_BYTES
        expected_inputs = (
            self.width * self.embedding_row_bytes,
            self.width * self.context * 4,
            self.width * HEAD_DIM * HEAD_DIM * 4,
            cache_bytes,
            cache_bytes,
        )
        if self.descriptor.input_bytes[:5] != expected_inputs:
            raise PrefillRuntimeError(
                "wide input descriptor does not match embeddings/mask/RoPE/KV")
        expected_outputs = [None, None, None]
        expected_outputs[self.publisher.k_output_slot] = \
            self.publisher.kv_output_bytes
        expected_outputs[self.publisher.v_output_slot] = \
            self.publisher.kv_output_bytes
        expected_outputs[self.publisher.hidden_output_slot] = \
            self.publisher.hidden_output_bytes
        if self.descriptor.output_bytes != tuple(expected_outputs):
            raise PrefillRuntimeError(
                "wide output descriptor does not match publisher ABI")

    def verify_loaded_descriptor(
            self, descriptor: tuple[Sequence[int], Sequence[int]]) -> None:
        inputs, outputs = descriptor
        if tuple(inputs) != self.descriptor.input_bytes \
                or tuple(outputs) != self.descriptor.output_bytes:
            raise PrefillRuntimeError(
                f"S{self.width} model {self.model_index} ready descriptor drift")


@dataclass(frozen=True)
class WideInputWrite:
    input_slot: int
    offset: int
    payload: bytes


@dataclass(frozen=True)
class ResidentHidden:
    model_index: int
    output_slot: int
    byte_size: int
    position: int


@dataclass(frozen=True)
class WideBlockResult:
    width: int
    start: int
    stop: int
    hidden: ResidentHidden
    prepare_ms: float
    prefix_copy_ms: float
    execute_ms: float
    publish_ms: float


class WidePrefillTransport(Protocol):
    """Minimal transport boundary; fake tests and Merged implement this."""

    def wide_copy_prefix(
            self, *, source_model: int, destination_model: int,
            token_count: int) -> None: ...

    def wide_execute(
            self, *, spec: WideHandleSpec,
            writes: tuple[WideInputWrite, ...]) -> None: ...

    def wide_publish_kv(self, *, spec: WideHandleSpec, start: int) -> None: ...

    def wide_discard_session(
            self, *, spec: WideHandleSpec, stage: str,
            error: Exception) -> None: ...


@dataclass(frozen=True)
class WidePrefillHandler:
    """Prepare and transactionally execute one exact-width resident block.

    Mask row ``j`` uses the decode convention for absolute position
    ``start+j``: columns ``[0, start+j)`` are the absolute K/V prefix and the
    final context column is the current-token sentinel.  The qualified wide
    graph must dynamically append earlier rows from the same block at their
    absolute cache slots before a later row consumes that prefix.  The runtime
    never host-repacks those intra-block rows.
    """

    spec: WideHandleSpec

    def validate_range(self, start: int) -> int:
        """Return the exclusive stop after validating the publisher range.

        This check is deliberately transport-free so a complete request plan
        can be rejected before the first model execute is submitted.
        """
        if type(start) is not int or start < 1:
            raise PrefillRuntimeError("wide blocks are steady-only (start >= 1)")
        stop = start + self.spec.width
        # Canonical decode inputs contain context-1 rows. Opcode 6 has no
        # source-row-stride/count override, so a terminal partial publication
        # cannot be represented safely by this v1 dispatcher.
        if stop > self.spec.context - 1:
            raise PrefillRuntimeError(
                "wide block cannot publish beyond context-1 canonical cache")
        return stop

    def execute(
        self,
        *,
        transport: WidePrefillTransport,
        canonical_model: int,
        start: int,
        token_ids: Sequence[int],
        embedding_row: Callable[[int, int], bytes],
        rope_row: Callable[[int], bytes],
    ) -> WideBlockResult:
        width = self.spec.width
        tokens = tuple(token_ids)
        if len(tokens) != width or any(type(token) is not int for token in tokens):
            raise PrefillRuntimeError(
                f"S{width} requires exactly {width} integer token IDs")
        stop = self.validate_range(start)

        stage = "prepare-inputs"
        try:
            began = time.perf_counter()
            embeddings = b"".join(
                embedding_row(token, self.spec.embedding_row_bytes)
                for token in tokens)
            expected_embeddings = width * self.spec.embedding_row_bytes
            if len(embeddings) != expected_embeddings:
                raise PrefillRuntimeError(
                    "wide embedding payload does not match descriptor")

            negative = struct.pack("<f", float(self.spec.mask_negative))
            zero = struct.pack("<f", 0.0)
            mask = b"".join(
                zero * position
                + negative * (self.spec.context - 1 - position)
                + zero
                for position in range(start, stop)
            )

            ropes = b"".join(rope_row(position) for position in range(start, stop))
            expected_ropes = width * HEAD_DIM * HEAD_DIM * 4
            if len(ropes) != expected_ropes:
                raise PrefillRuntimeError(
                    "wide RoPE payload does not match descriptor")
            writes = (
                WideInputWrite(self.spec.embedding_input_slot, 0, embeddings),
                WideInputWrite(self.spec.mask_input_slot, 0, mask),
                WideInputWrite(self.spec.rope_input_slot, 0, ropes),
            )
            prepared_at = time.perf_counter()

            stage = "copy-prefix"
            transport.wide_copy_prefix(
                source_model=canonical_model,
                destination_model=self.spec.model_index,
                token_count=start)
            copied_at = time.perf_counter()

            stage = "execute"
            transport.wide_execute(spec=self.spec, writes=writes)
            executed_at = time.perf_counter()

            stage = "publish-kv"
            transport.wide_publish_kv(spec=self.spec, start=start)
            published_at = time.perf_counter()
        except Exception as error:
            try:
                transport.wide_discard_session(
                    spec=self.spec, stage=stage, error=error)
            except Exception as discard_error:
                error = RuntimeError(f"{error}; discard also failed: {discard_error}")
            raise WideBlockExecutionError(
                width=width, start=start, stage=stage, cause=error) from error

        return WideBlockResult(
            width=width,
            start=start,
            stop=stop,
            hidden=ResidentHidden(
                model_index=self.spec.model_index,
                output_slot=self.spec.publisher.hidden_output_slot,
                byte_size=self.spec.publisher.hidden_output_bytes,
                position=stop - 1,
            ),
            prepare_ms=(prepared_at - began) * 1000.0,
            prefix_copy_ms=(copied_at - prepared_at) * 1000.0,
            execute_ms=(executed_at - copied_at) * 1000.0,
            publish_ms=(published_at - executed_at) * 1000.0,
        )


@dataclass(frozen=True)
class PrefillRuntimeRegistry:
    """Immutable startup result consumed by every request planner."""

    context: int
    activation_manifest: Path | None
    activation_report: dict[str, object] | None
    qualified_widths: tuple[int, ...]
    enabled_widths: tuple[int, ...]
    handler_widths: tuple[int, ...]
    unavailable: dict[str, str]
    handlers: tuple[WidePrefillHandler, ...] = ()
    strict_s1_anchor: activation_contract.StrictS1Anchor | None = None
    activated_model_sha256: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.context) is not int or self.context < 2:
            raise PrefillRuntimeError("prefill runtime context must be >= 2")
        for label, widths in (
                ("qualified", self.qualified_widths),
                ("enabled", self.enabled_widths),
                ("handler", self.handler_widths)):
            if any(type(width) is not int for width in widths):
                raise PrefillRuntimeError(f"{label} widths must be integers")
            if len(widths) != len(set(widths)):
                raise PrefillRuntimeError(f"{label} widths must not repeat")
            if set(widths) - set(schedule_contract.CANDIDATE_WIDTHS):
                raise PrefillRuntimeError(f"{label} widths are unsupported")
        if 1 not in self.qualified_widths or 1 not in self.enabled_widths:
            raise PrefillRuntimeError("strict S1 fallback must remain enabled")
        if 1 in self.handler_widths:
            raise PrefillRuntimeError("strict S1 is built in, not a wide handler")
        hashes = dict(self.activated_model_sha256)
        if len(hashes) != len(self.activated_model_sha256) \
                or any(width not in WIDE_WIDTHS \
                       or _SHA256.fullmatch(value) is None
                       for width, value in self.activated_model_sha256):
            raise PrefillRuntimeError(
                "activated wide model identities are malformed")
        handler_by_width: dict[int, WidePrefillHandler] = {}
        for handler in self.handlers:
            if not isinstance(handler, WidePrefillHandler):
                raise PrefillRuntimeError(
                    "wide runtime handlers must be WidePrefillHandler objects")
            width = handler.spec.width
            if width in handler_by_width:
                raise PrefillRuntimeError(f"duplicate S{width} runtime handler")
            if handler.spec.context != self.context:
                raise PrefillRuntimeError(
                    f"S{width} handler context does not match registry")
            handler_by_width[width] = handler
        canonical_handlers = tuple(
            width for width in schedule_contract.CANDIDATE_WIDTHS
            if width > 1 and width in handler_by_width)
        if self.handler_widths != canonical_handlers:
            raise PrefillRuntimeError(
                "handler_widths do not match registered handler declarations")
        expected_enabled = tuple(
            width for width in schedule_contract.CANDIDATE_WIDTHS
            if width == 1 or (
                width in self.qualified_widths and width in self.handler_widths))
        if self.enabled_widths != expected_enabled:
            raise PrefillRuntimeError(
                "enabled widths must be qualified and have a runtime handler")

    def plan(self, start: int, stop: int, *, hard_boundaries=()):
        """Plan one exact request and reject every unhandled wide segment."""
        # A canonical decode cache has only ``context - 1`` writable rows.
        # Keep the terminal context position on strict S1: S1 intentionally
        # omits its unused final scatter, whereas opcode 6 cannot partially
        # publish a wide output. Do this for the complete plan before the
        # caller submits even the position-zero bootstrap execute.
        boundaries = set(hard_boundaries)
        if start < self.context - 1 < stop:
            boundaries.add(self.context - 1)
        schedule = schedule_contract.plan_prefill(
            start, stop, context=self.context,
            enabled_widths=self.enabled_widths,
            hard_boundaries=tuple(sorted(boundaries)))
        self.require_executable(schedule)
        return schedule

    def require_executable(self, schedule) -> None:
        """Last invariant immediately before any model execute is submitted."""
        for segment in schedule.segments:
            if segment.width > 1 and segment.width not in self.handler_widths:
                raise PrefillRuntimeError(
                    f"scheduled S{segment.width} has no registered wide-handle "
                    "executor")

    def handler_for(self, width: int) -> WidePrefillHandler:
        for handler in self.handlers:
            if handler.spec.width == width:
                return handler
        raise PrefillRuntimeError(
            f"scheduled S{width} has no registered wide-handle executor")

    def validate_loaded_handlers(
            self, descriptors: Sequence[tuple[Sequence[int], Sequence[int]]],
            models: Sequence[Path]) -> None:
        """Bind declarations to the exact live model table and descriptors."""
        for handler in self.handlers:
            spec = handler.spec
            if spec.model_index >= len(models) or spec.model_index >= len(descriptors):
                raise PrefillRuntimeError(
                    f"S{spec.width} model index is absent from live table")
            if Path(models[spec.model_index]) != spec.model_path:
                raise PrefillRuntimeError(
                    f"S{spec.width} model index/path binding drift")
            spec.verify_loaded_descriptor(descriptors[spec.model_index])

    @staticmethod
    def _live_sha256(path: Path, label: str) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
        except OSError as error:
            raise PrefillRuntimeError(
                f"cannot rehash live {label}: {error}") from error
        return digest.hexdigest()

    def validate_live_startup_identity(
        self, *, executable: Path, decode: Path, prefill: Path | None,
        head: Path, embedding: Path, runner: Path,
    ) -> None:
        """Rebind release evidence to the exact process about to be spawned.

        Qualification happens before deployment.  This final host-side check
        resolves and hashes every executable-route artifact immediately before
        ``probe._start``.  It catches stale or passively replaced deployments,
        but it is not an inherited-fd handoff: a malicious writer could still
        race the executor's later path opens.  Activation therefore requires a
        trusted deployment tree kept read-only/immutable for the process
        lifetime.  S1-only releases without an activation manifest retain their
        legacy startup surface.
        """
        anchor = self.strict_s1_anchor
        if anchor is None:
            if self.activation_manifest is None and not self.handlers:
                return
            raise PrefillRuntimeError(
                "native-prefill activation requires a live strict-S1 route "
                "anchor")

        def exact_path(value: Path, expected: Path, label: str) -> Path:
            try:
                live = Path(value).expanduser().resolve(strict=True)
            except OSError as error:
                raise PrefillRuntimeError(
                    f"live {label} is unavailable: {error}") from error
            if live != expected:
                raise PrefillRuntimeError(
                    f"live {label} path does not match activation")
            return live

        bootstrap = decode if prefill is None else prefill
        route = (
            (exact_path(Path(executable), anchor.executor, "executor"),
             anchor.identity.executor_sha256, "executor"),
            (exact_path(Path(decode), anchor.canonical_decode_model,
                        "canonical decode OM"),
             anchor.identity.canonical_decode_om_sha256,
             "canonical decode OM"),
            (exact_path(Path(bootstrap), anchor.bootstrap_model,
                        "bootstrap OM"),
             anchor.identity.bootstrap_om_sha256, "bootstrap OM"),
            (exact_path(Path(head), anchor.head_model, "head OM"),
             anchor.identity.head_om_sha256, "head OM"),
            (exact_path(Path(embedding), anchor.embedding, "embedding"),
             anchor.identity.embedding_sha256, "embedding"),
            (exact_path(Path(runner), anchor.runner, "runner"),
             anchor.identity.runner_sha256, "runner"),
            (anchor.build_manifest,
             anchor.identity.build_manifest_sha256, "build manifest"),
            (anchor.bootstrap_ready_descriptor,
             anchor.identity.bootstrap_ready_descriptor_sha256,
             "bootstrap ready descriptor"),
            (anchor.canonical_ready_descriptor,
             anchor.identity.canonical_ready_descriptor_sha256,
             "canonical ready descriptor"),
        )
        for path, expected_hash, label in route:
            if self._live_sha256(path, label) != expected_hash:
                raise PrefillRuntimeError(
                    f"live {label} SHA-256 does not match activation")

        wide_hashes = dict(self.activated_model_sha256)
        for handler in self.handlers:
            expected_hash = wide_hashes.get(handler.spec.width)
            if expected_hash is None or self._live_sha256(
                    handler.spec.model_path,
                    f"S{handler.spec.width} model") != expected_hash:
                raise PrefillRuntimeError(
                    f"live S{handler.spec.width} model SHA-256 does not match "
                    "activation")

    def to_dict(self) -> dict[str, object]:
        if self.activation_manifest is None:
            status = "strict-s1-default"
        elif any(width > 1 for width in self.enabled_widths):
            status = "wide-handler-enabled"
        elif any(width > 1 for width in self.qualified_widths):
            status = "qualified-wide-handler-unavailable"
        else:
            status = "activation-s1-only"
        return {
            "schema": SCHEMA,
            "status": status,
            "context": self.context,
            "activation_manifest": (
                str(self.activation_manifest)
                if self.activation_manifest is not None else None),
            "qualified_widths": list(self.qualified_widths),
            "enabled_widths": list(self.enabled_widths),
            "handler_widths": list(self.handler_widths),
            "handlers": [
                {
                    "width": handler.spec.width,
                    "context": handler.spec.context,
                    "model_index": handler.spec.model_index,
                    "model": str(handler.spec.model_path),
                    "ready_descriptor_sha256": (
                        handler.spec.descriptor.ready_descriptor_sha256),
                    "publisher": {
                        "dtype": handler.spec.publisher.dtype,
                        "layout": handler.spec.publisher.layout,
                        "shape": [1, CHANNELS, handler.spec.width, HEAD_DIM],
                        "k_output_slot": (
                            handler.spec.publisher.k_output_slot),
                        "v_output_slot": (
                            handler.spec.publisher.v_output_slot),
                        "hidden_output_slot": (
                            handler.spec.publisher.hidden_output_slot),
                    },
                }
                for handler in self.handlers
            ],
            "unavailable": dict(self.unavailable),
            "activation": self.activation_report,
        }


def _strict_s1_registry(context: int) -> PrefillRuntimeRegistry:
    unavailable = {
        f"S{width}": "no activation manifest and no registered wide-handle executor"
        for width in WIDE_WIDTHS
    }
    return PrefillRuntimeRegistry(
        context=context,
        activation_manifest=None,
        activation_report=None,
        qualified_widths=(1,),
        enabled_widths=(1,),
        handler_widths=(),
        unavailable=unavailable,
        handlers=(),
    )


def load_runtime_registry(
    *,
    activation_manifest: Path | None,
    deployment_root: Path,
    context: int,
    available_bytes: int | None,
    base_resident_bytes: int | None,
    reserve_bytes: int | None,
    handlers: Sequence[WidePrefillHandler] = (),
) -> PrefillRuntimeRegistry:
    """Load the optional v4 activation and expose only executable widths.

    The memory values form one all-or-nothing live snapshot.  Supplying them
    without a manifest, or omitting any value with a manifest, is rejected so
    an operator cannot accidentally believe a stale/default number was used.
    """
    memory = (available_bytes, base_resident_bytes, reserve_bytes)
    if activation_manifest is None:
        if any(value is not None for value in memory):
            raise PrefillRuntimeError(
                "MMZ available/base/reserve inputs require an activation manifest")
        if tuple(handlers):
            raise PrefillRuntimeError(
                "wide handlers require a release activation manifest")
        return _strict_s1_registry(context)
    if any(value is None for value in memory):
        raise PrefillRuntimeError(
            "activation manifest requires live available/base/reserve bytes")

    activation = activation_contract.load_activation(
        activation_manifest,
        deployment_root=deployment_root,
        context=context,
        available_bytes=available_bytes,
        base_resident_bytes=base_resident_bytes,
        reserve_bytes=reserve_bytes,
    )
    qualified = tuple(activation.enabled_widths)
    if 1 not in qualified:
        raise PrefillRuntimeError(
            "activation result removed the strict S1 fallback")

    supplied_handlers = tuple(handlers)
    strict_anchor = getattr(activation, "strict_s1", None)
    if supplied_handlers and strict_anchor is None:
        raise PrefillRuntimeError(
            "registered wide handlers require the live strict-S1 route anchor")
    handler_by_width: dict[int, WidePrefillHandler] = {}
    activated_blocks = {
        block.width: block for block in getattr(activation, "blocks", ())}
    for handler in supplied_handlers:
        if not isinstance(handler, WidePrefillHandler):
            raise PrefillRuntimeError(
                "registered wide handler has the wrong type")
        spec = handler.spec
        if spec.width in handler_by_width:
            raise PrefillRuntimeError(
                f"duplicate registered S{spec.width} handler")
        block = activated_blocks.get(spec.width)
        if block is None or spec.width not in qualified:
            raise PrefillRuntimeError(
                f"S{spec.width} handler has no activated release block")
        try:
            live_model = spec.model_path.expanduser().resolve(strict=True)
        except OSError as error:
            raise PrefillRuntimeError(
                f"S{spec.width} handler model is unavailable: {error}") from error
        if live_model != block.model \
                or spec.descriptor.ready_descriptor_sha256 != \
                block.ready_descriptor_sha256:
            raise PrefillRuntimeError(
                f"S{spec.width} handler artifact/descriptor identity mismatch")
        handler_by_width[spec.width] = handler
    canonical_handlers = tuple(
        width for width in schedule_contract.CANDIDATE_WIDTHS
        if width > 1 and width in handler_by_width)
    canonical_handler_objects = tuple(
        handler_by_width[width] for width in canonical_handlers)
    enabled = tuple(
        width for width in schedule_contract.CANDIDATE_WIDTHS
        if width == 1 or width in canonical_handlers)
    unavailable = dict(getattr(activation, "disabled", {}))
    for width in WIDE_WIDTHS:
        if width in canonical_handlers:
            unavailable.pop(f"S{width}", None)
        elif width in qualified:
            unavailable[f"S{width}"] = (
                "release-qualified but no wide-handle executor is registered")
        else:
            unavailable.setdefault(f"S{width}", "not release-activated")

    return PrefillRuntimeRegistry(
        context=context,
        activation_manifest=Path(activation_manifest),
        activation_report=activation.to_dict(),
        qualified_widths=qualified,
        enabled_widths=enabled,
        handler_widths=canonical_handlers,
        unavailable=unavailable,
        handlers=canonical_handler_objects,
        strict_s1_anchor=strict_anchor,
        activated_model_sha256=(tuple(
            (block.width, block.model_sha256)
            for block in activation.blocks) if supplied_handlers else ()),
    )


def strict_s1_runtime(context: int) -> PrefillRuntimeRegistry:
    """Compatibility constructor for direct/legacy ``Merged`` callers."""
    return _strict_s1_registry(context)
