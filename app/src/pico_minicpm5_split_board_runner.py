#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Persistent JSONL runner for the MiniCPM5 24-layer split decode graph.

This is an orchestration boundary, not a fused-model claim.  Every A_raw,
shared R_dynamic, B, C and final-head invocation is executed independently
with a byte-exact DDR hand-off.  The default compatibility path uses
``svp_acl_board_probe`` per segment.  ``--persistent-executor`` instead keeps
one position-selected model table resident in a native ACL process while this
module retains the qualified 97-stage/token schedule.  Position zero and
steady decode use separate <=64-handle processes because real Hi3403 admission
rejects the 65th simultaneous model.  The module is deliberately Python-
standard-library-only so it can run in the board image.

The checked configuration must keep ``production_ready`` false until the
final dynamic OMs and a real board numeric run have been attached.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO, Mapping, Protocol, Sequence, TextIO


CONFIG_SCHEMA = "pico.minicpm5.split_board_runner.config.v1"
RUNNER_PROTOCOL = "pico.minicpm5.runner.v1"
MODEL_ID = "minicpm5-1b"
A_OUTPUT_ORDER = ("q", "k", "v")
R_OUTPUT_ORDER = ("current_v", "q_rope", "current_k")
REQUEST_FIELDS = frozenset({
    "protocol", "request_id", "op", "model", "input_ids",
    "max_new_tokens", "temperature", "eos_token_ids", "reset_kv",
})
SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,159}\Z")
MAX_JSONL_LINE_BYTES = 16 << 20
MAX_EOS_IDS = 8
PERSISTENT_PROTOCOL_VERSION = 1
PERSISTENT_READY_MAGIC = 0x31525850
PERSISTENT_REQUEST_MAGIC = 0x31514550
PERSISTENT_RESPONSE_MAGIC = 0x31534550
PERSISTENT_OP_EXECUTE = 1
PERSISTENT_OP_SHUTDOWN = 2
# Resident-input ops.  Each model's device input buffers are allocated once and
# already survive across executes; op 1 just overwrites all of them every call.
# For decode that is the dominant cost -- the KV window grows by one row per
# token, so re-sending it whole spends ~34% of the token budget pushing
# unchanged bytes through a pipe.  Op 3 writes only the bytes that moved, op 4
# executes against the retained buffers.  Needs an executor built from
# pypto/tools/pico_persistent_acl_executor.c at or after the resident-op change.
PERSISTENT_OP_WRITE_INPUT = 3
PERSISTENT_OP_EXECUTE_RESIDENT = 4
# Reduce a vocab-sized FP32 output to (index, value) inside the executor.  The
# head publishes 522 KB of logits only so the host can take an argmax; shipping
# that across the pipe costs 2.4 ms/token, and this board's Python has no numpy
# so the reduction after it is worse than the transfer.
PERSISTENT_OP_ARGMAX = 5
# Convert contiguous channel-major FP32 outputs to FP16 rows and scatter them
# into a resident model input.  The fixed record is generic layout metadata;
# tensor roles and MiniCPM geometry remain in the caller.
PERSISTENT_OP_SCATTER_F32_TO_F16 = 6
# Executor-owned resident-input snapshots. SAVE copies explicitly named input
# ranges into bounded host memory; RESTORE copies them back without crossing
# the protocol pipe. These opcodes are generic and contain no model semantics.
PERSISTENT_OP_SNAPSHOT_INPUTS = 7
PERSISTENT_OP_RESTORE_INPUTS = 8
# Generic resident input-to-input copy.  The protocol is intentionally scoped
# to one live executor/model table: it has no handle-generation field and does
# not claim that indices remain valid after a process or phase replacement.
PERSISTENT_OP_COPY_INPUTS = 9
PERSISTENT_MAX_MODELS = 128
# The binary protocol can describe 128 models, but a real Hi3403 admission
# rejected the 65th resident model with runtime error 500004.  Runtime model
# tables therefore fail closed at the observed 64-handle device limit.  The
# larger protocol limit remains useful for host-only protocol qualification.
PERSISTENT_RUNTIME_MODEL_LIMIT = 64
PERSISTENT_PHASE_STARTUP = "startup"
PERSISTENT_PHASE_STEADY = "steady"
PERSISTENT_PHASES = (PERSISTENT_PHASE_STARTUP, PERSISTENT_PHASE_STEADY)
PERSISTENT_MAX_MODEL_IO = 64
PERSISTENT_MAX_INPUT_COPY_RECORDS = 256
PERSISTENT_MAX_ERROR_BYTES = 4096
PERSISTENT_MAX_TENSOR_BYTES = 1 << 34
_PERSISTENT_READY = struct.Struct("<IHHII")
_PERSISTENT_COUNTS = struct.Struct("<II")
_PERSISTENT_REQUEST = struct.Struct("<IHHIIII")
_PERSISTENT_RESPONSE = struct.Struct("<IHHIIII")
_PERSISTENT_U64 = struct.Struct("<Q")
# One embedded write inside a resident execute frame: input index, flags,
# destination offset, byte length.  A PAYLOAD write is followed by that many
# bytes; a CHAIN write is followed by _PERSISTENT_CHAIN naming the producing
# model/output, and the bytes are copied device-to-device without ever crossing
# the pipe -- which is what a split decode schedule needs at each layer
# boundary, where the hidden state would otherwise be read back and written
# forward for no reason.
_PERSISTENT_WRITE = struct.Struct("<IIQQ")
_PERSISTENT_CHAIN = struct.Struct("<IIQ")
_PERSISTENT_SCATTER_F32_TO_F16 = struct.Struct("<IIIIQQIIQ")
_PERSISTENT_SNAPSHOT_RANGE = struct.Struct("<IIQQ")
_PERSISTENT_INPUT_COPY = struct.Struct("<IIQIIQQI")
PERSISTENT_WRITE_FLAG_PAYLOAD = 0
PERSISTENT_WRITE_FLAG_CHAIN = 1


class ConfigError(ValueError):
    """The split-runner configuration is incomplete or inconsistent."""


class ExecutionError(RuntimeError):
    """One native segment or a byte-layout contract failed."""


@dataclass(frozen=True)
class ResidentInputCopy:
    """One byte-exact input-to-input copy in the current resident table.

    Model identifiers are executor-local indices.  Callers must rebuild these
    records after replacing an executor process or changing its phase; protocol
    v1 deliberately has no model-table generation guard.
    """

    destination_model: int
    destination_input: int
    destination_offset: int
    source_model: int
    source_input: int
    source_offset: int
    length: int


@dataclass(frozen=True)
class Geometry:
    hidden: int
    context: int
    vocab: int
    layer_count: int
    query_heads: int
    kv_heads: int
    head_dim: int
    rope_elements: int
    window_rows: int = 2

    def validate(self) -> None:
        values = (
            self.hidden, self.context, self.vocab, self.layer_count,
            self.query_heads, self.kv_heads, self.head_dim,
            self.rope_elements, self.window_rows,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ConfigError("geometry values must be positive integers")
        if self.window_rows != 2:
            raise ConfigError("the native B contract requires a two-row window")
        if self.query_heads % self.kv_heads:
            raise ConfigError("query_heads must be divisible by kv_heads")
        if self.rope_elements != self.head_dim:
            raise ConfigError("rope_elements must equal head_dim")

    @property
    def hidden_carrier_bytes(self) -> int:
        return self.hidden * 4 * 4

    @property
    def q_f32_bytes(self) -> int:
        return self.query_heads * self.head_dim * 4

    @property
    def current_kv_f32_bytes(self) -> int:
        return self.kv_heads * self.head_dim * 4

    @property
    def cache_row_f16_bytes(self) -> int:
        return self.head_dim * 2

    @property
    def cache_layer_f16_bytes(self) -> int:
        return self.kv_heads * self.context * self.cache_row_f16_bytes

    @property
    def cache_kind_f16_bytes(self) -> int:
        return self.layer_count * self.cache_layer_f16_bytes

    @property
    def combined_window_f16_bytes(self) -> int:
        return (2 * self.layer_count * self.window_rows * self.kv_heads
                * self.cache_row_f16_bytes)

    @property
    def cache_window_f32_bytes(self) -> int:
        return self.combined_window_f16_bytes * 2

    @property
    def update_kv_f32_bytes(self) -> int:
        return (2 * self.layer_count * self.kv_heads * self.head_dim * 4)


MINICPM5_GEOMETRY = Geometry(
    hidden=1536,
    context=4096,
    vocab=130560,
    layer_count=24,
    query_heads=16,
    kv_heads=2,
    head_dim=128,
    rope_elements=128,
)


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    bytes: int | None = None
    sha256: str | None = None

    def verify(self, *, expected_bytes: int | None = None) -> None:
        if not self.path.is_file():
            raise ConfigError(f"artifact is not a file: {self.path}")
        size = self.path.stat().st_size
        if self.bytes is not None and size != self.bytes:
            raise ConfigError(
                f"artifact size drift for {self.path}: {size} != {self.bytes}")
        if expected_bytes is not None and size != expected_bytes:
            raise ConfigError(
                f"artifact geometry drift for {self.path}: "
                f"{size} != {expected_bytes}")
        if self.sha256 is not None:
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != self.sha256:
                raise ConfigError(f"artifact sha256 drift: {self.path}")


@dataclass(frozen=True)
class LayerModels:
    index: int
    a: ArtifactRef
    c: ArtifactRef
    v_override: ArtifactRef | None = None
    startup_a: ArtifactRef | None = None
    startup_c: ArtifactRef | None = None
    startup_v_override: ArtifactRef | None = None


@dataclass(frozen=True)
class RunnerConfig:
    source: Path
    geometry: Geometry
    probe: ArtifactRef
    library_paths: tuple[Path, ...]
    embedding: ArtifactRef
    rope_cos: ArtifactRef
    rope_sin: ArtifactRef
    layers: tuple[LayerModels, ...]
    r_model: ArtifactRef
    b_model: ArtifactRef
    final_head: ArtifactRef
    work_dir: Path
    timeout_seconds: int
    max_new_tokens: int
    mask_value: float
    head_aux_bytes: int
    head_workspace_bytes: int
    logits_carrier_multiplier: float
    startup_head: ArtifactRef | None = None

    def validate(
        self, *, verify_files: bool = True, enforce_minicpm5: bool = True,
    ) -> None:
        self.geometry.validate()
        if enforce_minicpm5 and self.geometry != MINICPM5_GEOMETRY:
            raise ConfigError("CLI configuration must use MiniCPM5-1B C4096 geometry")
        if tuple(layer.index for layer in self.layers) != tuple(
                range(self.geometry.layer_count)):
            raise ConfigError("models.layers must contain all ordered layer indices")
        for layer in self.layers:
            if (layer.startup_a is None) != (layer.startup_c is None):
                raise ConfigError(
                    f"layer{layer.index} startup_a/startup_c must be paired")
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be a positive integer")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens must be a positive integer")
        if self.head_aux_bytes <= 0 or self.head_workspace_bytes <= 0:
            raise ConfigError("final-head auxiliary sizes must be positive")
        if not math.isfinite(self.mask_value) or self.mask_value >= 0:
            raise ConfigError("mask_value must be finite and negative")
        if (not math.isfinite(self.logits_carrier_multiplier)
                or self.logits_carrier_multiplier <= 0):
            raise ConfigError("logits carrier multiplier must be positive")
        if verify_files:
            self.probe.verify()
            if not os.access(self.probe.path, os.X_OK):
                raise ConfigError(f"probe is not executable: {self.probe.path}")
            self.embedding.verify(
                expected_bytes=self.geometry.vocab * self.geometry.hidden * 2)
            rope_bytes = self.geometry.context * self.geometry.rope_elements * 2
            self.rope_cos.verify(expected_bytes=rope_bytes)
            self.rope_sin.verify(expected_bytes=rope_bytes)
            self.r_model.verify()
            self.b_model.verify()
            self.final_head.verify()
            if self.startup_head is not None:
                self.startup_head.verify()
            for layer in self.layers:
                layer.a.verify()
                layer.c.verify()
                if layer.v_override is not None:
                    layer.v_override.verify()
                if layer.startup_v_override is not None:
                    layer.startup_v_override.verify()
                if layer.startup_a is not None:
                    layer.startup_a.verify()
                    assert layer.startup_c is not None
                    layer.startup_c.verify()
            for directory in self.library_paths:
                if not directory.is_dir():
                    raise ConfigError(f"library path is not a directory: {directory}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _resolve(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _artifact(base: Path, value: object, name: str) -> ArtifactRef:
    if isinstance(value, str):
        return ArtifactRef(_resolve(base, value, name))
    item = _mapping(value, name)
    unknown = set(item) - {"path", "bytes", "sha256"}
    if unknown:
        raise ConfigError(f"unknown {name} fields: {sorted(unknown)}")
    declared_bytes = item.get("bytes")
    if declared_bytes is not None:
        declared_bytes = _integer(declared_bytes, f"{name}.bytes")
        if declared_bytes <= 0:
            raise ConfigError(f"{name}.bytes must be positive")
    sha = item.get("sha256")
    if sha is not None and (not isinstance(sha, str)
                            or not re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise ConfigError(f"{name}.sha256 must be lowercase hex")
    return ArtifactRef(
        _resolve(base, item.get("path"), f"{name}.path"), declared_bytes, sha)


def _optional_artifact(
    base: Path, item: Mapping[str, object], key: str, name: str,
) -> ArtifactRef | None:
    value = item.get(key)
    return None if value is None else _artifact(base, value, name)


def load_config(
    path: str | Path, *, verify_files: bool = True,
    enforce_minicpm5: bool = True,
) -> RunnerConfig:
    """Load and strictly validate one split-runner configuration."""
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read runner config: {source}") from exc
    root = _mapping(raw, "config")
    expected_root = {
        "schema", "production_ready", "model", "geometry", "probe",
        "assets", "models", "work_dir", "limits", "numeric_contract",
        "head_abi",
    }
    if set(root) != expected_root:
        raise ConfigError(
            f"config fields mismatch: got={sorted(root)}, "
            f"expected={sorted(expected_root)}")
    if root["schema"] != CONFIG_SCHEMA or root["model"] != MODEL_ID:
        raise ConfigError("split-runner schema/model mismatch")
    if root["production_ready"] is not False:
        raise ConfigError("candidate split runner must declare production_ready=false")

    geometry_raw = _mapping(root["geometry"], "geometry")
    geometry_fields = {
        "hidden", "context", "vocab", "layer_count", "query_heads",
        "kv_heads", "head_dim", "rope_elements", "window_rows",
    }
    if set(geometry_raw) != geometry_fields:
        raise ConfigError("geometry fields mismatch")
    geometry = Geometry(**{
        key: _integer(geometry_raw[key], f"geometry.{key}")
        for key in geometry_fields
    })

    probe_raw = _mapping(root["probe"], "probe")
    if set(probe_raw) != {"path", "library_paths"}:
        raise ConfigError("probe fields mismatch")
    libraries = probe_raw["library_paths"]
    if not isinstance(libraries, list) or not all(
            isinstance(value, str) for value in libraries):
        raise ConfigError("probe.library_paths must be a path array")

    assets = _mapping(root["assets"], "assets")
    if set(assets) != {"embedding", "rope_cos", "rope_sin"}:
        raise ConfigError("assets fields mismatch")
    models = _mapping(root["models"], "models")
    required_model_fields = {
        "layers", "a_output_order", "r", "r_output_order", "b",
        "final_head",
    }
    if not required_model_fields.issubset(models) \
            or set(models) - required_model_fields - {"startup_head"}:
        raise ConfigError("models fields mismatch")
    a_output_order = models["a_output_order"]
    if not isinstance(a_output_order, list) \
            or tuple(a_output_order) != A_OUTPUT_ORDER:
        raise ConfigError("models.a_output_order must be [q,k,v]")
    r_output_order = models["r_output_order"]
    if not isinstance(r_output_order, list) \
            or tuple(r_output_order) != R_OUTPUT_ORDER:
        raise ConfigError(
            "models.r_output_order must be "
            "[current_v,q_rope,current_k]")
    layer_values = models["layers"]
    if not isinstance(layer_values, list):
        raise ConfigError("models.layers must be an array")
    layers = []
    for offset, value in enumerate(layer_values):
        item = _mapping(value, f"models.layers[{offset}]")
        required_layer_fields = {"index", "a", "c"}
        optional_layer_fields = {
            "v_override", "startup_a", "startup_c", "startup_v_override"}
        if not required_layer_fields.issubset(item) or \
                set(item) - required_layer_fields - optional_layer_fields:
            raise ConfigError(f"models.layers[{offset}] fields mismatch")
        layers.append(LayerModels(
            _integer(item["index"], f"models.layers[{offset}].index"),
            _artifact(source.parent, item["a"], f"models.layers[{offset}].a"),
            _artifact(source.parent, item["c"], f"models.layers[{offset}].c"),
            _optional_artifact(
                source.parent, item, "v_override",
                f"models.layers[{offset}].v_override"),
            _optional_artifact(
                source.parent, item, "startup_a",
                f"models.layers[{offset}].startup_a"),
            _optional_artifact(
                source.parent, item, "startup_c",
                f"models.layers[{offset}].startup_c"),
            _optional_artifact(
                source.parent, item, "startup_v_override",
                f"models.layers[{offset}].startup_v_override"),
        ))

    limits = _mapping(root["limits"], "limits")
    if set(limits) != {"timeout_seconds", "max_new_tokens"}:
        raise ConfigError("limits fields mismatch")
    numeric = _mapping(root["numeric_contract"], "numeric_contract")
    if set(numeric) != {"mask_value"}:
        raise ConfigError("numeric_contract fields mismatch")
    head = _mapping(root["head_abi"], "head_abi")
    if set(head) != {
            "aux_input_bytes", "workspace_input_bytes",
            "logits_dtype", "logits_output_bytes",
            "carrier_to_logical_multiplier"}:
        raise ConfigError("head_abi fields mismatch")
    if head["logits_dtype"] != "F32":
        raise ConfigError("current final-head public logits ABI must be F32")
    expected_logits = geometry.vocab * 4
    if _integer(head["logits_output_bytes"],
                "head_abi.logits_output_bytes") != expected_logits:
        raise ConfigError("final-head logits output byte geometry drift")

    config = RunnerConfig(
        source=source,
        geometry=geometry,
        probe=_artifact(source.parent, probe_raw["path"], "probe.path"),
        library_paths=tuple(
            _resolve(source.parent, value, "probe.library_paths[]")
            for value in libraries),
        embedding=_artifact(source.parent, assets["embedding"],
                            "assets.embedding"),
        rope_cos=_artifact(source.parent, assets["rope_cos"],
                           "assets.rope_cos"),
        rope_sin=_artifact(source.parent, assets["rope_sin"],
                           "assets.rope_sin"),
        layers=tuple(layers),
        r_model=_artifact(source.parent, models["r"], "models.r"),
        b_model=_artifact(source.parent, models["b"], "models.b"),
        final_head=_artifact(source.parent, models["final_head"],
                             "models.final_head"),
        work_dir=_resolve(source.parent, root["work_dir"], "work_dir"),
        timeout_seconds=_integer(limits["timeout_seconds"],
                                 "limits.timeout_seconds"),
        max_new_tokens=_integer(limits["max_new_tokens"],
                                "limits.max_new_tokens"),
        mask_value=_number(numeric["mask_value"],
                           "numeric_contract.mask_value"),
        head_aux_bytes=_integer(head["aux_input_bytes"],
                                "head_abi.aux_input_bytes"),
        head_workspace_bytes=_integer(head["workspace_input_bytes"],
                                      "head_abi.workspace_input_bytes"),
        logits_carrier_multiplier=_number(
            head["carrier_to_logical_multiplier"],
            "head_abi.carrier_to_logical_multiplier"),
        startup_head=_optional_artifact(
            source.parent, models, "startup_head", "models.startup_head"),
    )
    config.validate(
        verify_files=verify_files, enforce_minicpm5=enforce_minicpm5)
    return config


def f16_to_f32_bytes(source: bytes) -> bytes:
    """Convert dense little-endian binary16 values to binary32."""
    if len(source) % 2:
        raise ExecutionError("F16 buffer has an odd byte count")
    output = bytearray((len(source) // 2) * 4)
    for index, (value,) in enumerate(struct.iter_unpack("<e", source)):
        struct.pack_into("<f", output, index * 4, float(value))
    return bytes(output)


def embedding_f16_to_c4_f32(source: bytes, hidden: int) -> bytes:
    """Place one dense FP16 embedding row into lane 0 of an F32 C4 carrier."""
    if len(source) != hidden * 2:
        raise ExecutionError("embedding row byte geometry drift")
    output = bytearray(hidden * 16)
    for index, (value,) in enumerate(struct.iter_unpack("<e", source)):
        struct.pack_into("<f", output, index * 16, float(value))
    return bytes(output)


def validate_c4_f32_carrier(source: bytes, hidden: int) -> None:
    """Validate the physical C4 carrier used between embedding/A/C/head."""
    if len(source) != hidden * 16:
        raise ExecutionError("hidden C4 carrier byte geometry drift")
    zero_padding = bytes(12)
    for channel in range(hidden):
        base = channel * 16
        if source[base + 4:base + 16] != zero_padding:
            raise ExecutionError("hidden C4 carrier has nonzero padding lanes")


def two_row_plan(position: int, context: int) -> tuple[int, int]:
    if position < 0 or position >= context:
        raise ExecutionError("decode position escapes configured context")
    start = position & ~1
    if start + 1 >= context:
        start = context - 2
    return start, position - start


def cache_row_range(
    geometry: Geometry, layer: int, head: int, position: int,
) -> tuple[int, int]:
    if layer not in range(geometry.layer_count) \
            or head not in range(geometry.kv_heads) \
            or position not in range(geometry.context):
        raise ExecutionError("cache row coordinate is outside geometry")
    offset = (((layer * geometry.kv_heads + head) * geometry.context
               + position) * geometry.cache_row_f16_bytes)
    return offset, offset + geometry.cache_row_f16_bytes


def build_b_inputs(
    geometry: Geometry, k_cache: bytearray, v_cache: bytearray,
    layer: int, position: int, current_k_f32: bytes, current_v_f32: bytes,
) -> tuple[bytes, bytes, bytes]:
    """Materialize B's F32 two-row carrier and per-layer update carrier."""
    if len(k_cache) != geometry.cache_kind_f16_bytes \
            or len(v_cache) != geometry.cache_kind_f16_bytes:
        raise ExecutionError("full-cache byte geometry drift")
    if len(current_k_f32) != geometry.current_kv_f32_bytes \
            or len(current_v_f32) != geometry.current_kv_f32_bytes:
        raise ExecutionError("current K/V byte geometry drift")
    window_start, local_position = two_row_plan(position, geometry.context)
    windows = bytearray()
    updates = bytearray()
    for kind, cache in enumerate((k_cache, v_cache)):
        for layer_index in range(geometry.layer_count):
            local_rows: list[bytes] = []
            for local in range(geometry.window_rows):
                row = bytearray()
                for head in range(geometry.kv_heads):
                    begin, end = cache_row_range(
                        geometry, layer_index, head, window_start + local)
                    row.extend(cache[begin:end])
                local_rows.append(bytes(row))
                windows.extend(f16_to_f32_bytes(local_rows[-1]))
            update = f16_to_f32_bytes(local_rows[local_position])
            if layer_index == layer:
                update = current_k_f32 if kind == 0 else current_v_f32
            updates.extend(update)
    if len(windows) != geometry.cache_window_f32_bytes \
            or len(updates) != geometry.update_kv_f32_bytes:
        raise AssertionError("B carrier construction size drift")
    position_i32 = struct.pack("<i", local_position) + bytes(12)
    return bytes(windows), position_i32, bytes(updates)


def apply_combined_window(
    geometry: Geometry, k_cache: bytearray, v_cache: bytearray,
    position: int, combined_f16: bytes,
) -> None:
    """Byte-copy all B window rows back into the two full-cache banks."""
    if len(combined_f16) != geometry.combined_window_f16_bytes:
        raise ExecutionError("combined B output byte geometry drift")
    window_start, _local_position = two_row_plan(position, geometry.context)
    segment = geometry.cache_row_f16_bytes
    cursor = 0
    for cache in (k_cache, v_cache):
        for layer in range(geometry.layer_count):
            for local in range(geometry.window_rows):
                for head in range(geometry.kv_heads):
                    begin, end = cache_row_range(
                        geometry, layer, head, window_start + local)
                    cache[begin:end] = combined_f16[cursor:cursor + segment]
                    cursor += segment
    if cursor != len(combined_f16):
        raise AssertionError("combined-window copy did not consume output")


def causal_mask_f32(position: int, context: int, mask_value: float) -> bytes:
    if position not in range(context):
        raise ExecutionError("causal-mask position escapes context")
    return (bytes((position + 1) * 4)
            + struct.pack("<f", mask_value) * (context - position - 1))


def argmax_f32(logits: bytes, vocab: int) -> int:
    if len(logits) != vocab * 4:
        raise ExecutionError("F32 logits byte geometry drift")
    best_index = -1
    best_value = -math.inf
    for index, (value,) in enumerate(struct.iter_unpack("<f", logits)):
        if not math.isfinite(value):
            raise ExecutionError("F32 logits contain a non-finite value")
        if best_index < 0 or value > best_value:
            best_index, best_value = index, value
    if best_index < 0:
        raise ExecutionError("F32 logits are empty")
    return best_index


class SegmentExecutor(Protocol):
    def execute(
        self, model: Path, inputs: Sequence[bytes],
        expected_output_bytes: Sequence[int], *, tag: str,
    ) -> tuple[bytes, ...]:
        """Execute one model and return all public output buffers."""


class ProbeExecutor:
    """Execute independent OM segments through ``svp_acl_board_probe``."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self, model: Path, inputs: Sequence[bytes],
        expected_output_bytes: Sequence[int], *, tag: str,
    ) -> tuple[bytes, ...]:
        directory = Path(tempfile.mkdtemp(
            prefix=f"{tag}.", dir=str(self.config.work_dir)))
        try:
            command = [str(self.config.probe.path), "--om", str(model)]
            for index, payload in enumerate(inputs):
                path = directory / f"input.{index}.bin"
                path.write_bytes(payload)
                command.extend(("--input", str(path)))
            prefix = directory / "output"
            command.extend(("--output-prefix", str(prefix)))
            environment = os.environ.copy()
            configured = ":".join(str(path) for path in self.config.library_paths)
            inherited = environment.get("LD_LIBRARY_PATH", "")
            environment["LD_LIBRARY_PATH"] = ":".join(
                value for value in (configured, inherited) if value)
            completed = subprocess.run(
                command, text=True, capture_output=True,
                timeout=self.config.timeout_seconds, env=environment,
                check=False)
            if completed.returncode or "probe.result=ok" not in completed.stdout:
                detail = (completed.stderr or completed.stdout)[-1000:].strip()
                raise ExecutionError(
                    f"segment {tag} failed rc={completed.returncode}: {detail}")
            outputs = tuple(
                (directory / f"output.{index}.bin").read_bytes()
                for index in range(len(expected_output_bytes)))
            actual = tuple(len(value) for value in outputs)
            expected = tuple(int(value) for value in expected_output_bytes)
            if actual != expected:
                raise ExecutionError(
                    f"segment {tag} output sizes {actual} != {expected}")
            return outputs
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(f"segment {tag} timed out") from exc
        except OSError as exc:
            raise ExecutionError(f"segment {tag} I/O failed: {exc}") from exc
        finally:
            shutil.rmtree(directory, ignore_errors=True)


def _persistent_model_paths(config: RunnerConfig) -> tuple[Path, ...]:
    """Return the stable union of every startup and steady model path.

    This is a provenance/disk table, not a board-resident table.  The Hi3403
    runtime limit applies independently to each table returned by
    :func:`_persistent_phase_model_paths`.
    """
    ordered: list[Path] = []
    seen: set[Path] = set()
    for layer in config.layers:
        paths = [layer.a.path, layer.c.path]
        if layer.v_override is not None:
            paths.append(layer.v_override.path)
        if layer.startup_v_override is not None:
            paths.append(layer.startup_v_override.path)
        if layer.startup_a is not None:
            assert layer.startup_c is not None
            paths.extend((layer.startup_a.path, layer.startup_c.path))
        for path in paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)
    shared = [config.r_model, config.b_model, config.final_head]
    if config.startup_head is not None:
        shared.append(config.startup_head)
    for artifact in shared:
        resolved = artifact.path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return tuple(ordered)


def _persistent_phase_model_paths(
    config: RunnerConfig, phase: str,
) -> tuple[Path, ...]:
    """Return one fail-closed board-resident model table.

    Position zero uses startup A/C overrides when they exist and the startup
    head when it exists.  The steady table contains only default A/C routes
    and the steady head.  Shared R/B are present in both phases; startup uses
    ``startup_v_override`` when present and otherwise falls back to the steady
    V override.  Resolved paths are de-duplicated in stable execution order.
    """
    if phase not in PERSISTENT_PHASES:
        raise ConfigError(f"unknown persistent executor phase: {phase}")
    ordered: list[Path] = []
    seen: set[Path] = set()

    def append(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    for layer in config.layers:
        if phase == PERSISTENT_PHASE_STARTUP:
            a_model = layer.startup_a if layer.startup_a is not None else layer.a
            c_model = layer.startup_c if layer.startup_c is not None else layer.c
        else:
            a_model, c_model = layer.a, layer.c
        append(a_model.path)
        append(c_model.path)
        v_model = (layer.startup_v_override
                   if phase == PERSISTENT_PHASE_STARTUP
                   and layer.startup_v_override is not None
                   else layer.v_override)
        if v_model is not None:
            append(v_model.path)
    append(config.r_model.path)
    append(config.b_model.path)
    if phase == PERSISTENT_PHASE_STARTUP and config.startup_head is not None:
        append(config.startup_head.path)
    else:
        append(config.final_head.path)
    if not 0 < len(ordered) <= PERSISTENT_RUNTIME_MODEL_LIMIT:
        raise ConfigError(
            f"persistent {phase} phase exceeds observed Hi3403 runtime limit "
            f"{PERSISTENT_RUNTIME_MODEL_LIMIT}: {len(ordered)} unique handles")
    return tuple(ordered)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise ExecutionError("persistent ACL executor closed its protocol pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def _read_exact_until(stream: BinaryIO, size: int, deadline: float) -> bytes:
    """Read one protocol field with a single monotonic deadline."""
    chunks = bytearray()
    while len(chunks) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ExecutionError("persistent ACL executor response timed out")
        try:
            readable, _, _ = select.select([stream], [], [], remaining)
        except InterruptedError:
            continue
        except (OSError, ValueError) as exc:
            raise ExecutionError(
                f"persistent ACL executor protocol wait failed: {exc}") from exc
        if not readable:
            raise ExecutionError("persistent ACL executor response timed out")
        try:
            chunk = os.read(stream.fileno(), size - len(chunks))
        except OSError as exc:
            raise ExecutionError(
                f"persistent ACL executor protocol read failed: {exc}") from exc
        if not chunk:
            raise ExecutionError("persistent ACL executor closed its protocol pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    """Write a whole binary protocol field even when the pipe short-writes."""
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = stream.write(view[written:])
        if count is None or count <= 0:
            raise ExecutionError(
                "persistent ACL executor protocol pipe short-wrote")
        written += count


class PersistentAclExecutor:
    """Reuse native ACL model handles and datasets across all split segments.

    The executor does no model emulation: every successful response is the
    direct byte payload returned by ``svp_acl_mdl_execute``.  Container-
    internal descriptor inputs remain zero-filled, matching ``ProbeExecutor``;
    only the compiler-qualified public input/output prefix crosses the pipe.
    """

    def __init__(
        self, config: RunnerConfig, executable: str | Path, *, device: int = 0,
        phase: str | None = None,
    ) -> None:
        if type(device) is not int or device < 0:
            raise ConfigError("persistent executor device must be nonnegative")
        self.config = config
        self.executable = Path(executable).expanduser().resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ConfigError(
                f"persistent ACL executor is not executable: {self.executable}")
        self.phase = phase
        self.model_paths = (_persistent_phase_model_paths(config, phase)
                            if phase is not None
                            else _persistent_model_paths(config))
        self.model_indices = {
            path: index for index, path in enumerate(self.model_paths)
        }
        if not 0 < len(self.model_paths) <= PERSISTENT_RUNTIME_MODEL_LIMIT:
            raise ConfigError(
                "persistent ACL executor model table exceeds observed Hi3403 "
                f"runtime limit {PERSISTENT_RUNTIME_MODEL_LIMIT}: "
                f"{len(self.model_paths)}")
        command = [str(self.executable), "--device", str(device)]
        for path in self.model_paths:
            command.extend(("--model", str(path)))
        environment = os.environ.copy()
        configured = ":".join(str(path) for path in config.library_paths)
        inherited = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = ":".join(
            value for value in (configured, inherited) if value)
        self._command = tuple(command)
        self._environment = environment
        self.process: subprocess.Popen[bytes] | None = None
        self._closed = False
        self.descriptors: tuple[
            tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
        self._start_process()

    def _start_process(self) -> None:
        if self._closed:
            raise ExecutionError("persistent ACL executor is closed")
        if self.process is not None:
            raise ExecutionError("persistent ACL executor is already running")
        try:
            self.process = subprocess.Popen(
                self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, bufsize=0, env=self._environment)
        except OSError as exc:
            raise ExecutionError(
                f"cannot start persistent ACL executor: {exc}") from exc
        try:
            self.descriptors = self._read_ready()
        except Exception:
            self._dispose_process()
            raise

    @property
    def _stdin(self) -> BinaryIO:
        if self.process is None or self.process.stdin is None:
            raise ExecutionError("persistent ACL executor stdin is unavailable")
        return self.process.stdin

    @property
    def _stdout(self) -> BinaryIO:
        if self.process is None or self.process.stdout is None:
            raise ExecutionError("persistent ACL executor stdout is unavailable")
        return self.process.stdout

    def _read_ready(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        deadline = time.monotonic() + self.config.timeout_seconds
        read = lambda size: _read_exact_until(self._stdout, size, deadline)
        magic, version, status, model_count, error_bytes = \
            _PERSISTENT_READY.unpack(read(_PERSISTENT_READY.size))
        if magic != PERSISTENT_READY_MAGIC \
                or version != PERSISTENT_PROTOCOL_VERSION \
                or model_count > PERSISTENT_MAX_MODELS \
                or error_bytes > PERSISTENT_MAX_ERROR_BYTES:
            raise ExecutionError("persistent ACL executor ready frame is invalid")
        if status:
            if model_count != 0:
                raise ExecutionError(
                    "failed persistent ready frame carries a model table")
            message = read(error_bytes).decode(
                "utf-8", errors="replace")
            raise ExecutionError(
                f"persistent ACL executor startup failed: {message}")
        if error_bytes != 0 or model_count != len(self.model_paths):
            raise ExecutionError(
                "persistent ACL executor model table does not match request")
        descriptors: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for _model_index in range(model_count):
            input_count, output_count = _PERSISTENT_COUNTS.unpack(
                read(_PERSISTENT_COUNTS.size))
            if not (0 < input_count <= PERSISTENT_MAX_MODEL_IO
                    and 0 < output_count <= PERSISTENT_MAX_MODEL_IO):
                raise ExecutionError(
                    "persistent ACL executor descriptor count is invalid")
            inputs = tuple(_PERSISTENT_U64.unpack(
                read(_PERSISTENT_U64.size))[0]
                for _ in range(input_count))
            outputs = tuple(_PERSISTENT_U64.unpack(
                read(_PERSISTENT_U64.size))[0]
                for _ in range(output_count))
            if any(size == 0 or size > PERSISTENT_MAX_TENSOR_BYTES
                   for size in (*inputs, *outputs)):
                raise ExecutionError(
                    "persistent ACL executor descriptor size is invalid")
            descriptors.append((inputs, outputs))
        return tuple(descriptors)

    def _read_response(
        self, expected_model: int, expected_output_sizes: Sequence[int], *,
        timeout_seconds: float | None = None,
    ) -> tuple[bytes, ...]:
        timeout = (self.config.timeout_seconds if timeout_seconds is None
                   else timeout_seconds)
        deadline = time.monotonic() + timeout
        read = lambda size: _read_exact_until(self._stdout, size, deadline)
        (magic, version, status, model_index, output_count,
         error_bytes, reserved) = _PERSISTENT_RESPONSE.unpack(
             read(_PERSISTENT_RESPONSE.size))
        if magic != PERSISTENT_RESPONSE_MAGIC \
                or version != PERSISTENT_PROTOCOL_VERSION \
                or model_index != expected_model or reserved != 0 \
                or output_count > PERSISTENT_MAX_MODEL_IO \
                or error_bytes > PERSISTENT_MAX_ERROR_BYTES:
            raise ExecutionError("persistent ACL executor response frame is invalid")
        if status:
            if output_count != 0:
                raise ExecutionError(
                    "failed persistent response carries tensor outputs")
            message = read(error_bytes).decode(
                "utf-8", errors="replace")
            raise ExecutionError(
                f"persistent ACL model[{model_index}] failed: {message}")
        if error_bytes != 0:
            raise ExecutionError("successful persistent response carries an error")
        sizes = tuple(_PERSISTENT_U64.unpack(
            read(_PERSISTENT_U64.size))[0]
            for _ in range(output_count))
        expected = tuple(int(value) for value in expected_output_sizes)
        if output_count != len(expected) or sizes != expected \
                or any(size <= 0 or size > PERSISTENT_MAX_TENSOR_BYTES
                       for size in sizes):
            raise ExecutionError(
                f"persistent ACL model[{model_index}] response sizes "
                f"{sizes} != {expected}")
        return tuple(read(size) for size in sizes)

    def copy_resident_inputs(
        self, records: Sequence[ResidentInputCopy], *,
        tag: str = "resident-input-copy",
    ) -> None:
        """Copy validated byte ranges between inputs in this live process.

        All records are validated and serialized before the request header is
        written, matching the native executor's no-destination-write behavior
        for a bad record.  The operation carries no model-table generation;
        executor-local indices must not be retained across a process restart or
        :class:`PhasedPersistentAclExecutor` phase transition.
        """
        if self._closed:
            raise ExecutionError("persistent ACL executor is closed")
        if self.process is None:
            self._start_process()
        fixed = tuple(records)
        if not 0 < len(fixed) <= PERSISTENT_MAX_INPUT_COPY_RECORDS:
            raise ExecutionError(
                f"{tag} record count must be in "
                f"[1, {PERSISTENT_MAX_INPUT_COPY_RECORDS}]")
        encoded: list[bytes] = []
        for index, record in enumerate(fixed):
            if not isinstance(record, ResidentInputCopy):
                raise ExecutionError(
                    f"{tag} record[{index}] is not a ResidentInputCopy")
            values = (
                record.destination_model, record.destination_input,
                record.destination_offset, record.source_model,
                record.source_input, record.source_offset, record.length)
            if any(type(value) is not int for value in values):
                raise ExecutionError(
                    f"{tag} record[{index}] fields must be integers")
            if not 0 <= record.destination_model < len(self.descriptors) \
                    or not 0 <= record.source_model < len(self.descriptors):
                raise ExecutionError(
                    f"{tag} record[{index}] model index is out of range")
            destination_inputs = self.descriptors[
                record.destination_model][0]
            source_inputs = self.descriptors[record.source_model][0]
            if not 0 <= record.destination_input < len(destination_inputs) \
                    or not 0 <= record.source_input < len(source_inputs):
                raise ExecutionError(
                    f"{tag} record[{index}] input index is out of range")
            if record.destination_offset < 0 or record.source_offset < 0 \
                    or not 0 < record.length <= PERSISTENT_MAX_TENSOR_BYTES:
                raise ExecutionError(
                    f"{tag} record[{index}] offset or length is invalid")
            destination_capacity = destination_inputs[
                record.destination_input]
            source_capacity = source_inputs[record.source_input]
            if record.destination_offset > destination_capacity \
                    or record.length > (destination_capacity -
                                          record.destination_offset) \
                    or record.source_offset > source_capacity \
                    or record.length > (source_capacity -
                                          record.source_offset):
                raise ExecutionError(
                    f"{tag} record[{index}] exceeds source or destination "
                    "input")
            encoded.append(_PERSISTENT_INPUT_COPY.pack(
                record.destination_model, record.destination_input,
                record.destination_offset, record.source_model,
                record.source_input, record.source_offset, record.length, 0))
        frame = _PERSISTENT_REQUEST.pack(
            PERSISTENT_REQUEST_MAGIC, PERSISTENT_PROTOCOL_VERSION,
            PERSISTENT_OP_COPY_INPUTS, 0, len(encoded), 0, 0) + b"".join(encoded)
        try:
            _write_all(self._stdin, frame)
            self._stdin.flush()
            outputs = self._read_response(0, ())
            if outputs:
                raise ExecutionError(
                    f"{tag} returned unexpected tensor outputs")
        except ExecutionError:
            self._dispose_process()
            raise
        except (BrokenPipeError, OSError) as exc:
            self._dispose_process()
            raise ExecutionError(
                f"{tag} persistent ACL transport failed: {exc}") from exc

    def execute(
        self, model: Path, inputs: Sequence[bytes],
        expected_output_bytes: Sequence[int], *, tag: str,
    ) -> tuple[bytes, ...]:
        if self._closed:
            raise ExecutionError("persistent ACL executor is closed")
        if self.process is None:
            self._start_process()
        resolved = model.resolve()
        model_index = self.model_indices.get(resolved)
        if model_index is None:
            raise ExecutionError(f"segment {tag} model is not resident: {resolved}")
        descriptor_inputs, descriptor_outputs = self.descriptors[model_index]
        input_sizes = tuple(len(value) for value in inputs)
        output_sizes = tuple(int(value) for value in expected_output_bytes)
        if len(input_sizes) > len(descriptor_inputs) \
                or input_sizes != descriptor_inputs[:len(input_sizes)]:
            raise ExecutionError(
                f"segment {tag} public input prefix {input_sizes} disagrees "
                f"with descriptor {descriptor_inputs}")
        if len(output_sizes) > len(descriptor_outputs) \
                or output_sizes != descriptor_outputs[:len(output_sizes)]:
            raise ExecutionError(
                f"segment {tag} public output prefix {output_sizes} disagrees "
                f"with descriptor {descriptor_outputs}")
        header = _PERSISTENT_REQUEST.pack(
            PERSISTENT_REQUEST_MAGIC, PERSISTENT_PROTOCOL_VERSION,
            PERSISTENT_OP_EXECUTE, model_index, len(inputs),
            len(output_sizes), 0)
        try:
            _write_all(self._stdin, header)
            for size in (*input_sizes, *output_sizes):
                _write_all(self._stdin, _PERSISTENT_U64.pack(size))
            for payload in inputs:
                _write_all(self._stdin, payload)
            self._stdin.flush()
            outputs = self._read_response(model_index, output_sizes)
        except ExecutionError:
            self._dispose_process()
            raise
        except (BrokenPipeError, OSError) as exc:
            self._dispose_process()
            raise ExecutionError(
                f"segment {tag} persistent ACL transport failed: {exc}") from exc
        actual = tuple(len(value) for value in outputs)
        if actual != output_sizes:
            raise ExecutionError(
                f"segment {tag} output sizes {actual} != {output_sizes}")
        return outputs

    def _terminate(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def _dispose_process(self) -> None:
        """Terminate one poisoned session and release both protocol pipes."""
        process = self.process
        if process is None:
            return
        self._terminate()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self.process = None
        self.descriptors = ()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                _write_all(self._stdin, _PERSISTENT_REQUEST.pack(
                    PERSISTENT_REQUEST_MAGIC, PERSISTENT_PROTOCOL_VERSION,
                    PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0))
                self._stdin.flush()
                outputs = self._read_response(
                    0, (), timeout_seconds=10)
                if outputs:
                    raise ExecutionError(
                        "persistent ACL shutdown returned tensor outputs")
                process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired,
                ExecutionError):
            self._terminate()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            self.process = None
            self.descriptors = ()


class PhasedPersistentAclExecutor:
    """Own one <=64-handle native executor for each decode lifecycle phase.

    Transitioning closes the old ACL process before admitting the new table,
    while KV remains in :class:`SplitBoardRunner` host memory.  A failed new
    admission leaves no fallback process and propagates an execution error.
    """

    def __init__(
        self, config: RunnerConfig, executable: str | Path, *, device: int = 0,
    ) -> None:
        self.config = config
        self.executable = Path(executable).expanduser().resolve()
        self.device = device
        self.phase: str | None = None
        self.executor: PersistentAclExecutor | None = None
        self._closed = False
        self.ensure_phase(PERSISTENT_PHASE_STARTUP)

    def ensure_phase(self, phase: str) -> None:
        if phase not in PERSISTENT_PHASES:
            raise ConfigError(f"unknown persistent executor phase: {phase}")
        if self._closed:
            raise ExecutionError("phased persistent ACL executor is closed")
        if self.phase == phase and self.executor is not None:
            return
        previous = self.executor
        self.executor = None
        self.phase = None
        if previous is not None:
            previous.close()
        try:
            replacement = PersistentAclExecutor(
                self.config, self.executable, device=self.device, phase=phase)
        except Exception:
            # PersistentAclExecutor disposes a partially started subprocess;
            # keep this wrapper phase-less so later reset can retry startup.
            raise
        self.executor = replacement
        self.phase = phase

    def execute(
        self, model: Path, inputs: Sequence[bytes],
        expected_output_bytes: Sequence[int], *, tag: str,
    ) -> tuple[bytes, ...]:
        if self._closed or self.executor is None or self.phase is None:
            raise ExecutionError("phased persistent ACL executor is unavailable")
        return self.executor.execute(
            model, inputs, expected_output_bytes, tag=tag)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        executor = self.executor
        self.executor = None
        self.phase = None
        if executor is not None:
            executor.close()


class SplitBoardRunner:
    """Stateful token decoder over 24 A_raw/R_dynamic/B/C stages."""

    def __init__(
        self, config: RunnerConfig, executor: SegmentExecutor | None = None,
    ) -> None:
        config.validate(verify_files=False, enforce_minicpm5=False)
        self.config = config
        self.geometry = config.geometry
        self.executor = executor if executor is not None else ProbeExecutor(config)
        self.k_cache = bytearray(self.geometry.cache_kind_f16_bytes)
        self.v_cache = bytearray(self.geometry.cache_kind_f16_bytes)
        self.position = 0
        self._embedding_stream: BinaryIO | None = None
        self._cos_stream: BinaryIO | None = None
        self._sin_stream: BinaryIO | None = None

    def __enter__(self) -> "SplitBoardRunner":
        self._embedding_stream = self.config.embedding.path.open("rb")
        self._cos_stream = self.config.rope_cos.path.open("rb")
        self._sin_stream = self.config.rope_sin.path.open("rb")
        return self

    def __exit__(self, *_exc: object) -> None:
        closer = getattr(self.executor, "close", None)
        if callable(closer):
            closer()
        for stream in (self._embedding_stream, self._cos_stream, self._sin_stream):
            if stream is not None:
                stream.close()
        self._embedding_stream = self._cos_stream = self._sin_stream = None

    def reset(self) -> None:
        ensure_phase = getattr(self.executor, "ensure_phase", None)
        if callable(ensure_phase):
            # A new request always begins at p0 and therefore must re-admit
            # the startup route before clearing the host-owned KV state.
            ensure_phase(PERSISTENT_PHASE_STARTUP)
        self.k_cache[:] = bytes(len(self.k_cache))
        self.v_cache[:] = bytes(len(self.v_cache))
        self.position = 0

    @staticmethod
    def _read_row(stream: BinaryIO | None, offset: int, size: int) -> bytes:
        if stream is None:
            raise ExecutionError("runner assets are not open")
        stream.seek(offset)
        result = stream.read(size)
        if len(result) != size:
            raise ExecutionError("runtime asset row is truncated")
        return result

    def _embedding(self, token_id: int) -> bytes:
        row_bytes = self.geometry.hidden * 2
        source = self._read_row(
            self._embedding_stream, token_id * row_bytes, row_bytes)
        return embedding_f16_to_c4_f32(source, self.geometry.hidden)

    def _rope(self, position: int) -> tuple[bytes, bytes]:
        row_bytes = self.geometry.rope_elements * 2
        offset = position * row_bytes
        cosine = self._read_row(self._cos_stream, offset, row_bytes)
        sine = self._read_row(self._sin_stream, offset, row_bytes)
        return f16_to_f32_bytes(cosine), f16_to_f32_bytes(sine)

    def _layer_cache(self, cache: bytearray, layer: int) -> bytes:
        begin = layer * self.geometry.cache_layer_f16_bytes
        end = begin + self.geometry.cache_layer_f16_bytes
        return bytes(cache[begin:end])

    def _execute(
        self, model: Path, inputs: Sequence[bytes],
        expected_output_bytes: Sequence[int], *, tag: str,
    ) -> tuple[bytes, ...]:
        outputs = tuple(self.executor.execute(
            model, inputs, expected_output_bytes, tag=tag))
        expected = tuple(int(value) for value in expected_output_bytes)
        actual = tuple(len(value) for value in outputs)
        if len(outputs) != len(expected) or actual != expected:
            raise ExecutionError(
                f"segment {tag} output sizes {actual} != {expected}")
        return outputs

    def decode_token(self, token_id: int) -> bytes:
        """Consume one token and return the final-head public F32 carrier."""
        if token_id not in range(self.geometry.vocab):
            raise ExecutionError("token id escapes vocabulary")
        position = self.position
        if position >= self.geometry.context:
            raise ExecutionError("decode position exceeds context")
        hidden = self._embedding(token_id)
        validate_c4_f32_carrier(hidden, self.geometry.hidden)
        rope_cos, rope_sin = self._rope(position)
        mask = causal_mask_f32(
            position, self.geometry.context, self.config.mask_value)
        zero_residual = bytes(self.geometry.hidden_carrier_bytes)
        for layer in self.config.layers:
            startup_route = position == 0 and layer.startup_a is not None
            a_model = layer.startup_a if startup_route else layer.a
            c_model = layer.startup_c if startup_route else layer.c
            assert a_model is not None and c_model is not None
            # A_dequant's board-qualified Reportop slots are q/k/v.  K and V
            # have identical byte geometry, so this ordering is a named ABI,
            # not an inference from output sizes.
            raw_q, raw_k, raw_v = self._execute(
                a_model.path, (hidden, zero_residual),
                (self.geometry.q_f32_bytes,
                 self.geometry.current_kv_f32_bytes,
                 self.geometry.current_kv_f32_bytes),
                tag=(f"p{position}.l{layer.index}.A_startup"
                     if startup_route else f"p{position}.l{layer.index}.A"))
            startup_v_route = position == 0 \
                and layer.startup_v_override is not None
            v_model = (layer.startup_v_override if startup_v_route
                       else layer.v_override)
            if v_model is not None:
                # Some indexed layers need an independently calibrated native
                # V projection.  It consumes the same hidden/zero-residual ABI
                # as A and replaces only A's V public slot.  No host scaling or
                # numeric repair is permitted at this boundary.
                (raw_v,) = self._execute(
                    v_model.path, (hidden, zero_residual),
                    (self.geometry.current_kv_f32_bytes,),
                    tag=(f"p{position}.l{layer.index}.V_override_startup"
                         if startup_v_route else
                         f"p{position}.l{layer.index}.V_override"))
            # R's board-qualified physical report slots do not follow the
            # semantic graph declaration.  Bind by this fixed slot contract;
            # two outputs are both 1024 bytes, so size-based sorting is unsafe.
            current_v, q, current_k = self._execute(
                self.config.r_model.path,
                (raw_q, raw_k, raw_v, rope_cos, rope_sin),
                (self.geometry.current_kv_f32_bytes,
                 self.geometry.q_f32_bytes,
                 self.geometry.current_kv_f32_bytes),
                tag=f"p{position}.l{layer.index}.R")
            b_inputs = build_b_inputs(
                self.geometry, self.k_cache, self.v_cache,
                layer.index, position, current_k, current_v)
            (combined,) = self._execute(
                self.config.b_model.path, b_inputs,
                (self.geometry.combined_window_f16_bytes,),
                tag=f"p{position}.l{layer.index}.B")
            apply_combined_window(
                self.geometry, self.k_cache, self.v_cache,
                position, combined)
            (hidden,) = self._execute(
                c_model.path,
                (q, self._layer_cache(self.k_cache, layer.index),
                 self._layer_cache(self.v_cache, layer.index), mask, hidden),
                (self.geometry.hidden_carrier_bytes,),
                tag=(f"p{position}.l{layer.index}.C_startup"
                     if startup_route else f"p{position}.l{layer.index}.C"))
            validate_c4_f32_carrier(hidden, self.geometry.hidden)
        startup_head_route = position == 0 \
            and self.config.startup_head is not None
        head_model = (self.config.startup_head if startup_head_route
                      else self.config.final_head)
        assert head_model is not None
        (logits,) = self._execute(
            head_model.path,
            (hidden, zero_residual, bytes(self.config.head_aux_bytes),
             bytes(self.config.head_workspace_bytes)),
            (self.geometry.vocab * 4,),
            tag=(f"p{position}.head_startup" if startup_head_route
                 else f"p{position}.head"))
        if position == 0:
            ensure_phase = getattr(self.executor, "ensure_phase", None)
            if callable(ensure_phase):
                # The position-0 logits are already in host memory.  Close
                # every startup handle, admit the steady table, and retain KV
                # entirely in this runner.  Do not advance position unless
                # the replacement table was admitted successfully.
                ensure_phase(PERSISTENT_PHASE_STEADY)
        self.position += 1
        return logits

    def generate(self, request: Mapping[str, object]) -> dict[str, object]:
        validated = validate_request(request, self.config)
        self.reset()
        logits = b""
        for token_id in validated["input_ids"]:
            logits = self.decode_token(token_id)
        output_ids: list[int] = []
        eos = set(validated["eos_token_ids"])
        finish_reason = "length"
        for generation_index in range(validated["max_new_tokens"]):
            token = argmax_f32(logits, self.geometry.vocab)
            output_ids.append(token)
            if token in eos:
                finish_reason = "stop"
                break
            if generation_index + 1 < validated["max_new_tokens"]:
                logits = self.decode_token(token)
        return {
            "protocol": RUNNER_PROTOCOL,
            "request_id": validated["request_id"],
            "ok": True,
            "output_ids": output_ids,
            "finish_reason": finish_reason,
        }


def validate_request(
    request: Mapping[str, object], config: RunnerConfig,
) -> dict[str, object]:
    if set(request) != REQUEST_FIELDS:
        raise ValueError("runner request fields mismatch")
    if request["protocol"] != RUNNER_PROTOCOL \
            or request["op"] != "generate" \
            or request["model"] != MODEL_ID \
            or request["reset_kv"] is not True:
        raise ValueError("runner request contract mismatch")
    request_id = request["request_id"]
    if not isinstance(request_id, str) or not SAFE_REQUEST_ID.fullmatch(request_id):
        raise ValueError("request_id contains unsupported characters")
    temperature = request["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not math.isfinite(float(temperature)) or float(temperature) != 0.0:
        raise ValueError("only temperature=0 is supported")
    input_ids = request["input_ids"]
    eos_ids = request["eos_token_ids"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("input_ids must be a non-empty array")
    if not isinstance(eos_ids, list) or len(eos_ids) > MAX_EOS_IDS:
        raise ValueError("eos_token_ids must be an array")
    if any(type(value) is not int or value not in range(config.geometry.vocab)
           for value in (*input_ids, *eos_ids)):
        raise ValueError("token id escapes vocabulary")
    max_new = request["max_new_tokens"]
    if type(max_new) is not int or max_new <= 0 \
            or max_new > config.max_new_tokens:
        raise ValueError("max_new_tokens escapes configured limit")
    if len(input_ids) + max_new - 1 > config.geometry.context:
        raise ValueError("request exceeds decode context")
    return {
        "request_id": request_id,
        "input_ids": tuple(input_ids),
        "eos_token_ids": tuple(eos_ids),
        "max_new_tokens": max_new,
    }


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def serve_jsonl(
    runner: SplitBoardRunner, reader: TextIO, writer: TextIO,
) -> None:
    """Serve the service-compatible line protocol until EOF."""
    while True:
        line = reader.readline(MAX_JSONL_LINE_BYTES + 1)
        if not line:
            break
        request_id = "unknown"
        try:
            if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                raise ValueError("runner request line exceeds size limit")
            request = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(request, dict):
                raise ValueError("runner request root must be an object")
            candidate = request.get("request_id")
            if isinstance(candidate, str) and SAFE_REQUEST_ID.fullmatch(candidate):
                request_id = candidate
            response = runner.generate(request)
        except (ValueError, ConfigError) as exc:
            response = {
                "protocol": RUNNER_PROTOCOL, "request_id": request_id,
                "ok": False,
                "error": {"code": "invalid_request", "message": str(exc)},
            }
        except ExecutionError as exc:
            response = {
                "protocol": RUNNER_PROTOCOL, "request_id": request_id,
                "ok": False,
                "error": {"code": "execution_error", "message": str(exc)},
            }
        writer.write(json.dumps(
            response, ensure_ascii=False, separators=(",", ":")) + "\n")
        writer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--serve-jsonl", action="store_true")
    parser.add_argument(
        "--persistent-executor", type=Path,
        help="native multi-model ACL executor; keeps all segment OMs resident")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.check_config:
            print(json.dumps({
                "schema": CONFIG_SCHEMA,
                "config": str(config.source),
                "validated": True,
                "production_ready": False,
                "model_execution": False,
            }, separators=(",", ":")))
            return 0
        executor = (PhasedPersistentAclExecutor(
            config, args.persistent_executor, device=args.device)
            if args.persistent_executor is not None else None)
        with SplitBoardRunner(config, executor) as runner:
            segments_per_token = 97 + sum(
                layer.v_override is not None
                or layer.startup_v_override is not None
                for layer in config.layers)
            startup_route_layers = sum(
                layer.startup_a is not None for layer in config.layers)
            print(
                "split_runner=ready production_ready=0 "
                f"segments_per_token={segments_per_token} "
                f"startup_route_layers={startup_route_layers} "
                f"startup_head={int(config.startup_head is not None)} "
                f"executor={'persistent_acl' if executor else 'probe'}",
                file=sys.stderr, flush=True)
            serve_jsonl(runner, sys.stdin, sys.stdout)
        return 0
    except (ConfigError, ExecutionError) as exc:
        print(f"split runner startup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
