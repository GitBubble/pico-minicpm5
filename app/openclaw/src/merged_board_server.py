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
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
if os.environ.get("PICO_MINICPM5_SHARED_SRC"):
    sys.path.insert(0, os.environ["PICO_MINICPM5_SHARED_SRC"])
sys.path.insert(0, str(HERE))

import qualify_minicpm_greedy_chain as gc  # noqa: E402
import pico_minicpm5_split_board_runner as runner  # noqa: E402
import probe_om_execute_latency as probe  # noqa: E402

HIDDEN, KV_HEADS, HEAD_DIM, LAYERS = gc.HIDDEN, gc.KV_HEADS, gc.HEAD_DIM, gc.LAYERS
CHANNELS = LAYERS * KV_HEADS
MIN_BOARD_SAFE_SHORT_CONTEXT = 128
C8192_CONTEXT = 8192
C8192_DESCRIPTOR_INPUTS = 7
# The SS928 ACL ready descriptor expands the synthesized control input from
# libinstsim's 112-byte view to 352 bytes.  The zero-once experiment is a board
# runtime contract and therefore admits only the live ACL size.
C8192_CONTROL_INPUT_BYTES = 352
C8192_WORKSPACE_INPUT_BYTES = 206_127_104


def _validated_context(value):
    value = int(value)
    if value < 2:
        raise ValueError("context must be at least 2")
    return value


def _validated_short_context(value, *, allow_unsafe=False):
    value = int(value)
    if value < 2:
        raise ValueError("short context must be at least 2")
    if value < MIN_BOARD_SAFE_SHORT_CONTEXT and not allow_unsafe:
        raise ValueError(
            f"short context {value} is below the board-qualified floor "
            f"{MIN_BOARD_SAFE_SHORT_CONTEXT}; pass "
            "--allow-unsafe-short-context only for characterization")
    return value


def _validate_short_route(context, short_context, *,
                          allow_c8192_short_characterization=False):
    if short_context >= context:
        raise ValueError(
            f"short context {short_context} must be smaller than long context "
            f"{context}")
    if context >= C8192_CONTEXT \
            and not allow_c8192_short_characterization:
        raise ValueError(
            f"C{context}+short routing is not MMZ-qualified; pass "
            "--allow-c8192-short-characterization only for characterization")


def _validate_workspace_zero_once_mode(
        context, *, enabled, executor_uncached, decode_no_cache,
        decode_short):
    if not enabled:
        return
    if context != C8192_CONTEXT:
        raise ValueError(
            "decode workspace zero-once characterization requires C8192")
    if executor_uncached or not decode_no_cache:
        raise ValueError(
            "decode workspace zero-once characterization requires only "
            "--decode-no-cache, not global --executor-uncached")
    if decode_short is not None:
        raise ValueError(
            "decode workspace zero-once characterization does not admit a "
            "short decode handle")


def _validate_resume_preflight(context, start, kv_in):
    """Reject an impossible resume before loading or executing any model."""
    context = _validated_context(context)
    start = int(start)
    if start < 0:
        raise ValueError("start position must be nonnegative")
    if start >= context:
        raise ValueError(
            f"start position {start} exceeds context {context}")
    if start > 0 and kv_in is None:
        raise ValueError("nonzero start position requires an exact KV snapshot")
    if kv_in is not None:
        expected = 2 * CHANNELS * (context - 1) * HEAD_DIM * 2
        actual = Path(kv_in).stat().st_size
        if actual != expected:
            raise ValueError(
                f"KV snapshot has {actual} bytes; expected exactly {expected}")


def _requires_head(position, prompt_length, *, kv_out=None, stop_after=None):
    """Return whether this step must produce a model-selected next token.

    Before the final prompt token, the next token is teacher-forced by the
    caller.  Running the 130k-vocabulary head and argmax there cannot affect
    hidden/KV state or the eventual generated sequence.  Snapshot probes keep
    the legacy prediction when they explicitly stop on this position.
    """
    teacher_forced = position + 1 < prompt_length
    snapshot_prediction = (kv_out is not None and stop_after is not None
                           and position == stop_after)
    return not teacher_forced or snapshot_prediction


class Merged:
    def __init__(self, *, executable, decode, prefill, head, library_paths,
                 embedding, context, timeout, tokenizer=None,
                 resident_kv=True, decode_short=None,
                 short_context=MIN_BOARD_SAFE_SHORT_CONTEXT,
                 short_kv_slots=None, allow_unsafe_short_context=False,
                 allow_c8192_short_characterization=False,
                 executor_uncached=False, decode_no_cache=False,
                 characterize_decode_workspace_zero_once=False):
        self.context = _validated_context(context)
        self.past = self.context - 1
        self.timeout = timeout
        self.row_f16 = HEAD_DIM * 2
        self.cache_bytes = CHANNELS * self.past * HEAD_DIM * 2
        self.resident_kv = bool(resident_kv)
        if executor_uncached and decode_no_cache:
            raise ValueError(
                "global no-cache and decode-only no-cache are mutually exclusive")
        _validate_workspace_zero_once_mode(
            self.context,
            enabled=characterize_decode_workspace_zero_once,
            executor_uncached=executor_uncached,
            decode_no_cache=decode_no_cache,
            decode_short=decode_short)
        if decode_short and not resident_kv:
            raise ValueError("short-context decode requires resident K/V")
        self.models = [decode] + ([decode_short] if decode_short else []) \
            + ([prefill] if prefill else []) + [head]
        self.decode_index = 0
        self.short_decode_index = 1 if decode_short else None
        self.short_context = (_validated_short_context(
            short_context, allow_unsafe=allow_unsafe_short_context)
            if decode_short else None)
        if self.short_context is not None:
            _validate_short_route(
                self.context, self.short_context,
                allow_c8192_short_characterization=
                allow_c8192_short_characterization)
        self.kv_slot_overrides = {}
        if self.short_decode_index is not None and short_kv_slots is not None:
            slots = tuple(int(slot) for slot in short_kv_slots)
            if len(slots) != 2 or len(set(slots)) != 2 \
                    or min(slots) < 0:
                raise ValueError("short K/V slots must be two distinct indices")
            self.kv_slot_overrides[self.short_decode_index] = slots
        self.prefill_index = (1 + int(bool(decode_short))) if prefill else 0
        self.head_index = len(self.models) - 1
        self.embed = open(embedding, "rb")
        self._embedding_half = struct.Struct(f"<{HIDDEN}e")
        self._embedding_f32 = struct.Struct(f"<{HIDDEN}f")
        self._embedding_c4 = struct.Struct("<" + "f12x" * HIDDEN)
        self._rope_inverse = tuple(
            1.0 / (gc.ROPE_THETA ** ((2 * i) / HEAD_DIM))
            for i in range(HEAD_DIM // 2))
        self.tokenizer = None
        if tokenizer:
            from tokenizers import Tokenizer  # noqa: PLC0415
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
        executor_args = (
            ("--no-cache",) if executor_uncached else
            (("--no-cache-model", "0",
              "--retain-input", "0:6:5:206127104")
             if characterize_decode_workspace_zero_once else
             (("--no-cache-model", "0") if decode_no_cache else ())))
        self.process = probe._start(
            executable, self.models, library_paths, 0,
            extra_executor_args=executor_args)
        self._closed = False
        self._deadline = time.monotonic() + timeout
        self.descriptors = probe._read_ready(
            self.process.stdout, len(self.models), self._deadline)
        self._validate_long_model_contract()
        if characterize_decode_workspace_zero_once:
            self._validate_workspace_zero_once_contract()
        if self.short_decode_index is not None:
            self._validate_short_model_contract()
        self.kv_slots = self._identify_kv()

    # -- transport -------------------------------------------------------
    def _respond(self, expected_model, out_sizes):
        outputs, _responded_at = probe._read_response(
            self.process.stdout, expected_model=expected_model,
            expected_sizes=out_sizes, deadline=self._deadline)
        return list(outputs)

    def _validate_decode_model_contract(self, index, expected_context, label):
        """Reject an OM that is not the packed 24-layer decode ABI."""
        desc_in, desc_out = self.descriptors[index]
        if len(desc_in) < 5:
            raise RuntimeError(
                f"{label} model has {len(desc_in)} inputs; expected at least 5")

        expected_hidden = HIDDEN * 16
        if desc_in[0] != expected_hidden:
            raise RuntimeError(
                f"{label} hidden input size {desc_in[0]} != {expected_hidden}")

        mask_bytes = desc_in[1]
        if mask_bytes % 4:
            raise RuntimeError(
                f"{label} mask has invalid byte size {mask_bytes}")
        actual_context = mask_bytes // 4
        if actual_context != expected_context:
            raise RuntimeError(
                f"{label} mask context {actual_context} != {expected_context}")

        expected_rope = HEAD_DIM * HEAD_DIM * 4
        if desc_in[2] != expected_rope:
            raise RuntimeError(
                f"{label} rope input size {desc_in[2]} != {expected_rope}")

        expected_cache = CHANNELS * (expected_context - 1) * self.row_f16
        cache_sizes = (desc_in[3], desc_in[4])
        if cache_sizes != (expected_cache, expected_cache):
            raise RuntimeError(
                f"{label} cache input sizes {cache_sizes} != "
                f"{(expected_cache, expected_cache)} for N24 ctx"
                f"{expected_context}")

        expected_output = CHANNELS * HEAD_DIM * 4
        if len(desc_out) != 3 or any(
                size != expected_output for size in desc_out):
            raise RuntimeError(
                f"{label} output ABI {desc_out} must contain exactly "
                f"three {expected_output}-byte hidden/K/V outputs")

    def _validate_long_model_contract(self):
        self._validate_decode_model_contract(
            self.decode_index, self.context, "long decode")

    def _validate_workspace_zero_once_contract(self):
        """Bind the fixed characterization to the C8192 tmp-buffer ABI."""
        desc_in, _desc_out = self.descriptors[self.decode_index]
        if len(desc_in) != C8192_DESCRIPTOR_INPUTS:
            raise RuntimeError(
                "C8192 zero-once characterization requires exactly "
                f"{C8192_DESCRIPTOR_INPUTS} decode inputs, got {len(desc_in)}")
        tail = (desc_in[5], desc_in[6])
        expected = (C8192_CONTROL_INPUT_BYTES, C8192_WORKSPACE_INPUT_BYTES)
        if tail != expected:
            raise RuntimeError(
                f"C8192 zero-once decode tail {tail} != {expected}")

    def _validate_short_model_contract(self):
        """Reject a short OM that is not the packed 24-layer decode ABI."""
        index = self.short_decode_index
        self._validate_decode_model_contract(
            index, self.short_context, "short decode")

        override = self.kv_slot_overrides.get(index)
        desc_out = self.descriptors[index][1]
        if override is not None and max(override) >= len(desc_out):
            raise RuntimeError(
                f"short decode K/V slots {override} exceed three-output ABI")

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
        # Stream the resident request field-by-field.  Building gc._frame()
        # materializes the full payload more than once, which is wasteful for
        # C4096/C8192 KV snapshot hydration (roughly 96/192 MiB).  The emitted
        # byte sequence remains identical to gc._frame().
        writes = tuple(writes)
        chains = tuple(chains)
        runner._write_all(
            self.process.stdin,
            runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_EXECUTE_RESIDENT,
                model, min(5, len(desc_in)), len(output_sizes),
                len(writes) + len(chains)))
        for size in output_sizes:
            runner._write_all(
                self.process.stdin, runner._PERSISTENT_U64.pack(size))
        for slot, offset, data in writes:
            payload = memoryview(data)
            runner._write_all(
                self.process.stdin,
                runner._PERSISTENT_WRITE.pack(
                    slot, runner.PERSISTENT_WRITE_FLAG_PAYLOAD,
                    offset, len(payload)))
            runner._write_all(self.process.stdin, payload)
        for slot, offset, length, source_model, source_output in chains:
            runner._write_all(
                self.process.stdin,
                runner._PERSISTENT_WRITE.pack(
                    slot, runner.PERSISTENT_WRITE_FLAG_CHAIN,
                    offset, length))
            runner._write_all(
                self.process.stdin,
                runner._PERSISTENT_CHAIN.pack(
                    source_model, source_output, 0))
        self.process.stdin.flush()
        return self._respond(model, output_sizes)

    def _model_context(self, model):
        size = self.descriptors[model][0][1]
        if size % 4:
            raise RuntimeError(f"model {model}: invalid mask bytes {size}")
        return size // 4

    def _decode_model(self, position):
        if self.short_decode_index is not None and position < self.short_context:
            return self.short_decode_index
        return self.decode_index

    def _scatter_destinations(self, position):
        destinations = []
        if position < self.past:
            destinations.append(self.decode_index)
        if self.short_decode_index is not None \
                and position < self.short_context - 1:
            destinations.append(self.short_decode_index)
        return tuple(destinations)

    def _scatter_kv(self, source_model, position, destination_model):
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
            destination_model, 2, 0, 0)
        base = position * self.row_f16
        past = self._model_context(destination_model) - 1
        stride = past * self.row_f16
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
        self._respond(destination_model, ())

    def _identify_kv(self):
        """RoPE moves k_cur and not v_cur.  Same structural test the split
        runner uses, applied once to the merged model."""
        hidden = self._hidden_input(
            0, self.descriptors[self.decode_index][0][0])
        out = {}
        transformer_models = {self.decode_index, self.prefill_index}
        if self.short_decode_index is not None:
            transformer_models.add(self.short_decode_index)
        for model in transformer_models:
            model_context = self._model_context(model)
            cache_bytes = CHANNELS * (model_context - 1) * self.row_f16
            zero_k = bytes(cache_bytes)
            mask = bytearray(
                struct.pack("<f", gc.MASK_NEG) * model_context)
            struct.pack_into("<f", mask, (model_context - 1) * 4, 0.0)
            seen = []
            for position in (0, 1):
                writes = [(0, 0, hidden), (1, 0, bytes(mask)),
                          (2, 0, self._rope_matrix_bytes(position)),
                          (3, 0, zero_k), (4, 0, zero_k)]
                seen.append(self._run(model, writes))
            moved = [i for i in range(2) if seen[0][i] != seen[1][i]]
            if len(moved) != 1:
                override = self.kv_slot_overrides.get(model)
                if override is None:
                    raise RuntimeError(
                        f"model {model}: RoPE moved {len(moved)} of 2 KV outputs; "
                        "the k/v identification is not safe")
                if max(override) >= len(self.descriptors[model][1]):
                    raise RuntimeError(
                        f"model {model}: K/V override exceeds published outputs")
                print(
                    f"warning: model {model} RoPE probe moved {len(moved)} slots; "
                    f"using qualified K/V override {override}",
                    file=sys.stderr, flush=True)
                out[model] = override
            else:
                out[model] = (moved[0], 1 - moved[0])
        return out

    def _validate_generate_start(self, prompt_ids, start, kv_in):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if start < 0:
            raise ValueError("start position must be nonnegative")
        if start >= len(prompt_ids):
            raise ValueError(
                f"start position {start} exceeds prompt length "
                f"{len(prompt_ids)}")
        if start >= self.context:
            raise ValueError(
                f"start position {start} exceeds context {self.context}")
        if start > 0 and kv_in is None:
            raise ValueError("nonzero start position requires an exact KV snapshot")

    def _host_kv_mirrors(self, host_kv, kv_in):
        if not host_kv:
            return None, None
        k_cache = bytearray(self.cache_bytes)
        v_cache = bytearray(self.cache_bytes)
        if kv_in is not None:
            expected = 2 * self.cache_bytes
            actual = Path(kv_in).stat().st_size
            if actual != expected:
                raise ValueError(
                    f"KV snapshot has {actual} bytes; expected exactly "
                    f"{expected}")

            def read_exact_into(stream, target):
                view = memoryview(target)
                offset = 0
                while offset < len(view):
                    count = stream.readinto(view[offset:])
                    if count is None or count <= 0:
                        raise ValueError("KV snapshot was truncated while reading")
                    offset += count

            with Path(kv_in).open("rb", buffering=0) as stream:
                read_exact_into(stream, k_cache)
                read_exact_into(stream, v_cache)
                if stream.read(1):
                    raise ValueError("KV snapshot grew while reading")
        return k_cache, v_cache

    def _host_kv_writes(self, position, start, desc_in, k_cache, v_cache):
        if position == 0:
            # Prefill may deliberately retain its qualified C1024 ABI while the
            # long decode handle is C4096/C8192.  Position zero has no past, so
            # write descriptor-sized zero caches instead of the long mirrors.
            return [(3, 0, bytes(desc_in[3])), (4, 0, bytes(desc_in[4]))]
        if position == 1 or position == start:
            # A resumed run starts with fresh executor datasets.  Hydrate the
            # entire validated snapshot once; later steps only update one row.
            return [(3, 0, memoryview(k_cache)),
                    (4, 0, memoryview(v_cache))]

        prior = position - 1
        writes = []
        for slot, mirror in ((3, k_cache), (4, v_cache)):
            for channel in range(CHANNELS):
                at = (channel * self.past + prior) * self.row_f16
                writes.append(
                    (slot, at, memoryview(mirror)[at:at + self.row_f16]))
        return writes

    # -- decode ----------------------------------------------------------
    def generate(self, prompt_ids, max_new, eos, *, start=0,
                 kv_in=None, kv_out=None, stop_after=None,
                 capture_dir=None, capture_position=None):
        self.last_phase_steps = []
        self._validate_generate_start(prompt_ids, start, kv_in)
        host_kv = not self.resident_kv or kv_in is not None or kv_out is not None
        k_cache, v_cache = self._host_kv_mirrors(host_kv, kv_in)
        token = prompt_ids[start]
        produced, produced_ids, steps = 0, [], []
        total = len(prompt_ids) + max_new
        for position in range(start, total):
            if position >= self.context:
                return "context", produced_ids, steps
            began = time.perf_counter()
            self._deadline = time.monotonic() + self.timeout
            model = (self.prefill_index if position == 0
                     else self._decode_model(position))
            desc_in = self.descriptors[model][0]
            model_context = self._model_context(model)

            mask = bytearray(
                struct.pack("<f", gc.MASK_NEG) * model_context)
            for slot in range(position):
                struct.pack_into("<f", mask, slot * 4, 0.0)
            struct.pack_into("<f", mask, (model_context - 1) * 4, 0.0)

            writes = [(0, 0, self._hidden_input(token, desc_in[0])),
                      (1, 0, bytes(mask)),
                      (2, 0, self._rope_matrix_bytes(position))]
            if host_kv:
                writes.extend(self._host_kv_writes(
                    position, start, desc_in, k_cache, v_cache))
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
            if host_kv and position < self.past:
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
                for destination in self._scatter_destinations(position):
                    self._scatter_kv(model, position, destination)
            kv_at = time.perf_counter()

            if not _requires_head(
                    position, len(prompt_ids), kv_out=kv_out,
                    stop_after=stop_after):
                # This position is still inside the supplied prompt.  Its
                # transformer/KV result is needed by the next position, but
                # its logits are unobservable because the caller has already
                # supplied the next token.
                steps.append((kv_at - began) * 1000.0)
                self.last_phase_steps.append({
                    "position": position,
                    "prepare_ms": (prepared_at - began) * 1000.0,
                    "transformer_ms": (transformer_at - prepared_at) * 1000.0,
                    "kv_ms": (kv_at - transformer_at) * 1000.0,
                    "head_execute_ms": 0.0,
                    "argmax_ms": 0.0,
                    "total_ms": (kv_at - began) * 1000.0,
                    "head_skipped_teacher_forcing": True,
                })
                token = prompt_ids[position + 1]
                continue

            head_in = self.descriptors[self.head_index][0]
            src = self._hidden_output(model)
            runner._write_all(self.process.stdin, gc._frame(
                self.head_index, min(3, len(head_in)), (), [],
                [(0, 0, min(head_in[0], self.descriptors[model][1][src]),
                  model, src)]))
            self.process.stdin.flush()
            self._respond(self.head_index, ())
            head_at = time.perf_counter()
            runner._write_all(self.process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_ARGMAX, self.head_index, 0, 0, 0))
            self.process.stdin.flush()
            (payload,) = self._respond(self.head_index, (8,))
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
            if int(predicted) in eos:
                return "eos", produced_ids, steps
            if produced >= max_new:
                return "max", produced_ids, steps
            token = int(predicted)
        return "max", produced_ids, steps

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        process = self.process
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
                outputs = self._respond(0, ())
                if outputs:
                    raise RuntimeError(
                        "persistent executor shutdown returned tensor outputs")
                # The ACK precedes native model/device teardown.  Waiting for
                # natural process exit closes that teardown before lifecycle
                # scripts report MMZ cleanup complete.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persistent-executor", type=Path, required=True)
    ap.add_argument(
        "--executor-uncached", action="store_true",
        help="start the persistent executor with its fixed --no-cache mode")
    ap.add_argument(
        "--decode-no-cache", action="store_true",
        help="allocate only long decode model[0] without CPU caching; keep "
             "prefill/head cached")
    ap.add_argument(
        "--characterize-decode-workspace-zero-once", action="store_true",
        help="characterization only: with --decode-no-cache, zero C8192 "
             "decode model0/input6 only at allocation and reuse it")
    ap.add_argument("--decode-model", type=Path, required=True)
    ap.add_argument(
        "--decode-short-model", type=Path,
        help="optional short-context decode OM used below --short-context")
    ap.add_argument(
        "--short-context", type=int, default=MIN_BOARD_SAFE_SHORT_CONTEXT)
    ap.add_argument(
        "--allow-unsafe-short-context", action="store_true",
        help="allow a short context below the board-qualified C128 floor; "
             "characterization only")
    ap.add_argument(
        "--allow-c8192-short-characterization", action="store_true",
        help="characterization only: allow an unqualified C8192-or-larger "
             "long decode and short-context handle in one residency")
    ap.add_argument(
        "--short-kv-slots", default=None,
        help="qualified short-model output slots as K,V (for example 0,1)")
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
    ap.add_argument("--max-new", type=int, default=16)
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
    if args.decode_short_model is not None \
            and (args.host_kv or args.kv_in is not None
                 or args.kv_out is not None):
        ap.error(
            "--decode-short-model is incompatible with host K/V options")
    try:
        context = _validated_context(args.context)
        _validate_workspace_zero_once_mode(
            context,
            enabled=args.characterize_decode_workspace_zero_once,
            executor_uncached=args.executor_uncached,
            decode_no_cache=args.decode_no_cache,
            decode_short=args.decode_short_model)
        _validate_resume_preflight(
            context, args.start_position, args.kv_in)
        if args.decode_short_model is not None:
            short_context = _validated_short_context(
                args.short_context,
                allow_unsafe=args.allow_unsafe_short_context)
            _validate_short_route(
                context, short_context,
                allow_c8192_short_characterization=
                args.allow_c8192_short_characterization)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    short_kv_slots = None
    if args.short_kv_slots is not None:
        try:
            short_kv_slots = tuple(
                int(value) for value in args.short_kv_slots.split(","))
        except ValueError:
            ap.error("--short-kv-slots must contain integer K,V indices")
        if args.decode_short_model is None:
            ap.error("--short-kv-slots requires --decode-short-model")

    began = time.perf_counter()
    session = Merged(executable=args.persistent_executor,
                     decode=args.decode_model, prefill=args.prefill_model,
                     head=args.head_model, library_paths=args.library_path,
                     embedding=args.embedding, context=args.context,
                     timeout=args.timeout, tokenizer=args.tokenizer,
                     resident_kv=not args.host_kv,
                     decode_short=args.decode_short_model,
                     short_context=args.short_context,
                     short_kv_slots=short_kv_slots,
                     allow_unsafe_short_context=
                     args.allow_unsafe_short_context,
                     allow_c8192_short_characterization=
                     args.allow_c8192_short_characterization,
                     executor_uncached=args.executor_uncached,
                     decode_no_cache=args.decode_no_cache,
                     characterize_decode_workspace_zero_once=
                     args.characterize_decode_workspace_zero_once)
    load_ms = (time.perf_counter() - began) * 1000.0
    print(f"loaded {len(session.models)} handles in {load_ms / 1000:.1f} s; "
          f"kv slots {session.kv_slots}", flush=True)
    results = []
    try:
        for spec in args.prompt:
            ids = [0] + list(session.tokenizer.encode(
                spec, add_special_tokens=False).ids)
            reason, out, steps = session.generate(
                ids, args.max_new, {1, 130073}, start=args.start_position,
                kv_in=args.kv_in, kv_out=args.kv_out,
                stop_after=args.stop_after, capture_dir=args.capture_dir,
                capture_position=args.capture_position)
            text = session.tokenizer.decode(out, skip_special_tokens=True)
            gen = steps[len(ids) - 1:]
            print(f"prompt={spec!r}\n  ids={out}\n  text={text!r}\n"
                  f"  reason={reason} steps_ms={[round(s,1) for s in gen]}",
                  flush=True)
            results.append({"prompt": spec, "ids": out, "text": text,
                            "reason": reason, "step_ms": steps,
                            "phase_ms": session.last_phase_steps})
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
    finally:
        session.close()
    if args.report:
        args.report.write_text(json.dumps(
            {"schema": "pico.minicpm5.merged_board.v1",
             "load_seconds": load_ms / 1000.0, "runs": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
