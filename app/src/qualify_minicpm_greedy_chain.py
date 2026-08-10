#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-token greedy decode on board, scored against the float greedy golden.

Single-token argmax is a degenerate judge for this checkpoint: 21 of 21
arbitrary tokens greedily predict 5 with no context, so a broken implementation
passes by emitting 5.  The project's own note says acceptance has to be a
context-carrying greedy sequence.  This is that test.

    "The capital of France is"  ->  " Paris, a city renowned for its rich ..."

Every position runs the real decode path: the token's embedding goes in, 24
layers run chained on device, the head produces logits, and the executor's
argmax picks the next token -- which is then embedded and fed back.  Nothing is
seeded from the reference after position 0, so the first divergence is real and
everything after it is consequence.

KV handling.  The layer graph appends the current row at the END of its window
(``Concat(k_cache, k_cur) axis=2``), so position p is stored at cache index p
and the mask opens ``[0, p)`` plus the final slot.  Attention is permutation
invariant once RoPE is applied, so index order does not matter -- only which
slots the mask admits.  ``k_cur``/``v_cur`` come back as FP32 while the cache is
FP16, so the row is converted host-side; ``struct`` handles that without numpy,
which the board does not have.

Only the changed row is pushed after the first position, so per-token pipe
traffic is ~24 KB rather than the ~25 MB a full cache rewrite would cost.

Three modes:

    prepare  (host)  prompt ids, rope tables per position, golden continuation
    execute  (board) the greedy loop
    score    (host)  generated ids vs the float golden

Three defects fixed 2026-08-06, all of which would have corrupted the first
passing run rather than the failing ones:

* **No eos stop.**  ``eos_token_id`` is ``[1, 130073]`` and the loop ran a fixed
  number of steps regardless, feeding the eos token back in as context.  Latent
  while the score is 0/16 -- the run never reached an eos -- and silently wrong
  the moment the chain starts producing the reference continuation on a prompt
  the reference terminates.
* **One token too many.**  ``total = len(prompt_ids) + max_new`` runs the loop
  once past the last generated token, because the first generated token is
  produced at position ``len(prompt_ids) - 1``, not ``len(prompt_ids)``.  The
  archived ``greedy_calibfix_score.json`` shows the symptom: 17 board tokens
  against 16 expected.  ``score`` zips, so it was invisible; the extra step also
  costs a full 24-layer pass.
* **Unpinned tokenisation.**  The prompt ids are taken from the oracle, and the
  oracle is now the artifact that pins them, but nothing here checked what
  arrived.  ``prepare`` refuses ids that do not carry exactly one leading BOS,
  which is the shape the double-BOS mistake produces
  (``[0, 0, 608, ...]`` instead of ``[0, 608, ...]``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pico_minicpm5_split_board_runner as runner  # noqa: E402
import probe_om_execute_latency as probe  # noqa: E402

HIDDEN = 1536
KV_HEADS = 2
HEAD_DIM = 128
LAYERS = 24
VOCAB = 130560
ROPE_THETA = 5_000_000.0
MASK_NEG = -64.0
EMBED_BIN = "/opt/pico-minicpm5/assets/token_embedding.f16.bin"
BOS_TOKEN_ID = 0
BOS_POLICY = "pinned-single"
# scripts/build_minicpm5_greedy_oracle.py records these from model.config; the
# fallback is only used when an older reference without the field is supplied.
FALLBACK_EOS = (1, 130073)


def _apply_spec(path: Path) -> None:
    """Override the module geometry/contract from a ModelSpec JSON.

    Only runs under ``--spec``; without it the MiniCPM5-1B constants above are
    untouched.  Imported lazily so an old board deployment without
    pico_model_spec keeps working.
    """
    global HIDDEN, KV_HEADS, HEAD_DIM, LAYERS, VOCAB, ROPE_THETA, MASK_NEG
    global EMBED_BIN, BOS_TOKEN_ID, BOS_POLICY, FALLBACK_EOS
    from pico_model_spec import load_spec  # noqa: PLC0415
    spec = load_spec(path)
    geo, con = spec.geometry, spec.contract
    HIDDEN, KV_HEADS, HEAD_DIM = geo.hidden, geo.kv_heads, geo.head_dim
    LAYERS, VOCAB, ROPE_THETA = geo.layers, geo.vocab, geo.rope_theta
    MASK_NEG = con.mask_negative
    EMBED_BIN = con.board.embed_bin
    BOS_POLICY = con.bos_policy
    if con.bos_token_id is not None:
        BOS_TOKEN_ID = con.bos_token_id
    FALLBACK_EOS = tuple(con.eos_ids)


def _numpy():
    import numpy as np  # noqa: PLC0415
    return np


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_oracle(golden: Path, prompt_index: int) -> tuple[dict, dict]:
    """Accept the FP64 oracle, or the older FP32 float_greedy_reference."""
    reference = json.loads(golden.read_text())
    schema = reference.get("schema", "")
    if "prompts" in reference:                      # greedy_oracle.v1
        entry = reference["prompts"][prompt_index]
    elif "greedy" in reference:                     # float_greedy_reference.v1
        entry = reference["greedy"][prompt_index]
    else:
        raise SystemExit(f"{golden}: no 'prompts' or 'greedy' array")
    provenance = {
        "schema": schema,
        "path": str(golden),
        "sha256": _sha256_file(golden),
        "dtype": reference.get("dtype"),
        "generated_utc": reference.get("generated_utc"),
        "git_commit": reference.get("git_commit"),
    }
    eos = reference.get("eos_token_id")
    if eos is None:
        eos = (reference.get("config") or {}).get("eos_token_id")
    if eos is None:
        eos = list(FALLBACK_EOS)
        provenance["eos_source"] = "hard-coded fallback: reference carries none"
    else:
        provenance["eos_source"] = "reference"
    entry = dict(entry)
    entry["_eos"] = [int(v) for v in
                     (eos if isinstance(eos, (list, tuple)) else [eos])]
    return entry, provenance


def _check_tokenisation(prompt_ids: Sequence[int]) -> None:
    """Refuse the double-BOS shape before 24 layers run on it.

    ``tokenizer('<s>The capital of France is')`` under the default
    ``add_special_tokens=True`` yields ``[0, 0, 608, ...]`` and decodes to a
    different, coherent-looking continuation -- the failure that reads like a
    missing scale factor somewhere in the NPU stack.
    """
    if not prompt_ids:
        raise SystemExit("prompt_token_ids is empty")
    if BOS_POLICY == "none":
        # The spec pins no BOS; the tokenizer's own convention is authoritative
        # and there is no double-BOS shape to refuse.
        return
    if prompt_ids[0] != BOS_TOKEN_ID:
        raise SystemExit(
            f"prompt_token_ids must start with BOS {BOS_TOKEN_ID}, "
            f"got {list(prompt_ids[:4])}")
    if len(prompt_ids) > 1 and prompt_ids[1] == BOS_TOKEN_ID:
        raise SystemExit(
            f"double BOS in {list(prompt_ids[:4])}: the oracle was built from "
            f"a '<s>'-prefixed literal. Rebuild it with "
            f"scripts/build_minicpm5_greedy_oracle.py, which refuses that "
            f"spelling.")


def prepare(*, prompt_index: int, golden: Path, context: int, max_new: int,
            work: Path) -> dict[str, Any]:
    np = _numpy()
    entry, provenance = _read_oracle(golden, prompt_index)
    prompt_ids = list(entry["prompt_token_ids"])
    _check_tokenisation(prompt_ids)
    eos = entry["_eos"]
    expected = list(entry["generated_token_ids"])[:max_new]
    # The first generated token is produced at position len(prompt_ids) - 1 --
    # the step that consumes the LAST prompt token -- so the loop needs
    # len(prompt_ids) - 1 prefill positions plus max_new generating ones.  The
    # old expression ran one 24-layer pass too many and produced max_new + 1
    # tokens.
    total = len(prompt_ids) + max_new - 1

    # Llama RoPE: inverse frequencies duplicated across the two halves, so the
    # elementwise form x*cos + rotate_half(x)*sin is the standard rotation.
    inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, np.float64)
                                / HEAD_DIM))
    work.mkdir(parents=True, exist_ok=True)
    cos_rows, sin_rows = [], []
    for position in range(total):
        angle = np.concatenate((inv, inv)) * position
        cos_rows.append(np.cos(angle).astype(np.float32))
        sin_rows.append(np.sin(angle).astype(np.float32))
    (work / "rope_cos.bin").write_bytes(
        np.ascontiguousarray(np.stack(cos_rows)).tobytes())
    (work / "rope_sin.bin").write_bytes(
        np.ascontiguousarray(np.stack(sin_rows)).tobytes())

    manifest = {
        "schema": "pico.minicpm5.greedy_chain.inputs.v2",
        "prompt": entry.get("prompt"), "prompt_token_ids": prompt_ids,
        "expected_token_ids": expected, "expected_text": entry.get(
            "generated_text"),
        "expected_stopped_on_eos": bool(entry.get("stopped_on_eos", False)),
        "eos_token_ids": eos,
        "oracle": provenance,
        "context": context, "max_new_tokens": max_new, "positions": total,
        "hidden": HIDDEN, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "layers": LAYERS, "mask_negative": MASK_NEG, "embedding": EMBED_BIN,
    }
    (work / "greedy.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prompt   : {entry.get('prompt')!r}")
    print(f"ids      : {prompt_ids}")
    print(f"expected : {expected}")
    print(f"eos      : {eos} ({provenance['eos_source']})")
    print(f"oracle   : {provenance['schema']} dtype={provenance['dtype']}")
    print(f"positions: {total} (rope rows written), "
          f"{len(prompt_ids) - 1} prefill + {max_new} generating")
    return manifest


def rope_matrix_bytes(position: int) -> bytes:
    """RoPE as a 128x128 right-multiply matrix, standard library only.

    The ``--rope matmul`` decode layer replaces the elementwise cos/sin pair
    (inputs 4 and 5) with a single ``[1,1,128,128]`` FP32 matrix at input 4,
    because deleting the BinaryMath class from the RoPE subgraph is what makes
    the layer execute correctly on silicon.  ``x @ R`` equals
    ``x*cos + rotate_half(x)*sin``.

    Verified byte-identical to the numpy staging path
    (scratchpad/minicpm5/ropematmul_full/stage_ropematmul_inputs.py::rope_matrix)
    at positions 1-5, which is what funcsim and the standalone board probe
    consumed.  numpy is unavailable on the board, hence the reimplementation.
    """
    import math  # noqa: PLC0415
    half = HEAD_DIM // 2
    inverse = [1.0 / (ROPE_THETA ** ((2 * i) / HEAD_DIM)) for i in range(half)]
    angle = [position * v for v in inverse] * 2
    cos = [math.cos(a) for a in angle]
    sin = [math.sin(a) for a in angle]
    matrix = [0.0] * (HEAD_DIM * HEAD_DIM)
    for i in range(HEAD_DIM):
        matrix[i * HEAD_DIM + i] = cos[i]
    for i in range(half):
        matrix[(i + half) * HEAD_DIM + i] = -sin[i]
        matrix[i * HEAD_DIM + (i + half)] = sin[i + half]
    return struct.pack(f"<{HEAD_DIM * HEAD_DIM}f", *matrix)


def rope_writes(rope: str, position: int, cos: bytes, sin: bytes
                ) -> tuple[list[tuple[int, int, bytes]], int]:
    """(writes, public_input_count) for the layer's RoPE contract.

    elementwise: input[4]=cos, input[5]=sin, 6 public inputs.
    matmul:      input[4]=R (128x128), 5 public inputs -- inputs 5 and 6 are
                 ATC-synthesised and the executor zeroes everything at or after
                 the public count, which is exactly what the standalone probe
                 does when it is handed five files.
    """
    if rope == "matmul":
        return [(4, 0, rope_matrix_bytes(position))], 5
    return [(4, 0, cos), (5, 0, sin)], 6


def identify_kv_slots(*, process, descriptors, respond, layers: int,
                      rope: str, hidden: bytes, context: int,
                      cache_bytes: int) -> list[tuple[int, int]]:
    """Decide which published output is k_cur and which is v_cur, per layer.

    ATC reorders graph outputs.  The ONNX declares (next_hidden, k_cur, v_cur)
    and every OM in the composed 24-layer build publishes
    (v_cur, k_cur, next_hidden) -- so the obvious ``k_cur, v_cur = respond(...)``
    files V into the K cache and K into the V cache.  The two tensors have the
    same shape and dtype, so the descriptor cannot tell them apart.

    This decides it STRUCTURALLY, with no reference and no hard-coded order:
    RoPE is applied to k_cur and not to v_cur, so running the layer twice with
    identical hidden, caches and mask while changing ONLY the RoPE input
    (position 0 = identity rotation, then position 1) moves exactly one of the
    two outputs.  That one is k_cur.

    Costs 2 executes per layer once at startup.  Raises if both outputs move or
    neither does, because then the assumption behind the whole KV feedback path
    is wrong and guessing would corrupt the run.
    """
    zero_cache = bytes(cache_bytes)
    mask = bytearray(struct.pack("<f", MASK_NEG) * context)
    struct.pack_into("<f", mask, (context - 1) * 4, 0.0)
    slots: list[tuple[int, int]] = []
    for layer in range(layers):
        desc_in, desc_out = descriptors[layer]
        seen = []
        for position in (0, 1):
            cos_row, sin_row = _rope_row(position)
            cos = struct.pack(f"<{HEAD_DIM}f", *cos_row)
            sin = struct.pack(f"<{HEAD_DIM}f", *sin_row)
            rope_extra, public = rope_writes(rope, position, cos, sin)
            writes = [(0, 0, _to_c4(hidden, desc_in[0])),
                      (1, 0, zero_cache), (2, 0, zero_cache),
                      (3, 0, bytes(mask))] + rope_extra
            runner._write_all(process.stdin, _frame(
                layer, min(public, len(desc_in)), tuple(desc_out[:2]),
                writes, []))
            process.stdin.flush()
            seen.append(respond(desc_out[:2]))
        moved = [i for i in range(2) if seen[0][i] != seen[1][i]]
        if len(moved) != 1:
            raise RuntimeError(
                f"layer {layer}: RoPE moved {len(moved)} of 2 KV outputs "
                f"(expected exactly 1 -- k_cur is rotated, v_cur is not); "
                f"cannot identify k_cur from v_cur")
        k_slot = moved[0]
        slots.append((k_slot, 1 - k_slot))
    return slots


def _rope_row(position: int) -> tuple[list[float], list[float]]:
    import math  # noqa: PLC0415
    half = HEAD_DIM // 2
    inverse = [1.0 / (ROPE_THETA ** ((2 * i) / HEAD_DIM)) for i in range(half)]
    angle = [position * v for v in inverse] * 2
    return ([math.cos(a) for a in angle], [math.sin(a) for a in angle])


def _to_c4(payload: bytes, want: int, element: int = 4) -> bytes:
    if want == len(payload):
        return payload
    if want != len(payload) * 4:
        raise ValueError(f"cannot map {len(payload)} onto {want}")
    pad = bytes(element * 3)
    return b"".join(payload[i:i + element] + pad
                    for i in range(0, len(payload), element))


def _frame(model: int, public_inputs: int, out_sizes: Sequence[int],
           writes: Sequence[tuple[int, int, bytes]],
           chains: Sequence[tuple[int, int, int, int, int]]) -> bytes:
    trailer = bytearray(
        b"".join(runner._PERSISTENT_U64.pack(s) for s in out_sizes))
    for slot, offset, data in writes:
        trailer += runner._PERSISTENT_WRITE.pack(
            slot, runner.PERSISTENT_WRITE_FLAG_PAYLOAD, offset, len(data))
        trailer += data
    for slot, offset, length, src, out in chains:
        trailer += runner._PERSISTENT_WRITE.pack(
            slot, runner.PERSISTENT_WRITE_FLAG_CHAIN, offset, length)
        trailer += runner._PERSISTENT_CHAIN.pack(src, out, 0)
    return runner._PERSISTENT_REQUEST.pack(
        runner.PERSISTENT_REQUEST_MAGIC, runner.PERSISTENT_PROTOCOL_VERSION,
        runner.PERSISTENT_OP_EXECUTE_RESIDENT, model, public_inputs,
        len(out_sizes), len(writes) + len(chains)) + bytes(trailer)


def execute(*, executable: Path, models: Sequence[Path],
            library_paths: Sequence[Path], work: Path, timeout: float,
            rope: str = "elementwise", seed_cache: Path | None = None,
            seed_until: int = 0) -> None:
    manifest = json.loads((work / "greedy.json").read_text())
    context = manifest["context"]
    past = context - 1
    prompt_ids = manifest["prompt_token_ids"]
    total = manifest["positions"]
    row_f16 = HEAD_DIM * 2          # one head's row inside the FP16 cache
    cache_bytes = KV_HEADS * past * HEAD_DIM * 2
    cos_all = (work / "rope_cos.bin").read_bytes()
    sin_all = (work / "rope_sin.bin").read_bytes()
    rope_stride = HEAD_DIM * 4
    embed = open(manifest["embedding"], "rb")

    process = probe._start(executable, models, library_paths, 0)
    assert process.stdin is not None and process.stdout is not None
    deadline = time.monotonic() + timeout
    descriptors = probe._read_ready(process.stdout, len(models), deadline)
    read = lambda n: runner._read_exact_until(process.stdout, n, deadline)

    def hidden_output(index: int) -> int:
        for slot, size in enumerate(descriptors[index][1]):
            if size in (HIDDEN * 4, HIDDEN * 16):
                return slot
        raise RuntimeError(f"model {index} publishes no hidden")

    def respond(out_sizes: Sequence[int]) -> list[bytes]:
        (_m, _v, status, _i, out_count, error_bytes, _r) = \
            runner._PERSISTENT_RESPONSE.unpack(
                read(runner._PERSISTENT_RESPONSE.size))
        if status:
            raise RuntimeError(read(error_bytes).decode("utf-8", "replace"))
        for _ in range(out_count):
            read(runner._PERSISTENT_U64.size)
        return [read(size) for size in out_sizes]

    # Which published output is k_cur.  Decided on device, structurally: see
    # identify_kv_slots.  Reading them in descriptor order is WRONG for this
    # build -- every layer publishes (v_cur, k_cur, next_hidden).
    embed.seek(prompt_ids[0] * HIDDEN * 2)
    probe_row = struct.unpack(f"<{HIDDEN}e", embed.read(HIDDEN * 2))
    kv_slots = identify_kv_slots(
        process=process, descriptors=descriptors, respond=respond,
        layers=LAYERS, rope=rope,
        hidden=struct.pack(f"<{HIDDEN}f", *probe_row),
        context=context, cache_bytes=cache_bytes)
    orders = sorted({f"out{k}=k_cur,out{v}=v_cur" for k, v in kv_slots})
    print(f"KV output identification: {orders} "
          f"({len(set(kv_slots))} distinct across {LAYERS} layers)", flush=True)

    # Host-side mirror of every layer's KV window, so only the row that changed
    # has to cross the pipe.
    k_cache = [bytearray(cache_bytes) for _ in range(LAYERS)]
    v_cache = [bytearray(cache_bytes) for _ in range(LAYERS)]
    generated: list[int] = []
    token = prompt_ids[0]
    step_ms: list[float] = []
    eos = set(int(v) for v in manifest.get("eos_token_ids") or FALLBACK_EOS)
    stopped_on_eos = False

    try:
        for position in range(total):
            began = time.perf_counter()
            deadline = time.monotonic() + timeout
            embed.seek(token * HIDDEN * 2)
            row = struct.unpack(f"<{HIDDEN}e", embed.read(HIDDEN * 2))
            hidden = struct.pack(f"<{HIDDEN}f", *row)
            mask = bytearray(struct.pack("<f", MASK_NEG) * context)
            for slot in range(position):
                struct.pack_into("<f", mask, slot * 4, 0.0)
            struct.pack_into("<f", mask, (context - 1) * 4, 0.0)
            cos = cos_all[position * rope_stride:(position + 1) * rope_stride]
            sin = sin_all[position * rope_stride:(position + 1) * rope_stride]

            for layer in range(LAYERS):
                desc_in, desc_out = descriptors[layer]
                writes: list[tuple[int, int, bytes]] = []
                chains: list[tuple[int, int, int, int, int]] = []
                if layer == 0:
                    writes.append((0, 0, _to_c4(hidden, desc_in[0])))
                else:
                    src = hidden_output(layer - 1)
                    chains.append((0, 0, min(desc_in[0],
                                             descriptors[layer - 1][1][src]),
                                   layer - 1, src))
                if position == 0:
                    writes.append((1, 0, bytes(k_cache[layer])))
                    writes.append((2, 0, bytes(v_cache[layer])))
                else:
                    # Only the row written at the previous position moved.
                    prior = position - 1
                    for slot, mirror in ((1, k_cache[layer]),
                                         (2, v_cache[layer])):
                        for head in range(KV_HEADS):
                            start = (head * past + prior) * row_f16
                            writes.append((slot, start,
                                           bytes(mirror[start:start + row_f16])))
                writes.append((3, 0, bytes(mask)))
                rope_extra, public = rope_writes(rope, position, cos, sin)
                writes.extend(rope_extra)
                runner._write_all(process.stdin, _frame(
                    layer, min(public, len(desc_in)), tuple(desc_out[:2]),
                    writes, chains))
                process.stdin.flush()
                published = respond(desc_out[:2])
                k_slot, v_slot = kv_slots[layer]
                k_cur, v_cur = published[k_slot], published[v_slot]
                # FP32 out, FP16 cache: convert and file the row at this
                # position so the next step sees it as past.
                for mirror, blob in ((k_cache[layer], k_cur),
                                     (v_cache[layer], v_cur)):
                    values = struct.unpack(f"<{KV_HEADS * HEAD_DIM}f", blob)
                    for head in range(KV_HEADS):
                        start = (head * past + position) * row_f16
                        mirror[start:start + row_f16] = struct.pack(
                            f"<{HEAD_DIM}e",
                            *values[head * HEAD_DIM:(head + 1) * HEAD_DIM])
                # DIAGNOSTIC ONLY.  Overwrite the row this position just
                # produced with the FP64 reference row, for positions below
                # --seed-cache-until.  Attributes a decode failure to the
                # prefill rows rather than to the generating steps: with
                # --seed-cache-until 1 the run is the board's own everywhere
                # except position 0's cached row.  Never use for acceptance --
                # a seeded run is not a board result.
                if seed_cache is not None and position < seed_until:
                    src = seed_cache / f"pos{position + 1}"
                    for kind, mirror in (("k", k_cache[layer]),
                                         ("v", v_cache[layer])):
                        blob = (src / f"{kind}_cache_{layer:02d}.f16.bin"
                                ).read_bytes()
                        for head in range(KV_HEADS):
                            start = (head * past + position) * row_f16
                            mirror[start:start + row_f16] = \
                                blob[start:start + row_f16]

            head_index = len(models) - 1
            desc_in, desc_out = descriptors[head_index]
            src = hidden_output(LAYERS - 1)
            runner._write_all(process.stdin, _frame(
                head_index, min(3, len(desc_in)), (),
                [], [(0, 0, min(desc_in[0], descriptors[LAYERS - 1][1][src]),
                      LAYERS - 1, src)]))
            process.stdin.flush()
            respond(())
            runner._write_all(process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_ARGMAX, head_index, 0, 0, 0))
            process.stdin.flush()
            (payload,) = respond((8,))
            predicted, value = struct.unpack("<If", payload)
            step_ms.append((time.perf_counter() - began) * 1000)

            if position + 1 < len(prompt_ids):
                token = prompt_ids[position + 1]
                note = f"prompt[{position + 1}]"
            else:
                token = predicted
                generated.append(int(predicted))
                note = "generated"
            print(f"  pos {position:>3}  -> {predicted:>6} "
                  f"(logit {value:>9.4f})  {note}  "
                  f"{step_ms[-1]:.0f} ms", flush=True)
            # eos terminates the sequence.  Without this the loop feeds the eos
            # token back in as context and keeps decoding past the end of the
            # reference continuation, so a run that reproduced the reference
            # exactly would still be scored against tokens the reference never
            # emitted.
            if note == "generated" and int(predicted) in eos:
                stopped_on_eos = True
                print(f"  eos {int(predicted)} at generated index "
                      f"{len(generated) - 1}; stopping", flush=True)
                break
    finally:
        embed.close()
        try:
            runner._write_all(process.stdin, runner._PERSISTENT_REQUEST.pack(
                runner.PERSISTENT_REQUEST_MAGIC,
                runner.PERSISTENT_PROTOCOL_VERSION,
                runner.PERSISTENT_OP_SHUTDOWN, 0, 0, 0, 0))
            process.stdin.flush()
        except Exception:
            pass
        process.terminate()
        process.wait(timeout=30)

    (work / "generated.json").write_text(json.dumps({
        "schema": "pico.minicpm5.greedy_chain.generated.v2",
        "generated_token_ids": generated,
        "stopped_on_eos": stopped_on_eos,
        "eos_token_ids": sorted(eos),
        "step_ms": step_ms,
    }, indent=2) + "\n")
    print(f"\ngenerated: {generated}")
    if stopped_on_eos:
        print("stopped on eos")


def score(*, work: Path, output: Path | None) -> dict[str, Any]:
    manifest = json.loads((work / "greedy.json").read_text())
    produced = json.loads((work / "generated.json").read_text())
    got = produced["generated_token_ids"]
    want = manifest["expected_token_ids"]
    prefix = 0
    for a, b in zip(got, want):
        if a != b:
            break
        prefix += 1
    matches = sum(1 for a, b in zip(got, want) if a == b)
    steps = produced.get("step_ms") or []
    # eos: the run is only equivalent to the reference if it terminated the same
    # way.  A board run that emitted the right 16 tokens but kept going where
    # the reference stopped is not the same sequence.
    want_eos = bool(manifest.get("expected_stopped_on_eos", False))
    got_eos = bool(produced.get("stopped_on_eos", False))
    report = {
        "schema": "pico.minicpm5.greedy_chain.v2",
        "criterion": ("exact_prefix == len(expected_token_ids) and the run "
                      "terminates the same way the reference does"),
        "prompt": manifest.get("prompt"),
        "oracle": manifest.get("oracle"),
        "eos_token_ids": manifest.get("eos_token_ids"),
        "expected_token_ids": want, "board_token_ids": got,
        "expected_text": manifest.get("expected_text"),
        "expected_stopped_on_eos": want_eos,
        "board_stopped_on_eos": got_eos,
        "eos_agreement": want_eos == got_eos,
        "exact_prefix": prefix, "positional_matches": matches,
        "total": len(want),
        "count_agreement": len(got) == len(want),
        "exact": bool(got == want and want_eos == got_eos),
        "median_step_ms": (sorted(steps)[len(steps) // 2] if steps else None),
    }
    print(f"prompt   : {report['prompt']!r}")
    print(f"expected : {want}")
    print(f"board    : {got}")
    print()
    print(f"exact prefix        : {prefix}/{len(want)}   <- the criterion")
    print(f"positional matches  : {matches}/{len(want)}")
    print(f"token count         : board {len(got)} vs expected {len(want)}")
    print(f"eos                 : board {got_eos} vs expected {want_eos}")
    print(f"exact sequence      : {report['exact']}")
    if report["median_step_ms"]:
        print(f"median step         : {report['median_step_ms']:.0f} ms")
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "score"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--golden", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "scratchpad/minicpm5/greedy_oracle/oracle.json",
                        help="FP64 oracle from build_minicpm5_greedy_oracle.py "
                             "(the older FP32 float_greedy_reference is also "
                             "accepted)")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--persistent-executor", type=Path)
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument("--library-path", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--rope", choices=("elementwise", "matmul"),
                        default="elementwise",
                        help="layer RoPE input contract; 'matmul' for builds "
                             "made with --rope matmul, which take a 128x128 "
                             "matrix at input[4] and have no input[5] cos/sin")
    parser.add_argument("--seed-cache", type=Path,
                        help="DIAGNOSTIC: FP64 multipos reference directory "
                             "whose k/v rows replace the board's own for "
                             "positions below --seed-cache-until")
    parser.add_argument("--seed-cache-until", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spec", type=Path,
                        help="ModelSpec JSON (scripts/model_specs/*.json); "
                             "overrides the module geometry/contract constants "
                             "(hidden/kv_heads/head_dim/layers/vocab/rope_theta/"
                             "mask_negative/embedding/bos/eos).  Absent: "
                             "MiniCPM5-1B constants, unchanged.")
    args = parser.parse_args(argv)
    if args.spec is not None:
        _apply_spec(args.spec)
    if args.mode == "prepare":
        prepare(prompt_index=args.prompt_index, golden=args.golden,
                context=args.context, max_new=args.max_new, work=args.work)
        return 0
    if args.mode == "execute":
        execute(executable=args.persistent_executor, models=args.model,
                library_paths=args.library_path, work=args.work,
                timeout=args.timeout, rope=args.rope,
                seed_cache=args.seed_cache, seed_until=args.seed_cache_until)
        return 0
    report = score(work=args.work, output=args.output)
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
