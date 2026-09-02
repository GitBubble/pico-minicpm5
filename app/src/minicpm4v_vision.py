#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-4v-0.5B vision-language inference, ported from the HiSpark sample.

The reference is C++: ``samples/common/infer/preprocess/minicpm_preprocess.cpp``
in HiSpark modelzoo, driven by ``MiniCpmInfer.cpp``. Everything numeric here is
a transcription of it -- the patch reshape, the 200-token prefill template, the
two-pass attention mask, the greedy-longest-match tokenizer and the decode
step's K/V handoff. Where a constant appears below it is the constant the C++
uses, not one chosen here.

Four OM handles run in sequence: ``vision`` turns the image into hidden states,
``resample`` compresses them to 64 vision tokens, ``prefill_decode`` ingests one
200-token window, and ``decode`` runs autoregressively until ``<|im_end|>``. The
executor session is injected, so everything above that boundary is testable
without a board.

The board this targets has numpy and PIL and nothing else -- no OpenCV, no
tokenizers -- and under a gigabyte free against a 300 MB embedding table, so
the tokenizer is the reference's own greedy matcher and table rows are read by
seeking rather than loaded.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

#: Image geometry: one 512x512 image rearranged into 16x16 patches, which is
#: why the vision handle declares [1, 3, 16, 16384].
SOURCE_TGT_SIZE = 512
CHANNEL_NUM = 3
PATCH_SIZE = 16
NORMALIZE_VALUE = 255.0
HALF_VALUE = 0.5

#: Language geometry.
EMB_DIM = 1024
VISION_TOKEN_LEN = 64
TOTAL_PREFILL_LEN = 200
MAX_TEXT_LEN = 118
MAX_TOKEN_LEN = 1024
ROTARY_DIM = 64
MAX_POSITION_IDX = 32768
TOTAL_TOKEN_NUM = 73448
MASK_MIN_VALUE = -9999999.0

#: Special ids, transcribed from the C++ SpecialTokens namespace:
#: <s><|im_start|>_user\n<image_id>_0</image_id> <image>
PRE_TEMPLATE = (1, 73441, 3060, 5, 113, 59320, 59344, 114, 101)
#: </image>_\n
MID_TEMPLATE = (102, 59320, 5)
#: <|im_end|>_\n<|im_start|>_assistant\n
POST_TEMPLATE = (73440, 59320, 5, 73441, 16434, 5)
END_TOKEN = 73440

#: Decode input order, from the C++ input-id constants.
EMBEDDING_INPUT_ID = 0
ATTENTION_MASK_INPUT_ID = 1
LOOP_IDX_INPUT_ID = 2
ROTARY_POSITION_0_INPUT_ID = 3
ROTARY_POSITION_1_INPUT_ID = 4
KV_START_INPUT_ID = 5


class VisionError(RuntimeError):
    """The vision pipeline cannot proceed with the inputs it was given."""


def reshape_by_patch(chw: np.ndarray, patch: int = PATCH_SIZE) -> np.ndarray:
    """Rearrange a CHW image into the layout the vision handle expects.

    The reference writes each pixel to
    ``c*patch*outWidth + ph*outWidth + (i*patchNumW + j)*patch + pw``
    while reading ``c*H*W + (i*patch + ph)*W + (j*patch + pw)``. Destination
    order therefore runs the intra-patch row ``ph`` outside the patch grid,
    which is a transpose of the two middle axes.
    """
    channels, height, width = chw.shape
    if height % patch or width % patch:
        raise VisionError(f"image {height}x{width} is not a multiple of {patch}")
    patch_h, patch_w = height // patch, width // patch
    grid = chw.reshape(channels, patch_h, patch, patch_w, patch)
    out = grid.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(
        out.reshape(channels, patch, patch_h * patch_w * patch),
        dtype=np.float32)


def preprocess_image(path) -> np.ndarray:
    """Load an image and produce the vision handle's [1, 3, 16, 16384] input.

    PIL stands in for the reference's OpenCV: it decodes to RGB directly, so
    the C++ BGR-to-RGB step has no counterpart, and ``Image.BICUBIC`` is the
    counterpart of ``INTER_CUBIC``. Both are cubic convolution resamplers but
    they do not agree bit for bit; that is a difference in the resampled
    input, not in this contract.
    """
    from PIL import Image

    source = Path(path)
    if not source.is_file():
        raise VisionError(f"image not found: {source}")
    try:
        with Image.open(source) as handle:
            image = handle.convert("RGB")
            if image.size != (SOURCE_TGT_SIZE, SOURCE_TGT_SIZE):
                image = image.resize(
                    (SOURCE_TGT_SIZE, SOURCE_TGT_SIZE), Image.BICUBIC)
            pixels = np.asarray(image, dtype=np.float32)
    except OSError as error:
        raise VisionError(f"image cannot be read: {source}") from error

    normalized = (pixels / NORMALIZE_VALUE - HALF_VALUE) / HALF_VALUE
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return reshape_by_patch(chw)[np.newaxis, ...]


class VocabTable:
    """The reference tokenizer: greedy longest match over a flat vocabulary.

    Deliberately not BPE. The C++ walks the string taking the longest prefix
    present in the vocabulary and falls back to a single character, so a
    byte-level merge table would produce different ids. Reproducing it also
    keeps the worker free of a library the board does not have.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        if not vocab:
            raise VisionError("vocabulary is empty")
        self.vocab = dict(vocab)
        self.max_token_len = max(len(token) for token in self.vocab)
        self.id_to_token = {index: token for token, index in self.vocab.items()}

    @classmethod
    def from_tokenizer_json(cls, path) -> VocabTable:
        source = Path(path)
        if not source.is_file():
            raise VisionError(f"tokenizer not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        vocab = payload.get("model", {}).get("vocab")
        if not isinstance(vocab, dict):
            raise VisionError("tokenizer.json has no model.vocab object")
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            limit = min(self.max_token_len, len(text) - cursor)
            for length in range(limit, 0, -1):
                candidate = text[cursor:cursor + length]
                if candidate in self.vocab:
                    ids.append(self.vocab[candidate])
                    cursor += length
                    break
            else:
                # No candidate matched: the reference emits the unknown id.
                ids.append(0)
                cursor += 1
        return ids

    def decode(self, ids) -> str:
        pieces = [self.id_to_token.get(int(index), "") for index in ids]
        return "".join(pieces).replace("▁", " ")


class RowTable:
    """Seek-based row lookup over a flat float32 table.

    The embedding table is 300 MB against under a gigabyte of free board
    memory, so rows are read on demand exactly as the reference does rather
    than materialised.
    """

    def __init__(self, path, dim: int) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise VisionError(f"table not found: {self.path}")
        self.dim = int(dim)
        self.row_bytes = self.dim * 4
        size = self.path.stat().st_size
        if size % self.row_bytes:
            raise VisionError(
                f"{self.path.name} is not a whole number of "
                f"{self.dim}-wide float32 rows")
        self.rows = size // self.row_bytes
        # Held open on purpose: this table is 300 MB and is read one row
        # at a time by seek. close() and __exit__ own its lifetime.
        self._handle = open(self.path, "rb")  # noqa: SIM115

    def row(self, index: int) -> np.ndarray:
        position = int(index)
        if not 0 <= position < self.rows:
            raise VisionError(
                f"row {position} is outside {self.path.name} [0, {self.rows})")
        self._handle.seek(position * self.row_bytes)
        raw = self._handle.read(self.row_bytes)
        if len(raw) != self.row_bytes:
            raise VisionError(f"short read at row {position} of {self.path.name}")
        return np.frombuffer(raw, dtype=np.float32).copy()

    def gather(self, indices) -> np.ndarray:
        rows = [self.row(index) for index in indices]
        return np.stack(rows) if rows else np.zeros((0, self.dim), np.float32)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> RowTable:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_attention_mask(prefill_len: int,
                         fixed: int = TOTAL_PREFILL_LEN) -> np.ndarray:
    """The prefill mask: causal, then the padding region masked a second time.

    The reference writes the causal triangle, then walks rows at and beyond
    ``prefill_len`` masking columns ``[prefill_len, i]``. That second pass is
    what stops a padded row attending to other padded rows; without it the
    padding attends to itself and the window is not equivalent to a shorter
    one.
    """
    if not 0 < prefill_len <= fixed:
        raise VisionError(f"prefill length {prefill_len} outside (0, {fixed}]")
    rows = np.arange(fixed).reshape(-1, 1)
    cols = np.arange(fixed).reshape(1, -1)
    mask = np.where(cols <= rows, 0.0, MASK_MIN_VALUE).astype(np.float32)
    padding = (rows >= prefill_len) & (cols >= prefill_len) & (cols <= rows)
    mask[padding] = MASK_MIN_VALUE
    return mask.reshape(1, 1, fixed, fixed)


@dataclass(frozen=True)
class PrefillInputs:
    """One assembled prefill window and the length that is real."""

    inputs_embeds: np.ndarray
    attention_mask: np.ndarray
    prefill_len: int


def build_prefill_inputs(vision_tokens: np.ndarray, text: str,
                         vocab: VocabTable, embeddings: RowTable,
                         continuation: Sequence[int] = ()) -> PrefillInputs:
    """Assemble the 200-token window: template, image, template, text, template.

    The layout is fixed by the reference: nine ids opening the user turn and
    the image, sixty-four vision tokens copied verbatim from the resampler,
    three ids closing the image, the question truncated to 118 ids, and six
    ids that close the turn and open the assistant's.

    ``continuation`` appends tokens the model has already produced, which is
    how this deployment generates at all: ``decode.om`` declares more ports
    than the SDK will load, so each new token is taken from the logits of a
    fresh prefill over the prompt plus everything said so far. The window is
    200 rows and a question is short, so there is normally room for a hundred
    tokens of answer; a caller that fills it is told rather than truncated.
    """
    if vision_tokens.size != VISION_TOKEN_LEN * EMB_DIM:
        raise VisionError(
            f"resample output must hold {VISION_TOKEN_LEN}x{EMB_DIM} floats, "
            f"got {vision_tokens.size}")
    text_ids = vocab.encode(text)[:MAX_TEXT_LEN]
    window = np.zeros((TOTAL_PREFILL_LEN, EMB_DIM), dtype=np.float32)

    cursor = len(PRE_TEMPLATE)
    window[:cursor] = embeddings.gather(PRE_TEMPLATE)
    window[cursor:cursor + VISION_TOKEN_LEN] = vision_tokens.reshape(
        VISION_TOKEN_LEN, EMB_DIM)
    cursor += VISION_TOKEN_LEN
    for group in (MID_TEMPLATE, tuple(text_ids), POST_TEMPLATE,
                  tuple(int(token) for token in continuation)):
        if not group:
            continue
        if cursor + len(group) > TOTAL_PREFILL_LEN:
            raise VisionError(
                f"prefill window holds {TOTAL_PREFILL_LEN} rows; "
                f"{cursor + len(group)} were assembled")
        window[cursor:cursor + len(group)] = embeddings.gather(group)
        cursor += len(group)

    return PrefillInputs(
        inputs_embeds=window.reshape(1, TOTAL_PREFILL_LEN, EMB_DIM),
        attention_mask=build_attention_mask(cursor),
        prefill_len=cursor)


def decode_mask(loop_id: int, width: int = MAX_TOKEN_LEN) -> np.ndarray:
    """First-step decode mask: visible through ``loop_id``, masked beyond."""
    if not 0 <= loop_id < width:
        raise VisionError(f"loop id {loop_id} outside [0, {width})")
    mask = np.full(width, MASK_MIN_VALUE, dtype=np.float32)
    mask[:loop_id + 1] = 0.0
    return mask.reshape(1, 1, 1, width)


def prefill_kv_slice(prefill_kv: np.ndarray, loop_id: int) -> np.ndarray:
    """Take the first ``loop_id`` rows of each layer out of the prefill K/V.

    The prefill handle emits a fixed 200-row window per layer; only the rows
    the prompt actually filled carry into the decode cache. That is the
    reference's ``prefillKeyValueSize`` against ``fixPrefillKeyValueSize``
    distinction, and copying the whole window instead would let the decode
    attend to uninitialised rows.
    """
    if prefill_kv.ndim < 2:
        raise VisionError("prefill K/V must be at least [layers, sequence]")
    if not 0 < loop_id <= prefill_kv.shape[1]:
        raise VisionError(
            f"loop id {loop_id} outside (0, {prefill_kv.shape[1]}]")
    return np.ascontiguousarray(prefill_kv[:, :loop_id])
