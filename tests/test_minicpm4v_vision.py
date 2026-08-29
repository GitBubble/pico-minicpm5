from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "app" / "src" / "minicpm4v_vision.py"


def _vision():
    spec = importlib.util.spec_from_file_location("minicpm4v_vision_test",
                                                  MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_reshape(chw, patch):
    """The C++ ReshapeByPatch, transcribed as literal loops."""
    channels, height, width = chw.shape
    patch_w, patch_h = width // patch, height // patch
    out_width = patch_h * patch_w * patch
    out = np.zeros((channels * patch * out_width,), dtype=np.float32)
    flat = chw.reshape(-1)
    for c in range(channels):
        for ph in range(patch):
            for pw in range(patch):
                for i in range(patch_h):
                    for j in range(patch_w):
                        src = c * height * width + (i * patch + ph) * width \
                            + (j * patch + pw)
                        dst = (c * patch * out_width) + ph * out_width \
                            + (i * patch_w + j) * patch + pw
                        out[dst] = flat[src]
    return out.reshape(channels, patch, out_width)


def _reference_mask(prefill_len, fixed, mask_min):
    """The C++ SetAttentionMask, transcribed as literal loops."""
    mask = np.zeros((fixed, fixed), dtype=np.float32)
    for i in range(fixed):
        for j in range(i + 1):
            mask[i][j] = 0.0
        for j in range(i + 1, fixed):
            mask[i][j] = mask_min
    for i in range(prefill_len, fixed):
        for j in range(prefill_len, i + 1):
            mask[i][j] = mask_min
    return mask


def test_patch_reshape_matches_the_reference_index_arithmetic() -> None:
    """Every element must land where the C++ index expression puts it."""
    vision = _vision()
    # Distinguishable values so any permutation error is visible.
    chw = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)

    ported = vision.reshape_by_patch(chw, patch=4)
    expected = _reference_reshape(chw, 4)

    assert ported.shape == expected.shape
    assert np.array_equal(ported, expected)


def test_patch_reshape_at_the_shipped_geometry() -> None:
    vision = _vision()
    chw = np.random.default_rng(20260829).standard_normal(
        (vision.CHANNEL_NUM, vision.SOURCE_TGT_SIZE, vision.SOURCE_TGT_SIZE)
    ).astype(np.float32)

    ported = vision.reshape_by_patch(chw)

    # [3, 16, 16384] is what the vision handle declares.
    assert ported.shape == (3, 16, 16384)
    assert np.array_equal(ported, _reference_reshape(chw, vision.PATCH_SIZE))


def test_reshape_rejects_a_geometry_the_patch_grid_cannot_tile() -> None:
    vision = _vision()
    with pytest.raises(vision.VisionError, match="multiple of"):
        vision.reshape_by_patch(np.zeros((3, 17, 16), dtype=np.float32))


@pytest.mark.parametrize("prefill_len,fixed", [(7, 12), (1, 4), (12, 12)])
def test_attention_mask_reproduces_both_reference_passes(
        prefill_len: int, fixed: int) -> None:
    """The padding region is masked twice; missing that pass is invisible
    until a short prompt attends to its own padding."""
    vision = _vision()

    ported = vision.build_attention_mask(prefill_len, fixed)[0, 0]
    expected = _reference_mask(prefill_len, fixed, vision.MASK_MIN_VALUE)

    assert np.array_equal(ported, expected)


def test_attention_mask_at_the_shipped_window() -> None:
    vision = _vision()
    ported = vision.build_attention_mask(118)[0, 0]
    expected = _reference_mask(118, vision.TOTAL_PREFILL_LEN,
                               vision.MASK_MIN_VALUE)
    assert ported.shape == (200, 200)
    assert np.array_equal(ported, expected)
    # A padded row sees nothing at all, including itself.
    assert float(ported[150, 150]) == vision.MASK_MIN_VALUE


def test_attention_mask_fails_closed_on_an_impossible_length() -> None:
    vision = _vision()
    for bad in (0, -1, 201):
        with pytest.raises(vision.VisionError, match="prefill length"):
            vision.build_attention_mask(bad)


def _fixture_vocab():
    return {"hello": 10, "he": 11, "l": 12, "o": 13, "▁world": 14, " ": 15}


def test_tokenizer_takes_the_longest_match_not_the_first() -> None:
    vision = _vision()
    table = vision.VocabTable(_fixture_vocab())

    # "hello" is present, so the greedy matcher must not stop at "he".
    assert table.encode("hello") == [10]
    assert table.encode("hell") == [11, 12, 12]


def test_tokenizer_falls_back_to_the_unknown_id_one_character_at_a_time() -> None:
    vision = _vision()
    table = vision.VocabTable(_fixture_vocab())

    assert table.encode("hezo") == [11, 0, 13]


def test_tokenizer_decode_restores_the_underscore_convention() -> None:
    vision = _vision()
    table = vision.VocabTable(_fixture_vocab())

    assert table.decode([10, 14]) == "hello world"


def test_an_empty_vocabulary_fails_closed() -> None:
    vision = _vision()
    with pytest.raises(vision.VisionError, match="empty"):
        vision.VocabTable({})


def _row_table(tmp_path, name, rows, dim, module):
    data = np.arange(rows * dim, dtype=np.float32).reshape(rows, dim)
    path = tmp_path / name
    path.write_bytes(data.tobytes())
    return module.RowTable(path, dim), data


def test_row_table_seeks_rather_than_loading(tmp_path) -> None:
    vision = _vision()
    table, data = _row_table(tmp_path, "emb.bin", 64, 8, vision)

    assert table.rows == 64
    assert np.array_equal(table.row(0), data[0])
    assert np.array_equal(table.row(63), data[63])
    assert np.array_equal(table.gather([3, 1, 3]), data[[3, 1, 3]])
    table.close()


def test_row_table_refuses_an_index_outside_the_table(tmp_path) -> None:
    vision = _vision()
    table, _ = _row_table(tmp_path, "emb.bin", 4, 8, vision)

    with pytest.raises(vision.VisionError, match="outside"):
        table.row(4)
    with pytest.raises(vision.VisionError, match="outside"):
        table.row(-1)
    table.close()


def test_row_table_refuses_a_table_that_is_not_whole_rows(tmp_path) -> None:
    vision = _vision()
    path = tmp_path / "ragged.bin"
    path.write_bytes(b"\x00" * (8 * 4 + 3))

    with pytest.raises(vision.VisionError, match="whole number"):
        vision.RowTable(path, 8)


def test_row_table_refuses_a_missing_file(tmp_path) -> None:
    vision = _vision()
    with pytest.raises(vision.VisionError, match="not found"):
        vision.RowTable(tmp_path / "absent.bin", 8)


def test_prefill_window_lays_out_template_image_text_template(tmp_path) -> None:
    """9 + 64 + 3 + len(text) + 6, with the vision tokens copied verbatim."""
    vision = _vision()
    embeddings, data = _row_table(
        tmp_path, "emb.bin", vision.TOTAL_TOKEN_NUM, vision.EMB_DIM, vision)
    vocab = vision.VocabTable({"a": 5, "b": 6})
    tokens = np.full((vision.VISION_TOKEN_LEN, vision.EMB_DIM), 7.0,
                     dtype=np.float32)

    built = vision.build_prefill_inputs(tokens, "ab", vocab, embeddings)

    assert built.prefill_len == 9 + 64 + 3 + 2 + 6
    window = built.inputs_embeds[0]
    assert np.array_equal(window[:9], data[list(vision.PRE_TEMPLATE)])
    assert np.array_equal(window[9:73], tokens)
    assert np.array_equal(window[73:76], data[list(vision.MID_TEMPLATE)])
    assert np.array_equal(window[76:78], data[[5, 6]])
    assert np.array_equal(window[78:84], data[list(vision.POST_TEMPLATE)])
    # Everything past the prompt stays zero.
    assert not window[built.prefill_len:].any()
    embeddings.close()


def test_prefill_truncates_the_question_at_the_template_budget(tmp_path) -> None:
    """200 - 9 - 64 - 3 - 6 leaves 118 for the question."""
    vision = _vision()
    embeddings, _ = _row_table(
        tmp_path, "emb.bin", vision.TOTAL_TOKEN_NUM, vision.EMB_DIM, vision)
    vocab = vision.VocabTable({"a": 5})

    built = vision.build_prefill_inputs(
        np.zeros((vision.VISION_TOKEN_LEN, vision.EMB_DIM), np.float32),
        "a" * 400, vocab, embeddings)

    assert built.prefill_len == vision.TOTAL_PREFILL_LEN
    assert built.attention_mask.shape == (1, 1, 200, 200)
    embeddings.close()


def test_a_continuation_is_appended_after_the_closing_template(
        tmp_path) -> None:
    """Generation here is re-prefill, so the answer so far rides in the window."""
    vision = _vision()
    embeddings, data = _row_table(
        tmp_path, "emb.bin", vision.TOTAL_TOKEN_NUM, vision.EMB_DIM, vision)
    vocab = vision.VocabTable({"a": 5, "b": 6})
    tokens = np.zeros((vision.VISION_TOKEN_LEN, vision.EMB_DIM), np.float32)

    built = vision.build_prefill_inputs(tokens, "ab", vocab, embeddings,
                                        [11, 12, 13])

    assert built.prefill_len == 9 + 64 + 3 + 2 + 6 + 3
    window = built.inputs_embeds[0]
    assert np.array_equal(window[84:87], data[[11, 12, 13]])
    assert not window[built.prefill_len:].any()
    # The mask has to grow with the continuation or the newest token is
    # generated from a position the model cannot attend to.
    assert built.attention_mask.shape == (1, 1, 200, 200)
    embeddings.close()


def test_an_empty_continuation_is_the_plain_prompt(tmp_path) -> None:
    vision = _vision()
    embeddings, _ = _row_table(
        tmp_path, "emb.bin", vision.TOTAL_TOKEN_NUM, vision.EMB_DIM, vision)
    vocab = vision.VocabTable({"a": 5})
    tokens = np.zeros((vision.VISION_TOKEN_LEN, vision.EMB_DIM), np.float32)

    plain = vision.build_prefill_inputs(tokens, "a", vocab, embeddings)
    empty = vision.build_prefill_inputs(tokens, "a", vocab, embeddings, [])

    assert plain.prefill_len == empty.prefill_len
    assert np.array_equal(plain.inputs_embeds, empty.inputs_embeds)
    embeddings.close()


def test_a_continuation_that_overflows_the_window_is_refused(tmp_path) -> None:
    """Silently truncating would drop the newest token and loop forever."""
    vision = _vision()
    embeddings, _ = _row_table(
        tmp_path, "emb.bin", vision.TOTAL_TOKEN_NUM, vision.EMB_DIM, vision)
    vocab = vision.VocabTable({"a": 5})
    tokens = np.zeros((vision.VISION_TOKEN_LEN, vision.EMB_DIM), np.float32)

    with pytest.raises(vision.VisionError, match="200 rows"):
        vision.build_prefill_inputs(tokens, "a", vocab, embeddings,
                                    list(range(200)))
    embeddings.close()


def test_prefill_refuses_a_resample_output_of_the_wrong_size(tmp_path) -> None:
    vision = _vision()
    embeddings, _ = _row_table(
        tmp_path, "emb.bin", vision.TOTAL_TOKEN_NUM, vision.EMB_DIM, vision)
    vocab = vision.VocabTable({"a": 5})

    with pytest.raises(vision.VisionError, match="resample output"):
        vision.build_prefill_inputs(
            np.zeros((32, vision.EMB_DIM), np.float32), "a", vocab, embeddings)
    embeddings.close()


def test_decode_mask_opens_exactly_the_positions_already_written() -> None:
    vision = _vision()
    mask = vision.decode_mask(3, width=8)[0, 0, 0]

    assert list(mask[:4]) == [0.0, 0.0, 0.0, 0.0]
    assert set(mask[4:]) == {vision.MASK_MIN_VALUE}


def test_decode_mask_fails_closed_outside_the_window() -> None:
    vision = _vision()
    with pytest.raises(vision.VisionError, match="loop id"):
        vision.decode_mask(8, width=8)


def test_first_decode_step_carries_only_the_rows_the_prompt_filled() -> None:
    """Copying the whole 200-row window would let decode attend to
    uninitialised rows."""
    vision = _vision()
    layers, sequence, dim = 4, 200, 6
    prefill_kv = np.arange(layers * sequence * dim, dtype=np.float32).reshape(
        layers, sequence, dim)

    carried = vision.prefill_kv_slice(prefill_kv, loop_id=84)

    assert carried.shape == (layers, 84, dim)
    assert np.array_equal(carried, prefill_kv[:, :84])


def test_prefill_kv_slice_fails_closed_past_the_window() -> None:
    vision = _vision()
    prefill_kv = np.zeros((4, 200, 6), dtype=np.float32)

    with pytest.raises(vision.VisionError, match="loop id"):
        vision.prefill_kv_slice(prefill_kv, loop_id=201)
    with pytest.raises(vision.VisionError, match="loop id"):
        vision.prefill_kv_slice(prefill_kv, loop_id=0)


def test_image_preprocess_normalises_and_reshapes(tmp_path) -> None:
    vision = _vision()
    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "solid.png"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(path)

    staged = vision.preprocess_image(path)

    assert staged.shape == (1, 3, 16, 16384)
    # (255/255 - 0.5)/0.5 = 1 on the red plane, -1 on the other two.
    assert np.allclose(staged[0, 0], 1.0)
    assert np.allclose(staged[0, 1], -1.0)
    assert np.allclose(staged[0, 2], -1.0)


def test_image_preprocess_fails_closed_on_a_missing_or_broken_file(
        tmp_path) -> None:
    vision = _vision()
    with pytest.raises(vision.VisionError, match="not found"):
        vision.preprocess_image(tmp_path / "absent.png")

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(vision.VisionError, match="cannot be read"):
        vision.preprocess_image(broken)


def test_tokenizer_json_loading_is_fail_closed(tmp_path) -> None:
    vision = _vision()
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"model": {"type": "BPE"}}), encoding="utf-8")

    with pytest.raises(vision.VisionError, match="model.vocab"):
        vision.VocabTable.from_tokenizer_json(path)

    path.write_text(json.dumps({"model": {"vocab": {"a": 1}}}),
                    encoding="utf-8")
    assert vision.VocabTable.from_tokenizer_json(path).encode("a") == [1]
