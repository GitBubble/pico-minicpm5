"""Invariants of the family clip presets, as documented in
docs/QUANTIZATION_CONTRACT.md.

These are preset-level invariants only. Whether ATC adopts a Clip bound or the
smaller of bound and observed range is an open question (see section 5 of that
document), and nothing here can pin ATC's behaviour. What these tests do
guarantee is that a preset regeneration cannot silently move a prefill bound
below the anchor that currently declares the range, cannot hand decode a
position-zero range, and cannot smuggle an anchor-valued bound onto a tensor
where the layer.py guard would not drop it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PRESETS = PROJECT / "configs" / "calibration"
ANCHOR = math.sqrt(1536)
ANCHOR_TENSORS = ("normed", "post_normed")


def _preset(family: str) -> dict:
    return json.loads(
        (PRESETS / f"{family}-clips.json").read_text(encoding="utf-8"))


def _effective(family: str) -> dict[tuple[str, str], float]:
    """Bounds that actually become a Clip node.

    Mirrors the guard in onnx/layer.py: a preset bound exactly equal to
    sqrt(H) on an anchored tensor is dropped, because the ExtendRMSNorm
    anchor Clip already declares it.
    """
    effective: dict[tuple[str, str], float] = {}
    for layer, block in _preset(family)["layers"].items():
        for name, bound in block["bounds"].items():
            if name in ANCHOR_TENSORS and abs(bound - ANCHOR) < 1e-5:
                continue
            effective[(layer, name)] = float(bound)
    return effective


def test_presets_declare_their_family_and_provenance() -> None:
    for family, source in (("prefill", "qualified-pos0"),
                           ("decode", "qualified-multiposition")):
        preset = _preset(family)
        assert preset["schema"] == "pico.minicpm5.clip-preset.v1"
        assert preset["family"] == family
        assert preset["source"] == source
        assert len(preset["layers"]) == 24


def test_prefill_is_never_tighter_than_decode() -> None:
    """The position-zero family carries the larger activations, so a bound
    that is tighter in prefill than in decode would mean the preset was
    derived from the wrong reference positions."""
    prefill, decode = _effective("prefill"), _effective("decode")
    shared = set(prefill) & set(decode)
    tighter = {key: (prefill[key], decode[key])
               for key in shared if prefill[key] < decode[key]}
    assert not tighter, f"prefill must never be tighter than decode: {tighter}"
    differing = [key for key in shared if prefill[key] != decode[key]]
    assert len(differing) == 64, (
        "the differing-bound count moved; docs/QUANTIZATION_CONTRACT.md "
        "section 4 quotes 64 and attributes the OM's param-head delta to them")


def test_prefill_only_clips_are_structurally_redundant() -> None:
    """The prefill-only bounds sit on anchored tensors at or above sqrt(H),
    where the ExtendRMSNorm anchor already declares the range.

    A regeneration that emits one BELOW the anchor makes it the binding
    declaration on that tensor and must be re-qualified, so that check runs
    first: a bare count mismatch must not mask a liveness regression.
    """
    prefill, decode = _effective("prefill"), _effective("decode")
    only = {key: bound for key, bound in prefill.items() if key not in decode}
    off_anchor = {key: bound for key, bound in only.items()
                  if key[1] not in ANCHOR_TENSORS}
    assert not off_anchor, (
        f"prefill-only clip on a non-anchored tensor: {off_anchor}")
    live = {key: bound for key, bound in only.items() if bound < ANCHOR}
    assert not live, (
        f"prefill-only clip below the anchor {ANCHOR:.6f} is live and binds "
        f"the declared range: {live}")
    assert len(only) == 33, (
        "prefill-only clip count moved; the count itself is not a contract, "
        "but docs/QUANTIZATION_CONTRACT.md section 5 quotes it and its cost "
        "accounting (851 layer records, 2529 bytes) must be re-measured")


def test_non_anchored_tensors_never_carry_the_anchor_bound() -> None:
    """onnx/layer.py drops an anchor-valued bound only on normed and
    post_normed. On any other tensor such a bound becomes a real Clip node,
    which is almost always a preset-derivation mistake rather than intent."""
    for family in ("prefill", "decode"):
        for (layer, name), bound in _effective(family).items():
            if name in ANCHOR_TENSORS:
                continue
            assert abs(bound - ANCHOR) > 1e-5, (
                f"{family} L{layer} {name} carries the anchor bound but is "
                "not an anchored tensor, so it survives the layer.py guard")
