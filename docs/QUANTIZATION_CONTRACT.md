# Quantization contract

[中文](QUANTIZATION_CONTRACT.zh-CN.md)

This document records how the int8/int16 quantization of the three OM handles
is decided: who runs the calibration, what the in-graph `Clip` nodes do to it,
and why the pipeline builds two calibration families from one graph.

Provenance: code facts cite files in this repository. Activation measurements
are recomputed from the float reference dump (`reference capture`, six
positions x 24 layers) and the ATC records quoted in section 2 come from an ATC
work directory in the integration monorepo — neither is redistributed here, so
those numbers cannot be re-derived from a release checkout alone.

## 1. Who quantizes

ATC does, by itself. `src/pico_minicpm5/compiler/atc.py` is the only backend
that produces a real `.om` (a `FakeCompiler` stub exists for CI and is
selectable with `--backend fake`), and its `compile()` performs exactly one
`subprocess.run` on the ATC binary. The pipeline hands ATC a float ONNX graph
plus one `--image_list` calibration corpus per input; ATC observes the graph
over that corpus, runs its IFMR range search internally, and bakes scale/offset
into the `.om`. The `calibration_param.txt` that appears next to each `.om` is
an ATC **output**, not a developer-authored input — the vendor's
`atc_param_conf.json` exposes no key that could consume a file of that name.

Consequences worth stating plainly:

- PICO ATC is IFMR-only. It cannot ingest QDQ, AMCT or QAT scales — there is no
  `AscendQuant` / `QuantizeLinear` in its op table.
- ATC itself does support one external-scale channel, `--gfpq_param_file`, but
  `AtcCompiler.command()` builds a fixed argv with no hook for it. Supplying
  external scales would require a code change here, not a new tool.
- The datapath mode is selected by `--compile_mode`, not by a dedicated flag.
  The vendor enum names the two values `Low-bandwidth` (0) and `High-precision`
  (1); this project calls them A8W8 and A16W8, meaning 8- or 16-bit activations
  against 8-bit weights. Artifact evidence for the naming is indirect: a
  `compile_mode=1` build records `calc_data_type: S16` with `weight { S8 }`,
  and a `compile_mode=0` build of a *different* model records `S8`/`S8`. No
  controlled A/B on one graph exists.
- `atc.py` takes `compile_mode: int = 1` as a default constructor argument and
  no CLI flag exposes it, so every OM this pipeline builds is compiled in the
  A16W8 mode. The release repository ships no `.om` and no build log, so that
  is a property of the code path, not something a release checkout can verify.

## 2. What a Clip does to the quantizer

A preset `Clip` is not metadata and not a hint. `_insert_clip`
(`src/pico_minicpm5/onnx/layer.py`) splices a real ONNX `Clip` node and rewires
**every** consumer of the tensor to read the clipped value, so the clamp is
unconditional at inference. Two properties follow directly, and one widely
repeated corollary does not.

**Evidenced: ATC adopts the Clip's constant bound as the quantizer range of
its output.** In an ATC record for layer 0, the `in_rmsnorm_clip` bound of
`+-sqrt(1536)` produces a recorded `q_proj` input quant of scale `52.2430191`,
offset `0` — and `52.2430191` is the bound's step reciprocal computed in
float32:

```
step  = float32(2 * float32(sqrt(1536))) / 4095   # 4095 levels, [-2048, 2047]
1/step = 52.243019104   ->  printed 52.2430191
```

Note the arithmetic is a reciprocal-of-step in float32, not a plain division:
`4095 / (2 x 39.191837)` equals `52.2430219`, one ULP away from the recorded
value. The same correction applies to the attention mask, whose recorded
`63.9843712` is `float32(1/(64/4095))`, not `4095/64 = 63.984375`.

**Evidenced: a symmetric Clip yields dequant offset exactly 0.** Confirmed on
`q_proj`, `k_proj`, `v_proj`, `gate_proj` and `up_proj` of the same record.

**Not established: that a Clip can only tighten.** An upstream note states the
rule as `min(inferred_from_image_list, clip_bound)`. The only artifact ever
offered as its proof contradicts it: the observed worst element on that tensor
is `37.8636`, so `min()` would have selected `37.86` and a scale of `54.10`;
the record instead carries the clip bound's `52.2430191`. What the artifact
shows is that the *bound* won, not that the *smaller of the two* won. Whether a
bound looser than the data widens the emitted range is untested, and section 5
turns on that open question.

What is certain either way is that the clamp is a real runtime op, so
calibration and inference see the same distribution: a value that would have
overflowed is clamped in both passes rather than silently wrapped in one.

## 3. Two tiers of Clip

**Provable anchor.** After each `ExtendRMSNorm`, the graph clips to
`+-sqrt(H)`. `sqrt(1536) = 39.191835884530846`, stored in the graph as the
float32 `39.191837310791016`. This is a mathematical bound, not an empirical
one: RMSNorm gives unit RMS per row, so `sum(x^2) = H` and no single element
can exceed `sqrt(H)`. It is emitted unconditionally for both families. The
largest element measured over all 24 layers and six reference positions is
`37.8636` (layer 19, position 0), so the anchor never actually clamps anything
— it exists to *declare* the range to ATC.

**Empirical family preset.** `configs/calibration/{prefill,decode}-clips.json`
carry per-layer, per-tensor bounds derived from measured activations. This tier
is the only thing the `family` parameter changes in the graph.

A preset bound within `1e-5` of `sqrt(H)` is dropped — but only on `normed` and
`post_normed` (`onnx/layer.py`). The tolerance matters: the presets store the
float32 value, which sits `1.4e-6` from `sqrt(1536)`, so an exact-equality test
would drop nothing. An anchor-valued bound on any other tensor still becomes a
real Clip node.

## 4. Why two families exist

The two OMs are the same graph. `family` reaches it through exactly one door —
which clip preset is loaded — and reaches calibration through exactly one line:
prefill samples reference position 0 only, decode samples positions `>= 1`
only. The shipped ctx1024 binaries differ by 2,529 bytes out of 687 MB.

They cannot be merged, because position 0's activations are not slightly
different but *orders of magnitude* larger. Recomputed from the float
reference, layer 0 (absolute maxima; ratios computed before rounding):

| L0 tensor | position 0 | position 1 | ratio |
|---|---:|---:|---:|
| hidden (input) | 0.1025 | 0.1191 | 0.86 |
| attn_residual | 1.1073 | 1.1462 | 0.97 |
| swiglu | 28.54 | 0.6262 | **45.6x** |
| down_out | 84.93 | 2.878 | **29.5x** |
| next_hidden | 86.04 | 4.024 | 21.4x |

The preset bounds record the same story: of the 64 shared-name bounds that
differ between families, prefill is looser on all 64 and tighter on none, with
extremes at L4 `down_out` (`4000.0` vs `7.5`) and L5/L6/L8 `attn_residual`
(`4000.0` vs `18.0`). Every one of those raised bounds sits on a
residual-carrier or MLP-branch tensor; none is `v_cur`, `context_grouped` or
`o_out`. A merged calibration would hand decode a range two orders of magnitude
too wide and destroy its resolution. Omitting the prefill family altogether
reproduces the historical whole-model failure byte for byte.

### The mechanism is the MLP, not attention

It is tempting to explain this with attention: at position 0 the KV cache is
empty, so softmax attends only to the current token and cannot average. The
first half is exactly true, and more strongly than needed —
`context_grouped == broadcast(v_cur)` with max absolute difference `0.0` at
**every** one of the 24 layers. **The explanation is still wrong**, because
`v_cur` is itself smaller at position 0 (pos0/pos1 absmax `0.089` at L1,
`0.238` at L4, `0.318` at L7). The two effects cancel: over all 24 layers the
attention output ratio pos0/pos1 spans `0.28`-`1.27` for `context_grouped` and
`0.47`-`2.83` for `o_out`. The single largest is L20's `o_out` at `2.8x` —
nowhere near the 10x-500x the preset bounds demand, and the median is below 1.

What actually happens is that **position 0 detonates in the MLP branch of layer
0**, and again at layer 4, after which the residual stream carries the
magnitude forward. Everything upstream of layer 0's MLP is normal (hidden
`0.86x`, attn_residual `0.97x`); the SwiGLU output jumps `45.6x` in one step.
From layer 1 onward `hidden` and `attn_residual` merely inherit that carrier
(L1 `21.4x`, L4 `15.6x`, L7 `448x`, L23 `13.9x`) while each layer's attention
branch stays near 1x, because `ExtendRMSNorm` is scale-invariant and the norm
gammas are folded into the q/k/v/gate/up projections. Layer 4's measured
position-0 values (`swiglu 1018.8`, `down_out 2777.0`) are exactly why that
layer carries a `4000.0` bound.

MiniCPM5-1B has no `scale_emb`, `scale_depth` or muP factor — the config is a
plain `LlamaForCausalLM`. The `x32` pre-scale in front of layer 0's two
anchored norms is this pipeline's own qualified choice, not a model property.

## 5. The 33 redundant prefill clips, and one open question

The prefill preset produces 33 more effective `Clip` nodes than decode: 15 on
`normed`, 18 on `post_normed`, all with bound `50.0`, all landing on a tensor
the anchor Clip already produced at `39.1918`. Separately, the two families
also differ on 12 shared anchored keys, and on 11 of those prefill sits at the
same inert `50.0` while decode tightens to `27.0`-`33.0`.

They are structurally redundant. Under identity gamma with gamma folded into
the consuming projections, `RMS(out) = 1` exactly, so no element could exceed
`sqrt(H)` even if every Clip on that tensor were deleted; measured worst is
`37.86`.

Their cost in the emitted OM is small and bounded: both shipped binaries carry
an identical **851** layer records, so ATC fuses each extra Clip into the
anchor's existing layer rather than adding one. The 2,529-byte size delta lands
entirely in the net-def item region (the 33 size-growing blocks' deltas sum to
exactly 2,529). The two OMs are not otherwise identical — they also differ in
562,052 param-head bytes and 301 instruction-stream bytes — but those follow
from the 64 differing live bounds, not from these 33.

**What is not settled is whether they change the quantizer.** If section 2's
unproven `min()` rule held, a `50.0` bound above a `39.19` anchor would be a
no-op. If instead ATC adopts the last Clip's bound, prefill would declare `50.0`
where the anchor declared `39.19` — a 28% wider range and roughly 0.35 bit of
lost resolution on `normed`/`post_normed`. The available evidence does not
separate the two: 6 of the 120 projection quant records differ between the
families (`L9_q_proj`, `L0_k_proj`, `L9_k_proj`, `L4_gate_proj`,
`L23_gate_proj`, `L4_up_proj`; `v_proj` is identical in all 24), which is
equally consistent with the 64 differing live bounds.

The experiment that would settle it is small: rebuild one prefill layer with
the 33 bounds removed and diff the emitted `calibration_param.txt`. Identical
records prove `min()`; a scale change on `normed`/`post_normed` proves
last-Clip-wins and makes their removal a free accuracy gain. Until then this
section records an open question, not a verdict.

`tests/test_clip_contract.py` pins the preset-level invariants behind sections
3-5, so a regeneration cannot quietly emit a prefill bound below the anchor, or
a decode bound looser than prefill's. It does not and cannot pin ATC's
behaviour.
