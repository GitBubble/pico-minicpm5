# OM composition contract

The shipping method composes graphs, then compiles the result once. This lets
the graph compiler own allocation, instruction scheduling, runtime TaskInfo
and the physical hidden bridge.

## Graph rules

1. Prefix every private node, tensor and initializer with `L{index}_`.
2. Preserve `hidden`, `attention_mask` and `rope_r` as public/shared names.
3. Replace layer `i>0`'s hidden input with `L{i-1}_next_hidden`.
4. Pack 24 per-layer K inputs into `k_cache_all` and split on channel axis 1;
   do the same for V.
5. Concatenate current K/V rows in layer-major, then KV-head-major order.
6. Publish only the final hidden and the two packed current-row tensors.

For 24 layers this is a stable five-input/three-output public ABI. ATC's
observed SS928 descriptor has seven runtime inputs because it synthesizes two
auxiliary inputs; they are runtime implementation details, not graph inputs.

## External data

The real 24-layer ONNX exceeds protobuf's 2 GiB message ceiling. A single
external blob is also unsafe in the qualified ATC path because offsets beyond
signed 32-bit range are rejected. Therefore every sufficiently large
initializer receives its own external file and offset zero. The compiler
backend also exposes relative links in its work directory for ATC's lookup
behavior.

## Input packing and output packing are independent

Packed inputs solve the compiler's input-count limit. Packed outputs create a
shared quantization domain. The accepted public prefill V tensor scores
`0.996646`, while its low-energy layer-2 diagnostic slice is `0.971119`.
Therefore public-output gating and per-layer diagnostics must both be reported.
If future policy requires every slice above 0.98, retain packed inputs but
publish per-layer outputs or otherwise split the output quantization domains.

## Why binary post-link is not the default

Binary OM linking must simultaneously relocate parameter feeders and the
cumulative weight table, all absolute branch PCs, END lifecycle, TaskInfo,
Item/Net IO slots and ordinals, workspace declarations and physical layer
bridges. A model can load and execute while still running layer zero repeatedly.
The historical V7 DLD/DSTR-offset method was disproved. Any future linker must
live behind an explicit experimental command and pass N=1 identity plus N=2
liveness before deeper tests.
