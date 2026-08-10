# pico-minicpm5 v0.1.0

This source release captures the reproducible MiniCPM5-1B → ONNX → packed
24-layer PICO OM workflow and the accepted SS928 ctx1024 three-handle artifact
contract.

The corresponding qualified artifacts are identified by size and SHA256 in
`release-manifest.json`. They are not embedded in the public source archive.
The portable `qualification.json` retains raw-output hashes, public tensor
cosines, greedy token evidence and performance while omitting the board address.
`release assemble` can build a local model bundle from user-supplied artifacts
only after every hash and policy check passes.

The default compiler route is graph-level 24-layer composition followed by one
ATC invocation per family. Binary OM post-linking is not a production path.
