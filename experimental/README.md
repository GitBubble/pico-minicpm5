# Experimental OM post-linking

Binary OM post-linking is not part of the supported build pipeline.

The accepted three-handle release composes all 24 decoder layers at the ONNX
graph level and lets ATC allocate and compile the whole graph. Historical
post-link experiments were useful for identifying relocation domains, but a
general linker must simultaneously close parameter heaps and cumulative
weight tables, all absolute branch PCs, Item graph PCs/slots/ordinals,
TaskInfo/runtime tables and workspace declarations. Load or execute success is
not proof that layer 1+ ran with its own parameters.

No post-linker is shipped in v0.1.0. A future implementation belongs here only
after it passes N=1 byte parity, N=2 liveness and parameter isolation, full
24-layer public-output cosine `>0.98`, token exactness, and board validation.
