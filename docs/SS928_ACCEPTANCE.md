# Frozen SS928 ctx1024 acceptance

The accepted 2026-08-09 three-handle candidate is identified in
`release/v0.1.0/release-manifest.json`.

| Model | Bytes | Minimum public cosine |
|---|---:|---:|
| Prefill, position 0 | 686,999,901 | 0.996646 |
| Decode, position 1 | 686,997,372 | 0.998023 |
| Dense vocabulary head | 202,651,666 | covered by 48/48 greedy exact |

Three prompt runs matched all 48 FP64 greedy tokens. EOS and Chinese text paths
passed. Median generated-token latency was 116.3–122.0 ms, or 8.20–8.60 tok/s,
approximately 1.67x the accepted 49-handle baseline.

This document records evidence for the frozen hashes. A source rebuild with a
regenerated calibration corpus is a new candidate, even if its graph is
semantically identical, and must repeat the gate.
