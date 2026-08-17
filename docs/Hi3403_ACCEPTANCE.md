# Frozen Hi3403 ctx1024 acceptance

[中文](Hi3403_ACCEPTANCE.zh-CN.md)

This records the 2026-08-09 acceptance of the three-handle candidate
identified in `release/v0.1.0/release-manifest.json`. The three OM hashes are
unchanged in `v0.2.0`; the throughput figures below were superseded by the
`v0.2.0` executor and are kept here as the historical record. Current numbers
are in [the performance board](../release/perf/README.md).

| Model | Bytes | Minimum public cosine |
|---|---:|---:|
| Prefill, position 0 | 686,999,901 | 0.996646 |
| Decode, position 1 | 686,997,372 | 0.998023 |
| Dense vocabulary head | 202,651,666 | covered by 48/48 greedy exact |

Three prompt runs matched all 48 FP64 greedy tokens. EOS and Chinese text paths
passed. The incrementally refreshed resident-K/V runtime measured
105.5–106.1 ms per generated token, or 9.42–9.48 tok/s, approximately 1.91x
the accepted 49-handle baseline. The three accepted OM hashes are unchanged.

This document records evidence for the frozen hashes. A source rebuild with a
regenerated calibration corpus is a new candidate, even if its graph is
semantically identical, and must repeat the gate.
