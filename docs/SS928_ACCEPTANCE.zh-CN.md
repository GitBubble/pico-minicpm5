# 冻结的 SS928 ctx1024 验收

[English](SS928_ACCEPTANCE.md)

2026-08-09 验收的三句柄候选由 `release/v0.1.0/release-manifest.json` 唯一标识。

| 模型 | 字节数 | 最低公开 cosine |
|---|---:|---:|
| Prefill，position 0 | 686,999,901 | 0.996646 |
| Decode，position 1 | 686,997,372 | 0.998023 |
| 词表 head | 202,651,666 | 由 48/48 greedy exact 覆盖 |

三个 prompt 共 48 个 FP64 greedy token 全部一致；EOS 与中文路径通过。生成 token
中位延迟 116.3–122.0 ms，即 8.20–8.60 tok/s，约为 49 句柄基线的 1.67 倍。

该证据只适用于冻结哈希。即使语义图相同，只要重新生成 calibration 并重编译，
就属于新候选，必须重新执行全部门禁。
