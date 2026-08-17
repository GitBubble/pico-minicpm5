# Native prefill 发布资格 v4

Native S16/S32/S128 宽块刻意区分两套资格契约：

- `pico.minicpm5.prefill-block-qualification.v2` 仅保留给开发采集和兼容读取，
  不具备发布激活资格；
- `pico.minicpm5.prefill-block-qualification.v4` 是
  `pico.minicpm5.prefill-activation.v4` 唯一接受的发布资格格式；发布 v3 已是历史
  格式，会被明确拒绝。

旧命令因此明确属于开发用途：

```bash
pico-minicpm5 qualify-prefill-block \
  --evidence work/prefill/dev-evidence.json \
  --out work/prefill/dev-qualification.json
```

发布资格必须使用显式的 v4 入口：

```bash
pico-minicpm5 qualify-prefill-block-release \
  --evidence work/prefill/release-evidence.json \
  --out work/prefill/release-qualification.json
```

验收 S16 之前，必须先生成 content-bound strict-S1 baseline：

```bash
pico-minicpm5 qualify-prefill-s1-release \
  --evidence work/prefill/s1-release-evidence.json \
  --out work/prefill/s1-release-qualification.json
```

## v4 绑定的证据

发布 evidence index 用精确的 `path`、`bytes` 和小写 `sha256` 指向真实文件。
路径必须是相对路径、始终位于 evidence 目录下，并且不能穿越符号链接。资格生成
过程会实际读取每个文件；文件缺失、字节数漂移、hash 漂移、JSON 重复键或多余字段
都会 fail closed。

一份 v4 资格记录会绑定：

- 候选 OM、实际 head OM、embedding artifact 和 build manifest；
- 板端 runner、native executor 和 ready descriptor；
- required absolute position 矩阵中每个位置的真实 JSON capture artifact；
- EOS、英文、中文和 context-boundary workload artifact；
- 包含 warm-up 与 measured 样本数组的真实性能测量 artifact；均值和 speedup
  由资格代码重算，而不是接受自报值；
- 完整 baseline qualification + baseline OM：S16 对 S1，S32 对 S16，S128
  同时对 S32 和 S16；
- clean-board MMZ before/after 观测。

每个 workload artifact 都绑定该次运行实际使用的 head OM 与 embedding SHA-256。
`prompt_sha256` 和 `output_tokens_sha256` 对“有序 token ID 序列按 little-endian
uint32 紧密打包后的字节串”做 SHA-256，契约名固定为
`sha256-le32-u32-token-id-sequence`。比较对象已经是精确 token ID，因此解释或复现
该 hash 不需要 tokenizer artifact。精确定义是对
`b"".join(struct.pack("<I", token_id) for token_id in token_ids)` 计算 SHA-256，
不添加数量、分隔符或其他前缀，且每个 ID 必须位于 `0..2^32-1`。
`pico_minicpm5.prefill_blocks.token_id_sequence_sha256()` 是规范的生成实现。

宽块 baseline qualification 会递归验证为完整 v4 记录。S1 trust anchor 必须是完整的
`pico.minicpm5.strict-s1-baseline-qualification.v4`：分别绑定 position 0 的
bootstrap OM 与 position>=1 的 canonical decode OM、实际 head OM 与 embedding、两套
ready descriptor、build manifest、runner、executor、绝对位置 capture 矩阵、精确 48-token、
EOS、英文、中文、context-boundary workload 和 clean-board MMZ 观测。旧的六字段
`strict-s1-baseline-qualification.v1` 仅保留为开发格式；单 OM 的 v2 与双路 v3 都仅
保留为历史格式；宽块 builder 与发布 activation 都会明确拒绝。
builder 和 verifier 还会递归提取每条 baseline 的完整 S1 identity；S128 的 S32/S16
只要落到不同 S1，即使两条资格各自为 PASS，也会在生成 release PASS 前整体拒绝。

## MMZ admission

v4 qualification builder 不接受外部传入的 `admission_bytes`，而是固定按下式推导：

```text
before_available_bytes - after_available_bytes
```

MMZ 观测必须为 `PASS`、板型为 Hi3403、`clean_board=true`，并绑定候选 OM、
runner、executor 和 ready descriptor 的 hash。发布激活还会要求 manifest 中的
`admission_bytes` 与该差值完全一致。每个激活宽度独立计费；重复 residency group
会被禁用，不能用来给第二个模型打折。

在某个宽度没有真实 v4 证据 artifact 之前，该宽度保持 release blocked，运行时
继续保留 strict S1 fallback。

strict-S1 MMZ 观测使用 `role=base-resident` 与
`accounting=included-in-base_resident_bytes`；其 `resident_bytes` 是 base accounting
的实测证据，不会再作为一份 wide-block admission 重复收费。

## 实际常驻 strict-S1 activation identity

Activation v4 不再接受旧的 `"strict_s1": true` 布尔标志。manifest 必须显式指向
板端实际常驻的 trust anchor：

```json
{
  "schema": "pico.minicpm5.prefill-activation.v4",
  "context": 4096,
  "deployment_mode": "trusted-read-only-process-lifetime",
  "strict_s1": {
    "bootstrap_model": "models/prefill-position0.om",
    "canonical_decode_model": "models/decode.om",
    "head_model": "models/head.om",
    "embedding": "assets/token_embedding.f16.bin",
    "qualification": "evidence/s1-qualification.json",
    "qualification_sha256": "<64 位小写十六进制>",
    "build_manifest": "evidence/s1-build.json",
    "runner": "app/src/pico_minicpm5_split_board_runner.py",
    "executor": "app/pico_persistent_acl_executor",
    "bootstrap_ready_descriptor": "evidence/prefill-ready-descriptor.bin",
    "canonical_ready_descriptor": "evidence/decode-ready-descriptor.bin"
  },
  "blocks": []
}
```

每个宽块的直接或递归 baseline 都必须解析到同一个 S1 qualification、两只路由 OM、
head OM、embedding、build、runner、executor 和两套 descriptor identity；每份宽块
资格本身也必须与 live S1 使用完全相同的 runner/executor 和 head/embedding identity。
不同 anchor 只禁用对应宽度。只要使用 activation manifest，即使没有注册 wide handler，
runtime 也会在 `probe._start` 紧前再次 resolve 并哈希实际 executor、imported protocol
runner、bootstrap OM、canonical decode OM、head OM、embedding、两套 descriptor
artifact 和每只已注册 wide OM。

`deployment_mode` 的精确值是必填的运维信任断言；缺失或换值会整体拒绝 manifest，
但 runtime 无法凭这个声明证明文件系统确实不可变。这次 preflight 能发现陈旧或被动
替换的文件，并尽量缩短 qualification 到加载的间隔，
但它不是 inherited-fd handoff：executor 之后仍会按路径重新打开模型，因此主动恶意
写入者仍可能在最终 hash 与 open 之间竞态。发布 activation 因而对部署前提 fail closed：
部署树必须可信，并在整个进程生命周期内保持只读/不可变。如果调用方
传入的 `base_resident_bytes` 小于 S1 实测 `resident_bytes`，所有宽块都会被禁用，
但已验证的 strict-S1 fallback 仍保留在报告中。若顶层 S1 anchor 本身缺失、不可读、
不是 UTF-8、被篡改或不完整，activation 会整体拒绝 manifest，不会宣称一个未经验证的
fallback。

内存门禁固定使用
`effective_base_resident_bytes = max(base_resident_bytes, s1.resident_bytes)`。
如果该有效下界加 reserve 已超过 available MMZ，activation 会整体拒绝 manifest，
因为此时连声明的 S1 fallback 本身都无法完成 admission。
