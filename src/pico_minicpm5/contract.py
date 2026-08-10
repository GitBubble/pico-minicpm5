"""Pinned checkpoint contract and drift checks.

The checks deliberately inspect metadata before loading any learned tensor.
They make a wrong revision, renamed shard, changed geometry or truncated
safetensors payload fail before ONNX construction starts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

HF_REPO_ID = "openbmb/MiniCPM5-1B"
HF_REVISION = "4e9de7a0778dc1c362e983e6858f0e77542cbdca"
CONFIG_SHA256 = "6a6509b646cb3169616c5ffc3196e7ccaf9d4d6bc17b266581d241a31c217714"
WEIGHT_INDEX_SHA256 = "162add042e75abc3d571c4a8679523fa4f1ffc55d1fea25fc6658a19d6e957ee"
WEIGHT_SHARD = "model-00000-of-00001.safetensors"
WEIGHT_PAYLOAD_BYTES = 2_161_265_664
WEIGHT_FILE_BYTES = 2_161_290_912
WEIGHT_FILE_SHA256 = "7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d"
TOKENIZER_SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
PUBLIC_COSINE_THRESHOLD_EXCLUSIVE = 0.98
SAFETENSORS_HEADER_BYTES = 25_240
SAFETENSORS_PREFIX_SHA256 = (
    "ecbdf640e2ce4cf283b3d6d7c758d7cf71cfbbd30c3aea04601f36f8999c5675"
)


@dataclass(frozen=True)
class MiniCPM5Contract:
    architecture: str = "LlamaForCausalLM"
    model_type: str = "llama"
    hidden_size: int = 1536
    intermediate_size: int = 4608
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 128
    num_hidden_layers: int = 24
    vocab_size: int = 130560
    max_position_embeddings: int = 131072
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    torch_dtype: str = "bfloat16"
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.architecture != "LlamaForCausalLM" or self.model_type != "llama":
            raise ValueError("MiniCPM5-1B must use the standard Llama architecture")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotate-half RoPE")
        if self.hidden_act != "silu":
            raise ValueError("the qualified graph requires SwiGLU/SiLU")

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def kv_cache_bytes(self, context: int, dtype_bytes: int = 2) -> int:
        if not 1 <= context <= self.max_position_embeddings:
            raise ValueError("context is outside max_position_embeddings")
        return (
            self.num_hidden_layers
            * 2
            * self.num_key_value_heads
            * context
            * self.head_dim
            * dtype_bytes
        )


OFFICIAL_CONTRACT = MiniCPM5Contract()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def contract_from_hf_config(config: Mapping[str, Any]) -> MiniCPM5Contract:
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("config.architectures must contain exactly one entry")
    candidate = MiniCPM5Contract(
        architecture=str(architectures[0]),
        model_type=str(config.get("model_type")),
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        vocab_size=int(config["vocab_size"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        rope_theta=float(config["rope_theta"]),
        rms_norm_eps=float(config["rms_norm_eps"]),
        hidden_act=str(config["hidden_act"]),
        torch_dtype=str(config["torch_dtype"]),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
    )
    if candidate != OFFICIAL_CONTRACT:
        expected, actual = asdict(OFFICIAL_CONTRACT), asdict(candidate)
        differences = {
            key: [expected[key], actual[key]]
            for key in expected
            if expected[key] != actual[key]
        }
        raise ValueError(f"MiniCPM5-1B config drift: {differences}")
    if config.get("rope_scaling") is not None:
        raise ValueError("qualified MiniCPM5-1B config has rope_scaling=null")
    return candidate


def expected_weight_shapes(
    contract: MiniCPM5Contract = OFFICIAL_CONTRACT,
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (contract.vocab_size, contract.hidden_size),
        "model.norm.weight": (contract.hidden_size,),
        "lm_head.weight": (contract.vocab_size, contract.hidden_size),
    }
    for layer in range(contract.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (contract.hidden_size,),
                prefix + "post_attention_layernorm.weight": (contract.hidden_size,),
                prefix + "self_attn.q_proj.weight": (
                    contract.query_width,
                    contract.hidden_size,
                ),
                prefix + "self_attn.k_proj.weight": (
                    contract.kv_width,
                    contract.hidden_size,
                ),
                prefix + "self_attn.v_proj.weight": (
                    contract.kv_width,
                    contract.hidden_size,
                ),
                prefix + "self_attn.o_proj.weight": (
                    contract.hidden_size,
                    contract.query_width,
                ),
                prefix + "mlp.gate_proj.weight": (
                    contract.intermediate_size,
                    contract.hidden_size,
                ),
                prefix + "mlp.up_proj.weight": (
                    contract.intermediate_size,
                    contract.hidden_size,
                ),
                prefix + "mlp.down_proj.weight": (
                    contract.hidden_size,
                    contract.intermediate_size,
                ),
            }
        )
    return shapes


def read_safetensors_header(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError("safetensors file is shorter than its length field")
        header_bytes = struct.unpack("<Q", raw_length)[0]
        if header_bytes != SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"safetensors header size drift: {header_bytes}")
        payload = stream.read(header_bytes)
    if len(payload) != header_bytes:
        raise ValueError("safetensors file does not contain its complete header")
    prefix_hash = hashlib.sha256(raw_length + payload).hexdigest()
    if prefix_hash != SAFETENSORS_PREFIX_SHA256:
        raise ValueError(f"safetensors header hash drift: {prefix_hash}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("safetensors header must be a JSON object")
    return value


def validate_safetensors_header(header: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_weight_shapes()
    tensors = {str(name): entry for name, entry in header.items() if name != "__metadata__"}
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))[:8]
        extra = sorted(set(tensors) - set(expected))[:8]
        raise ValueError(f"safetensors symbols drift: missing={missing} extra={extra}")
    ranges: list[tuple[int, int, str]] = []
    for name, shape in expected.items():
        entry = tensors[name]
        if entry.get("dtype") != "BF16" or tuple(entry.get("shape", ())) != shape:
            raise ValueError(f"safetensors tensor contract drift for {name}: {entry}")
        begin, end = (int(value) for value in entry["data_offsets"])
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - begin != elements * 2:
            raise ValueError(f"BF16 byte span disagrees with shape for {name}")
        ranges.append((begin, end, name))
    ranges.sort()
    cursor = 0
    for begin, end, name in ranges:
        if begin != cursor:
            raise ValueError(f"safetensors payload gap before {name}: {cursor} != {begin}")
        cursor = end
    if cursor != WEIGHT_PAYLOAD_BYTES:
        raise ValueError(f"safetensors payload size drift: {cursor}")
    return {"tensor_count": len(tensors), "dtype": "BF16", "payload_bytes": cursor}


def validate_weight_index(index: Mapping[str, Any]) -> dict[str, Any]:
    metadata, weight_map = index.get("metadata"), index.get("weight_map")
    if not isinstance(metadata, Mapping) or not isinstance(weight_map, Mapping):
        raise ValueError("invalid safetensors index structure")
    expected = set(expected_weight_shapes())
    actual = {str(name) for name in weight_map}
    if actual != expected:
        raise ValueError(
            "weight index symbols drift: "
            f"missing={sorted(expected - actual)[:8]} extra={sorted(actual - expected)[:8]}"
        )
    if int(metadata.get("total_size", -1)) != WEIGHT_PAYLOAD_BYTES:
        raise ValueError("weight index total_size drift")
    shards = {str(value) for value in weight_map.values()}
    if shards != {WEIGHT_SHARD}:
        raise ValueError(f"unexpected weight shards: {sorted(shards)}")
    return {"tensor_count": len(actual), "total_size": WEIGHT_PAYLOAD_BYTES}


def verify_checkpoint(model_dir: Path, *, full_hash: bool = False) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    shard_path = model_dir / WEIGHT_SHARD
    required = (
        config_path,
        index_path,
        shard_path,
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete: {missing}")
    if sha256_file(config_path) != CONFIG_SHA256:
        raise ValueError("config.json SHA256 drift")
    if sha256_file(index_path) != WEIGHT_INDEX_SHA256:
        raise ValueError("model.safetensors.index.json SHA256 drift")
    tokenizer_digest = sha256_file(model_dir / "tokenizer.json")
    if tokenizer_digest != TOKENIZER_SHA256:
        raise ValueError("tokenizer.json SHA256 drift")
    if shard_path.stat().st_size != WEIGHT_FILE_BYTES:
        raise ValueError(f"weight shard size drift: {shard_path.stat().st_size}")
    contract_from_hf_config(_load_json(config_path))
    index_report = validate_weight_index(_load_json(index_path))
    tensor_report = validate_safetensors_header(read_safetensors_header(shard_path))
    report: dict[str, Any] = {
        "schema": "pico.minicpm5.checkpoint-verification.v1",
        "repository": HF_REPO_ID,
        "revision": HF_REVISION,
        "model_dir": str(model_dir),
        "config_sha256": CONFIG_SHA256,
        "weight_index_sha256": WEIGHT_INDEX_SHA256,
        "tokenizer_sha256": tokenizer_digest,
        "weight_shard": WEIGHT_SHARD,
        "weight_shard_bytes": shard_path.stat().st_size,
        "index": index_report,
        "safetensors": tensor_report,
        "status": "PASS",
    }
    if full_hash:
        digest = sha256_file(shard_path)
        if digest != WEIGHT_FILE_SHA256:
            raise ValueError(f"weight shard SHA256 drift: {digest}")
        report["weight_shard_sha256"] = digest
    return report
