#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strict runtime-profile loader for the dependency-free Hi3403 application."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


SCHEMA = "pico.minicpm5.runtime-profile.v2"
KNOWN_CONTEXTS = (128, 1024, 4096, 8192, 10240, 16384)
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


class ProfileError(ValueError):
    """A runtime profile is malformed or incompatible with the requested mode."""


def _integer(value, label, minimum, maximum):
    if type(value) is not int or value < minimum or value > maximum:
        raise ProfileError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _boolean(value, label):
    if type(value) is not bool:
        raise ProfileError(f"{label} must be boolean")
    return value


def _relative_path(value, label):
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ProfileError(f"{label} must stay under the deployment root")
    return path


@dataclass(frozen=True)
class RuntimeProfile:
    source: Path
    name: str
    status: str
    context: int
    past_length: int
    prefill_window: int
    chat: bool
    agent: bool
    completion: bool
    default_max_new: int
    max_new_limit: int
    reserve_tokens: int
    compact_at_tokens: int
    max_tool_steps: int
    max_tool_output_chars: int
    decode: Path
    prefill: Path
    head: Path
    kv_output_slots: tuple[int, int]
    hidden_output_slot: int
    minimum_cosine_exclusive: float

    def require_mode(self, mode: str, *, allow_unqualified: bool = False) -> None:
        enabled = {
            "agent": self.agent,
            "chat": self.chat,
            "completion": self.completion,
        }.get(mode)
        if enabled is None:
            raise ProfileError(f"unknown runtime mode: {mode}")
        if not enabled:
            if mode == "agent" and self.name == "ctx128":
                raise ProfileError(
                    "ctx128 is chat-only; use chat.sh --profile ctx128. "
                    "Agent mode requires ctx1024, ctx4096 or ctx8192")
            raise ProfileError(f"profile {self.name} does not support {mode} mode")
        if self.status != "qualified" and not allow_unqualified:
            raise ProfileError(
                f"profile {self.name} status is {self.status!r}; pass "
                "--allow-unqualified-profile only for controlled development")

    def model_paths(self, deployment_root: Path) -> dict[str, Path]:
        root = deployment_root.expanduser().resolve()
        return {
            "decode": root / self.decode,
            "prefill": root / self.prefill,
            "head": root / self.head,
        }


def _profile_path(value: str, profiles_dir: Path) -> Path:
    if not value:
        raise ProfileError("profile name/path is empty")
    supplied = Path(value).expanduser()
    if supplied.is_absolute() or "/" in value or value.endswith(".json"):
        return supplied.resolve()
    if _NAME.fullmatch(value) is None:
        raise ProfileError("profile name contains unsupported characters")
    return (profiles_dir / f"{value}.json").resolve()


def load_runtime_profile(value: str, profiles_dir: Path) -> RuntimeProfile:
    path = _profile_path(value, profiles_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProfileError(f"cannot read runtime profile {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProfileError(f"invalid runtime profile JSON {path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ProfileError(f"runtime profile must use schema {SCHEMA}")
    expected = {
        "schema", "name", "status", "context", "capabilities", "generation",
        "agent", "models", "transformer_outputs", "numeric_gate",
    }
    if set(raw) != expected:
        raise ProfileError(
            f"runtime profile fields mismatch: expected={sorted(expected)}")
    name = raw["name"]
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ProfileError("profile name is invalid")
    status = raw["status"]
    if status not in {"qualified", "pending"}:
        raise ProfileError("profile status must be qualified or pending")

    context = raw["context"]
    if not isinstance(context, dict) or set(context) != {
            "capacity", "past_length", "prefill_window"}:
        raise ProfileError(
            "context must contain capacity, past_length and prefill_window")
    capacity = _integer(context["capacity"], "context.capacity", 2, 65536)
    if capacity not in KNOWN_CONTEXTS:
        raise ProfileError(f"context.capacity must be one of {KNOWN_CONTEXTS}")
    past = _integer(context["past_length"], "context.past_length", 1, capacity)
    if past != capacity - 1:
        raise ProfileError("context.past_length must equal capacity - 1")
    window = _integer(
        context["prefill_window"], "context.prefill_window", 2, capacity)
    if window not in KNOWN_CONTEXTS:
        raise ProfileError(
            f"context.prefill_window must be one of {KNOWN_CONTEXTS}")

    capabilities = raw["capabilities"]
    if not isinstance(capabilities, dict) or set(capabilities) != {
            "chat", "agent", "completion"}:
        raise ProfileError("capabilities fields mismatch")
    chat = _boolean(capabilities["chat"], "capabilities.chat")
    agent_enabled = _boolean(capabilities["agent"], "capabilities.agent")
    completion = _boolean(capabilities["completion"], "capabilities.completion")
    if capacity == 128 and agent_enabled:
        raise ProfileError("ctx128 must be chat-only and cannot enable agent mode")

    generation = raw["generation"]
    if not isinstance(generation, dict) or set(generation) != {
            "default_max_new", "max_new_limit", "reserve_tokens",
            "compact_at_tokens"}:
        raise ProfileError("generation fields mismatch")
    max_limit = _integer(
        generation["max_new_limit"], "generation.max_new_limit", 1, capacity - 1)
    default_max = _integer(
        generation["default_max_new"], "generation.default_max_new", 1, max_limit)
    reserve = _integer(
        generation["reserve_tokens"], "generation.reserve_tokens", 0, capacity - 1)
    compact_at = _integer(
        generation["compact_at_tokens"], "generation.compact_at_tokens", 1,
        capacity - 1)
    if compact_at + reserve > capacity:
        raise ProfileError("compact_at_tokens + reserve_tokens exceeds context")

    agent_policy = raw["agent"]
    if not isinstance(agent_policy, dict) or set(agent_policy) != {
            "max_tool_steps", "max_tool_output_chars"}:
        raise ProfileError("agent fields mismatch")
    tool_steps = _integer(
        agent_policy["max_tool_steps"], "agent.max_tool_steps", 0, 16)
    tool_chars = _integer(
        agent_policy["max_tool_output_chars"],
        "agent.max_tool_output_chars", 64, 16384)
    if agent_enabled and tool_steps < 1:
        raise ProfileError("agent-enabled profiles need at least one tool step")

    models = raw["models"]
    if not isinstance(models, dict) or set(models) != {"decode", "prefill", "head"}:
        raise ProfileError("models must contain decode, prefill and head")
    outputs = raw["transformer_outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"k", "v", "hidden"}:
        raise ProfileError("transformer_outputs must contain k, v and hidden")
    k_slot = _integer(outputs["k"], "transformer_outputs.k", 0, 2)
    v_slot = _integer(outputs["v"], "transformer_outputs.v", 0, 2)
    hidden_slot = _integer(outputs["hidden"], "transformer_outputs.hidden", 0, 2)
    if {k_slot, v_slot, hidden_slot} != {0, 1, 2}:
        raise ProfileError("transformer output slots must be distinct and cover 0,1,2")
    gate = raw["numeric_gate"]
    if not isinstance(gate, dict) or set(gate) != {"minimum_cosine_exclusive"}:
        raise ProfileError("numeric_gate fields mismatch")
    threshold = gate["minimum_cosine_exclusive"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not 0.0 < float(threshold) < 1.0:
        raise ProfileError("minimum cosine must be a real number in (0,1)")

    return RuntimeProfile(
        source=path, name=name, status=status, context=capacity,
        past_length=past, prefill_window=window, chat=chat, agent=agent_enabled,
        completion=completion, default_max_new=default_max,
        max_new_limit=max_limit, reserve_tokens=reserve,
        compact_at_tokens=compact_at, max_tool_steps=tool_steps,
        max_tool_output_chars=tool_chars,
        decode=_relative_path(models["decode"], "models.decode"),
        prefill=_relative_path(models["prefill"], "models.prefill"),
        head=_relative_path(models["head"], "models.head"),
        kv_output_slots=(k_slot, v_slot), hidden_output_slot=hidden_slot,
        minimum_cosine_exclusive=float(threshold),
    )
