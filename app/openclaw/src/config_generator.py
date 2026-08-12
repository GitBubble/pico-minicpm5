#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the lean OpenClaw contract for a board-local MiniCPM5-1B.

The generator is deliberately render-only: it never invokes an installer,
opens a network connection, or modifies a board.  It emits a pinned OpenClaw
configuration plus online-install, offline-bundle, and staged E2E command
templates for a later, explicitly authorized deployment step.

``context_window`` is a required input because it must describe the actual
compiled KV-cache/runtime contract.  OpenClaw hard-blocks local-model contexts
below 4096 tokens, so this generator fails closed instead of copying the
checkpoint's much larger architectural maximum into a board configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Sequence


OPENCLAW_VERSION = "2026.7.1"
NODE_VERSION = "24.15.0"
MIN_CONTEXT_WINDOW = 4096
DEFAULT_MAX_TOKENS = 256
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_INSTALL_PREFIX = "/opt/openclaw"
# The generated OpenClaw workspace exceeds the usable prompt budget of the
# board's 16K decode contract.  Keep a small, deterministic project-context
# allowance instead of inheriting desktop-sized bootstrap defaults.
BOARD_BOOTSTRAP_MAX_CHARS = 1024
BOARD_BOOTSTRAP_TOTAL_MAX_CHARS = 4096
BOARD_COMPACTION_RESERVE_CAP = 4096
BOARD_COMPACTION_RESERVE_MIN = 1024
PROVIDER_ID = "pico-minicpm"
MODEL_ID = "minicpm5-1b"
MODEL_REF = f"{PROVIDER_ID}/{MODEL_ID}"
MANIFEST_SCHEMA = "pico.openclaw.minicpm5.v1"
DEFAULT_TOOL_NAME = "session_status"
TOOL_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _require_absolute_prefix(value: str) -> str:
    prefix = str(value).rstrip("/") or "/"
    path = PurePosixPath(prefix)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("install_prefix must be an absolute normalized path")
    return prefix


@dataclass(frozen=True)
class OpenClawMiniCPM5Contract:
    """Actual board runtime limits and the corresponding OpenClaw contract."""

    context_window: int
    max_tokens: int = DEFAULT_MAX_TOKENS
    port: int = DEFAULT_PORT
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    install_prefix: str = DEFAULT_INSTALL_PREFIX
    supports_tools: bool = False
    tool_name: str = DEFAULT_TOOL_NAME

    def __post_init__(self) -> None:
        context_window = int(self.context_window)
        max_tokens = int(self.max_tokens)
        port = int(self.port)
        timeout_seconds = int(self.timeout_seconds)
        if context_window < MIN_CONTEXT_WINDOW:
            raise ValueError(
                f"actual context_window must be at least {MIN_CONTEXT_WINDOW}")
        if max_tokens <= 0 or max_tokens > context_window:
            raise ValueError("max_tokens must be positive and no larger than context_window")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in the range 1..65535")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (not isinstance(self.tool_name, str) or
                TOOL_NAME_RE.fullmatch(self.tool_name) is None):
            raise ValueError("tool_name must be one simple OpenClaw tool name")
        object.__setattr__(self, "context_window", context_window)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(
            self, "install_prefix", _require_absolute_prefix(self.install_prefix))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def openclaw_bin(self) -> str:
        return f"{self.install_prefix}/bin/openclaw"

    @property
    def offline_bundle_name(self) -> str:
        return (
            f"openclaw-{OPENCLAW_VERSION}-node-{NODE_VERSION}-linux-arm64.tar.gz")

    def openclaw_config(self) -> dict[str, object]:
        """Return a network-quiet, single-agent local configuration."""
        reserve_tokens = min(
            BOARD_COMPACTION_RESERVE_CAP,
            max(BOARD_COMPACTION_RESERVE_MIN, self.context_window // 4))
        return {
            "gateway": {
                "mode": "local",
                "bind": "loopback",
                "controlUi": {"enabled": False},
            },
            "models": {
                "mode": "replace",
                # OpenClaw 2026.7.1 uses ``models.pricing`` for the
                # background pricing-catalog bootstrap.  ``catalogRefresh``
                # belongs to a later schema and must not leak into this
                # version-pinned contract.
                "pricing": {"enabled": False},
                "providers": {
                    PROVIDER_ID: {
                        "baseUrl": self.base_url,
                        "apiKey": "pico-local-no-auth",
                        "api": "openai-completions",
                        "timeoutSeconds": self.timeout_seconds,
                        "models": [{
                            "id": MODEL_ID,
                            "name": "MiniCPM5-1B PicoNPU",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": self.context_window,
                            "contextTokens": self.context_window,
                            "maxTokens": self.max_tokens,
                            "compat": {
                                "supportsTools": bool(self.supports_tools),
                                "requiresStringContent": True,
                                # Tool history needs assistant.tool_calls and
                                # tool.tool_call_id to survive OpenClaw's
                                # provider normalization.
                                "strictMessageKeys": not self.supports_tools,
                            },
                        }],
                    },
                },
            },
            "agents": {
                "defaults": {
                    "model": {"primary": MODEL_REF},
                    "maxConcurrent": 1,
                    "timeoutSeconds": self.timeout_seconds,
                    "heartbeat": {"every": "0m"},
                    "sandbox": {"mode": "off"},
                    "bootstrapMaxChars": BOARD_BOOTSTRAP_MAX_CHARS,
                    "bootstrapTotalMaxChars": BOARD_BOOTSTRAP_TOTAL_MAX_CHARS,
                    # A native 1B deployment cannot afford OpenClaw's generated
                    # workspace bootstrap on the first turn.  Keep all agent
                    # workspace files out of the model prompt at every context
                    # bucket; explicit user messages and tool results remain.
                    "contextInjection": "never",
                    "skipBootstrap": True,
                    "startupContext": {"enabled": False},
                    "compaction": {
                        "reserveTokens": reserve_tokens,
                        "reserveTokensFloor": reserve_tokens,
                        "keepRecentTokens": reserve_tokens,
                    },
                    # The release surface uses native OpenClaw tools, not
                    # prompt-injected skills.  Keep the skill allowlist empty
                    # in both text and tool modes: removing this field in tool
                    # mode makes OpenClaw advertise every eligible desktop
                    # skill and can overflow the actual C8192 prompt budget.
                    "skills": [],
                    "experimental": {"localModelLean": True},
                },
            },
            # Tool mode is a deliberately single-tool release surface.  Lean
            # mode would otherwise add Tool Search controls, and OpenClaw's
            # rolling loop detector is off by default, so make both policies
            # explicit here as a second guard behind the service's one-call
            # per-active-user-turn budget.
            **({"tools": {
                "profile": "minimal",
                "allow": [self.tool_name],
                "toolSearch": False,
                "loopDetection": {
                    "enabled": True,
                    "historySize": 8,
                    "warningThreshold": 1,
                    "unknownToolThreshold": 1,
                    "criticalThreshold": 2,
                    "globalCircuitBreakerThreshold": 3,
                    "detectors": {
                        "genericRepeat": True,
                        "knownPollNoProgress": True,
                        "pingPong": True,
                    },
                    "postCompactionGuard": {"windowSize": 1},
                },
            }}
               if self.supports_tools else {}),
        }

    def install_manifest(self) -> dict[str, object]:
        installer = (
            "curl -fsSL --proto '=https' --tlsv1.2 "
            "https://openclaw.ai/install-cli.sh | "
            f"bash -s -- --prefix {shlex.quote(self.install_prefix)} "
            f"--version {OPENCLAW_VERSION} --node-version {NODE_VERSION} "
            "--no-onboard"
        )
        bundle = self.offline_bundle_name
        return {
            "pinned_openclaw_version": OPENCLAW_VERSION,
            "pinned_node_version": NODE_VERSION,
            "target": "linux-arm64-glibc",
            "online_install_command_template": installer,
            "online_command_requires_network_when_executed": True,
            "offline_bundle": {
                "filename": bundle,
                "format": "tar.gz",
                "sha256_sidecar_required": f"{bundle}.sha256",
                "extract_root": self.install_prefix,
                "required_relative_paths": [
                    "bin/openclaw",
                    f"tools/node-v{NODE_VERSION}/bin/node",
                ],
                "deploy_command_templates": [
                    _shell_join(["sha256sum", "-c", f"{bundle}.sha256"]),
                    _shell_join(["install", "-d", self.install_prefix]),
                    _shell_join([
                        "tar", "-xzf", bundle, "-C", self.install_prefix]),
                    _shell_join(["test", "-x", self.openclaw_bin]),
                    _shell_join([self.openclaw_bin, "--version"]),
                ],
            },
        }

    def e2e_commands(self) -> dict[str, object]:
        chat_payload = json.dumps({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Reply exactly: pong"}],
            "stream": False,
            "max_tokens": min(8, self.max_tokens),
        }, separators=(",", ":"))
        return {
            "models": _shell_join([
                "curl", "-fsS", "--connect-timeout", "5", "--max-time",
                "10", f"{self.base_url}/models"]),
            "chat": _shell_join([
                "curl", "-fsS", "--connect-timeout", "5", "--max-time",
                str(self.timeout_seconds), "-H", "Content-Type: application/json",
                "-d", chat_payload, f"{self.base_url}/chat/completions",
            ]),
            "infer_local": _shell_join([
                self.openclaw_bin, "infer", "model", "run", "--local",
                "--model", MODEL_REF, "--prompt", "Reply exactly: pong", "--json",
            ]),
            "gateway_start": _shell_join([
                self.openclaw_bin, "gateway", "run", "--bind", "loopback",
                "--port", "18789",
            ]),
            "infer_gateway": _shell_join([
                self.openclaw_bin, "infer", "model", "run", "--gateway",
                "--model", MODEL_REF, "--prompt", "Reply exactly: pong", "--json",
            ]),
            "agent_local": _shell_join([
                self.openclaw_bin, "agent", "--local", "--session-id",
                "minicpm5-e2e-local", "--model", MODEL_REF, "--message",
                "只回复 PICO_AGENT_OK", "--thinking", "off", "--timeout",
                str(self.timeout_seconds), "--json",
            ]),
            "agent": _shell_join([
                self.openclaw_bin, "agent", "--session-id",
                "minicpm5-e2e-gateway", "--model", MODEL_REF, "--message",
                "只回复 PICO_AGENT_OK", "--thinking", "off", "--timeout",
                str(self.timeout_seconds), "--json",
            ]),
            "expected_gates": {
                "models": f"HTTP success and model id {MODEL_ID!r} is present",
                "chat": "assistant content is exactly 'pong'",
                "infer_local": "JSON status is 'ok' and final is 'pong'",
                "infer_gateway": "JSON status is 'ok' through Gateway routing",
                "agent_local": (
                    "embedded JSON response is non-error and contains 'PICO_AGENT_OK'"),
                "agent": (
                    "Gateway JSON response is non-error and contains 'PICO_AGENT_OK'"),
                "tools": (
                    "structured tool_calls and toolSummary.calls >= 1 are required"
                    if self.supports_tools else
                    "disabled for the initial text-only agent gate"),
            },
        }

    def manifest(self) -> dict[str, object]:
        config = self.openclaw_config()
        defaults = config["agents"]["defaults"]
        compaction = defaults["compaction"]
        return {
            "schema": MANIFEST_SCHEMA,
            "render_only": True,
            "generator_side_effects": {
                "network_commands_executed": False,
                "board_commands_executed": False,
            },
            "runtime_contract": {
                "provider": PROVIDER_ID,
                "model": MODEL_ID,
                "model_ref": MODEL_REF,
                "base_url": self.base_url,
                "api": "openai-completions",
                "context_window": self.context_window,
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout_seconds,
                "supports_tools": bool(self.supports_tools),
                "tool_allowlist": (
                    [self.tool_name] if self.supports_tools else []),
                "minimum_openclaw_context_window": MIN_CONTEXT_WINDOW,
                "catalog_contract": (
                    "models.mode=replace with one explicit loopback provider; "
                    "models.pricing.enabled=false disables the 2026.7.1 "
                    "background pricing-catalog fetch"),
                "bootstrap_contract": (
                    f"per_file={BOARD_BOOTSTRAP_MAX_CHARS},"
                    f"total={BOARD_BOOTSTRAP_TOTAL_MAX_CHARS},"
                    f"context_injection={defaults['contextInjection']}; "
                    "sized for the actual "
                    "board-local context rather than generated desktop defaults"),
                "compaction_contract": (
                    f"reserve={compaction['reserveTokens']},"
                    f"floor={compaction['reserveTokensFloor']},"
                    f"keep_recent={compaction['keepRecentTokens']}; "
                    "overrides OpenClaw's desktop-sized reserve floor"),
            },
            "openclaw_config": config,
            "installation": self.install_manifest(),
            "e2e": self.e2e_commands(),
        }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-window", type=int, required=True,
        help="actual compiled KV-cache/runtime token limit (must be >=4096)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--install-prefix", default=DEFAULT_INSTALL_PREFIX)
    parser.add_argument(
        "--supports-tools", action="store_true",
        help="declare only after the endpoint passes structured tool-call gates")
    parser.add_argument(
        "--tool-name", default=DEFAULT_TOOL_NAME,
        help="single OpenClaw tool exposed when --supports-tools is enabled")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--config-out", type=Path)
    args = parser.parse_args(argv)

    contract = OpenClawMiniCPM5Contract(
        context_window=args.context_window,
        max_tokens=args.max_tokens,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        install_prefix=args.install_prefix,
        supports_tools=args.supports_tools,
        tool_name=args.tool_name,
    )
    manifest = contract.manifest()
    if args.manifest_out is not None:
        _write_json(args.manifest_out, manifest)
    if args.config_out is not None:
        _write_json(args.config_out, contract.openclaw_config())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
