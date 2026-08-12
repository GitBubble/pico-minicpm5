#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Safe local lifecycle helper for the MiniCPM5 OpenClaw adapter.

The helper reads a strict JSON runtime contract.  It never evaluates shell
text: a native runner command is represented as a JSON array and is passed to
``subprocess.Popen`` unchanged by the OpenAI service.  The HTTP endpoint is
intentionally restricted to IPv4 loopback because it has no authentication.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import time
from typing import Iterator, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from config_generator import OPENCLAW_VERSION, OpenClawMiniCPM5Contract
from merged_jsonl_runner import build_parser as build_merged_runner_parser


RUNTIME_SCHEMA = "pico.minicpm5.openclaw-runtime.v1"
PROFILE_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PROFILE = "pico-minicpm"
DEFAULT_WAIT_SECONDS = 60.0
APP_ROOT = Path(__file__).resolve().parent.parent
SERVICE_SCRIPT = Path(__file__).resolve().with_name("openai_service.py")


class ContractError(ValueError):
    """The local runtime JSON is unsafe or incomplete."""


def _absolute_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{label} must be an absolute normalized path")
    return path


def _plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    return value


@dataclass(frozen=True)
class RuntimeContract:
    tokenizer_json: Path
    chat_template: Path
    context_window: int
    max_tokens: int
    port: int
    enable_tools: bool
    runner_socket: Path | None = None
    runner_command: tuple[str, ...] | None = None

    @classmethod
    def load(cls, path: Path) -> "RuntimeContract":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractError(
                f"runtime config not found: {path}; copy config/runtime.example.json "
                "to config/runtime.json and edit the absolute paths") from exc
        except json.JSONDecodeError as exc:
            raise ContractError(f"runtime config is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ContractError("runtime config must be a JSON object")
        allowed = {
            "schema", "tokenizer_json", "chat_template", "context_window",
            "max_tokens", "host", "port", "enable_tools", "runner",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ContractError(f"unknown runtime keys: {sorted(unknown)}")
        if raw.get("schema") != RUNTIME_SCHEMA:
            raise ContractError(f"schema must be {RUNTIME_SCHEMA!r}")
        if raw.get("host", LOOPBACK_HOST) != LOOPBACK_HOST:
            raise ContractError(
                "host must remain 127.0.0.1; use an SSH tunnel for remote access")
        context_window = _plain_int(raw.get("context_window"), "context_window")
        max_tokens = _plain_int(raw.get("max_tokens"), "max_tokens")
        port = _plain_int(raw.get("port", 8000), "port")
        if context_window < 4096:
            raise ContractError("OpenClaw requires context_window >= 4096")
        if not 1 <= max_tokens <= context_window:
            raise ContractError("max_tokens must be in 1..context_window")
        if not 1 <= port <= 65535:
            raise ContractError("port must be in 1..65535")
        enable_tools = raw.get("enable_tools", False)
        if not isinstance(enable_tools, bool):
            raise ContractError("enable_tools must be a boolean")

        runner = raw.get("runner")
        if not isinstance(runner, dict) or set(runner) not in ({"socket"}, {"command"}):
            raise ContractError(
                "runner must contain exactly one of socket or command")
        runner_socket: Path | None = None
        runner_command: tuple[str, ...] | None = None
        if "socket" in runner:
            runner_socket = _absolute_path(runner["socket"], "runner.socket")
        else:
            command = runner["command"]
            if (not isinstance(command, list) or not command or
                    any(not isinstance(part, str) or not part for part in command)):
                raise ContractError("runner.command must be a non-empty string array")
            executable = _absolute_path(command[0], "runner.command[0]")
            runner_command = (str(executable), *command[1:])

        return cls(
            tokenizer_json=_absolute_path(raw.get("tokenizer_json"), "tokenizer_json"),
            chat_template=_absolute_path(raw.get("chat_template"), "chat_template"),
            context_window=context_window,
            max_tokens=max_tokens,
            port=port,
            enable_tools=enable_tools,
            runner_socket=runner_socket,
            runner_command=runner_command,
        )

    @property
    def base_url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}"

    def service_argv(self, python: str = sys.executable) -> list[str]:
        argv = [
            python, str(SERVICE_SCRIPT),
            "--tokenizer-json", str(self.tokenizer_json),
            "--chat-template", str(self.chat_template),
            "--context-window", str(self.context_window),
            "--max-tokens", str(self.max_tokens),
            "--host", LOOPBACK_HOST,
            "--port", str(self.port),
        ]
        if self.enable_tools:
            argv.append("--enable-tools")
        if self.runner_socket is not None:
            argv.extend(["--runner-socket", str(self.runner_socket)])
        else:
            assert self.runner_command is not None
            argv.append("--runner-command")
            argv.extend(self.runner_command)
        return argv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _state_lock(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state_dir, 0o700)
    lock_path = state_dir / "lifecycle.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        os.chmod(lock_path, 0o600)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_command(pid: int) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="], check=False,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def _owned_service(pid: int) -> bool:
    command = _process_command(pid)
    return bool(command) and str(SERVICE_SCRIPT) in command


def _load_state(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ContractError(f"state file is corrupt: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"state file must contain an object: {path}")
    return value


def _health(contract: RuntimeContract, timeout: float = 2.0) -> dict[str, object]:
    with urlopen(f"{contract.base_url}/healthz", timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("health endpoint returned a non-object")
    return value


def _health_matches(contract: RuntimeContract, health: Mapping[str, object]) -> bool:
    return (
        health.get("status") == "ok" and
        health.get("model") == "minicpm5-1b" and
        health.get("context_window") == contract.context_window and
        health.get("supportsTools") is contract.enable_tools
    )


def _wait_health(contract: RuntimeContract, pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return False
        try:
            if _health_matches(contract, _health(contract)):
                return True
        except (OSError, URLError, RuntimeError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    return False


def _preflight(contract: RuntimeContract) -> None:
    for path, label in (
            (contract.tokenizer_json, "tokenizer_json"),
            (contract.chat_template, "chat_template")):
        if not path.is_file():
            raise ContractError(f"{label} is not a regular file: {path}")
    if contract.runner_socket is not None:
        try:
            socket_ok = stat.S_ISSOCK(contract.runner_socket.stat().st_mode)
        except FileNotFoundError:
            socket_ok = False
        if not socket_ok:
            raise ContractError(
                f"runner socket is absent or not a socket: {contract.runner_socket}")
    else:
        assert contract.runner_command is not None
        executable = Path(contract.runner_command[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ContractError(f"runner executable is not executable: {executable}")
        _preflight_merged_runner(contract)
    for module in ("jinja2", "tokenizers"):
        if importlib.util.find_spec(module) is None:
            raise ContractError(f"missing Python dependency: {module}")


def _preflight_merged_runner(contract: RuntimeContract) -> None:
    """Fail closed on a packaged merged-runner command before HTTP startup."""
    command = contract.runner_command
    if command is None:
        return
    script_indices = [
        index for index, part in enumerate(command)
        if Path(part).name == "merged_jsonl_runner.py"
    ]
    if not script_indices:
        return
    if script_indices != [1]:
        raise ContractError(
            "merged_jsonl_runner.py must be command[1] after an explicit Python")
    configured_script = Path(command[1]).resolve()
    packaged_script = Path(__file__).resolve().with_name("merged_jsonl_runner.py")
    if configured_script != packaged_script:
        raise ContractError(
            "runner command must use this package's merged_jsonl_runner.py: "
            f"{packaged_script}")
    try:
        args = build_merged_runner_parser().parse_args(list(command[2:]))
    except SystemExit as exc:
        raise ContractError("merged JSONL runner arguments are invalid") from exc
    packaged_runtime = Path(__file__).resolve().with_name("merged_board_server.py")
    if Path(args.runtime_module).resolve() != packaged_runtime:
        raise ContractError(
            "--runtime-module must use this package's merged_board_server.py: "
            f"{packaged_runtime}; do not use the legacy board REPL")
    packaged_source = Path(__file__).resolve().parent
    if packaged_source not in {Path(path).resolve() for path in args.python_path}:
        raise ContractError(
            "--python-path must include this package's src directory: "
            f"{packaged_source}")
    if args.context != contract.context_window:
        raise ContractError(
            "merged runner --context must equal service context_window")
    if args.max_new_limit < contract.max_tokens:
        raise ContractError(
            "merged runner --max-new-limit is smaller than service max_tokens")
    required_files = {
        "runtime module": args.runtime_module,
        "persistent executor": args.persistent_executor,
        "decode OM": args.decode_model,
        "prefill OM": args.prefill_model,
        "head OM": args.head_model,
        "embedding": args.embedding,
    }
    if args.decode_short_model is not None:
        required_files["short decode OM"] = args.decode_short_model
    for label, path in required_files.items():
        if not Path(path).is_file():
            raise ContractError(f"{label} is not a regular file: {path}")
    if not os.access(args.persistent_executor, os.X_OK):
        raise ContractError(
            f"persistent executor is not executable: {args.persistent_executor}")
    for label, paths in (
            ("python path", args.python_path),
            ("library path", args.library_path)):
        for path in paths:
            if not Path(path).is_dir():
                raise ContractError(f"{label} is not a directory: {path}")
    if args.characterize_decode_workspace_zero_once:
        if (args.context != 8192 or not args.decode_no_cache or
                args.executor_uncached or args.decode_short_model is not None):
            raise ContractError(
                "zero-once requires C8192, decode-no-cache, no global no-cache "
                "and no short decode model")


def _terminate_owned(pid: int, seconds: float = 40.0) -> None:
    if not _pid_alive(pid):
        return
    if not _owned_service(pid):
        raise ContractError(
            f"refusing to signal PID {pid}: it is not this package's service")
    # SIGINT lets the Python service leave serve_forever through its
    # KeyboardInterrupt handler.  Its finally block then closes runner stdin,
    # waits for JSONL EOF cleanup, and gives the native runtime a chance to
    # release resident MMZ before any forced process-group signal.
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    if not _owned_service(pid):
        raise ContractError(
            f"service PID {pid} changed identity during graceful shutdown")
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        return
    target_group = process_group == pid
    if target_group:
        os.killpg(process_group, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)
    forced_deadline = time.monotonic() + 5.0
    while time.monotonic() < forced_deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    if target_group:
        os.killpg(process_group, signal.SIGKILL)
    else:
        os.kill(pid, signal.SIGKILL)


def start_service(
        config_path: Path, state_dir: Path,
        wait_seconds: float = DEFAULT_WAIT_SECONDS) -> dict[str, object]:
    contract = RuntimeContract.load(config_path)
    _preflight(contract)
    state_path = state_dir / "service.json"
    with _state_lock(state_dir):
        state = _load_state(state_path)
        if state is not None:
            pid = state.get("pid")
            if isinstance(pid, int) and _pid_alive(pid):
                if not _owned_service(pid):
                    raise ContractError(
                        f"PID file points to an unrelated process: {pid}")
                health = _health(contract)
                if not _health_matches(contract, health):
                    raise ContractError(
                        "a service is already running but its runtime contract differs")
                return {"status": "already_running", "pid": pid, "health": health}
            state_path.unlink(missing_ok=True)

        try:
            foreign_health = _health(contract, timeout=0.5)
        except (OSError, URLError, RuntimeError, json.JSONDecodeError):
            pass
        else:
            raise ContractError(
                "the configured HTTP endpoint already responds without an owned "
                f"service state: {json.dumps(foreign_health, sort_keys=True)}")

        log_path = state_dir / "service.log"
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                contract.service_argv(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True,
                close_fds=True)
        state_payload: dict[str, object] = {
            "schema": "pico.minicpm5.openclaw-service-state.v1",
            "pid": process.pid,
            "service_script": str(SERVICE_SCRIPT),
            "config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "started_unix": int(time.time()),
            "base_url": contract.base_url,
            "log": str(log_path.resolve()),
        }
        _atomic_json(state_path, state_payload)
        if not _wait_health(contract, process.pid, wait_seconds):
            try:
                _terminate_owned(process.pid, seconds=3)
            finally:
                state_path.unlink(missing_ok=True)
            tail = ""
            try:
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
            except OSError:
                pass
            raise RuntimeError(
                f"service did not become healthy within {wait_seconds:g}s"
                + (f"; log tail:\n{tail}" if tail else ""))
        health = _health(contract)
        return {"status": "started", "pid": process.pid, "health": health,
                "log": str(log_path.resolve())}


def stop_service(state_dir: Path) -> dict[str, object]:
    state_path = state_dir / "service.json"
    with _state_lock(state_dir):
        state = _load_state(state_path)
        if state is None:
            return {"status": "already_stopped"}
        pid = state.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            raise ContractError("state file has an invalid PID")
        if _pid_alive(pid):
            _terminate_owned(pid)
        state_path.unlink(missing_ok=True)
        return {"status": "stopped", "pid": pid}


def configure_openclaw(
        contract: RuntimeContract, profile: str, home: Path,
        force: bool = False) -> dict[str, object]:
    if PROFILE_RE.fullmatch(profile) is None:
        raise ContractError("profile must match [A-Za-z0-9_-]{1,64}")
    profile_dir = home / f".openclaw-{profile}"
    config_path = profile_dir / "openclaw.json"
    manifest_path = profile_dir / "pico-minicpm5.manifest.json"
    if not force and (config_path.exists() or manifest_path.exists()):
        raise ContractError(
            f"profile files already exist in {profile_dir}; pass --force to replace")
    generated = OpenClawMiniCPM5Contract(
        context_window=contract.context_window,
        max_tokens=contract.max_tokens,
        port=contract.port,
        timeout_seconds=3600,
        supports_tools=contract.enable_tools,
    )
    _atomic_json(config_path, generated.openclaw_config())
    _atomic_json(manifest_path, generated.manifest())
    return {
        "status": "configured",
        "profile": profile,
        "config": str(config_path),
        "manifest": str(manifest_path),
        "supports_tools": contract.enable_tools,
    }


def doctor(
        config_path: Path, state_dir: Path, profile: str | None,
        require_running: bool = False) -> tuple[dict[str, object], bool]:
    checks: list[dict[str, object]] = []

    def check(name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({
            "name": name, "ok": bool(ok), "required": required,
            "detail": detail,
        })

    try:
        contract = RuntimeContract.load(config_path)
        check("runtime_config", True, str(config_path.resolve()))
    except (OSError, ContractError) as exc:
        check("runtime_config", False, str(exc))
        return {"status": "failed", "checks": checks}, False

    for path, name in (
            (contract.tokenizer_json, "tokenizer_json"),
            (contract.chat_template, "chat_template")):
        check(name, path.is_file(), str(path))
    for module in ("jinja2", "tokenizers"):
        check(f"python_dependency:{module}",
              importlib.util.find_spec(module) is not None, module)
    if contract.runner_socket is not None:
        try:
            socket_ok = stat.S_ISSOCK(contract.runner_socket.stat().st_mode)
        except FileNotFoundError:
            socket_ok = False
        check("runner_socket", socket_ok, str(contract.runner_socket))
    else:
        assert contract.runner_command is not None
        executable = Path(contract.runner_command[0])
        check("runner_executable",
              executable.is_file() and os.access(executable, os.X_OK),
              str(executable))
        try:
            _preflight_merged_runner(contract)
            check("merged_runner_contract", True, "validated")
        except ContractError as exc:
            check("merged_runner_contract", False, str(exc))

    state = _load_state(state_dir / "service.json")
    running = False
    if state is not None and isinstance(state.get("pid"), int):
        pid = int(state["pid"])
        running = _pid_alive(pid) and _owned_service(pid)
        check("service_process", running, f"pid={pid}", required=require_running)
    else:
        check("service_process", not require_running, "not started",
              required=require_running)
    if running:
        try:
            health = _health(contract)
            check("service_health", _health_matches(contract, health),
                  json.dumps(health, sort_keys=True))
        except Exception as exc:  # doctor must report, not obscure, diagnostics
            check("service_health", False, str(exc))

    if profile is not None:
        if PROFILE_RE.fullmatch(profile) is None:
            check("openclaw_profile", False, "invalid profile name")
        else:
            home = Path.home()
            profile_config = home / f".openclaw-{profile}" / "openclaw.json"
            check("openclaw_profile", profile_config.is_file(), str(profile_config))
            executable = _find_openclaw()
            check("openclaw_binary", executable is not None,
                  str(executable) if executable else "not found")
            if executable is not None:
                try:
                    version = subprocess.run(
                        [str(executable), "--version"], check=False, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=15)
                    version_text = version.stdout.strip()
                    check("openclaw_version", version.returncode == 0,
                          version_text or f"exit={version.returncode}")
                    check("openclaw_version_pin",
                          OPENCLAW_VERSION in version_text,
                          f"expected {OPENCLAW_VERSION}; actual {version_text}",
                          required=False)
                    validation = subprocess.run(
                        [str(executable), "--profile", profile, "config",
                         "validate", "--json"],
                        check=False, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=30)
                    validation_text = validation.stdout.strip()
                    validation_ok = validation.returncode == 0
                    if validation_ok:
                        try:
                            validation_json = json.loads(validation_text)
                            validation_ok = validation_json.get("valid") is True
                        except json.JSONDecodeError:
                            validation_ok = False
                    check("openclaw_config_validate", validation_ok,
                          validation_text or f"exit={validation.returncode}")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    check("openclaw_cli", False, str(exc))

    ok = all(item["ok"] for item in checks if item["required"])
    return {"status": "ok" if ok else "failed", "checks": checks}, ok


def _default_state_dir() -> Path:
    override = os.environ.get("PICO_OPENCLAW_STATE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "pico-minicpm5-openclaw"
    return Path.home() / ".local" / "state" / "pico-minicpm5-openclaw"


def _find_openclaw() -> Path | None:
    override = os.environ.get("PICO_OPENCLAW_BIN")
    candidates = (
        Path(override).expanduser() if override else None,
        Path(value) if (value := shutil.which("openclaw")) else None,
        Path("/opt/openclaw/bin/openclaw"),
    )
    for candidate in candidates:
        if (candidate is not None and candidate.is_file() and
                os.access(candidate, os.X_OK)):
            return candidate.resolve()
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=APP_ROOT / "config" / "runtime.json")
    parser.add_argument("--state-dir", type=Path, default=_default_state_dir())
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--wait-seconds", type=float, default=DEFAULT_WAIT_SECONDS)
    subparsers.add_parser("stop")
    configure = subparsers.add_parser("configure")
    configure.add_argument("--profile", default=DEFAULT_PROFILE)
    configure.add_argument("--home", type=Path, default=Path.home())
    configure.add_argument("--force", action="store_true")
    diagnose = subparsers.add_parser("doctor")
    diagnose.add_argument("--profile")
    diagnose.add_argument("--require-running", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "start":
            if args.wait_seconds <= 0:
                raise ContractError("--wait-seconds must be positive")
            result = start_service(args.config, args.state_dir, args.wait_seconds)
            ok = True
        elif args.command == "stop":
            result = stop_service(args.state_dir)
            ok = True
        elif args.command == "configure":
            contract = RuntimeContract.load(args.config)
            result = configure_openclaw(
                contract, args.profile, args.home.expanduser(), args.force)
            ok = True
        else:
            result, ok = doctor(
                args.config, args.state_dir, args.profile,
                args.require_running)
    except (ContractError, OSError, RuntimeError, URLError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
