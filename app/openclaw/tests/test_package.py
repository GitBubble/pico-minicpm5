# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


APP_ROOT = Path(__file__).resolve().parents[1]
SRC = APP_ROOT / "src"
sys.path.insert(0, str(SRC))

import config_generator  # noqa: E402
import lifecycle  # noqa: E402
import merged_jsonl_runner  # noqa: E402


class RuntimeContractTests(unittest.TestCase):
    def _write_config(self, root: Path, **updates: object) -> Path:
        tokenizer = root / "tokenizer.json"
        template = root / "chat_template.jinja"
        runner = root / "runner"
        tokenizer.write_text("fixture", encoding="utf-8")
        template.write_text("fixture", encoding="utf-8")
        runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        runner.chmod(0o700)
        payload: dict[str, object] = {
            "schema": lifecycle.RUNTIME_SCHEMA,
            "tokenizer_json": str(tokenizer),
            "chat_template": str(template),
            "context_window": 8192,
            "max_tokens": 128,
            "host": "127.0.0.1",
            "port": 18080,
            "enable_tools": False,
            "runner": {"command": [str(runner), "--literal", "a;$(id)"]},
        }
        payload.update(updates)
        path = root / "runtime.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_command_is_an_argv_and_never_shell_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory))
            contract = lifecycle.RuntimeContract.load(path)
            argv = contract.service_argv("/usr/bin/python3")
        marker = argv.index("--runner-command")
        self.assertEqual(argv[marker + 1 :], [
            str(Path(directory) / "runner"), "--literal", "a;$(id)"])
        self.assertNotIn("shell", argv)

    def test_rejects_non_loopback_bind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory), host="0.0.0.0")
            with self.assertRaisesRegex(lifecycle.ContractError, "127.0.0.1"):
                lifecycle.RuntimeContract.load(path)

    def test_rejects_context_smaller_than_openclaw_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(Path(directory), context_window=1024)
            with self.assertRaisesRegex(lifecycle.ContractError, ">= 4096"):
                lifecycle.RuntimeContract.load(path)

    def test_rejects_unknown_keys_and_ambiguous_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_config(root, surprise=True)
            with self.assertRaisesRegex(lifecycle.ContractError, "unknown"):
                lifecycle.RuntimeContract.load(path)
            path = self._write_config(root, runner={
                "socket": "/tmp/runner.sock", "command": ["/bin/false"]})
            with self.assertRaisesRegex(lifecycle.ContractError, "exactly one"):
                lifecycle.RuntimeContract.load(path)

    def test_configure_writes_private_isolated_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = lifecycle.RuntimeContract.load(self._write_config(root))
            result = lifecycle.configure_openclaw(
                contract, "pico-test", root / "home")
            config_path = Path(str(result["config"]))
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            provider = payload["models"]["providers"]["pico-minicpm"]
            self.assertEqual(provider["baseUrl"], "http://127.0.0.1:18080/v1")
            self.assertNotIn("tools", payload)
            self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(lifecycle.ContractError, "already exist"):
                lifecycle.configure_openclaw(
                    contract, "pico-test", root / "home")

    def test_c8192_command_example_uses_packaged_zero_once_runner(self) -> None:
        example = APP_ROOT / "config" / "runtime.command.example.json"
        contract = lifecycle.RuntimeContract.load(example)
        self.assertEqual(contract.context_window, 8192)
        self.assertEqual(contract.max_tokens, 128)
        self.assertIsNotNone(contract.runner_command)
        command = list(contract.runner_command or ())
        self.assertEqual(
            command[1],
            "/opt/pico-minicpm5/app/openclaw/src/merged_jsonl_runner.py")
        args = merged_jsonl_runner.build_parser().parse_args(command[2:])
        self.assertEqual(args.context, 8192)
        self.assertEqual(args.max_new_limit, 128)
        self.assertEqual(
            {args.decode_model.name, args.prefill_model.name, args.head_model.name},
            {"decode.ctx8192.om", "prefill.ctx1024.om", "head.om"})
        self.assertEqual(
            [str(path) for path in args.library_path],
            ["/root/pico_default_smoke/lib", "/opt/lib/npu"])
        self.assertEqual(args.embedding.name, "token_embedding.f16.bin")
        self.assertTrue(args.decode_no_cache)
        self.assertTrue(args.characterize_decode_workspace_zero_once)
        self.assertFalse(args.executor_uncached)
        self.assertIsNone(args.decode_short_model)

    def test_packaged_merged_runner_preflight_binds_runtime_files(self) -> None:
        example = lifecycle.RuntimeContract.load(
            APP_ROOT / "config" / "runtime.command.example.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "--persistent-executor": root / "executor",
                "--decode-model": root / "decode.om",
                "--prefill-model": root / "prefill.om",
                "--head-model": root / "head.om",
                "--embedding": root / "embedding.bin",
            }
            for path in files.values():
                path.write_bytes(b"fixture")
            files["--persistent-executor"].chmod(0o700)
            library_path = root / "lib"
            library_path.mkdir()
            command = list(example.runner_command or ())
            command[0] = sys.executable
            command[1] = str(SRC / "merged_jsonl_runner.py")
            command[command.index("--runtime-module") + 1] = str(
                SRC / "merged_board_server.py")
            for flag, path in files.items():
                command[command.index(flag) + 1] = str(path)
            command[command.index("--python-path") + 1] = str(SRC)
            for index, part in enumerate(command):
                if part == "--library-path":
                    command[index + 1] = str(library_path)
            bound = replace(example, runner_command=tuple(command))
            lifecycle._preflight_merged_runner(bound)
            with self.assertRaisesRegex(
                    lifecycle.ContractError, "must equal service"):
                lifecycle._preflight_merged_runner(
                    replace(bound, context_window=4096))

    def test_stop_prefers_graceful_service_interrupt(self) -> None:
        with (
            mock.patch.object(lifecycle, "_pid_alive", side_effect=[True, False]),
            mock.patch.object(lifecycle, "_owned_service", return_value=True),
            mock.patch.object(lifecycle.os, "kill") as kill,
        ):
            lifecycle._terminate_owned(1234, seconds=1)
        kill.assert_called_once_with(1234, lifecycle.signal.SIGINT)


class PackageIntegrityTests(unittest.TestCase):
    def test_monorepo_snapshots_match_when_source_tree_is_present(self) -> None:
        root = APP_ROOT.parents[3]
        sources = {
            "src/openai_service.py":
                root / "scripts" / "pico_minicpm5_openai_service.py",
            "src/config_generator.py":
                root / "scripts" / "pico_openclaw_minicpm5_config.py",
            "src/merged_jsonl_runner.py":
                root / "scripts" / "pico_minicpm5_merged_jsonl_runner.py",
            "src/merged_board_server.py":
                root / "scratchpad" / "minicpm5" / "merged24" /
                "merged_board_server.py",
            "src/pico_minicpm5_split_board_runner.py":
                root / "scripts" / "pico_minicpm5_split_board_runner.py",
            "src/probe_om_execute_latency.py":
                root / "scripts" / "probe_om_execute_latency.py",
            "src/qualify_minicpm_greedy_chain.py":
                root / "scripts" / "qualify_minicpm_greedy_chain.py",
        }
        if not all(path.is_file() for path in sources.values()):
            self.skipTest("standalone checkout has no monorepo source snapshots")
        for packaged, source in sources.items():
            self.assertEqual(
                (APP_ROOT / packaged).read_bytes(), source.read_bytes(), packaged)

    def test_source_snapshot_hashes_are_current(self) -> None:
        manifest = json.loads(
            (APP_ROOT / "SOURCE_SNAPSHOTS.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema"],
            "pico.minicpm5.openclaw-source-snapshots.v1")
        self.assertEqual(len(manifest["files"]), 7)
        for relative, metadata in manifest["files"].items():
            digest = hashlib.sha256((APP_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, metadata["sha256"], relative)

    def test_python_sources_compile(self) -> None:
        for source in sorted(SRC.glob("*.py")):
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(source)],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_shell_entrypoints_parse(self) -> None:
        for script in sorted((APP_ROOT / "bin").glob("*.sh")):
            result = subprocess.run(
                ["sh", "-n", str(script)], check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, f"{script}: {result.stderr}")

    def test_readme_bash_examples_parse(self) -> None:
        readme = (APP_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)\n```", readme, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 20)
        for index, block in enumerate(blocks):
            result = subprocess.run(
                ["bash", "-n"], input=block, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(
                result.returncode, 0,
                f"README bash block {index + 1}: {result.stderr}")

    def test_package_contains_no_model_or_sdk_binary(self) -> None:
        forbidden_suffixes = {
            ".om", ".so", ".a", ".bin", ".safetensors", ".onnx", ".tar",
            ".tgz", ".gz", ".zip",
        }
        offenders = [
            str(path.relative_to(APP_ROOT))
            for path in APP_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden_suffixes
        ]
        self.assertEqual(offenders, [])

    def test_source_snapshots_expose_entrypoints(self) -> None:
        service_spec = importlib.util.spec_from_file_location(
            "packaged_service", SRC / "openai_service.py")
        generator_spec = importlib.util.spec_from_file_location(
            "packaged_generator", SRC / "config_generator.py")
        self.assertIsNotNone(service_spec)
        self.assertIsNotNone(generator_spec)
        self.assertTrue(callable(config_generator.main))
        self.assertTrue(callable(lifecycle.main))

    def test_bundled_native_runtime_imports_with_exact_dependencies(self) -> None:
        module = merged_jsonl_runner._load_runtime_module(
            SRC / "merged_board_server.py", [SRC])
        self.assertTrue(callable(module.Merged))


if __name__ == "__main__":
    unittest.main()
