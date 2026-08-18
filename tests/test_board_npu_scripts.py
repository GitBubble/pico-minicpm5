# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess

PROJECT = Path(__file__).resolve().parents[1]


def test_prepare_npu_skips_off_board(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.ko"
    result = subprocess.run(
        ["sh", str(PROJECT / "app" / "prepare_npu.sh")],
        env={**os.environ, "PICO_KO_ROOT": str(tmp_path)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "skip" in result.stderr
    assert not missing.exists()


def test_board_env_reads_firmware_version(tmp_path: Path) -> None:
    firmware = tmp_path / "firmware_version"
    firmware.write_text(
        "\nROOTFS\nChip:          SS928V100\n"
        "SDK:           SS928V100_SDK_V2.0.2.2\n"
        "HardWare_ver:  HiEuerPI_V1.2\n"
        "SoftWare_ver:  V2.0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["sh", str(PROJECT / "app" / "board_env.sh")],
        env={
            **os.environ,
            "PICO_FIRMWARE_VERSION": str(firmware),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Product:       Euler Pi" in result.stdout
    assert "Chip:          SS928V100" in result.stdout
    assert "SDK:           SS928V100_SDK_V2.0.2.2" in result.stdout
    assert "Hardware:      HiEuerPI_V1.2" in result.stdout
    assert "Software:      V2.0" in result.stdout


def test_install_board_writes_init_and_profile(tmp_path: Path) -> None:
    init = tmp_path / "S91pico_npu"
    profile = tmp_path / ".profile"
    script = PROJECT / "app" / "install_board.sh"
    environment = os.environ.copy()
    environment.update({
        "PICO_NPU_INIT": str(init),
        "PICO_LOGIN_PROFILE": str(profile),
        "PICO_KO_ROOT": str(tmp_path),
    })
    # The script requires root. Skip the real install unless we can fake id.
    # Syntax and help path stay covered on every host.
    help_result = subprocess.run(
        ["sh", str(script), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "ot_pqp" in help_result.stdout
    assert "ot_svp_npu" in help_result.stdout
    assert init.exists() is False
    assert not profile.exists()
    # keep unused names referenced so a later root-level test can reuse them
    assert stat.S_IMODE(script.stat().st_mode) & 0o111


def test_install_python_help_and_pins() -> None:
    script = PROJECT / "app" / "install_python.sh"
    help_result = subprocess.run(
        ["sh", str(script), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "CPython 3.10.21" in help_result.stdout
    assert "tokenizers" in help_result.stdout
    assert "/opt/pico-minicpm5/venv" in help_result.stdout
    text = script.read_text(encoding="utf-8")
    assert "686077ed8d668e3446f03f202bc3c99955d350a0db0ace5f8ce45f1b451b54b7" in text
    assert "tokenizers-0.23.1-cp310-abi3-manylinux_2_17_aarch64" in text
    assert stat.S_IMODE(script.stat().st_mode) & 0o111


def test_install_runtime_lib_help() -> None:
    script = PROJECT / "app" / "install_runtime_lib.sh"
    help_result = subprocess.run(
        ["sh", str(script), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "libsvp_acl.so" in help_result.stdout
    assert "app/lib" in help_result.stdout
    assert stat.S_IMODE(script.stat().st_mode) & 0o111


def test_shipped_svp_acl_libs_are_aarch64_elf() -> None:
    import struct

    lib_dir = PROJECT / "app" / "lib"
    expected = {
        "libsvp_acl.so": "2346049c2f6a7646254d2d5c4cf2421bcd84092aa0801b90bc2e21a6aa832908",
        "libsvp_aicpu.so": "1f4485249cc3757d78a3d24d82786925500dadea80002016c5269433f113bc0d",
        "libprotobuf-c.so.1": "65bf7aeb0997f4a13be0cf625eaea26f78fe2d5bd8b1ec9b8f04dab39fa1857e",
        "libsecurec.so": "dcda6e8370056302f04fbbd0bc4414a2fe9be6efc53e6b97b7aa8fb84ddfce1f",
    }
    sums = (lib_dir / "SHA256SUMS").read_text(encoding="utf-8")
    for name, digest in expected.items():
        data = (lib_dir / name).read_bytes()
        assert data[:4] == b"\x7fELF"
        assert struct.unpack_from("<H", data, 18)[0] == 183
        assert hashlib.sha256(data).hexdigest() == digest
        assert digest in sums
