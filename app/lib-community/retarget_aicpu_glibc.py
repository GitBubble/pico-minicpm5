#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retarget Pegasus libsvp_aicpu.so fmod@GLIBC_2.38 → fmod@GLIBC_2.17.

Ubuntu 22.04 Jammy is glibc 2.35. The Pegasus gcc aicpu records one symbol
(fmod) against libm's GLIBC_2.38. The fmod ABI on AArch64 is the 2.17
baseline, so rewriting that one VERNAUX (dynstr name *and* ELF hash) lets
ld.so bind it. A name-only patch is not enough: glibc matches both fields.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

SRC_VER = b"GLIBC_2.38"
DST_VER = b"GLIBC_2.17"


def elf_hash(name: bytes) -> int:
    h = 0
    for byte in name:
        h = ((h << 4) + byte) & 0xFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g
        h &= 0xFFFFFFFF
    return h


def parse_sections(data: bytes):
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise SystemExit("not ELF64")
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]

    def shdr(index: int):
        off = e_shoff + index * e_shentsize
        return {
            "name": struct.unpack_from("<I", data, off)[0],
            "offset": struct.unpack_from("<Q", data, off + 24)[0],
            "size": struct.unpack_from("<Q", data, off + 32)[0],
        }

    shstr = shdr(e_shstrndx)
    names = data[shstr["offset"] : shstr["offset"] + shstr["size"]]
    sections = {}
    for index in range(e_shnum):
        section = shdr(index)
        name = names[section["name"] :].split(b"\0", 1)[0].decode()
        section["n"] = name
        sections[name] = section
    return sections


def iter_vernaux(data: bytes, sections):
    vr = sections[".gnu.version_r"]
    dynstr = sections[".dynstr"]
    strtab_off = dynstr["offset"]
    pos = vr["offset"]
    end = pos + vr["size"]
    while pos + 16 <= end:
        vn_version, vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from(
            "<HHIII", data, pos
        )
        file_name = data[strtab_off + vn_file :].split(b"\0", 1)[0]
        apos = pos + vn_aux
        for _ in range(vn_cnt):
            vna_hash, vna_flags, vna_other, vna_name, vna_next = struct.unpack_from(
                "<IHHII", data, apos
            )
            name = data[strtab_off + vna_name :].split(b"\0", 1)[0]
            yield {
                "file": file_name,
                "aux": apos,
                "hash": vna_hash,
                "flags": vna_flags,
                "other": vna_other,
                "name_off": strtab_off + vna_name,
                "name": name,
            }
            if vna_next == 0:
                break
            apos += vna_next
        if vn_next == 0:
            break
        pos += vn_next


def retarget(src: Path, dst: Path, check_only: bool) -> int:
    raw = src.read_bytes()
    sections = parse_sections(raw)
    matches = [
        entry
        for entry in iter_vernaux(raw, sections)
        if entry["name"] == SRC_VER
    ]
    if len(matches) != 1:
        print(
            f"expected exactly one {SRC_VER.decode()} VERNAUX, got {len(matches)}",
            file=sys.stderr,
        )
        for entry in matches:
            print(
                f"  {entry['file']!r} other={entry['other']} hash={entry['hash']:#x}",
                file=sys.stderr,
            )
        return 2
    entry = matches[0]
    expected_hash = elf_hash(SRC_VER)
    if entry["hash"] != expected_hash:
        print(
            f"VERNAUX hash {entry['hash']:#x} != elf_hash({SRC_VER}) {expected_hash:#x}",
            file=sys.stderr,
        )
        return 2
    if entry["file"] != b"libm.so.6":
        print(f"expected libm.so.6, got {entry['file']!r}", file=sys.stderr)
        return 2
    if check_only:
        print(
            f"ok: {src} has {SRC_VER.decode()} on {entry['file'].decode()} "
            f"other={entry['other']} hash={entry['hash']:#x}"
        )
        return 0
    if len(SRC_VER) != len(DST_VER):
        raise SystemExit("version strings must be the same length")
    out = bytearray(raw)
    new_hash = elf_hash(DST_VER)
    struct.pack_into("<I", out, entry["aux"], new_hash)
    name_off = entry["name_off"]
    out[name_off : name_off + len(DST_VER)] = DST_VER
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    verify = list(iter_vernaux(bytes(out), parse_sections(bytes(out))))
    patched = [item for item in verify if item["aux"] == entry["aux"]][0]
    if patched["name"] != DST_VER or patched["hash"] != new_hash:
        print("post-patch verify failed", file=sys.stderr)
        return 2
    leftover = [item for item in verify if item["name"] == SRC_VER]
    if leftover:
        print(f"still has {SRC_VER!r}", file=sys.stderr)
        return 2
    print(
        f"retargeted {src} -> {dst}: {SRC_VER.decode()}@{entry['file'].decode()} "
        f"other={entry['other']} hash {entry['hash']:#x}->{new_hash:#x}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    dst = args.dst or args.src
    return retarget(args.src, dst, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
