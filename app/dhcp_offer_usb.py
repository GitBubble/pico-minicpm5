#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal DHCP server: offer one IPv4 on a USB ethernet iface.

Orange Pi AIfly has no persistent address after reboot and waits for DHCP.
The Mac USB NIC is 192.168.138.1; this offers 192.168.138.10.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

MAGIC = b"c\x82Sc"
IP_BOUND_IF = 25  # macOS / some BSDs


def ip4(text: str) -> bytes:
    return socket.inet_aton(text)


def parse_options(buf: bytes) -> dict[int, bytes]:
    options: dict[int, bytes] = {}
    i = 0
    while i < len(buf):
        tag = buf[i]
        if tag == 0:
            i += 1
            continue
        if tag == 255:
            break
        if i + 1 >= len(buf):
            break
        length = buf[i + 1]
        options[tag] = buf[i + 2 : i + 2 + length]
        i += 2 + length
    return options


def build_reply(
    discover: bytes,
    *,
    msg_type: int,
    yiaddr: bytes,
    server: bytes,
    mask: bytes,
    lease: int,
) -> bytes:
    xid = discover[4:8]
    flags = discover[10:12]
    chaddr = discover[28:44]
    # op=2 BOOTREPLY, htype=1, hlen=6
    pkt = bytearray(240)
    pkt[0] = 2
    pkt[1] = 1
    pkt[2] = 6
    pkt[4:8] = xid
    pkt[10:12] = flags
    pkt[16:20] = yiaddr
    pkt[20:24] = server
    pkt[28:44] = chaddr
    pkt[236:240] = MAGIC
    opts = bytearray()
    opts += bytes([53, 1, msg_type])
    opts += bytes([54, 4]) + server
    opts += bytes([1, 4]) + mask
    opts += bytes([51, 4]) + struct.pack("!I", lease)
    opts += bytes([3, 4]) + server
    opts += bytes([6, 4]) + server
    opts += bytes([255])
    return bytes(pkt) + bytes(opts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iface", default="en10")
    parser.add_argument("--server", default="192.168.138.1")
    parser.add_argument("--offer", default="192.168.138.10")
    parser.add_argument("--mask", default="255.255.255.0")
    parser.add_argument("--lease", type=int, default=86400)
    parser.add_argument("--seconds", type=int, default=0, help="0 = run until killed")
    args = parser.parse_args()

    idx = socket.if_nametoindex(args.iface)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        sock.setsockopt(socket.IPPROTO_IP, IP_BOUND_IF, idx)
    except OSError as exc:
        print(f"dhcp: IP_BOUND_IF failed: {exc}", file=sys.stderr)
    sock.bind(("0.0.0.0", 67))
    sock.settimeout(1.0)
    server = ip4(args.server)
    offer = ip4(args.offer)
    mask = ip4(args.mask)
    deadline = time.time() + args.seconds if args.seconds else None
    print(
        f"dhcp: listening on {args.iface} offering {args.offer} "
        f"via {args.server}",
        flush=True,
    )
    while True:
        if deadline is not None and time.time() >= deadline:
            return 0
        try:
            data, addr = sock.recvfrom(1500)
        except socket.timeout:
            continue
        if len(data) < 240 or data[0] != 1:
            continue
        cookie = data[236:240]
        if cookie != MAGIC:
            continue
        options = parse_options(data[240:])
        mtype = options.get(53, b"\x00")[:1]
        mac = ":".join(f"{b:02x}" for b in data[28:34])
        xid = data[4:8].hex()
        if mtype == b"\x01":
            reply = build_reply(
                data, msg_type=2, yiaddr=offer, server=server, mask=mask, lease=args.lease
            )
            sock.sendto(reply, ("255.255.255.255", 68))
            print(f"dhcp: OFFER {args.offer} to {mac} xid={xid} from {addr}", flush=True)
        elif mtype == b"\x03":
            reply = build_reply(
                data, msg_type=5, yiaddr=offer, server=server, mask=mask, lease=args.lease
            )
            sock.sendto(reply, ("255.255.255.255", 68))
            print(f"dhcp: ACK {args.offer} to {mac} xid={xid} from {addr}", flush=True)
        else:
            print(f"dhcp: ignore type={mtype!r} {mac} {addr}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
