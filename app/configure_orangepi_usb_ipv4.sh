#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Host-side: DHCP-offer 192.168.138.10 on USB ethernet, SSH in, persist it.
#
#   ./app/configure_orangepi_usb_ipv4.sh
#   ./app/configure_orangepi_usb_ipv4.sh --iface en10 --board-ip 192.168.138.10
set -eu

IFACE=${PICO_USB_IFACE:-en10}
HOST_IP=${PICO_HOST_IP:-192.168.138.1}
BOARD_IP=${PICO_BOARD_IP:-192.168.138.10}
USER_NAME=${PICO_BOARD_USER:-orangepi}
PASSWORD=${PICO_BOARD_PASSWORD:-orangepi}
WAIT=${PICO_WAIT_SECONDS:-90}

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SSH_BASE='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8'

usage() {
  cat <<'EOF'
Usage: configure_orangepi_usb_ipv4.sh [--iface en10] [--board-ip 192.168.138.10]

  Host USB NIC must already be HOST/24 (default 192.168.138.1).
  Offers BOARD_IP over DHCP, waits for SSH, then installs
  app/install_orangepi_usb_ipv4.sh so the address returns on reboot.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --iface) IFACE=${2:?}; shift 2 ;;
    --board-ip) BOARD_IP=${2:?}; shift 2 ;;
    --board-ip=*) BOARD_IP=${1#--board-ip=}; shift ;;
    --host-ip) HOST_IP=${2:?}; shift 2 ;;
    --user) USER_NAME=${2:?}; shift 2 ;;
    --wait) WAIT=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! ifconfig "$IFACE" | grep -q 'status: active'; then
  echo "configure: $IFACE is not active" >&2
  ifconfig "$IFACE" >&2 || true
  exit 1
fi
if ! ifconfig "$IFACE" | grep -q "inet $HOST_IP"; then
  echo "configure: $IFACE lacks $HOST_IP (set it with networksetup -setmanual)" >&2
  ifconfig "$IFACE" >&2 || true
  exit 1
fi

echo "configure: DHCP on $IFACE offering $BOARD_IP (host $HOST_IP)"
python3 "$APP_DIR/dhcp_offer_usb.py" \
  --iface "$IFACE" --server "$HOST_IP" --offer "$BOARD_IP" --seconds "$WAIT" &
DHCP_PID=$!
trap 'kill "$DHCP_PID" 2>/dev/null || true' EXIT

echo "configure: waiting ${WAIT}s for $USER_NAME@$BOARD_IP"
ok=0
i=0
while [ "$i" -lt "$WAIT" ]; do
  if ping -c 1 -W 1 "$BOARD_IP" >/dev/null 2>&1; then
    if ssh $SSH_BASE -o BatchMode=yes "$USER_NAME@$BOARD_IP" 'echo SSH_OK' >/dev/null 2>&1; then
      ok=1
      break
    fi
    if command -v sshpass >/dev/null 2>&1; then
      if sshpass -p "$PASSWORD" ssh $SSH_BASE \
          -o PreferredAuthentications=password -o PubkeyAuthentication=no \
          "$USER_NAME@$BOARD_IP" 'echo SSH_OK' >/dev/null 2>&1; then
        ok=1
        break
      fi
    fi
  fi
  i=$((i + 1))
  sleep 1
done

if [ "$ok" -ne 1 ]; then
  echo "configure: no SSH on $BOARD_IP after ${WAIT}s" >&2
  echo "configure: $IFACE link is up but the board may not be sending DHCP." >&2
  echo "configure: on the HDMI console run:" >&2
  echo "  sudo ip link set eth0 up; sudo ip addr add $BOARD_IP/24 dev eth0" >&2
  echo "  (or end0 instead of eth0), then re-run this script." >&2
  exit 1
fi

echo "configure: SSH ok, installing persistent $BOARD_IP"
remote() {
  if ssh $SSH_BASE -o BatchMode=yes "$USER_NAME@$BOARD_IP" "$@"; then
    return 0
  fi
  if command -v sshpass >/dev/null 2>&1; then
    sshpass -p "$PASSWORD" ssh $SSH_BASE \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      "$USER_NAME@$BOARD_IP" "$@"
  else
    return 1
  fi
}

# Copy installer without sudo-on-stdin conflict: scp then sudo -S.
scp $SSH_BASE "$APP_DIR/install_orangepi_usb_ipv4.sh" \
  "$USER_NAME@$BOARD_IP:/tmp/install_orangepi_usb_ipv4.sh"
remote "chmod +x /tmp/install_orangepi_usb_ipv4.sh && echo $PASSWORD | sudo -S env PICO_USB_IPV4=$BOARD_IP/24 /tmp/install_orangepi_usb_ipv4.sh"

echo "configure: persistent USB IPv4 installed"
remote "ip -br a; systemctl is-enabled pico-usb-ipv4.service; echo PERSIST_OK"
