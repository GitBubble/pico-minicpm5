#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# One-shot Euler Pi board install: NPU bring-up, persist it past reboot,
# and print the board identity on interactive SSH login.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INIT_LINK=${PICO_NPU_INIT:-/etc/init.d/S91pico_npu}
PROFILE=${PICO_LOGIN_PROFILE:-/root/.profile}
USB_IPV4=${PICO_USB_IPV4:-}

usage() {
  cat <<'EOF'
Usage: install_board.sh [--usb-ipv4 ADDR/PREFIX]

  Unload factory ot_pqp, load ot_svp_npu, persist that swap after the
  Euler Pi S90autorun hook, and print the board environment on SSH login.

  --usb-ipv4 192.168.137.100/24
      Also keep a USB-ethernet address on eth0 (Windows ICS / Mac 192.168.137.1).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --usb-ipv4)
      USB_IPV4=${2:?--usb-ipv4 needs ADDR/PREFIX}
      shift 2
      ;;
    --usb-ipv4=*)
      USB_IPV4=${1#--usb-ipv4=}
      shift
      ;;
    *)
      echo "install_board: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "install_board: must run as root on the board" >&2
  exit 1
fi

chmod +x "$APP_DIR/prepare_npu.sh" "$APP_DIR/board_env.sh" \
  "$APP_DIR/chat.sh" "$APP_DIR/agent.sh" \
  "$APP_DIR/bin/pico_persistent_acl_executor.aarch64" 2>/dev/null || true

"$APP_DIR/prepare_npu.sh"

if [ -n "$USB_IPV4" ]; then
  iface=${PICO_USB_IFACE:-eth0}
  if ! ip -o -4 addr show "$iface" 2>/dev/null | grep -q " ${USB_IPV4} "; then
    ip addr add "$USB_IPV4" dev "$iface"
    echo "install_board: added $USB_IPV4 on $iface"
  fi
fi

cat > "$INIT_LINK" <<EOF
#!/bin/sh
# Installed by pico-minicpm5 app/install_board.sh
# Runs after /etc/init.d/S90autorun (load_ss928v100 -i), which inserts ot_pqp.
$APP_DIR/prepare_npu.sh
EOF
if [ -n "$USB_IPV4" ]; then
  cat >> "$INIT_LINK" <<EOF
ip addr add $USB_IPV4 dev \${PICO_USB_IFACE:-eth0} 2>/dev/null || true
EOF
fi
chmod 755 "$INIT_LINK"
echo "install_board: wrote $INIT_LINK"

marker="# pico-minicpm5 board environment"
if [ ! -f "$PROFILE" ]; then
  printf '%s\n' "#!/bin/sh" > "$PROFILE"
fi
if ! grep -q "$marker" "$PROFILE" 2>/dev/null; then
  cat >> "$PROFILE" <<EOF

$marker
if [ -x $APP_DIR/board_env.sh ]; then
  $APP_DIR/board_env.sh
fi
EOF
  echo "install_board: appended login banner to $PROFILE"
else
  echo "install_board: login banner already present in $PROFILE"
fi

echo
"$APP_DIR/board_env.sh"
echo
echo "install_board: done. Next interactive SSH login prints this environment."
