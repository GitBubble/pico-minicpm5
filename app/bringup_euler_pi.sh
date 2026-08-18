#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Host-side Euler Pi bring-up. Discovers the board on USB ethernet, copies
# the staged deployment (including app/lib), loads SVP NPU, installs Python
# if missing, and optionally smokes chat.sh.
#
#   ./app/bringup_euler_pi.sh --stage /tmp/pico-minicpm5-board-stage --smoke
set -eu

IFACE=${PICO_USB_IFACE:-en8}
BOARD_IP=${PICO_BOARD_IP:-192.168.137.100}
PASSWORD=${PICO_BOARD_PASSWORD:-ebaina}
STAGE=""
SMOKE=0
SKIP_COPY=0
SKIP_PYTHON=0

usage() {
  cat <<'EOF'
Usage: bringup_euler_pi.sh --stage DIR [--iface en8] [--board-ip 192.168.137.100]
                           [--smoke] [--skip-copy] [--skip-python]

  Run on the host after the USB ethernet link is up. Factory login is
  root / ebaina. The stage directory must already contain app/ (with
  app/lib), models/ and assets/.

  --smoke   after install, run chat.sh --prompt '只回复 PICO_OK' --max-new 8
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --stage) STAGE=${2:?}; shift 2 ;;
    --stage=*) STAGE=${1#--stage=}; shift ;;
    --iface) IFACE=${2:?}; shift 2 ;;
    --board-ip) BOARD_IP=${2:?}; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --skip-copy) SKIP_COPY=1; shift ;;
    --skip-python) SKIP_PYTHON=1; shift ;;
    *)
      echo "bringup: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -z "$STAGE" ]; then
  echo "bringup: --stage DIR is required" >&2
  exit 2
fi
if [ ! -e "$STAGE/app/lib/libsvp_acl.so" ] || [ ! -f "$STAGE/models/prefill.om" ]; then
  echo "bringup: $STAGE is missing app/lib/libsvp_acl.so or models/prefill.om" >&2
  exit 1
fi

SSH_BASE='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8'

discover_ll() {
  # print first IPv6 neighbor on IFACE that is not us
  ndp -an | awk -v iface="$IFACE" '
    $0 ~ iface && $1 ~ /^fe80:/ {
      mac=tolower($2)
      if (mac != "" && mac != "(incomplete)") print $1
    }' | while read -r addr; do
      case "$addr" in
        *%*) echo "$addr" ;;
        *) echo "$addr%$IFACE" ;;
      esac
    done | grep -v "$(ifconfig "$IFACE" | awk '/inet6 fe80/{print $2}' | sed 's/%.*//')" | head -1
}

remote() {
  target=$1
  shift
  if ssh $SSH_BASE -o BatchMode=yes -o IdentitiesOnly=yes \
      -i "$HOME/.ssh/id_ed25519" "root@$target" "$@" 2>/dev/null; then
    return 0
  fi
  if command -v sshpass >/dev/null 2>&1; then
    sshpass -p "$PASSWORD" ssh $SSH_BASE \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      "root@$target" "$@"
  else
    echo "bringup: ssh key rejected and sshpass is not installed" >&2
    return 1
  fi
}

install_key() {
  target=$1
  pub=$(cat "$HOME/.ssh/id_ed25519.pub")
  remote "$target" "mkdir -p /root/.ssh; chmod 700 /root/.ssh
grep -qxF '$pub' /root/.ssh/authorized_keys 2>/dev/null || echo '$pub' >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys"
}

echo "bringup: looking for a peer on $IFACE"
if ! ifconfig "$IFACE" | grep -q 'status: active'; then
  echo "bringup: $IFACE is not active" >&2
  exit 1
fi

LL=$(discover_ll || true)
if [ -z "$LL" ]; then
  echo "bringup: no IPv6 neighbor on $IFACE (is the board plugged in?)" >&2
  ndp -an | grep "$IFACE" || true
  exit 1
fi
echo "bringup: peer $LL"

if remote "$LL" 'echo LOGIN_OK'; then
  echo "bringup: login via link-local"
else
  echo "bringup: cannot log in as root@$LL (password $PASSWORD?)" >&2
  exit 1
fi

install_key "$LL"
remote "$LL" "ip addr add $BOARD_IP/24 dev eth0 2>/dev/null || true
ip -o -4 addr show eth0"

# prefer IPv4 after this
if ping -c 1 -W 1 "$BOARD_IP" >/dev/null 2>&1; then
  HOST=$BOARD_IP
else
  HOST=$LL
fi
echo "bringup: using root@$HOST"

if [ "$SKIP_COPY" -eq 0 ]; then
  echo "bringup: copy $STAGE -> $HOST:/opt/pico-minicpm5"
  remote "$HOST" 'mkdir -p /opt/pico-minicpm5'
  if ssh $SSH_BASE -o BatchMode=yes -o IdentitiesOnly=yes \
      -i "$HOME/.ssh/id_ed25519" "root@$HOST" 'true' 2>/dev/null; then
    tar cf - -C "$STAGE" . | ssh $SSH_BASE -o IdentitiesOnly=yes \
      -i "$HOME/.ssh/id_ed25519" "root@$HOST" 'tar xf - -C /opt/pico-minicpm5'
  else
    tar cf - -C "$STAGE" . | sshpass -p "$PASSWORD" ssh $SSH_BASE \
      -o PreferredAuthentications=password -o PubkeyAuthentication=no \
      "root@$HOST" 'tar xf - -C /opt/pico-minicpm5'
  fi
fi

remote "$HOST" 'chmod +x /opt/pico-minicpm5/app/*.sh /opt/pico-minicpm5/app/bin/pico_persistent_acl_executor.aarch64 /opt/pico-minicpm5/app/lib/*.so
/opt/pico-minicpm5/app/install_board.sh --usb-ipv4 '"$BOARD_IP"'/24
cd /opt/pico-minicpm5/app/lib && (sha256sum -c SHA256SUMS || busybox sha256sum -c SHA256SUMS)'

if [ "$SKIP_PYTHON" -eq 0 ]; then
  if ! remote "$HOST" 'test -x /opt/pico-minicpm5/venv/bin/python'; then
    echo "bringup: installing Python 3.10 onto the board"
    "$APP_DIR/install_python.sh" --board "root@$HOST"
  else
    echo "bringup: Python already present"
  fi
fi

echo "bringup: environment"
remote "$HOST" /opt/pico-minicpm5/app/board_env.sh

if [ "$SMOKE" -eq 1 ]; then
  echo "bringup: chat.sh smoke"
  remote "$HOST" 'cd /opt/pico-minicpm5 && PICO_MINICPM5_COLOR=never ./app/chat.sh --no-spinner --prompt "只回复 PICO_OK" --max-new 8'
fi

echo "bringup: done. ssh root@$BOARD_IP"
