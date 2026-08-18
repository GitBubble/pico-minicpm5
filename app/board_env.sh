#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Print the Euler Pi / SS928 board identity that pico-minicpm5 needs.
# Safe to source or execute. Does not change kernel modules.
set -eu

firmware=${PICO_FIRMWARE_VERSION:-/etc/firmware_version}

chip=""
sdk=""
hardware=""
software=""
if [ -r "$firmware" ]; then
  chip=$(awk -F: '/^Chip:/{sub(/^[[:space:]]+/, "", $2); print $2}' "$firmware")
  sdk=$(awk -F: '/^SDK:/{sub(/^[[:space:]]+/, "", $2); print $2}' "$firmware")
  hardware=$(awk -F: '/^HardWare_ver:/{sub(/^[[:space:]]+/, "", $2); print $2}' "$firmware")
  software=$(awk -F: '/^SoftWare_ver:/{sub(/^[[:space:]]+/, "", $2); print $2}' "$firmware")
fi

product="Euler Pi"
case "$hardware" in
  *Euer*|*Euler*|*HiEuer*|*HiEuler*) product="Euler Pi" ;;
esac

npu_dev="missing"
if [ -e /dev/svp_npu ]; then
  npu_dev="/dev/svp_npu"
fi
pqp_loaded=no
svp_loaded=no
if lsmod 2>/dev/null | awk '{print $1}' | grep -qx ot_pqp; then
  pqp_loaded=yes
fi
if lsmod 2>/dev/null | awk '{print $1}' | grep -qx ot_svp_npu; then
  svp_loaded=yes
fi

python="missing"
if [ -x /opt/pico-minicpm5/venv/bin/python ]; then
  python=/opt/pico-minicpm5/venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  python=$(command -v python3)
fi

lib="unset"
if [ -n "${PICO_RUNTIME_LIB:-}" ]; then
  lib=$PICO_RUNTIME_LIB
elif [ -e /opt/pico-minicpm5/app/lib/libsvp_acl.so ]; then
  lib=/opt/pico-minicpm5/app/lib
elif [ -e /root/pico_default_smoke/lib/libsvp_acl.so ]; then
  lib=/root/pico_default_smoke/lib
elif [ -e /opt/ss928-runtime/lib/libsvp_acl.so ]; then
  lib=/opt/ss928-runtime/lib
fi

ipv4=""
if command -v ip >/dev/null 2>&1; then
  ipv4=$(ip -o -4 addr show 2>/dev/null | awk '{print $2"="$4}' | tr '\n' ' ')
fi

cat <<EOF
pico-minicpm5 board environment
  Product:       $product
  Chip:          ${chip:-unknown}
  SDK:           ${sdk:-unknown}
  Hardware:      ${hardware:-unknown}
  Software:      ${software:-unknown}
  Kernel:        $(uname -srm 2>/dev/null || echo unknown)
  NPU device:    $npu_dev
  ot_pqp:        $pqp_loaded
  ot_svp_npu:    $svp_loaded
  Runtime lib:   $lib
  Python:        $python
  IPv4:          ${ipv4:-unknown}
EOF
