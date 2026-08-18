#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Euler Pi factory Linux loads ot_pqp.ko. That module is mutually exclusive
# with ot_svp_npu.ko, which pico-minicpm5 needs for /dev/svp_npu.
# Idempotent: unload pqp if present, load svp_npu if missing.
set -eu

KO_ROOT=${PICO_KO_ROOT:-/opt/ko}
SVP_KO=$KO_ROOT/svp_npu/ot_svp_npu.ko

module_loaded() {
  lsmod 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

if [ ! -f "$SVP_KO" ]; then
  echo "prepare_npu: skip — $SVP_KO is not on this system" >&2
  exit 0
fi

if module_loaded ot_pqp; then
  echo "prepare_npu: unloading ot_pqp (blocks svp_npu on Euler Pi factory image)"
  rmmod ot_pqp
fi

if ! module_loaded ot_svp_npu; then
  echo "prepare_npu: loading $SVP_KO"
  insmod "$SVP_KO"
fi

if [ ! -e /dev/svp_npu ]; then
  echo "prepare_npu: /dev/svp_npu is still missing after insmod" >&2
  if module_loaded ot_pqp; then
    echo "prepare_npu: ot_pqp is still loaded; unload it first" >&2
  fi
  exit 1
fi

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -e "$APP_DIR/lib/libsvp_acl.so" ]; then
  echo "prepare_npu: using shipped $APP_DIR/lib"
elif [ -e /root/pico_default_smoke/lib/libsvp_acl.so ]; then
  :
elif [ -e /opt/ss928-runtime/lib/libsvp_acl.so ]; then
  mkdir -p /root/pico_default_smoke
  ln -sfn /opt/ss928-runtime/lib /root/pico_default_smoke/lib
  echo "prepare_npu: linked /root/pico_default_smoke/lib -> /opt/ss928-runtime/lib"
else
  echo "prepare_npu: warning — libsvp_acl.so missing; expected $APP_DIR/lib" >&2
fi

echo "prepare_npu: ready (/dev/svp_npu present, ot_svp_npu loaded)"
