#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Verify the shipped app/lib SVP ACL runtime, or refresh it from an SDK tree.
# chat.sh prefers app/lib when libsvp_acl.so is present, so a normal deploy
# does not need this script.
#
#   ./app/install_runtime_lib.sh
#   ./app/install_runtime_lib.sh --board root@192.168.137.100
#   ./app/install_runtime_lib.sh --sdk-root /path/to/SS928V100_SDK_V2.0.2.2
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BOARD=""
SDK_ROOT=""
LIB_DIR=$APP_DIR/lib
DEST=/root/pico_default_smoke/lib

usage() {
  cat <<'EOF'
Usage: install_runtime_lib.sh [--board USER@HOST] [--sdk-root DIR] [--lib-dir DIR]

  The executor links libsvp_acl.so, libsvp_aicpu.so, libprotobuf-c.so.1 and
  libsecurec.so. They ship in app/lib/ (SS928V100_SDK_V2.0.2.2). Factory
  /opt/lib/npu is the Ascend stack and cannot replace them.

  With no arguments, verify app/lib against SHA256SUMS.
  --board copies those files onto the board (optional; chat.sh reads app/lib).
  --sdk-root refresh app/lib from an SDK checkout, then verify.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --board)
      BOARD=${2:?}
      shift 2
      ;;
    --board=*)
      BOARD=${1#--board=}
      shift
      ;;
    --sdk-root)
      SDK_ROOT=${2:?}
      shift 2
      ;;
    --sdk-root=*)
      SDK_ROOT=${1#--sdk-root=}
      shift
      ;;
    --lib-dir)
      LIB_DIR=${2:?}
      shift 2
      ;;
    --lib-dir=*)
      LIB_DIR=${1#--lib-dir=}
      shift
      ;;
    --dest)
      DEST=${2:?}
      shift 2
      ;;
    *)
      echo "install_runtime_lib: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -n "$SDK_ROOT" ]; then
  found=""
  for cand in \
    "$SDK_ROOT/smp/a55_linux/mpp/out/lib/svp_npu" \
    "$SDK_ROOT/out/lib/svp_npu" \
    "$SDK_ROOT/lib/svp_npu"
  do
    if [ -f "$cand/libsvp_acl.so" ]; then
      found=$cand
      break
    fi
  done
  if [ -z "$found" ]; then
    echo "install_runtime_lib: no libsvp_acl.so under $SDK_ROOT" >&2
    exit 1
  fi
  SECUREC=$found/libsecurec.so
  if [ ! -f "$SECUREC" ]; then
    for cand in "$found/../libsecurec.so" "$found/../../libsecurec.so"; do
      [ -f "$cand" ] && SECUREC=$cand && break
    done
  fi
  mkdir -p "$LIB_DIR"
  cp -p "$found/libsvp_acl.so" "$found/libsvp_aicpu.so" \
    "$found/libprotobuf-c.so.1" "$SECUREC" "$LIB_DIR/"
  echo "install_runtime_lib: refreshed $LIB_DIR from $found"
fi

for name in libsvp_acl.so libsvp_aicpu.so libprotobuf-c.so.1 libsecurec.so; do
  if [ ! -f "$LIB_DIR/$name" ]; then
    echo "install_runtime_lib: missing $LIB_DIR/$name" >&2
    exit 1
  fi
done

if [ -f "$LIB_DIR/SHA256SUMS" ]; then
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$LIB_DIR" && sha256sum -c SHA256SUMS)
  else
    (cd "$LIB_DIR" && shasum -a 256 -c SHA256SUMS)
  fi
fi

if [ -z "$BOARD" ]; then
  echo "install_runtime_lib: app/lib is ready (chat.sh will use it)"
  exit 0
fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/pico-svp-lib.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
cp "$LIB_DIR/libsvp_acl.so" "$LIB_DIR/libsvp_aicpu.so" \
  "$LIB_DIR/libprotobuf-c.so.1" "$LIB_DIR/libsecurec.so" "$STAGE/"

ssh "$BOARD" "if [ -L $DEST ]; then rm -f $DEST; fi
mkdir -p $DEST"
tar cf - -C "$STAGE" libsvp_acl.so libsvp_aicpu.so libprotobuf-c.so.1 libsecurec.so \
  | ssh "$BOARD" "tar xf - -C $DEST
chmod 755 $DEST/libsvp_acl.so $DEST/libsvp_aicpu.so $DEST/libprotobuf-c.so.1 $DEST/libsecurec.so
ls -l $DEST/libsvp_acl.so
echo install_runtime_lib: copied to $DEST"
