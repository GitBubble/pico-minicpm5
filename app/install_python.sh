#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Host-side installer: Euler Pi factory Linux has no python3 and no package
# manager (glibc 2.29). Download a relocatable CPython 3.10 + wheels here,
# then unpack to /opt/pico-minicpm5/venv on the board.
#
#   ./app/install_python.sh --board root@192.168.137.100
#
# Slow GitHub / PyPI:
#   PICO_GITHUB_MIRROR=https://ghfast.top \
#   PICO_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn \
#   ./app/install_python.sh --board root@192.168.137.100
set -eu

# Pinned 2026-08-18. CPython 3.10 matches the board server (stdlib + tokenizers).
# install_only_stripped needs glibc >= 2.17; Euler Pi factory is 2.29.
PYTHON_REL=20260814
PYTHON_VER=3.10.21
PYTHON_NAME="cpython-${PYTHON_VER}+${PYTHON_REL}-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_SHA256=686077ed8d668e3446f03f202bc3c99955d350a0db0ace5f8ce45f1b451b54b7
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_REL}/${PYTHON_NAME}"

TOKENIZERS_NAME=tokenizers-0.23.1-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
TOKENIZERS_SHA256=1bf13402aff9bc533c89cb849ec3b412dc3fbeacc9744840e423d7bf3f7dc0e3
TOKENIZERS_PATH=packages/6c/36/e006edf031154cba92b8416057d92c3abe3635e4c4b0aa0b5b9bb39dde70/${TOKENIZERS_NAME}

JINJA2_NAME=jinja2-3.1.6-py3-none-any.whl
JINJA2_SHA256=85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67
JINJA2_PATH=packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/${JINJA2_NAME}

MARKUPSAFE_NAME=markupsafe-3.0.3-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl
MARKUPSAFE_SHA256=1ba88449deb3de88bd40044603fafffb7bc2b055d626a330323a9ed736661695
MARKUPSAFE_PATH=packages/40/01/e560d658dc0bb8ab762670ece35281dec7b6c1b33f5fbc09ebb57a185519/${MARKUPSAFE_NAME}

BOARD=""
STAGE=""
SKIP_UPLOAD=0
DEST=/opt/pico-minicpm5/venv

usage() {
  cat <<'EOF'
Usage: install_python.sh [--board USER@HOST] [--stage DIR] [--skip-upload]

  Run on the host. Factory Euler Pi Linux has no python3. This downloads a
  relocatable CPython 3.10.21 (glibc 2.17+) plus tokenizers / jinja2 wheels
  and installs them to /opt/pico-minicpm5/venv so chat.sh finds
  $ROOT/venv/bin/python.

  --board root@192.168.137.100   copy the staged tree and verify on the board
  --stage DIR                    keep the download/stage tree (default: temp)
  --skip-upload                  only download and stage

Environment:
  PICO_GITHUB_MIRROR   prefix for the GitHub release URL (e.g. https://ghfast.top)
  PICO_PYPI_INDEX      PyPI origin (default https://files.pythonhosted.org)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --board)
      BOARD=${2:?--board needs USER@HOST}
      shift 2
      ;;
    --board=*)
      BOARD=${1#--board=}
      shift
      ;;
    --stage)
      STAGE=${2:?--stage needs DIR}
      shift 2
      ;;
    --stage=*)
      STAGE=${1#--stage=}
      shift
      ;;
    --skip-upload)
      SKIP_UPLOAD=1
      shift
      ;;
    --dest)
      DEST=${2:?--dest needs PATH}
      shift 2
      ;;
    *)
      echo "install_python: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

github_url() {
  url=$1
  if [ -n "${PICO_GITHUB_MIRROR:-}" ]; then
    echo "${PICO_GITHUB_MIRROR%/}/$url"
  else
    echo "$url"
  fi
}

pypi_url() {
  path=$1
  origin=${PICO_PYPI_INDEX:-https://files.pythonhosted.org}
  case "$origin" in
    */simple|*/simple/)
      origin=${origin%/simple}
      origin=${origin%/simple/}
      origin=${origin%/}
      echo "${origin}/packages/${path#packages/}"
      ;;
    *)
      echo "${origin%/}/$path"
      ;;
  esac
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

fetch() {
  url=$1
  dest=$2
  expect=$3
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$expect" ]; then
    echo "install_python: cached $(basename "$dest")"
    return 0
  fi
  echo "install_python: download $url"
  curl -fL --retry 5 --retry-delay 2 -o "$dest.part" "$url"
  got=$(sha256_of "$dest.part")
  if [ "$got" != "$expect" ]; then
    echo "install_python: sha256 mismatch for $(basename "$dest")" >&2
    echo "  expected $expect" >&2
    echo "  got      $got" >&2
    rm -f "$dest.part"
    exit 1
  fi
  mv "$dest.part" "$dest"
}

cleanup() {
  if [ -n "${TMP_OWNED:-}" ] && [ -d "${TMP_OWNED}" ]; then
    rm -rf "$TMP_OWNED"
  fi
}
trap cleanup EXIT

if [ -n "$STAGE" ]; then
  mkdir -p "$STAGE"
  WORK=$STAGE
else
  WORK=$(mktemp -d "${TMPDIR:-/tmp}/pico-board-python.XXXXXX")
  TMP_OWNED=$WORK
fi

CACHE=$WORK/cache
ROOT=$WORK/venv
mkdir -p "$CACHE" "$ROOT"

fetch "$(github_url "$PYTHON_URL")" "$CACHE/$PYTHON_NAME" "$PYTHON_SHA256"
fetch "$(pypi_url "$TOKENIZERS_PATH")" "$CACHE/$TOKENIZERS_NAME" "$TOKENIZERS_SHA256"
fetch "$(pypi_url "$JINJA2_PATH")" "$CACHE/$JINJA2_NAME" "$JINJA2_SHA256"
fetch "$(pypi_url "$MARKUPSAFE_PATH")" "$CACHE/$MARKUPSAFE_NAME" "$MARKUPSAFE_SHA256"

echo "install_python: extract CPython into $ROOT"
rm -rf "$ROOT"
mkdir -p "$ROOT"
tar xzf "$CACHE/$PYTHON_NAME" --strip-components=1 -C "$ROOT"

SITE=$ROOT/lib/python3.10/site-packages
if [ ! -d "$SITE" ]; then
  echo "install_python: missing $SITE after extract" >&2
  exit 1
fi
for wheel in "$CACHE/$TOKENIZERS_NAME" "$CACHE/$JINJA2_NAME" "$CACHE/$MARKUPSAFE_NAME"; do
  echo "install_python: unpack $(basename "$wheel")"
  python3 -m zipfile -e "$wheel" "$SITE"
done

if [ ! -e "$ROOT/bin/python" ] && [ -x "$ROOT/bin/python3" ]; then
  ln -s python3 "$ROOT/bin/python"
fi
if [ ! -e "$ROOT/bin/python3" ]; then
  echo "install_python: staged tree has no bin/python3" >&2
  exit 1
fi

echo "install_python: staged $ROOT"
ls -lh "$ROOT/bin/python3"

if [ "$SKIP_UPLOAD" -eq 1 ]; then
  echo "install_python: --skip-upload, tree left at $ROOT"
  TMP_OWNED=""
  exit 0
fi

if [ -z "$BOARD" ]; then
  echo "install_python: staged at $ROOT (pass --board USER@HOST to copy)" >&2
  echo "  tar cf - -C $WORK venv | ssh USER@HOST 'mkdir -p /opt/pico-minicpm5 && tar xf - -C /opt/pico-minicpm5'" >&2
  TMP_OWNED=""
  exit 0
fi

echo "install_python: upload to $BOARD:$DEST"
ssh "$BOARD" "mkdir -p $(dirname "$DEST") && rm -rf $DEST"
tar cf - -C "$WORK" venv | ssh "$BOARD" "tar xf - -C $(dirname "$DEST")"
ssh "$BOARD" "chmod +x $DEST/bin/python3 $DEST/bin/python
$DEST/bin/python3 - <<'PY'
import sys
from tokenizers import Tokenizer
import jinja2
print('python', sys.version.split()[0], sys.platform)
print('tokenizers', __import__('tokenizers').__version__)
print('jinja2', jinja2.__version__)
print('ok')
PY"
echo "install_python: done. chat.sh will use $DEST/bin/python"
