#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B ctx1024: resident-K/V three-handle SS928 demo.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=${PICO_MINICPM5_ROOT:-$(dirname "$APP_DIR")}
if [ -n "${PICO_RUNTIME_LIB:-}" ]; then
  LIB=$PICO_RUNTIME_LIB
elif [ -d /root/pico_default_smoke/lib ]; then
  LIB=/root/pico_default_smoke/lib
else
  LIB=/opt/ss928-runtime/lib
fi
if [ -n "${PYTHON:-}" ]; then
  PYTHON_BIN=$PYTHON
elif [ -x "$ROOT/venv/bin/python" ]; then
  PYTHON_BIN=$ROOT/venv/bin/python
else
  PYTHON_BIN=python3
fi
TOKENIZERS=${TOKENIZERS:-}
MAX_NEW=${MAX_NEW:-24}

# With no arguments and no PROMPT override, start a resident REPL. Explicit
# CLI arguments are forwarded unchanged so `chat.sh --prompt ...` does not also
# run a hidden default prompt.
if [ "$#" -eq 0 ]; then
  if [ "${PROMPT+x}" = x ]; then
    set -- --prompt "$PROMPT" --max-new "$MAX_NEW"
  else
    set -- --interactive --max-new "$MAX_NEW"
  fi
fi

PYTHONPATH_VALUE="$APP_DIR/src"
if [ -n "$TOKENIZERS" ]; then
  PYTHONPATH_VALUE="$TOKENIZERS:$PYTHONPATH_VALUE"
fi

exec env LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$PYTHONPATH_VALUE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -u "$APP_DIR/src/merged_board_server.py" \
    --persistent-executor "$APP_DIR/bin/pico_persistent_acl_executor.aarch64" \
    --decode-model "$ROOT/models/decode.om" \
    --prefill-model "$ROOT/models/prefill.om" \
    --head-model "$ROOT/models/head_flat.om" \
    --library-path "$LIB" \
    --embedding "$ROOT/assets/token_embedding.f16.bin" \
    --tokenizer "$ROOT/assets/tokenizer.json" \
    --context 1024 \
    "$@"
