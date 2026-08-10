#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B ctx1024: resident-K/V three-handle SS928 demo.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=${PICO_MINICPM5_ROOT:-$(dirname "$APP_DIR")}
LIB=${PICO_RUNTIME_LIB:-/opt/ss928-runtime/lib}
PYTHON=${PYTHON:-python3}
TOKENIZERS=${TOKENIZERS:-}
PROMPT=${PROMPT:-The capital of France is}
MAX_NEW=${MAX_NEW:-24}

PYTHONPATH_VALUE="$APP_DIR/src"
if [ -n "$TOKENIZERS" ]; then
  PYTHONPATH_VALUE="$TOKENIZERS:$PYTHONPATH_VALUE"
fi

exec env LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$PYTHONPATH_VALUE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -u "$APP_DIR/src/merged_board_server.py" \
    --persistent-executor "$APP_DIR/bin/pico_persistent_acl_executor.aarch64" \
    --decode-model "$ROOT/models/decode.om" \
    --prefill-model "$ROOT/models/prefill.om" \
    --head-model "$ROOT/models/head_flat.om" \
    --library-path "$LIB" \
    --embedding "$ROOT/assets/token_embedding.f16.bin" \
    --tokenizer "$ROOT/assets/tokenizer.json" \
    --context 1024 \
    --prompt "$PROMPT" \
    --max-new "$MAX_NEW" \
    "$@"
