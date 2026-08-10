#!/bin/sh
# MiniCPM5-1B merged 24-layer runtime: one prefill OM, one decode OM, one head.
set -eu

GATE="${GATE:-/root/minicpm5_gate_3handle}"
LIB="${LIB:-/root/pico_default_smoke/lib}"
TOKENIZERS="${TOKENIZERS:-/opt/pico-minicpm5/venv/lib/python3.10/site-packages}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_NEW="${MAX_NEW:-24}"

exec env LD_LIBRARY_PATH="$LIB" \
  PYTHONPATH="$TOKENIZERS:$GATE/src" \
  python3 -u "$GATE/src/merged_board_server.py" \
    --persistent-executor "$GATE/bin/pico_persistent_acl_executor.resident.aarch64" \
    --decode-model "$GATE/models/decode.om" \
    --prefill-model "$GATE/models/prefill.om" \
    --head-model "$GATE/models/head_flat.om" \
    --library-path "$LIB" \
    --embedding "$GATE/assets/token_embedding.f16.bin" \
    --tokenizer "$GATE/assets/tokenizer.json" \
    --context 1024 \
    --prompt "$PROMPT" \
    --max-new "$MAX_NEW" \
    "$@"
