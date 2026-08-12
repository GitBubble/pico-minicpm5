#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B multi-context resident-K/V Hi3403 demo.
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
MAX_NEW=${MAX_NEW:-}
PROFILE=${PICO_PROFILE:-ctx1024}
REUSE_SESSION_KV=${REUSE_SESSION_KV:-1}

case "$REUSE_SESSION_KV" in
  1|true|TRUE|on|ON)
    set -- --reuse-session-kv "$@"
    ;;
  0|false|FALSE|off|OFF|"")
    ;;
  *)
    echo "REUSE_SESSION_KV must be 0/1, false/true or off/on" >&2
    exit 2
    ;;
esac

# Chat remains the default when callers only pass display/runtime options.
# An explicit prompt or mode is forwarded unchanged for compatibility.
MODE_SET=0
for ARG in "$@"; do
  case "$ARG" in
    --prompt|--prompt=*|--prompt-ids|--prompt-ids=*|--interactive|--chat|--agent)
      MODE_SET=1
      ;;
  esac
done
if [ "$MODE_SET" -eq 0 ]; then
  if [ "$#" -eq 0 ] && [ "${PROMPT+x}" = x ]; then
    if [ -n "$MAX_NEW" ]; then
      set -- --prompt "$PROMPT" --max-new "$MAX_NEW"
    else
      set -- --prompt "$PROMPT"
    fi
  else
    if [ -n "$MAX_NEW" ]; then
      set -- --chat --max-new "$MAX_NEW" "$@"
    else
      set -- --chat "$@"
    fi
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
    --profile "$PROFILE" \
    --deployment-root "$ROOT" \
    --library-path "$LIB" \
    --embedding "$ROOT/assets/token_embedding.f16.bin" \
    --tokenizer "$ROOT/assets/tokenizer.json" \
    "$@"
