#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B configurable native tool-calling agent for Hi3403.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MAX_NEW=${MAX_NEW:-}
THINKING=${THINKING:-0}
FIXED_PREFIX_SNAPSHOTS=${FIXED_PREFIX_SNAPSHOTS:-1}

case "$THINKING" in
  1|true|TRUE|on|ON)
    set -- --thinking "$@"
    ;;
  0|false|FALSE|off|OFF|"")
    ;;
  *)
    echo "THINKING must be 0/1, false/true or off/on" >&2
    exit 2
    ;;
esac

case "$FIXED_PREFIX_SNAPSHOTS" in
  1|true|TRUE|on|ON)
    set -- --fixed-prefix-snapshots "$@"
    ;;
  0|false|FALSE|off|OFF|"")
    ;;
  *)
    echo "FIXED_PREFIX_SNAPSHOTS must be 0/1, false/true or off/on" >&2
    exit 2
    ;;
esac

# chat.sh owns the common three-handle deployment contract. Passing an
# explicit mode keeps its plain-chat default independent from this agent app.
if [ -n "$MAX_NEW" ]; then
  exec "$APP_DIR/chat.sh" --agent --max-new "$MAX_NEW" "$@"
fi
exec "$APP_DIR/chat.sh" --agent "$@"
