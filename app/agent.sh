#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B configurable native tool-calling agent for Hi3403.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MAX_NEW=${MAX_NEW:-}
THINKING=${THINKING:-0}
FIXED_PREFIX_SNAPSHOTS=${FIXED_PREFIX_SNAPSHOTS:-1}
CONTEXT_PROFILE=${CONTEXT_PROFILE:-}
# EAGER_TOOL_PREFILL=0|1 (default 0) is handled by chat.sh, which owns the
# command line; it is exported through this exec unchanged. Turning it on
# prefills a streaming tool's output while the tool still runs.

if [ -n "$CONTEXT_PROFILE" ]; then
  if [ -n "${PICO_PROFILE:-}" ] && [ "$PICO_PROFILE" != "$CONTEXT_PROFILE" ]; then
    echo "CONTEXT_PROFILE and PICO_PROFILE select different profiles" >&2
    exit 2
  fi
  PICO_PROFILE=$CONTEXT_PROFILE
  export PICO_PROFILE
fi

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
