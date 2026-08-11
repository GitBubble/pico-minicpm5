#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B ctx1024: native tool-calling agent for SS928.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MAX_NEW=${MAX_NEW:-128}

# chat.sh owns the common three-handle deployment contract. Passing an
# explicit mode keeps its plain-chat default independent from this agent app.
exec "$APP_DIR/chat.sh" --agent --max-new "$MAX_NEW" "$@"
