#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PICO_OPENCLAW_PYTHON:-python3}
CONFIG=${PICO_OPENCLAW_CONFIG:-$APP_DIR/config/runtime.json}

exec "$PYTHON" "$APP_DIR/src/lifecycle.py" \
  --config "$CONFIG" doctor "$@"
