#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# MiniCPM5-1B multi-context resident-K/V Hi3403 demo.
set -eu

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=${PICO_MINICPM5_ROOT:-$(dirname "$APP_DIR")}
# Standalone CPython ships its own terminfo. Factory Euler Pi has none, and
# GNU readline SIGSEGVs on the first input() of the interactive REPL.
if [ -z "${TERMINFO:-}" ] && [ -d "$ROOT/venv/share/terminfo" ]; then
  TERMINFO=$ROOT/venv/share/terminfo
  export TERMINFO
fi
# Euler Pi factory Linux inserts ot_pqp.ko, which blocks /dev/svp_npu.
# Host-side tests skip this: /opt/ko/svp_npu is only on the board image.
if [ ! -e /dev/svp_npu ] && [ -x "$APP_DIR/prepare_npu.sh" ] && [ -d /opt/ko/svp_npu ]; then
  "$APP_DIR/prepare_npu.sh"
fi
# Euler Pi commercial kernel wants app/lib (SS928V100_SDK). Orange Pi /
# Pegasus community (Jammy + /usr/lib/svp_npu) wants app/lib-community:
# commercial ACL returns svp_acl_init ret=100000 on the 12KB community
# ot_svp_npu, and stock Pegasus aicpu needs fmod@GLIBC_2.38 (Jammy is 2.35).
COMMUNITY=0
if [ -e /usr/lib/svp_npu/libsvp_acl.so ]; then
  COMMUNITY=1
fi
if [ -r /etc/os-release ] && grep -Eq 'UBUNTU_CODENAME=jammy|VERSION_ID="22.04"' /etc/os-release; then
  COMMUNITY=1
fi
if [ -d /opt/ko/svp_npu ]; then
  COMMUNITY=0
fi
if [ -n "${PICO_RUNTIME_LIB:-}" ]; then
  LIB=$PICO_RUNTIME_LIB
elif [ "$COMMUNITY" -eq 1 ] && [ -e "$APP_DIR/lib-community/libsvp_acl.so" ]; then
  LIB=$APP_DIR/lib-community
elif [ "$COMMUNITY" -eq 1 ]; then
  echo "chat.sh: community SDK needs $APP_DIR/lib-community (Pegasus ACL)." >&2
  echo "chat.sh: commercial app/lib fails svp_acl_init ret=100000 on this kernel." >&2
  echo "chat.sh: set PICO_RUNTIME_LIB to override." >&2
  exit 2
elif [ -e "$APP_DIR/lib/libsvp_acl.so" ]; then
  LIB=$APP_DIR/lib
elif [ -e /usr/lib/svp_npu/libsvp_acl.so ]; then
  LIB=/usr/lib/svp_npu
elif [ -e /root/pico_default_smoke/lib/libsvp_acl.so ]; then
  LIB=/root/pico_default_smoke/lib
elif [ -e /opt/ss928-runtime/lib/libsvp_acl.so ]; then
  LIB=/opt/ss928-runtime/lib
else
  LIB=$APP_DIR/lib
fi
# Community packages keep libsecurec in /usr/lib, not next to libsvp_acl.
LD_EXTRA=""
if [ ! -e "$LIB/libsecurec.so" ] && [ -e /usr/lib/libsecurec.so ]; then
  LD_EXTRA=/usr/lib
fi
if command -v pgrep >/dev/null 2>&1 \
    && pgrep -x Xorg >/dev/null 2>&1 \
    && [ "${PICO_ALLOW_GRAPHICS:-}" != "1" ]; then
  if [ "$(id -u)" -eq 0 ] && [ -x "$APP_DIR/prepare_community.sh" ]; then
    echo "chat.sh: stopping LightDM/Xorg (graphics and inference cannot coexist)" >&2
    "$APP_DIR/prepare_community.sh"
  fi
fi
if command -v pgrep >/dev/null 2>&1 \
    && pgrep -x Xorg >/dev/null 2>&1 \
    && [ "${PICO_ALLOW_GRAPHICS:-}" != "1" ]; then
  echo "chat.sh: community SDK cannot run graphics and inference together." >&2
  echo "chat.sh: sudo $APP_DIR/prepare_community.sh   # stops LightDM/Xorg only" >&2
  echo "chat.sh: do not rmmod gfbg/ot_vo/ot_hdmi — that hangs the kernel." >&2
  echo "chat.sh: set PICO_ALLOW_GRAPHICS=1 to skip this check." >&2
  exit 2
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
# Eager tool prefill is DEFAULT OFF. It is an agent-mode feature and the
# server refuses it without --agent, resident KV and --reuse-session-kv.
EAGER_TOOL_PREFILL=${EAGER_TOOL_PREFILL:-0}

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

case "$EAGER_TOOL_PREFILL" in
  1|true|TRUE|on|ON)
    EAGER_SET=0
    for ARG in "$@"; do
      case "$ARG" in
        --eager-tool-prefill) EAGER_SET=1 ;;
      esac
    done
    if [ "$EAGER_SET" -eq 0 ]; then
      set -- --eager-tool-prefill "$@"
    fi
    ;;
  0|false|FALSE|off|OFF|"")
    ;;
  *)
    echo "EAGER_TOOL_PREFILL must be 0/1, false/true or off/on" >&2
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

EXECUTOR=$APP_DIR/bin/pico_persistent_acl_executor.aarch64
if [ "$COMMUNITY" -eq 1 ] && [ -x "$APP_DIR/bin/pico_persistent_acl_executor.community" ] \
    && [ -x "$APP_DIR/glibc239/ld-linux-aarch64.so.1" ]; then
  EXECUTOR=$APP_DIR/bin/pico_persistent_acl_executor.community
  if [ -e /usr/lib/svp_npu/libsvp_acl.so ]; then
    LIB=/usr/lib/svp_npu
  fi
fi

exec env LD_LIBRARY_PATH="$LIB${LD_EXTRA:+:$LD_EXTRA}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$PYTHONPATH_VALUE${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -u "$APP_DIR/src/merged_board_server.py" \
    --persistent-executor "$EXECUTOR" \
    --profile "$PROFILE" \
    --deployment-root "$ROOT" \
    --library-path "$LIB" \
    --embedding "$ROOT/assets/token_embedding.f16.bin" \
    --tokenizer "$ROOT/assets/tokenizer.json" \
    "$@"
