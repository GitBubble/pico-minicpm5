#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Orange Pi AIfly: graphics (LightDM/Xorg + sample_gfbg) and SVP inference
# cannot run together. Stop the *userspace* graphics stack, then infer.
#
# sample_gfbg 0 0 0 is started by orangepi-hardware-optimization when
# BUILD_DESKTOP=yes. It installs SIGTERM/SIGINT handlers that join VO
# threads and call sample_comm_sys_exit(). SIGKILL skips that and has
# taken the board off the network. rmmod gfbg/ot_vo/ot_hdmi D-states sshd.
set -eu

if [ "$(id -u)" -ne 0 ]; then
  echo "prepare_community: must run as root (sudo $0)" >&2
  exit 1
fi

stop_lightdm() {
  if command -v systemctl >/dev/null 2>&1; then
    systemctl stop lightdm 2>/dev/null || true
    systemctl disable lightdm 2>/dev/null || true
    systemctl mask lightdm 2>/dev/null || true
    systemctl stop gdm 2>/dev/null || true
    systemctl stop sddm 2>/dev/null || true
  fi
  pkill -x Xorg 2>/dev/null || true
  i=0
  while [ "$i" -lt 10 ] && pgrep -x Xorg >/dev/null 2>&1; do
    sleep 1
    i=$((i + 1))
  done
}

# SIGTERM: sample_gfbg_handle_sig sets g_sample_gfbg_exit, getchar()
# returns, sample_comm_sys_exit() runs. Never SIGKILL.
stop_sample_gfbg() {
  pids=$(pgrep -x sample_gfbg || true)
  if [ -z "$pids" ]; then
    echo "prepare_community: sample_gfbg not running"
    return 0
  fi
  echo "prepare_community: SIGTERM sample_gfbg ($pids)"
  kill -TERM $pids 2>/dev/null || true
  i=0
  while [ "$i" -lt 15 ]; do
    pgrep -x sample_gfbg >/dev/null 2>&1 || return 0
    sleep 1
    i=$((i + 1))
  done
  for pid in $(pgrep -x sample_gfbg || true); do
    if [ -w "/proc/$pid/fd/0" ]; then
      printf 'q\n' > "/proc/$pid/fd/0" 2>/dev/null || true
    fi
  done
  sleep 3
  if pgrep -x sample_gfbg >/dev/null 2>&1; then
    echo "prepare_community: sample_gfbg still running after SIGTERM" >&2
    echo "prepare_community: do not kill -9; inference may still fail" >&2
    return 1
  fi
}

persist_no_desktop() {
  rel=/etc/orangepi-release
  if [ -f "$rel" ] && grep -q '^BUILD_DESKTOP=yes' "$rel"; then
    cp -a "$rel" "$rel.pico-bak" 2>/dev/null || true
    sed -i 's/^BUILD_DESKTOP=yes/BUILD_DESKTOP=no/' "$rel"
    echo "prepare_community: BUILD_DESKTOP=no in $rel (next boot skips sample_gfbg+lightdm)"
  fi
}

stop_lightdm
stop_sample_gfbg || true
persist_no_desktop

if [ ! -e /dev/svp_npu ]; then
  KO=/lib/modules/$(uname -r)/hi3403_mpp_ko/svp_npu/ot_svp_npu.ko
  if [ -f "$KO" ] && ! lsmod | awk '{print $1}' | grep -qx ot_svp_npu; then
    insmod "$KO"
  fi
fi

if [ ! -e /dev/svp_npu ]; then
  echo "prepare_community: /dev/svp_npu still missing" >&2
  exit 1
fi

chmod 666 /dev/svp_npu /dev/mmz_userdev /dev/sys /dev/ot_irq /dev/ot_proc 2>/dev/null || true

if pgrep -x Xorg >/dev/null 2>&1; then
  echo "prepare_community: Xorg still running; inference may fail" >&2
  exit 1
fi

echo "prepare_community: graphics userspace stopped, /dev/svp_npu ready"
echo "prepare_community: leaving gfbg/ot_vo/ot_hdmi loaded (do not rmmod)"
pgrep -a sample_gfbg || echo "prepare_community: sample_gfbg stopped"
lsmod | awk '/gfbg|ot_vo|ot_hdmi|ot_svp_npu/{print}'
