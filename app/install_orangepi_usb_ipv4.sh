#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Board-side: give the USB/GbE NIC a persistent 192.168.138.10/24 and
# bring it up on every boot. Run as root on Orange Pi AIfly.
set -eu

ADDR=${PICO_USB_IPV4:-192.168.138.10/24}
UNIT=/etc/systemd/system/pico-usb-ipv4.service
SCRIPT=/usr/local/sbin/pico-usb-ipv4.sh

if [ "$(id -u)" -ne 0 ]; then
  echo "install_orangepi_usb_ipv4: must run as root" >&2
  exit 1
fi

mkdir -p /usr/local/sbin
cat > "$SCRIPT" <<EOF
#!/bin/sh
# Add $ADDR to the first real Ethernet NIC. Never default-route it.
set -eu
ADDR="$ADDR"
is_wifi() {
  case "\$1" in
    wlan*|wl*|ap*) return 0 ;;
  esac
  [ -d "/sys/class/net/\$1/wireless" ]
}
skip_nic() {
  case "\$1" in
    lo|dummy*|veth*|docker*|br-*|virbr*|tun*|tap*|sit*|ip6tnl*|ip_vti*|wg*|tailscale*) return 0 ;;
  esac
  return 1
}
for nic in eth0 end0 \$(ls /sys/class/net); do
  [ -d "/sys/class/net/\$nic" ] || continue
  skip_nic "\$nic" && continue
  is_wifi "\$nic" && continue
  [ -f "/sys/class/net/\$nic/type" ] || continue
  [ "\$(cat /sys/class/net/\$nic/type)" = 1 ] || continue
  ip link set "\$nic" up || true
  ip addr add "\$ADDR" dev "\$nic" 2>/dev/null || true
  echo "pico-usb-ipv4: \$ADDR on \$nic"
  ip -br a show "\$nic"
  exit 0
done
echo "pico-usb-ipv4: no ethernet NIC found" >&2
ip -br a >&2 || true
exit 1
EOF
chmod 755 "$SCRIPT"

cat > "$UNIT" <<'EOF'
[Unit]
Description=pico-minicpm5 USB ethernet 192.168.138.10/24
After=network-pre.target systemd-udevd.service
Wants=network-pre.target
DefaultDependencies=yes

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/pico-usb-ipv4.sh
ExecStartPost=/bin/sleep 1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable pico-usb-ipv4.service
"$SCRIPT"

# NetworkManager: keep the address if NM later takes the NIC.
if command -v nmcli >/dev/null 2>&1; then
  nic=""
  for d in eth0 end0 $(ls /sys/class/net); do
    [ -d "/sys/class/net/$d" ] || continue
    case "$d" in
      lo|dummy*|veth*|docker*|br-*|virbr*|tun*|tap*|sit*|ip6tnl*|ip_vti*|wg*|tailscale*|wlan*|wl*|ap*) continue ;;
    esac
    [ -d "/sys/class/net/$d/wireless" ] && continue
    [ -f "/sys/class/net/$d/type" ] && [ "$(cat /sys/class/net/$d/type)" = 1 ] || continue
    nic=$d
    break
  done
  if [ -n "$nic" ]; then
    con=$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v n="$nic" '$2==n{print $1; exit}')
    if [ -z "$con" ]; then
      con=$(nmcli -t -f NAME,DEVICE connection show | awk -F: -v n="$nic" '$2==n{print $1; exit}')
    fi
    if [ -n "$con" ]; then
      nmcli connection modify "$con" \
        ipv4.method manual \
        ipv4.addresses "$ADDR" \
        ipv4.gateway "" \
        ipv4.never-default yes \
        ipv6.method auto \
        connection.autoconnect yes || true
      echo "install_orangepi_usb_ipv4: NetworkManager $con -> $ADDR"
    else
      nmcli connection add type ethernet ifname "$nic" con-name pico-usb \
        ipv4.method manual ipv4.addresses "$ADDR" ipv4.never-default yes \
        ipv6.method auto autoconnect yes || true
      echo "install_orangepi_usb_ipv4: added NM connection pico-usb on $nic"
    fi
  fi
fi

echo "install_orangepi_usb_ipv4: enabled pico-usb-ipv4.service"
systemctl is-enabled pico-usb-ipv4.service
ip -br a
