#!/usr/bin/env bash
# Start the edge service automatically at boot (systemd). Usage:
#   bash scripts/install_service.sh picam        # or: usb, usb:1
# Undo: sudo systemctl disable --now asl-edge && sudo rm /etc/systemd/system/asl-edge.service
set -euo pipefail
cd "$(dirname "$0")/.."
CAMERA="${1:-picam}"
DIR="$(pwd)"
sudo tee /etc/systemd/system/asl-edge.service > /dev/null << UNIT
[Unit]
Description=Group 3 anti-spoofing edge liveness service
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart="$DIR/.venv/bin/python" -m src.edge.app --camera $CAMERA --gpio
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now asl-edge
sleep 2
systemctl --no-pager status asl-edge | head -n 12
echo "Logs: journalctl -u asl-edge -f"
