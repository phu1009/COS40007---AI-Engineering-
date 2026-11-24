#!/usr/bin/env bash
# One-time setup of the edge liveness service on a Raspberry Pi.
# Target: Raspberry Pi 3 (or newer) + Raspberry Pi OS 64-bit. Run from anywhere:
#   bash scripts/setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # the "Group Project" folder

echo "== 1/5 Checking the board and OS"
board=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "unknown board")
. /etc/os-release
echo "Board: $board | OS: $PRETTY_NAME | arch: $(uname -m) | RAM: $(free -m | awk '/Mem:/{print $2}') MB"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: this needs Raspberry Pi OS (64-bit). ONNX Runtime has no 32-bit ARM build."
  echo "Re-flash the SD card with Raspberry Pi Imager -> Raspberry Pi OS (64-bit) or Lite (64-bit)."
  exit 1
fi

echo "== 2/5 Installing system packages (apt) - this takes a few minutes on a Pi 3"
sudo apt-get update
sudo apt-get install -y python3-venv python3-numpy python3-opencv python3-yaml v4l-utils curl
sudo apt-get install -y --no-install-recommends python3-picamera2 \
  || echo "WARN: python3-picamera2 not installed - USB webcams still work"
sudo apt-get install -y python3-gpiozero || echo "WARN: gpiozero not installed - LEDs disabled"
sudo apt-get install -y python3-lgpio || true

echo "== 3/5 Creating the Python environment (.venv, uses the apt OpenCV/NumPy/Picamera2)"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --disable-pip-version-check -r requirements-edge.txt
if ! .venv/bin/python -c "import cv2, sys; sys.exit(tuple(map(int, cv2.__version__.split('.')[:2])) < (4, 8))"; then
  echo "System OpenCV is older than 4.8 (Raspberry Pi OS Bookworm) - adding OpenCV 4.10 to .venv"
  .venv/bin/pip install --disable-pip-version-check opencv-python-headless==4.10.0.84
fi

echo "== 4/5 Downloading model files"
.venv/bin/python -m src.edge.fetch_models

echo "== 5/5 Self-test"
.venv/bin/python -m src.edge.selftest
echo
echo "Done. Start the service with:"
echo "  source .venv/bin/activate"
echo "  python -m src.edge.app --camera picam     # or: --camera usb"
