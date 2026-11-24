"""Download the model files listed in configs/pad_config.yaml into models/ and check their sha256.

    python -m src.edge.fetch_models

Weights are not committed to git (see .gitignore); they are attached to a GitHub Release of this
repo (tag `pad-model-v<model.version>`). Set MODELS_BASE_URL to use another location.
"""

import hashlib
import os
import sys
import urllib.request

from .pipeline import ROOT, load_config

REPO = "https://github.com/phu1009/COS40007---AI-Engineering-"
FALLBACK = {  # official upstream copy, used if the release is not reachable
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def download(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    cfg = load_config()
    base = os.environ.get("MODELS_BASE_URL", f"{REPO}/releases/download/pad-model-v{cfg['model']['version']}")
    files = list(cfg["model"]["files"]) + [cfg["preprocessing"]["detector_model"]]
    failed = 0
    for f in files:
        dest = ROOT / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and sha256_bytes(dest.read_bytes()) == f["sha256"]:
            print(f"[ OK ] {dest.name} already present")
            continue
        urls = [f"{base}/{dest.name}"] + ([FALLBACK[dest.name]] if dest.name in FALLBACK else [])
        for url in urls:
            try:
                data = download(url)
            except Exception as e:
                print(f"[....] {dest.name}: {url} -> {e}")
                continue
            if sha256_bytes(data) != f["sha256"]:
                print(f"[....] {dest.name}: sha256 mismatch from {url}")
                continue
            dest.write_bytes(data)
            print(f"[ OK ] {dest.name} ({len(data) / 1e6:.2f} MB)")
            break
        else:
            failed += 1
            print(f"[FAIL] {dest.name}: copy it into {dest.parent} by hand (see docs/edge/raspberry-pi-3-setup.md)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
