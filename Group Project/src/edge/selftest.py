"""Check that the Pi is ready: OS, Python packages, model files, models running, camera.

    python -m src.edge.selftest                    # software + models
    python -m src.edge.selftest --camera picam     # ... and grab frames from the camera
    python -m src.edge.selftest --image a.jpg b.jpg  # score photos (prints p_live per image)
"""

import argparse
import platform
import sys
import time

CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=None, help="picam | usb | usb:N")
    ap.add_argument("--image", nargs="*", default=[])
    args = ap.parse_args(argv)
    ctx = {}

    @check("64-bit OS (aarch64)")
    def _arch():
        m = platform.machine()
        assert m in ("aarch64", "x86_64", "AMD64"), f"{m}: install Raspberry Pi OS (64-bit)"
        return m

    @check("Python >= 3.11")
    def _py():
        assert sys.version_info >= (3, 11), sys.version
        return platform.python_version()

    @check("OpenCV with YuNet (>= 4.8)")
    def _cv():
        import cv2
        version = tuple(int(v) for v in cv2.__version__.split(".")[:2])
        assert hasattr(cv2, "FaceDetectorYN") and version >= (4, 8), \
            f"OpenCV {cv2.__version__} is too old for YuNet 2023mar (rerun scripts/setup_pi.sh)"
        return cv2.__version__

    @check("ONNX Runtime")
    def _ort():
        import onnxruntime
        return onnxruntime.__version__

    @check("Config + model files (sha256)")
    def _cfg():
        from .pipeline import ROOT, load_config
        from .vision import check_model_file
        cfg = load_config()
        files = [cfg["preprocessing"]["detector_model"]] + list(cfg["model"]["files"])
        for f in files:
            check_model_file(ROOT / f["path"], f.get("sha256"))
        ctx["cfg"] = cfg
        return f"config v{cfg['config_version']}, {len(files)} files ok"

    @check("Pipeline runs (detector + PAD)")
    def _pipe():
        import numpy as np
        from .pipeline import EdgePipeline
        pipe = EdgePipeline(ctx["cfg"])
        frame = np.full((480, 640, 3), 128, np.uint8)
        pipe.warmup(frame)
        t = time.perf_counter()
        for _ in range(10):
            pipe.pad.predict(frame, (220, 140, 200, 200))
        ctx["pipe"] = pipe
        return f"PAD {(time.perf_counter() - t) * 100:.1f} ms/frame ({len(pipe.pad.models)} models)"

    if args.camera:
        @check(f"Camera '{args.camera}'")
        def _cam():
            from .camera import open_source
            with open_source(args.camera) as src:
                for _ in range(5):
                    frame, _ = src.read()
            return f"{frame.shape[1]}x{frame.shape[0]} frames ok"

    failed = 0
    for name, fn in CHECKS:
        try:
            print(f"[ OK ] {name}: {fn()}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
            if name.startswith(("Config", "Pipeline")):
                break

    if args.image and "pipe" in ctx:
        import cv2
        thr = ctx["cfg"]["decision"]["threshold"]
        for path in args.image:
            frame = cv2.imread(path)
            if frame is None:
                print(f"{path}: cannot read")
                continue
            obs, face, m = ctx["pipe"].process(frame)
            verdict = "-" if obs.p_live is None else ("LIVE" if obs.p_live >= thr else "SPOOF")
            score = "" if obs.p_live is None else f" p_live={obs.p_live:.3f}"
            print(f"{path}: {obs.reason}{score} -> {verdict}")
    print("All checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
