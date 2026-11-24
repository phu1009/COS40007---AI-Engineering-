"""Per-frame pipeline: detect -> quality check -> crop -> PAD model -> FrameObservation."""

import time
from pathlib import Path

import yaml

from . import decision as D
from .vision import MiniFASNetPAD, YuNetDetector, check_model_file, check_quality, default_threads, to_training_box

ROOT = Path(__file__).resolve().parents[2]  # the "Group Project" folder
DEFAULT_CONFIG = ROOT / "configs" / "pad_config.yaml"


def load_config(path=DEFAULT_CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class EdgePipeline:
    def __init__(self, cfg, threads=None):
        self.cfg = cfg
        pre = cfg["preprocessing"]
        det = pre["detector_model"]
        self.quality = pre
        self.adapt = pre.get("box_to_training_geometry", [1.0, 1.0, 0.0])
        self.detector = YuNetDetector(check_model_file(ROOT / det["path"], det.get("sha256")),
                                      pre.get("detector_input_width", 320), pre.get("detector_score", 0.7))
        threads = threads or cfg.get("runtime", {}).get("onnx_threads") or default_threads()
        self.pad = MiniFASNetPAD(cfg["model"], ROOT, threads)
        self.model_version = f'{cfg["model"]["name"]}@{cfg["model"]["version"]}'
        self.config_version = cfg["config_version"]

    def process(self, frame):
        """Return (FrameObservation, main face or None, quality metrics)."""
        t0 = time.perf_counter()
        faces = self.detector.detect(frame)
        t1 = time.perf_counter()
        timings = {"detect": (t1 - t0) * 1000}
        if not faces:
            return D.FrameObservation(D.NO_FACE, None, timings), None, {}
        face = faces[0]
        if len(faces) > 1:
            return D.FrameObservation(D.MULTIPLE_FACES, None, timings), face, {}
        reason, metrics = check_quality(frame, face, self.quality)
        t2 = time.perf_counter()
        timings["quality"] = (t2 - t1) * 1000
        if reason != D.OK:
            return D.FrameObservation(reason, None, timings), face, metrics
        p_live = self.pad.predict(frame, to_training_box(face.box, self.adapt))
        timings["pad"] = (time.perf_counter() - t2) * 1000
        return D.FrameObservation(D.OK, p_live, timings), face, metrics

    def warmup(self, frame, n=3):
        """First ONNX Runtime calls are slow; run a few before timing anything."""
        h, w = frame.shape[:2]
        for _ in range(n):
            self.detector.detect(frame)
            self.pad.predict(frame, (w / 4, h / 4, w / 2, h / 2))
