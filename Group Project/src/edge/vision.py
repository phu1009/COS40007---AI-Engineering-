"""Face detection (OpenCV YuNet), frame quality checks and the MiniFASNet PAD model (ONNX Runtime)."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import decision as D


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_model_file(path, expected_sha):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model file missing: {path} (run scripts/download_models.sh)")
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError(f"model file {path.name} does not match its sha256 in the config")
    return path


@dataclass
class Face:
    box: tuple  # x, y, w, h in frame pixels
    score: float
    landmarks: np.ndarray  # 5 x 2: right eye, left eye, nose tip, mouth corners


class YuNetDetector:
    """OpenCV YuNet (face_detection_yunet_2023mar, OpenCV >= 4.8). Runs on a downscaled copy of
    the frame (`input_width` px wide) and returns faces in full-frame coordinates, largest first."""

    def __init__(self, model_path, input_width=320, score_threshold=0.7, nms_threshold=0.3):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(f"OpenCV {cv2.__version__} has no FaceDetectorYN (need >= 4.8)")
        self.input_width = input_width
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (input_width, input_width),
                                             score_threshold, nms_threshold, 20)
        self.size = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = self.input_width / w
        small = cv2.resize(frame, (self.input_width, int(round(h * s)))) if s < 1 else frame
        s = min(s, 1.0)
        size = (small.shape[1], small.shape[0])
        if size != self.size:
            self.det.setInputSize(size)
            self.size = size
        _, out = self.det.detect(small)
        faces = []
        for f in (out if out is not None else []):
            box = tuple(float(v) / s for v in f[:4])
            faces.append(Face(box, float(f[14]), (f[4:14].reshape(5, 2) / s)))
        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        return faces


def check_quality(frame, face, q):
    """Return (reason, metrics) for one face; reason is decision.OK when the frame is usable."""
    x, y, w, h = (int(round(v)) for v in face.box)
    x0, y0 = max(x, 0), max(y, 0)
    roi = frame[y0:y + h, x0:x + w]
    if roi.size == 0:
        return D.NO_FACE, {}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 112:  # sharpness is measured at a fixed size so it does not depend on distance
        gray = cv2.resize(gray, (112, int(gray.shape[0] * 112 / gray.shape[1])))
    m = {"face_px": w, "brightness": float(gray.mean()),
         "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}
    if w < q["min_face_px"]:
        return D.FACE_TOO_SMALL, m
    if m["brightness"] < q["min_brightness"]:
        return D.TOO_DARK, m
    if m["brightness"] > q.get("max_brightness", 235):
        return D.TOO_BRIGHT, m
    if m["sharpness"] < q["min_blur_variance"]:
        return D.BLURRY, m
    return D.OK, m


def to_training_box(box, adapt):
    """YuNet boxes are taller and narrower than the RetinaFace boxes MiniFASNet was trained on.
    Map them to the same geometry: width*kw, height*kh, centre moved down by dy*height."""
    kw, kh, dy = adapt
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2 + dy * h
    return cx - w * kw / 2, cy - h * kh / 2, w * kw, h * kh


def crop_patch(frame, box, scale, out_size=80):
    """Same crop as Silent-Face-Anti-Spoofing: enlarge the box `scale` times around its centre,
    shift it inside the image, resize to out_size x out_size."""
    src_h, src_w = frame.shape[:2]
    x, y, bw, bh = box
    scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = x + bw / 2, y + bh / 2
    left, top, right, bottom = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if left < 0:
        right, left = right - left, 0
    if top < 0:
        bottom, top = bottom - top, 0
    if right > src_w - 1:
        left, right = left - (right - src_w + 1), src_w - 1
    if bottom > src_h - 1:
        top, bottom = top - (bottom - src_h + 1), src_h - 1
    patch = frame[int(top):int(bottom) + 1, int(left):int(right) + 1]
    return cv2.resize(patch, (out_size, out_size))


class MiniFASNetPAD:
    """Ensemble of MiniFASNet ONNX models; p_live = mean softmax probability of the live class."""

    def __init__(self, model_cfg, root, threads=4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.live_idx = int(model_cfg.get("live_class_index", 1))
        self.models = []
        for m in model_cfg["files"]:
            path = check_model_file(Path(root) / m["path"], m.get("sha256"))
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            self.models.append((sess, float(m["crop_scale"]), sess.get_inputs()[0].name))

    def predict(self, frame, box):
        probs = []
        for sess, scale, name in self.models:
            x = crop_patch(frame, box, scale).transpose(2, 0, 1)[None].astype(np.float32)
            probs.append(sess.run(None, {name: x})[0][0])
        return float(np.mean([p[self.live_idx] for p in probs]))


def default_threads():
    return min(4, os.cpu_count() or 1)
