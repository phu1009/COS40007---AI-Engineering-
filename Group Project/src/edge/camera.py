"""Frame sources for the edge service.

Live cameras (Pi Camera Module through Picamera2, or a USB webcam) and recorded sources
(video file, image or folder of images) share one interface, so the same pipeline runs on
the Pi and in automated tests without a camera.
"""

import threading
import time
from pathlib import Path

import cv2

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class CameraError(RuntimeError):
    """Camera missing, busy, or not delivering frames."""


class FrameSource:
    live = False

    def open(self):
        return self

    def read(self):
        """Return (frame_bgr, timestamp). Raise CameraError on failure, StopIteration at the end."""
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class PiCameraSource(FrameSource):
    """Raspberry Pi Camera Module (v1.3, v2, v3, HQ) on the CSI port, through Picamera2."""

    live = True

    def __init__(self, width=640, height=480):
        self.size = (width, height)
        self.cam = None

    def open(self):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CameraError("picamera2 not installed (sudo apt install python3-picamera2)") from e
        try:
            self.cam = Picamera2()
            # "RGB888" gives pixels in B,G,R order - the layout OpenCV expects.
            cfg = self.cam.create_video_configuration(main={"size": self.size, "format": "RGB888"},
                                                      buffer_count=2)
            self.cam.configure(cfg)
            self.cam.start()
        except Exception as e:  # no camera, cable, or camera in use by another program
            raise CameraError(f"Pi camera not available: {e}") from e
        return self

    def read(self):
        try:
            return self.cam.capture_array("main"), time.time()
        except Exception as e:
            raise CameraError(f"Pi camera read failed: {e}") from e

    def close(self):
        if self.cam is not None:
            try:
                self.cam.stop()
                self.cam.close()
            except Exception:
                pass
            self.cam = None


class USBCameraSource(FrameSource):
    """UVC webcam through OpenCV + V4L2 (/dev/videoN)."""

    live = True

    def __init__(self, index=0, width=640, height=480):
        self.index, self.width, self.height = index, width, height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise CameraError(f"USB camera /dev/video{self.index} could not be opened")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # faster over USB 2.0
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("USB camera returned no frame")
        return frame, time.time()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFileSource(FrameSource):
    """Recorded video. `realtime=True` plays it at its own frame rate, like a live camera."""

    def __init__(self, path, loop=False, realtime=False):
        self.path, self.loop, self.realtime = str(path), loop, realtime
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise CameraError(f"cannot open video {self.path}")
        self.period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.next_t = time.monotonic()
        return self

    def read(self):
        if self.realtime:
            delay = self.next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_t = max(self.next_t + self.period, time.monotonic() - self.period)
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame, time.time()

    def close(self):
        if self.cap is not None:
            self.cap.release()


class ImageSource(FrameSource):
    """One image or every image in a folder; `loop=True` repeats them (for demos/benchmarks)."""

    def __init__(self, path, loop=False):
        p = Path(path)
        self.files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
        if not self.files:
            raise CameraError(f"no images in {path}")
        self.loop, self.i = loop, 0
        self.current = None

    def read(self):
        if self.i >= len(self.files):
            if not self.loop:
                raise StopIteration
            self.i = 0
        f = self.files[self.i]
        self.i += 1
        frame = cv2.imread(str(f))
        if frame is None:
            raise CameraError(f"cannot read image {f}")
        self.current = f
        return frame, time.time()


class LatestFrameReader(FrameSource):
    """Reads a live camera in a background thread and always hands out the newest frame,
    so slow processing never works on stale, buffered frames."""

    def __init__(self, source, timeout_s=2.0):
        self.source, self.timeout_s = source, timeout_s
        self.live = source.live
        self.frame, self.ts, self.seq, self.error = None, 0.0, 0, None
        self.cond = threading.Condition()
        self.running = False
        self.last_seq = 0

    def open(self):
        self.source.open()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while self.running:
            try:
                frame, ts = self.source.read()
            except Exception as e:  # keep the error and let read() raise it
                with self.cond:
                    self.error = e
                    self.cond.notify_all()
                return
            with self.cond:
                self.frame, self.ts, self.seq = frame, ts, self.seq + 1
                self.cond.notify_all()

    def read(self):
        with self.cond:
            ok = self.cond.wait_for(lambda: self.seq != self.last_seq or self.error is not None,
                                    timeout=self.timeout_s)
            if self.error is not None:
                raise self.error if isinstance(self.error, (CameraError, StopIteration)) \
                    else CameraError(str(self.error))
            if not ok:
                raise CameraError(f"camera timeout: no frame for {self.timeout_s:.1f} s")
            self.last_seq = self.seq
            return self.frame, self.ts

    def close(self):
        self.running = False
        self.source.close()


def open_source(spec, width=640, height=480, loop=False, realtime=False):
    """'picam', 'usb', 'usb:1', a video file, an image or a folder of images."""
    if spec == "picam":
        return LatestFrameReader(PiCameraSource(width, height))
    if spec.startswith("usb"):
        index = int(spec.split(":")[1]) if ":" in spec else 0
        return LatestFrameReader(USBCameraSource(index, width, height))
    p = Path(spec)
    if not p.exists():
        raise CameraError(f"unknown camera '{spec}' (use picam, usb, usb:N, a video or an image path)")
    if p.is_dir() or p.suffix.lower() in IMAGE_EXT:
        return ImageSource(p, loop=loop)
    return VideoFileSource(p, loop=loop, realtime=realtime)
