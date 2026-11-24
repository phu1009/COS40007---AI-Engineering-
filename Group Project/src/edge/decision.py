"""Decision engine for one login attempt (pure Python, no camera or model needed).

An attempt collects per-frame observations until it has enough valid frames or times out,
then returns one of four outcomes. Any technical error fails closed: it can never PASS.
"""

import statistics
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

PASS = "PASS"
SPOOF_SUSPECTED = "SPOOF_SUSPECTED"
RETRY = "RETRY"
TECH_FAILURE = "TECH_FAILURE"

# Per-frame reasons. Only OK frames carry a liveness score.
OK = "OK"
NO_FACE = "NO_FACE"
MULTIPLE_FACES = "MULTIPLE_FACES"
FACE_TOO_SMALL = "FACE_TOO_SMALL"
TOO_DARK = "TOO_DARK"
TOO_BRIGHT = "TOO_BRIGHT"
BLURRY = "BLURRY"

USER_HINTS = {
    NO_FACE: "No face found - look straight at the camera.",
    MULTIPLE_FACES: "More than one face - only one person in front of the camera.",
    FACE_TOO_SMALL: "Move closer to the camera.",
    TOO_DARK: "Too dark - add light in front of your face.",
    TOO_BRIGHT: "Too bright - avoid strong light behind or on the camera.",
    BLURRY: "Image is blurry - hold still.",
}


@dataclass
class FrameObservation:
    reason: str
    p_live: float | None = None
    timings_ms: dict = field(default_factory=dict)


@dataclass
class Decision:
    status: str
    score: float | None
    reason: str | None


def decide(observations, threshold, min_valid, aggregation="median", error=None):
    """Turn the frames of one attempt into a decision (fail closed)."""
    if error is not None:
        return Decision(TECH_FAILURE, None, str(error))
    scores = [o.p_live for o in observations if o.reason == OK and o.p_live is not None]
    if len(scores) < min_valid:
        bad = Counter(o.reason for o in observations if o.reason != OK)
        reason = bad.most_common(1)[0][0] if bad else NO_FACE
        return Decision(RETRY, None, reason)
    score = statistics.median(scores) if aggregation == "median" else statistics.fmean(scores)
    if score >= threshold:
        return Decision(PASS, score, None)
    return Decision(SPOOF_SUSPECTED, score, None)


class Attempt:
    """Collects frames for one login attempt."""

    def __init__(self, cfg, clock=time.monotonic):
        d = cfg["decision"]
        self.threshold = float(d["threshold"])
        self.frames_needed = int(d["frames_per_attempt"])
        self.min_valid = int(d.get("min_valid_frames", (self.frames_needed + 1) // 2))
        self.aggregation = d.get("aggregation", "median")
        self.timeout_s = float(d["attempt_timeout_s"])
        self.clock = clock
        self.started = clock()
        self.attempt_id = uuid.uuid4().hex[:12]
        self.observations = []
        self.error = None

    @property
    def valid_frames(self):
        return sum(1 for o in self.observations if o.reason == OK)

    def add(self, obs):
        self.observations.append(obs)

    def fail(self, error):
        self.error = error

    def done(self):
        return (self.error is not None or self.valid_frames >= self.frames_needed
                or self.clock() - self.started >= self.timeout_s)

    def last_hint(self):
        for o in reversed(self.observations[-5:]):
            if o.reason != OK:
                return USER_HINTS.get(o.reason, "")
        return "Hold still..."

    def result(self):
        dec = decide(self.observations, self.threshold, self.min_valid, self.aggregation, self.error)
        stages = {}
        for o in self.observations:
            for k, v in o.timings_ms.items():
                stages.setdefault(k, []).append(v)
        return {
            "attempt_id": self.attempt_id,
            "decision": dec.status,
            "score_live": None if dec.score is None else round(dec.score, 4),
            "threshold": self.threshold,
            "reason": dec.reason,
            "hint": USER_HINTS.get(dec.reason, "") if dec.status == RETRY else "",
            "frames_captured": len(self.observations),
            "frames_valid": self.valid_frames,
            "frame_reasons": dict(Counter(o.reason for o in self.observations)),
            "duration_ms": round((self.clock() - self.started) * 1000, 1),
            "median_stage_ms": {k: round(statistics.median(v), 2) for k, v in stages.items()},
        }


class RetryLimiter:
    """Locks the login after `retry_limit` failed attempts in a row."""

    def __init__(self, retry_limit, lockout_minutes, clock=time.monotonic):
        self.retry_limit = int(retry_limit)
        self.lockout_s = float(lockout_minutes) * 60
        self.clock = clock
        self.failures = 0
        self.locked_until = 0.0

    def locked_for(self):
        return max(0.0, self.locked_until - self.clock())

    def register(self, status):
        if status == PASS:
            self.failures = 0
            return
        if status == TECH_FAILURE:  # logged, but not the user's fault
            return
        self.failures += 1
        if self.failures >= self.retry_limit:
            self.locked_until = self.clock() + self.lockout_s
            self.failures = 0
