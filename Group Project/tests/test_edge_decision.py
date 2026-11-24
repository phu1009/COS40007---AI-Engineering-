"""Decision engine tests (no camera or model needed, so they run in CI)."""

from pathlib import Path

import yaml

from src.edge import decision as D

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "pad_config.yaml").read_text())


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def obs(reason=D.OK, p=None):
    return D.FrameObservation(reason, p)


def run(frames, clock=None):
    a = D.Attempt(CFG, clock=clock or FakeClock())
    for f in frames:
        a.add(f)
        if a.done():
            break
    return a


def test_live_frames_pass():
    a = run([obs(p=0.9)] * 20)
    assert a.done() and a.result()["decision"] == D.PASS
    assert a.result()["frames_valid"] == CFG["decision"]["frames_per_attempt"]


def test_spoof_frames_are_rejected():
    assert run([obs(p=0.1)] * 20).result()["decision"] == D.SPOOF_SUSPECTED


def test_median_ignores_a_few_outliers():
    frames = [obs(p=0.05)] * 9 + [obs(p=0.99)] * 3
    assert run(frames).result()["decision"] == D.SPOOF_SUSPECTED


def test_no_face_times_out_to_retry_with_reason():
    clock = FakeClock()
    a = D.Attempt(CFG, clock=clock)
    for _ in range(30):
        a.add(obs(D.NO_FACE))
    assert not a.done()
    clock.t = CFG["decision"]["attempt_timeout_s"]
    assert a.done()
    r = a.result()
    assert r["decision"] == D.RETRY and r["reason"] == D.NO_FACE and r["hint"]


def test_multiple_faces_never_pass():
    clock = FakeClock()
    a = D.Attempt(CFG, clock=clock)
    for _ in range(30):
        a.add(obs(D.MULTIPLE_FACES))
    clock.t = 99
    assert a.result()["decision"] == D.RETRY


def test_technical_error_fails_closed_even_with_live_frames():
    a = run([obs(p=0.99)] * 5)
    a.fail("camera timeout")
    assert a.done() and a.result()["decision"] == D.TECH_FAILURE


def test_retry_limiter_locks_after_limit_and_resets_on_pass():
    clock = FakeClock()
    lim = D.RetryLimiter(3, 5, clock=clock)
    lim.register(D.SPOOF_SUSPECTED)
    lim.register(D.PASS)
    lim.register(D.SPOOF_SUSPECTED)
    lim.register(D.RETRY)
    assert lim.locked_for() == 0
    lim.register(D.SPOOF_SUSPECTED)
    assert lim.locked_for() == 5 * 60
    clock.t = 5 * 60
    assert lim.locked_for() == 0


def test_config_lists_versioned_model_files():
    for m in CFG["model"]["files"] + [CFG["preprocessing"]["detector_model"]]:
        assert m["path"].startswith("models/") and len(m["sha256"]) == 64
    for m in CFG["model"]["files"]:
        assert m["crop_scale"] > 1
