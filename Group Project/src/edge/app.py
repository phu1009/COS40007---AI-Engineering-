"""Edge liveness service for the Raspberry Pi (Group 3, Anti-Spoofing Login).

    python -m src.edge.app --camera picam        # Pi Camera Module on the CSI port
    python -m src.edge.app --camera usb          # USB webcam (/dev/video0)
    python -m src.edge.app --camera clip.mp4     # recorded video (no camera needed)

Open http://<pi-address>:8000 in a browser on the same network to see the camera and start a
login attempt. Each attempt collects frames, checks quality, scores liveness on every valid frame
and returns PASS / SPOOF_SUSPECTED / RETRY / TECH_FAILURE. Results are appended to
logs/attempts.jsonl without any image data.
"""

import argparse
import datetime as dt
import json
import socket
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import decision as D
from . import system_info
from .camera import CameraError, open_source
from .pipeline import DEFAULT_CONFIG, ROOT, EdgePipeline, load_config

COLORS = {D.PASS: (60, 180, 75), D.SPOOF_SUSPECTED: (40, 40, 220), D.RETRY: (0, 165, 255),
          D.TECH_FAILURE: (160, 160, 160)}


class SharedState:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.jpeg_seq, self.clients = None, 0, 0
        self.status = {}
        self.attempt_requested = False

    def publish(self, jpeg):
        with self.cond:
            self.jpeg, self.jpeg_seq = jpeg, self.jpeg_seq + 1
            self.cond.notify_all()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Liveness check - Group 3</title>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#101418;color:#e8eaed}
header{padding:12px 20px;background:#1b2128;font-weight:600}
main{display:flex;flex-wrap:wrap;gap:20px;padding:20px}
img{max-width:100%;width:640px;border-radius:8px;background:#000}
aside{flex:1;min-width:260px;max-width:380px}
button{width:100%;padding:14px;font-size:17px;border:0;border-radius:8px;background:#2f6fde;color:#fff;cursor:pointer}
button:disabled{background:#555}
.card{background:#1b2128;border-radius:8px;padding:14px;margin-top:14px;font-size:14px;line-height:1.6}
.big{font-size:26px;font-weight:700}
.PASS{color:#4cc36b}.SPOOF_SUSPECTED{color:#ff5a5a}.RETRY{color:#ffa630}.TECH_FAILURE{color:#aaa}
td{padding:2px 8px 2px 0}
</style></head><body>
<header>Group 3 &middot; Anti-Spoofing Login &middot; Raspberry Pi edge node</header>
<main><img src="/stream.mjpg" alt="camera">
<aside><button id="go">Start login attempt</button>
<div class="card" id="result">No attempt yet.</div><div class="card" id="stats"></div></aside></main>
<script>
const $=id=>document.getElementById(id);
$('go').onclick=async()=>{const r=await fetch('/attempt',{method:'POST'});
  const j=await r.json(); if(!r.ok) $('result').textContent=j.message;};
function row(k,v){return `<tr><td>${k}</td><td>${v??'-'}</td></tr>`}
async function poll(){try{const s=await (await fetch('/status')).json(); const r=s.last_result;
 $('go').disabled=s.attempt_running||s.locked_for_s>0;
 $('go').textContent=s.locked_for_s>0?`Locked (${Math.ceil(s.locked_for_s)} s)`:
   (s.attempt_running?`Checking... ${s.attempt_valid}/${s.attempt_needed}`:'Start login attempt');
 if(s.attempt_running) $('result').innerHTML=`<div class="big">Checking...</div>${s.hint}`;
 else if(r) $('result').innerHTML=`<div class="big ${r.decision}">${r.decision}</div><table>`+
   row('score (live)',r.score_live)+row('threshold',r.threshold)+row('valid frames',`${r.frames_valid}/${r.frames_captured}`)+
   row('reason',r.reason)+row('hint',r.hint)+row('attempt time',r.duration_ms+' ms')+row('attempt id',r.attempt_id)+'</table>';
 $('stats').innerHTML='<table>'+row('FPS',s.fps)+row('detect ms',s.detect_ms)+row('PAD ms',s.pad_ms)+
   row('CPU temp',s.cpu_temp_c==null?'-':s.cpu_temp_c+' &deg;C')+row('throttled',s.throttled)+row('model',s.model_version)+
   row('config',s.config_version)+row('camera',s.camera_error||'ok')+'</table>';
}catch(e){} setTimeout(poll,500)}
poll();
</script></body></html>"""


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                with state.cond:
                    self._json(200, state.status)
            elif self.path.startswith("/stream.mjpg"):
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/attempt":
                return self.send_error(404)
            with state.cond:
                s = state.status
                if s.get("locked_for_s", 0) > 0:
                    return self._json(423, {"message": f"Too many failed attempts - locked for "
                                                       f"{int(s['locked_for_s'])} s."})
                if s.get("attempt_running"):
                    return self._json(409, {"message": "An attempt is already running."})
                state.attempt_requested = True
            self._json(202, {"message": "Attempt started."})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen = 0
            with state.cond:
                state.clients += 1
            try:
                while True:
                    with state.cond:
                        state.cond.wait_for(lambda: state.jpeg_seq != seen, timeout=5)
                        jpeg, seen = state.jpeg, state.jpeg_seq
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            finally:
                with state.cond:
                    state.clients -= 1

    return Handler


class Indicator:
    """Optional LEDs/button on the GPIO header (gpiozero): green = PASS, red = spoof/failure,
    red blinking = retry, green blinking = attempt running; the button starts an attempt."""

    def __init__(self, enabled, green=17, red=27, button=22, on_press=None):
        self.ok = False
        if not enabled:
            return
        try:
            from gpiozero import LED, Button

            self.green, self.red = LED(green), LED(red)
            self.button = Button(button)
            if on_press:
                self.button.when_pressed = on_press
            self.ok = True
        except Exception as e:
            print(f"[gpio] disabled: {e}")

    def busy(self):
        if self.ok:
            self.red.off()
            self.green.blink(0.1, 0.1)

    def show(self, status):
        if not self.ok:
            return
        self.green.off()
        self.red.off()
        if status == D.PASS:
            self.green.blink(3, 0, n=1)
        elif status == D.RETRY:
            self.red.blink(0.3, 0.3, n=4)
        else:
            self.red.blink(3, 0, n=1)


def draw(frame, face, obs, preview, attempt, last, info):
    img = frame.copy()
    h, w = img.shape[:2]
    if face is not None:
        x, y, bw, bh = (int(v) for v in face.box)
        if obs.reason == D.OK and preview is not None:
            live = preview >= info["threshold"]
            color, label = ((60, 180, 75) if live else (40, 40, 220)), f"{'live' if live else 'spoof?'} {preview:.2f}"
        else:
            color, label = (0, 165, 255), obs.reason.lower()
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(img, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(img, (0, 0), (w, 24), (24, 24, 24), -1)
    cv2.putText(img, f"{info['fps']:.1f} FPS | det {info['detect_ms']:.0f} ms | pad {info['pad_ms']:.0f} ms"
                f" | {'-' if info['temp'] is None else info['temp']} C", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    banner = None
    if attempt is not None:
        banner = (f"Checking {attempt.valid_frames}/{attempt.frames_needed} - {attempt.last_hint()}", (200, 120, 40))
    elif last is not None and time.monotonic() < last["_show_until"]:
        txt = last["decision"] + (f"  {last['score_live']:.2f}" if last["score_live"] is not None else "")
        if last["hint"]:
            txt += f"  - {last['hint']}"
        banner = (txt, COLORS[last["decision"]])
    if banner:
        cv2.rectangle(img, (0, h - 36), (w, h), banner[1], -1)
        cv2.putText(img, banner[0], (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def p95(values):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="picam", help="picam | usb | usb:N | video/image path")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-web", action="store_true", help="do not start the browser view")
    ap.add_argument("--show", action="store_true", help="also open a desktop window (Pi desktop only)")
    ap.add_argument("--gpio", action="store_true", help="use LEDs on GPIO17/27 and a button on GPIO22")
    ap.add_argument("--auto-attempt", type=float, default=0,
                    help="start an attempt every N seconds (unattended demos/tests)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run until Ctrl+C)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    rt = cfg.get("runtime", {})
    width, height = args.width or rt.get("camera_width", 640), args.height or rt.get("camera_height", 480)
    pipe = EdgePipeline(cfg)
    limiter = D.RetryLimiter(cfg["decision"]["retry_limit"], cfg["decision"]["lockout_minutes"])
    log_path = ROOT / cfg["logging"].get("attempts_log", "logs/attempts.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = SharedState()

    def request_attempt():
        with state.cond:
            state.attempt_requested = True

    gpio = Indicator(args.gpio, on_press=request_attempt)
    src = open_source(args.camera, width, height, loop=not args.camera.startswith(("picam", "usb")), realtime=True)
    src.open()
    first, _ = src.read()
    pipe.warmup(first)

    if not args.no_web:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[web] open http://{socket.gethostname()}.local:{args.port} (or http://<pi-ip>:{args.port})")
    if sys.stdin and sys.stdin.isatty():
        def stdin_loop():
            for _ in sys.stdin:
                request_attempt()
        threading.Thread(target=stdin_loop, daemon=True).start()
        print("[app] press Enter here (or the web button) to start a login attempt; Ctrl+C to quit")

    print(f"[app] camera={args.camera} {width}x{height} model={pipe.model_version} "
          f"config=v{pipe.config_version} threshold={cfg['decision']['threshold']}")
    attempt, last, preview = None, None, None
    fps, loop_ms, det_hist, pad_hist = 0.0, [], [], []
    health, health_t, next_auto = system_info.snapshot(), 0.0, time.monotonic() + args.auto_attempt
    camera_error, frames = None, 0
    frame = first
    try:
        while True:
            t0 = time.perf_counter()
            try:
                frame, _ = src.read()
                camera_error = None
            except StopIteration:
                break
            except CameraError as e:
                camera_error = str(e)
                if attempt is not None:
                    attempt.fail(f"camera: {e}")
                frame = None
            obs, face = D.FrameObservation(D.NO_FACE), None
            if frame is not None:
                try:
                    obs, face, _ = pipe.process(frame)
                except Exception as e:  # model/runtime error: fail closed
                    if attempt is not None:
                        attempt.fail(f"pipeline: {e}")
                    obs = D.FrameObservation(D.NO_FACE)
            preview = obs.p_live if preview is None or obs.p_live is None else 0.6 * preview + 0.4 * obs.p_live

            now = time.monotonic()
            if args.auto_attempt and now >= next_auto and attempt is None:
                request_attempt()
                next_auto = now + args.auto_attempt
            with state.cond:
                start = state.attempt_requested
                state.attempt_requested = False
            if start and attempt is None and limiter.locked_for() == 0:
                attempt = D.Attempt(cfg)
                gpio.busy()
            elif attempt is not None and frame is not None:
                attempt.add(obs)
            if attempt is not None and attempt.done():
                res = attempt.result()
                res.update({"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                            "device_id": socket.gethostname(), "model_version": pipe.model_version,
                            "config_version": pipe.config_version, "camera": args.camera})
                limiter.register(res["decision"])
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"[attempt] {res['decision']} score={res['score_live']} valid={res['frames_valid']}/"
                      f"{res['frames_captured']} reason={res['reason']} {res['duration_ms']} ms")
                gpio.show(res["decision"])
                last, attempt = dict(res, _show_until=time.monotonic() + 6), None

            if now - health_t > 2:
                health, health_t = system_info.snapshot(), now
            det_hist = (det_hist + [obs.timings_ms.get("detect", 0)])[-100:]
            if "pad" in obs.timings_ms:
                pad_hist = (pad_hist + [obs.timings_ms["pad"]])[-100:]
            info = {"fps": fps, "detect_ms": statistics.fmean(det_hist), "threshold": cfg["decision"]["threshold"],
                    "pad_ms": statistics.fmean(pad_hist) if pad_hist else 0.0, "temp": health["cpu_temp_c"]}
            with state.cond:
                want_jpeg = state.clients > 0
            if frame is not None and (want_jpeg or args.show):
                img = draw(frame, face, obs, preview, attempt, last, info)
                if want_jpeg:
                    state.publish(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
                if args.show:
                    cv2.imshow("Liveness check - press SPACE to start, Q to quit", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        request_attempt()
                    elif key in (ord("q"), 27):
                        break
            with state.cond:
                state.status = {
                    "fps": round(fps, 1), "detect_ms": round(info["detect_ms"], 1), "pad_ms": round(info["pad_ms"], 1),
                    "cpu_temp_c": health["cpu_temp_c"], "throttled": health["throttled"],
                    "model_version": pipe.model_version, "config_version": pipe.config_version,
                    "camera_error": camera_error, "attempt_running": attempt is not None,
                    "attempt_valid": attempt.valid_frames if attempt else 0,
                    "attempt_needed": attempt.frames_needed if attempt else 0,
                    "hint": attempt.last_hint() if attempt else "", "locked_for_s": round(limiter.locked_for(), 1),
                    "last_result": {k: v for k, v in last.items() if not k.startswith("_")} if last else None,
                }
            if camera_error:
                time.sleep(0.5)
            dt_ms = (time.perf_counter() - t0) * 1000
            loop_ms.append(dt_ms)
            fps = 1000 / dt_ms if fps == 0 else 0.9 * fps + 0.1 * (1000 / dt_ms)
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if args.show:
            cv2.destroyAllWindows()
    if loop_ms:
        print(f"[app] {frames} frames, loop median {statistics.median(loop_ms):.1f} ms, "
              f"p95 {p95(loop_ms):.1f} ms, ~{1000 / statistics.median(loop_ms):.1f} FPS")


if __name__ == "__main__":
    main()
