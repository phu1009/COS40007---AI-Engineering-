# Running the edge liveness service on a Raspberry Pi 3

This guide takes a bare Raspberry Pi 3 to a working liveness check that you open in a browser:
camera → YuNet face detection → quality checks → MiniFASNet anti-spoofing (ONNX Runtime) →
`PASS` / `SPOOF_SUSPECTED` / `RETRY` / `TECH_FAILURE`.

Time needed: about 1 hour, most of it waiting for downloads and updates.

```
camera (Pi Camera / USB) ─► YuNet face detector ─► quality check ─► crop 80×80 (×2 scales)
   640×480, newest frame      (on a 320-px copy)     size, light,      ─► MiniFASNet V2 + V1SE
                                                     blur, 1 face          (ONNX Runtime, 4 threads)
                                                                              │ p_live per frame
login attempt: 12 valid frames or 5 s ─► median p_live ≥ threshold ? PASS : SPOOF_SUSPECTED
        (too few valid frames → RETRY with a hint; any camera/model error → TECH_FAILURE, never PASS)
```

## 1. What you need

| Item | Notes |
|---|---|
| Raspberry Pi 3 Model B or B+ | 1 GB RAM is enough. The OS must be **64-bit** (ONNX Runtime has no 32-bit ARM build). |
| Power supply 5 V / 2.5 A, micro-USB | A phone charger often causes under-voltage and slows the Pi down. |
| microSD card, 16 GB or more (32 GB A1/A2 recommended) | Everything on it will be erased. |
| A camera | **Pi Camera Module** (v1.3, v2, v3 or HQ) on the ribbon-cable port, **or** any USB webcam (UVC). |
| Heatsink (recommended) | The benchmark runs the CPU at 100 %; a hot Pi 3 throttles at 80 °C. |
| Laptop on the same Wi-Fi | For SSH and for the browser view. A monitor and keyboard are optional. |
| Optional: 2 LEDs, 2 × 330 Ω resistors, 1 push button, jumper wires | Green/red result lights and a "start" button (section 9). |

> **Wi-Fi:** the Pi 3 Model **B** only has 2.4 GHz Wi-Fi; the 3 **B+** also has 5 GHz. If the Pi never
> appears on the network, check that your router has a 2.4 GHz network.

## 2. Flash the OS (on your laptop)

1. Install **Raspberry Pi Imager** from raspberrypi.com/software and open it.
2. **Device:** Raspberry Pi 3.
3. **OS:** *Raspberry Pi OS (other)* → **Raspberry Pi OS Lite (64-bit)**.
   Lite has no desktop, which leaves more of the 1 GB RAM for the model. Choose
   *Raspberry Pi OS (64-bit)* (with desktop) only if you will plug in a monitor.
   Do **not** pick a 32-bit image.
4. **Storage:** your microSD card.
5. In the customisation step set:
   - hostname: `asl-pi`
   - username and password (you will type them for SSH)
   - Wi-Fi name, password and country
   - time zone
   - **enable SSH** with password authentication
6. Write the card, then put it in the Pi.

## 3. Connect the camera (Pi switched off)

- **Pi Camera Module:** the camera connector is the long slot between the HDMI port and the audio jack.
  Lift its plastic latch gently, push the ribbon in with the **metal contacts facing the HDMI port**,
  press the latch down.
- **USB webcam:** plug it into any USB port.

Power the Pi on. The first boot takes 2–3 minutes.

## 4. Log in and update

From Windows PowerShell (or a macOS/Linux terminal):

```bash
ssh <your-user>@asl-pi.local
```

If `asl-pi.local` is not found, look up the Pi's IP address in your router's device list and use
`ssh <your-user>@<ip>`. Then update the system (10–20 minutes on a Pi 3) and reboot:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

Check that the OS is 64-bit and that the camera is seen:

```bash
uname -m                        # must print aarch64
rpicam-hello --list-cameras     # Pi Camera: lists ov5647 (v1.3), imx219 (v2), imx708 (v3) ...
v4l2-ctl --list-devices         # USB webcam: shows /dev/video0 (sudo apt install -y v4l-utils)
```

If `rpicam-hello` is missing: `sudo apt install -y rpicam-apps-lite`.

## 5. Get the code and install

```bash
sudo apt install -y git
git clone --depth 1 https://github.com/phu1009/COS40007---AI-Engineering-.git ~/asl
cd ~/asl/"Group Project"
bash scripts/setup_pi.sh
```

`setup_pi.sh` installs OpenCV, NumPy and Picamera2 from apt, creates `.venv` with ONNX Runtime, downloads
the three model files (checked by sha256) and runs a self-test. The end of the output should look like:

```
[ OK ] 64-bit OS (aarch64): aarch64
[ OK ] Python >= 3.11: 3.13.x
[ OK ] OpenCV with YuNet (>= 4.8): 4.10.0
[ OK ] ONNX Runtime: 1.32.0
[ OK ] Config + model files (sha256): config v2, 3 files ok
[ OK ] Pipeline runs (detector + PAD): PAD .. ms/frame (2 models)
All checks passed.
```

Test the camera on its own:

```bash
source .venv/bin/activate
python -m src.edge.selftest --camera picam      # or: --camera usb
```

## 6. Run the liveness check

```bash
cd ~/asl/"Group Project" && source .venv/bin/activate
python -m src.edge.app --camera picam           # Pi Camera Module
python -m src.edge.app --camera usb             # USB webcam (usb:1 for /dev/video1)
```

On your laptop or phone (same Wi-Fi) open **http://asl-pi.local:8000**. You see the live camera with
the face box, FPS and timings. Click **Start login attempt** (or press Enter in the SSH window) and
look at the camera for about 2 seconds.

| Try this | Expected result |
|---|---|
| Your own face, good light, 40–80 cm away | `PASS` |
| A photo of a face on a phone screen held to the camera | `SPOOF_SUSPECTED` |
| A printed photo of a face | `SPOOF_SUSPECTED` |
| Nobody in front / hand over the lens | `RETRY` – "No face found" or "Too dark" |
| Two people in view | `RETRY` – "More than one face" |
| Unplug the USB camera during an attempt | `TECH_FAILURE` (never `PASS`) |

After 3 failed attempts in a row the check locks for 5 minutes (`retry_limit`, `lockout_minutes` in
`configs/pad_config.yaml`). Every attempt is appended to `logs/attempts.jsonl` – decision, score, frame
counts, timings, model and config version, no images. Stop the service with Ctrl+C.

Other options: `--gpio` (LEDs and button, section 9), `--show` (desktop window when a monitor is
attached; SPACE starts an attempt), `--port 8080`, `--camera path/to/video.mp4` (replay a recording).

## 7. Measure latency and resources (for the report)

```bash
python -m src.edge.bench --camera picam --frames 500 --tag pi3-fp32-t4
python -m src.edge.bench --pad-only --frames 1000 --threads 1 --tag pi3-pad-t1   # repeat with 2, 3, 4
```

Sit in front of the camera during the pipeline run. Each run saves `logs/bench_<time>.csv` (every frame)
and `logs/bench_<time>.json`: median / p95 / mean per stage (capture, detect, quality, PAD, total),
throughput in FPS, CPU temperature, throttling flags, memory. `throttled_end` must be `0x0`; anything
else means under-voltage or overheating, and the numbers are not valid.

If the Pi 3 is too slow for you, these settings in `configs/pad_config.yaml` trade accuracy for speed
(re-run the benchmark after each change and report it):

- `detector_input_width: 240` instead of 320;
- keep only one entry under `model.files` (the V2 model), which roughly halves PAD time.

## 8. Start automatically at boot (optional)

```bash
bash scripts/install_service.sh picam        # or: usb
journalctl -u asl-edge -f                    # live logs
sudo systemctl disable --now asl-edge        # turn it off again
```

## 9. LEDs and button (optional, with `--gpio`)

| Part | Pi 3 header pin | Wiring |
|---|---|---|
| Green LED (PASS) | pin 11 = GPIO17 | GPIO17 → 330 Ω → LED long leg; LED short leg → GND (pin 9) |
| Red LED (spoof / failure; blinking = retry) | pin 13 = GPIO27 | GPIO27 → 330 Ω → LED long leg; LED short leg → GND (pin 14) |
| Push button (start attempt) | pin 15 = GPIO22 | one side to GPIO22, other side to GND (pin 20) |

The green LED blinks quickly while an attempt is running.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `setup_pi.sh` says the OS is not aarch64, or pip finds no `onnxruntime` | The card has a 32-bit OS. Re-flash with a **64-bit** image (section 2). |
| `Pi camera not available` | Check the ribbon direction and latch; run `rpicam-hello --list-cameras`; close any other program using the camera (only one can). |
| `USB camera /dev/video0 could not be opened` | `v4l2-ctl --list-devices`; try `--camera usb:1`. |
| Browser cannot open `asl-pi.local:8000` | Use the IP address; laptop and Pi must be on the same network (some university/guest Wi-Fi blocks device-to-device traffic – use a phone hotspot). |
| Very low FPS, `throttled` not `0x0` | Better power supply, add a heatsink, see the speed settings in section 7. |
| Always `RETRY – Too dark` / `Image is blurry` | Add light in front of the face; for Camera Module v1/v2 check focus. Thresholds: `min_brightness`, `min_blur_variance`. |
| Real faces get `SPOOF_SUSPECTED` | Light from the front, face 40–80 cm away, look straight. The threshold 0.50 is a placeholder (see below). |
| `model file missing` / `sha256` error | `python -m src.edge.fetch_models`. Without internet, copy the three `.onnx` files into `models/`. |
| OpenCV older than 4.8 (Raspberry Pi OS Bookworm) | `setup_pi.sh` installs OpenCV 4.10 into `.venv` automatically; re-run it. |

## 11. Current limits (state these in the report)

- The model is the **pre-trained MiniFASNet baseline** from Minivision (Apache-2.0), not yet a model trained
  on the project data (CelebA-Spoof / Replay-Attack). It is the Sprint 2 baseline the team model is compared to.
- `threshold: 0.50` is a placeholder. The final value must be chosen on validation data (BPCER ≤ 5 %).
- The Pi 3 camera and lighting differ from the training data (domain gap): test with the team's own
  live and attack recordings and report the results separately.
- Not yet implemented (Sprint 3): session nonce from the backend, signed results, HTTPS between Pi and
  backend. The browser page is a local demo; do not expose port 8000 to the internet.
- The blueprint targets a Raspberry Pi 5. On a Pi 3 the same code runs slower; report the Pi 3 numbers
  from section 7 against the latency targets.

## Files

| Path | Purpose |
|---|---|
| `src/edge/app.py` | The service: camera loop, login attempts, browser view, GPIO, attempt log |
| `src/edge/pipeline.py` | Per-frame pipeline: detect → quality → PAD |
| `src/edge/vision.py` | YuNet detector, quality checks, Silent-Face crop, MiniFASNet ONNX ensemble |
| `src/edge/decision.py` | Attempt aggregation, decision rules, retry limiter (fail closed) |
| `src/edge/camera.py` | Pi Camera / USB / video / image sources behind one interface |
| `src/edge/bench.py` | Latency and resource benchmark → CSV + JSON |
| `src/edge/selftest.py` | Installation and camera check |
| `src/edge/fetch_models.py` | Downloads and verifies the model files |
| `configs/pad_config.yaml` | Versioned settings: models + sha256, threshold, quality limits, retry limits |
| `models/README.md` | Model sources, licences, input/output contract, how to re-export |
| `tools/export_minifasnet_onnx.py` | PyTorch → ONNX export (run on a PC) |
| `scripts/setup_pi.sh`, `scripts/install_service.sh` | One-time install; autostart with systemd |
| `tests/test_edge_decision.py` | CI tests for the decision rules |
