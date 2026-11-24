"""Latency and resource benchmark on the target device (for the report).

    python -m src.edge.bench --camera picam --frames 500
    python -m src.edge.bench --camera samples/ --frames 500     # recorded images, no camera
    python -m src.edge.bench --pad-only --frames 1000           # model inference only

Writes logs/bench_<time>.csv (one row per frame) and logs/bench_<time>.json (median, p95,
throughput, CPU temperature, throttling flags, memory) and prints the summary.
"""

import argparse
import csv
import datetime as dt
import json
import statistics
import time

import numpy as np

from . import system_info
from .camera import open_source
from .pipeline import DEFAULT_CONFIG, ROOT, EdgePipeline, load_config


def summarise(values):
    if not values:
        return None
    s = sorted(values)
    return {"n": len(s), "median": round(statistics.median(s), 2),
            "p95": round(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], 2),
            "mean": round(statistics.fmean(s), 2), "max": round(s[-1], 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="picam")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads (default: config)")
    ap.add_argument("--pad-only", action="store_true", help="time the PAD models on a fixed crop only")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", default="", help="label stored in the summary, e.g. 'pi3-fp32-4threads'")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pipe = EdgePipeline(cfg, threads=args.threads)
    rt = cfg.get("runtime", {})
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    before = system_info.snapshot()
    rows, temps = [], []

    if args.pad_only:
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        box = (220, 140, 200, 200)
        for _ in range(args.warmup):
            pipe.pad.predict(frame, box)
        for i in range(args.frames):
            t = time.perf_counter()
            pipe.pad.predict(frame, box)
            rows.append({"frame": i, "pad_ms": (time.perf_counter() - t) * 1000})
            if i % 50 == 0:
                temps.append(system_info.cpu_temp_c())
    else:
        src = open_source(args.camera, rt.get("camera_width", 640), rt.get("camera_height", 480), loop=True)
        with src:
            frame, _ = src.read()
            pipe.warmup(frame)
            for _ in range(args.warmup):
                pipe.process(src.read()[0])
            t_start = time.perf_counter()
            for i in range(args.frames):
                t0 = time.perf_counter()
                frame, _ = src.read()
                t1 = time.perf_counter()
                obs, _, _ = pipe.process(frame)
                t2 = time.perf_counter()
                rows.append({"frame": i, "reason": obs.reason, "p_live": obs.p_live,
                             "capture_ms": (t1 - t0) * 1000, "detect_ms": obs.timings_ms.get("detect"),
                             "quality_ms": obs.timings_ms.get("quality"), "pad_ms": obs.timings_ms.get("pad"),
                             "process_ms": (t2 - t1) * 1000, "total_ms": (t2 - t0) * 1000})
                if i % 50 == 0:
                    temps.append(system_info.cpu_temp_c())
            wall = time.perf_counter() - t_start

    after = system_info.snapshot()
    keys = [k for k in rows[0] if k.endswith("_ms")]
    summary = {
        "tag": args.tag, "timestamp": stamp, "camera": None if args.pad_only else args.camera,
        "mode": "pad-only" if args.pad_only else "pipeline", "frames": len(rows),
        "onnx_threads": args.threads or rt.get("onnx_threads"), "model_version": pipe.model_version,
        "models": len(pipe.pad.models), "config_version": pipe.config_version,
        "latency_ms": {k: summarise([r[k] for r in rows if r.get(k) is not None]) for k in keys},
        "cpu_temp_c": {"start": before["cpu_temp_c"], "end": after["cpu_temp_c"],
                       "max": max([t for t in temps if t is not None], default=None)},
        "throttled_end": after["throttled"], "board": after["board"],
        "memory_mb": after["memory_mb"], "process_rss_mb": after["process_rss_mb"],
    }
    if not args.pad_only:
        summary["throughput_fps"] = round(len(rows) / wall, 2)
        summary["frames_by_reason"] = {r: sum(1 for x in rows if x["reason"] == r) for r in {x["reason"] for x in rows}}
    with open(out_dir / f"bench_{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / f"bench_{stamp}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved logs/bench_{stamp}.csv and .json")


if __name__ == "__main__":
    main()
