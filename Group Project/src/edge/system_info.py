"""Raspberry Pi health readings for benchmarks and the status page (safe on non-Pi machines)."""

import os
import platform
import subprocess
from pathlib import Path


def board_model():
    p = Path("/proc/device-tree/model")
    return p.read_text(errors="ignore").strip("\x00\n ") if p.exists() else platform.machine()


def cpu_temp_c():
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(p.read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def throttled():
    """`vcgencmd get_throttled`: 0x0 = fine. Bit 0 under-voltage now, bit 2 throttled now,
    bit 16 under-voltage has occurred, bit 18 throttling has occurred."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip().split("=")[-1] or None
    except (OSError, subprocess.SubprocessError):
        return None


def memory_mb():
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) // 1024
    except (OSError, ValueError):
        return None
    return {"total": info.get("MemTotal"), "available": info.get("MemAvailable")}


def process_rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def snapshot():
    return {
        "board": board_model(),
        "cpu_count": os.cpu_count(),
        "load_1m": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "cpu_temp_c": cpu_temp_c(),
        "throttled": throttled(),
        "memory_mb": memory_mb(),
        "process_rss_mb": process_rss_mb(),
    }
