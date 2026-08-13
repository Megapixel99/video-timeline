#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seth Wheeler
"""
Cross-check the converter's camera rates with an independent algorithm.

Real footage has no ground truth, so agreement between two methods that share no
code and no assumptions is the strongest available evidence. This script
measures displacement by FFT phase correlation — frequency domain, full-ish
resolution, no block matching, no proxy — and compares it with what
describe_camera reports from spatial SAD matching on a 96 px proxy.

It calibrates itself on fixture_motion.mp4 first, where the true rates are known
analytically. A validator you haven't validated is worth nothing.

Usage:  python3 tests/crosscheck_motion.py [video ...]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import vtl_convert as V  # noqa: E402

FPS = 8.0
PC_WIDTH = 480          # deliberately different from the converter's 96 px proxy


def decode_gray(path: Path, w: int, ss=None, dur=None) -> np.ndarray:
    """Decode to a w-wide grayscale stream, height preserved by aspect."""
    src = V.probe(path, do_hash=False)
    sw, sh = src.width or 1920, src.height or 1080
    h = max(2, int(round(sh * (w / sw) / 2)) * 2)
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", str(ss)]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-an", "-sn", "-vf", f"fps={FPS},scale={w}:{h}",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)


def phase_correlate(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Displacement of content from a to b, by phase correlation.

    Returns (dx, dy, peak_sharpness). Sign convention matches the converter's:
    positive dx means content moved right.
    """
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))          # kill edge wrap-around
    A = np.fft.rfft2((a - a.mean()) * win)
    B = np.fft.rfft2((b - b.mean()) * win)
    R = np.conj(A) * B          # conj(A)*B gives the a->b shift; A*conj(B) is b->a
    R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=(h, w))
    k = int(np.argmax(r))
    py, px = divmod(k, w)
    peak = float(r[py, px])
    # unwrap: a peak past the midpoint is a negative shift
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px

    # sub-pixel by parabolic fit on the three samples around the peak
    def refine(c0, cm, cp):
        den = cm - 2 * c0 + cp
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))
    sx = refine(peak, float(r[py, (px - 1) % w]), float(r[py, (px + 1) % w]))
    sy = refine(peak, float(r[(py - 1) % h, px]), float(r[(py + 1) % h, px]))
    return dx + sx, dy + sy, peak


def rates(path: Path, t0: float, t1: float) -> tuple[float, float, float]:
    """(pan, tilt, confidence) in the converter's camera-relative units."""
    fr = decode_gray(path, PC_WIDTH, t0, t1 - t0)
    if fr.shape[0] < 3:
        return 0.0, 0.0, 0.0
    h, w = fr.shape[1], fr.shape[2]
    dxs, dys, peaks = [], [], []
    for i in range(1, fr.shape[0]):
        dx, dy, pk = phase_correlate(fr[i - 1], fr[i])
        dxs.append(dx); dys.append(dy); peaks.append(pk)
    # same sign convention as describe_camera: pan = -dx, tilt = +dy
    return (-float(np.median(dxs)) / w * FPS,
            float(np.median(dys)) / h * FPS,
            float(np.median(peaks)))


def converter_rates(path: Path, t0: float, t1: float) -> tuple[str, float, float]:
    src = V.probe(path, do_hash=False)
    aw, ah = V.analysis_dims(src)
    vs = V.visual_signals(V.decode_proxy(src, FPS, aw, ah, None, None), FPS, 0.0)
    label, ev = V.describe_camera(vs, int(t0 * FPS), int(t1 * FPS))
    def field(name):
        try:
            return float(ev.split(f"{name} ")[1].split()[0])
        except (IndexError, ValueError):
            return 0.0
    return label, field("pan"), field("tilt")


def report(path: Path, segments, truth=None) -> None:
    print(f"\n{path.name}")
    print(f"  {'segment':>14}  {'converter (SAD, 96px)':<34} {'phase corr (FFT, 480px)':<26} agree?")
    for i, (t0, t1) in enumerate(segments):
        label, cp, ct = converter_rates(path, t0, t1)
        pp, pt, conf = rates(path, t0, t1)
        # agreement: both small, or same sign and within a factor of ~2.5
        def ok(a, b):
            if max(abs(a), abs(b)) < 0.015:
                return True
            if a * b <= 0:
                return False
            r = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-6)
            return r < 2.5
        agree = ok(cp, pp) and ok(ct, pt)
        extra = ""
        if truth:
            extra = f"   truth {truth[i]}"
        print(f"  {f'{t0:.1f}-{t1:.1f}s':>14}  {label:<12} pan{cp:+.3f} tilt{ct:+.3f}   "
              f"pan{pp:+.3f} tilt{pt:+.3f} (pk {conf:.2f})  {'YES' if agree else 'NO'}{extra}")


if __name__ == "__main__":
    fx = HERE / "fixture_motion.mp4"
    if fx.exists():
        print("=" * 96)
        print("CALIBRATION — fixture with analytically known rates")
        print("=" * 96)
        report(fx, [(0.3, 3.7), (12.3, 15.7), (16.3, 19.7)],
               truth=["pan 0 tilt 0", "pan +0.195", "tilt +0.139"])

    args = sys.argv[1:]
    if args:
        print("\n" + "=" * 96)
        print("REAL FOOTAGE — no ground truth; agreement between methods is the evidence")
        print("=" * 96)
    for a in args:
        p = Path(a).expanduser()
        import json
        b = Path("/tmp/_xcheck.vtl")
        subprocess.run([sys.executable, str(HERE.parent / "vtl_convert.py"), str(p),
                        "-o", str(b), "--force", "--ocr", "off", "--no-sheets",
                        "--no-hash"], check=True, capture_output=True)
        d = json.loads((b / "timeline.json").read_text())
        segs = [(s["start"] + 0.2, s["end"] - 0.2) for s in d["shots"]
                if s["duration"] > 1.0]
        report(p, segs)
