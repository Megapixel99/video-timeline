#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seth Wheeler
"""
vtl_convert.py — convert an MP4/MOV into a VTL (Video Timeline) bundle.

A VTL bundle is a directory of plain files that describes a video as a measured
timeline rather than a pile of screenshots. See SPEC.md for the format.

    python3 vtl_convert.py clip.mov
    python3 vtl_convert.py clip.mp4 -o /tmp/clip.vtl --max-frames 40

Requires: ffmpeg, numpy, Pillow.
Optional: tesseract (on-screen text), a whisper CLI (transcript).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FORMAT_VERSION = "VTL/1.0"
CONVERTER_VERSION = "1.0.0"

# Analysis proxy: every frame is decoded down to a box this many pixels wide/tall.
# All motion and cut metrics are computed on this proxy, not on full frames.
ANALYSIS_BOX = 96
COARSE_SEARCH = 4          # +/- px on the half-resolution proxy (=> +/-8 px full)
FINE_SEARCH = 2            # +/- px refinement at proxy resolution
BLOCK_SEARCH = 3           # +/- px each grid block may search from the global shift
DARK_LUMA = 16             # mean luma below this counts as "near black"
AUDIO_SR = 16000
AUDIO_HOP = 0.05           # seconds per audio analysis window


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def fmt_ts(t: float) -> str:
    """Seconds -> 00:01:23.400"""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_dur(t: float) -> str:
    if t >= 60:
        return f"{int(t // 60)}m{t % 60:04.1f}s"
    return f"{t:.2f}s"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # errors="replace": tesseract and ffmpeg both emit non-UTF-8 bytes on occasion
    # (OCR garbage, filenames), and a decode error must not kill the conversion.
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", **kw)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def log(msg: str) -> None:
    print(f"  {msg}", file=sys.stderr, flush=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# source probing  (ffprobe is not assumed — we parse `ffmpeg -i` stderr)
# --------------------------------------------------------------------------

@dataclass
class Source:
    path: str
    filename: str
    size_bytes: int
    sha256: str | None
    container_duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    rotation: float
    video_codec: str | None
    audio_codec: str | None
    audio_channels: str | None
    audio_rate: int | None
    has_audio: bool
    raw_probe: str = field(repr=False, default="")


def probe(path: Path, do_hash: bool = True) -> Source:
    p = run(["ffmpeg", "-hide_banner", "-i", str(path)])
    txt = p.stderr

    dur = None
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    rotation = 0.0
    m = re.search(r"rotation of\s*(-?[\d.]+)\s*degrees", txt)
    if m:
        rotation = float(m.group(1))

    vline = None
    aline = None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("Stream #") and " Video: " in s and vline is None:
            vline = s
        elif s.startswith("Stream #") and " Audio: " in s and aline is None:
            aline = s

    width = height = None
    fps = None
    vcodec = None
    if vline:
        m = re.search(r"Video:\s*([A-Za-z0-9_.\-]+)", vline)
        vcodec = m.group(1) if m else None
        # take the last WxH before the bitrate — SAR/DAR come after it
        dims = re.findall(r"\b(\d{2,5})x(\d{2,5})\b", vline)
        if dims:
            width, height = int(dims[0][0]), int(dims[0][1])
        m = re.search(r"([\d.]+)\s*fps", vline)
        if m:
            fps = float(m.group(1))
        elif (m := re.search(r"([\d.]+)\s*tbr", vline)):
            fps = float(m.group(1))

    acodec = achan = None
    arate = None
    has_audio = aline is not None
    if aline:
        m = re.search(r"Audio:\s*([A-Za-z0-9_.\-]+)", aline)
        acodec = m.group(1) if m else None
        m = re.search(r"(\d+)\s*Hz", aline)
        arate = int(m.group(1)) if m else None
        m = re.search(r"Hz,\s*([a-z0-9.()]+)", aline)
        achan = m.group(1) if m else None

    # ffmpeg auto-applies the display matrix on decode, so decoded frames are
    # already upright and the decoded dimensions are swapped for +/-90.
    if width and height and abs(rotation) in (90.0, 270.0):
        width, height = height, width

    digest = None
    if do_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()

    return Source(
        path=str(path.resolve()), filename=path.name, size_bytes=path.stat().st_size,
        sha256=digest, container_duration=dur, width=width, height=height, fps=fps,
        rotation=rotation, video_codec=vcodec, audio_codec=acodec,
        audio_channels=achan, audio_rate=arate, has_audio=has_audio, raw_probe=txt,
    )


# --------------------------------------------------------------------------
# visual analysis: one decode pass into a small RGB proxy stream
# --------------------------------------------------------------------------

def analysis_dims(src: Source) -> tuple[int, int]:
    w, h = src.width or 1920, src.height or 1080
    scale = ANALYSIS_BOX / max(w, h)
    aw = max(16, int(round(w * scale / 2)) * 2)
    ah = max(16, int(round(h * scale / 2)) * 2)
    return aw, ah


def decode_proxy(src: Source, fps: float, aw: int, ah: int,
                 ss: float | None, dur: float | None) -> np.ndarray:
    """Decode the whole video into an (T, ah, aw, 3) uint8 array."""
    cmd = ["ffmpeg", "-v", "error"]
    if ss:
        cmd += ["-ss", f"{ss}"]
    cmd += ["-i", src.path]
    if dur:
        cmd += ["-t", f"{dur}"]
    cmd += ["-an", "-sn", "-vf", f"fps={fps},scale={aw}:{ah}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0 and not raw:
        raise RuntimeError(f"ffmpeg decode failed:\n{err.decode(errors='replace')[-2000:]}")
    fsz = aw * ah * 3
    n = len(raw) // fsz
    if n == 0:
        raise RuntimeError("no frames decoded from the video")
    return np.frombuffer(raw[: n * fsz], dtype=np.uint8).reshape(n, ah, aw, 3)


def blur3(A: np.ndarray) -> np.ndarray:
    """Separable 3-tap gaussian over the last two axes.

    Downscaling a detailed frame to a 96 px proxy aliases high-frequency detail
    into a pattern that can repeat, and block matching then locks onto a false
    repeat of that pattern — confidently, with a low residual. Measured on a
    test pattern this turned a true +0.94 px/frame drift into -3.94. One blur
    pass removes the aliasing and recovers +0.97.
    """
    out = A
    p = np.pad(out, [(0, 0)] * (out.ndim - 2) + [(1, 1), (0, 0)], mode="edge")
    out = 0.25 * p[..., :-2, :] + 0.5 * p[..., 1:-1, :] + 0.25 * p[..., 2:, :]
    p = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(1, 1)], mode="edge")
    out = 0.25 * p[..., :-2] + 0.5 * p[..., 1:-1] + 0.25 * p[..., 2:]
    return out


def luma(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def color_hists(frames: np.ndarray, bins: int = 4) -> np.ndarray:
    """Normalized (T, bins^3) RGB histograms — used for scene similarity."""
    q = (frames.astype(np.uint16) * bins // 256).clip(0, bins - 1)
    idx = q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]
    T = frames.shape[0]
    flat = idx.reshape(T, -1)
    out = np.zeros((T, bins ** 3), np.float32)
    for i in range(T):
        out[i] = np.bincount(flat[i], minlength=bins ** 3)
    out /= flat.shape[1]
    return out


def luma_hists(L: np.ndarray, bins: int = 32) -> np.ndarray:
    q = (L / (256.0 / bins)).astype(np.int32).clip(0, bins - 1)
    T = L.shape[0]
    flat = q.reshape(T, -1)
    out = np.zeros((T, bins), np.float32)
    for i in range(T):
        out[i] = np.bincount(flat[i], minlength=bins)
    out /= flat.shape[1]
    return out


def _sad(a: np.ndarray, b: np.ndarray, m: int, dx: int, dy: int) -> float:
    h, w = a.shape
    return float(np.abs(a[m:h - m, m:w - m] -
                        b[m + dy:h - m + dy, m + dx:w - m + dx]).mean())


def estimate_motion_model(a: np.ndarray, b: np.ndarray, gdx: float, gdy: float,
                          trackable: list | None = None
                          ) -> tuple[float, float, float]:
    """Fit translation *and* scale together. Returns (scale, tx, ty), where
    scale is the fractional size change per frame (>0 zooming in) and tx/ty are
    the translation in proxy pixels.

    A 3x3 grid of overlapping half-size blocks is tracked, then a similarity
    model `u = tx + s*x` is fitted to their displacements. Under a zoom, blocks
    move away from the centre in proportion to their distance from it; under a
    pan they move together and s comes out ~0.

    Two reasons this beats measuring each separately:
      - Comparing just two halves (the obvious cheap zoom test) can't tell a
        zoom from one moving object; nine blocks with a trimmed fit outvote it.
      - A single whole-frame translation is a bad model *during* a zoom, so the
        SAD minimum wanders and invents a pan or tilt. Solving jointly and
        re-deriving translation from the residuals removes that phantom motion.
    """
    trackable = trackable if trackable is not None else [0.0]
    h, w = a.shape
    bw, bh = w // 2, h // 2
    if bw < 24 or bh < 16:
        return 0.0, gdx, gdy
    xs, ys, us, vs_ = [], [], [], []
    for fy in (0.25, 0.5, 0.75):
        for fx in (0.25, 0.5, 0.75):
            x0 = int(clamp(fx * w - bw / 2, 0, w - bw))
            y0 = int(clamp(fy * h - bh / 2, 0, h - bh))
            blk = a[y0:y0 + bh, x0:x0 + bw]
            if float(blk.std()) < 6.0:
                continue          # nothing to track in a flat patch
            u, v, _, edge = estimate_translation(blk, b[y0:y0 + bh, x0:x0 + bw],
                                                 fine=BLOCK_SEARCH, init=(gdx, gdy))
            if edge:
                continue          # ran out of search range: not a measurement
            xs.append(x0 + bw / 2 - w / 2)
            ys.append(y0 + bh / 2 - h / 2)
            us.append(u)
            vs_.append(v)
    trackable[0] = len(xs) / 9.0
    if len(xs) < 4:
        return 0.0, gdx, gdy      # too few usable blocks to fit anything
    X, Y = np.array(xs), np.array(ys)
    U, V = np.array(us), np.array(vs_)
    denom = float((X ** 2 + Y ** 2).sum())
    if denom < 1e-6:
        return 0.0, gdx, gdy

    tx, ty = float(np.median(U)), float(np.median(V))
    s = 0.0
    for _ in range(2):
        du, dv = U - tx, V - ty
        s = float((du * X + dv * Y).sum()) / denom
        resid = (du - s * X) ** 2 + (dv - s * Y) ** 2
        keep = np.ones(len(X), bool)
        n_drop = min(3, max(0, len(X) - 4))    # outvote the worst, keep >= 4
        if n_drop:
            keep[np.argsort(resid)[-n_drop:]] = False
        d2 = float((X[keep] ** 2 + Y[keep] ** 2).sum())
        if d2 > 1e-6:
            s = float((du[keep] * X[keep] + dv[keep] * Y[keep]).sum()) / d2
        tx = float(np.median(U - s * X))
        ty = float(np.median(V - s * Y))
    return s, tx, ty


def _parabolic(cm: float, c0: float, cp: float) -> float:
    """Sub-pixel offset of the minimum of three samples spaced one apart."""
    den = cm - 2 * c0 + cp
    if den <= 1e-9:
        return 0.0
    return clamp(0.5 * (cm - cp) / den, -0.5, 0.5)


def estimate_translation(a: np.ndarray, b: np.ndarray,
                         coarse: int = COARSE_SEARCH, fine: int = FINE_SEARCH,
                         init: tuple[float, float] | None = None
                         ) -> tuple[float, float, float, bool]:
    """Sub-pixel (dx, dy) displacement of image content between frames a and b,
    the residual error after compensating for it, and whether the search hit its
    own boundary.

    Convention: dx > 0 means *content* moved right, which means the *camera*
    moved left. Callers that talk about the camera must invert the sign.

    Two-level search — coarse pass at half resolution, then a fine pass with
    parabolic interpolation — so a 96 px proxy still resolves fractional shifts.

    The boundary flag matters more than it looks. Over a flat, textureless
    region the cost surface has no minimum, so the search slides to the edge of
    its range and returns that edge — a number that looks like a confident
    measurement and is really 'I could not tell'. Callers must discard those
    rather than average them in.
    """
    h, w = a.shape
    flat = float(np.abs(a - b).mean())
    if h < 16 or w < 16:
        return 0.0, 0.0, flat, True

    if init is not None:
        cx, cy = int(round(init[0])), int(round(init[1]))
    else:
        a2, b2 = a[::2, ::2], b[::2, ::2]
        m2 = coarse + 1
        if a2.shape[0] <= 2 * m2 or a2.shape[1] <= 2 * m2:
            cx = cy = 0
        else:
            best = (0, 0, float("inf"))
            for dy in range(-coarse, coarse + 1):
                for dx in range(-coarse, coarse + 1):
                    c = _sad(a2, b2, m2, dx, dy)
                    if c < best[2]:
                        best = (dx, dy, c)
            cx, cy = best[0] * 2, best[1] * 2

    m = abs(cx) + abs(cy) + fine + 1
    if h <= 2 * m or w <= 2 * m:
        return float(cx), float(cy), flat, True

    best = (cx, cy, float("inf"))
    for dy in range(cy - fine, cy + fine + 1):
        for dx in range(cx - fine, cx + fine + 1):
            c = _sad(a, b, m, dx, dy)
            if c < best[2]:
                best = (dx, dy, c)
    bx, by, cost = best
    at_edge = abs(bx - cx) >= fine or abs(by - cy) >= fine
    sx = _parabolic(_sad(a, b, m, bx - 1, by), cost, _sad(a, b, m, bx + 1, by))
    sy = _parabolic(_sad(a, b, m, bx, by - 1), cost, _sad(a, b, m, bx, by + 1))
    return bx + sx, by + sy, cost, at_edge


@dataclass
class VisualSignals:
    fps: float
    times: np.ndarray
    L: np.ndarray                  # (T,h,w) luma
    mean_luma: np.ndarray
    std_luma: np.ndarray
    delta: np.ndarray              # normalized mean abs pixel difference, 0..1
    ndelta: np.ndarray             # same, but brightness-invariant
    area: np.ndarray               # fraction of the frame that visibly changed
    hist_dist: np.ndarray          # luma-histogram L1 distance, 0..1
    cut_score: np.ndarray
    dx: np.ndarray                 # content displacement, proxy px per sample
    dy: np.ndarray                 # (positive dx = content right = camera left)
    divergence: np.ndarray         # fractional scale change per sample, >0 = zoom in
    residual: np.ndarray           # motion left over after global translation
    clipped: np.ndarray            # motion exceeded what the search can measure
    trackable: np.ndarray          # fraction of the frame with enough texture to track
    rendered: bool                 # frames are pixel-identical somewhere: not a camera
    chists: np.ndarray             # color histograms


def visual_signals(frames: np.ndarray, fps: float, t0: float) -> VisualSignals:
    T = frames.shape[0]
    L = luma(frames)
    times = t0 + np.arange(T) / fps

    mean_luma = L.mean(axis=(1, 2))
    std_luma = L.std(axis=(1, 2))

    # Contrast-normalised copy, used for everything motion-related. Removing the
    # per-frame mean and contrast is what stops a fade from registering as
    # motion. Rescaling by the clip's *median* contrast rather than a constant
    # keeps ndelta on the same scale as delta, so one gate value works for both
    # a flat cartoon and a high-contrast test pattern.
    ref_contrast = float(np.median(std_luma)) + 2.0
    Lnorm = ((L - mean_luma[:, None, None]) / (std_luma[:, None, None] + 2.0)
             * ref_contrast + 128.0)
    # "did anything change" is measured on the sharp normalised frames; "how did
    # it move" is matched on a blurred copy. Blurring before the change test
    # would hide exactly the slow motion the gate is meant to admit.
    Lm = blur3(Lnorm)

    delta = np.zeros(T, np.float32)
    ndelta = np.zeros(T, np.float32)
    area = np.zeros(T, np.float32)
    hist_dist = np.zeros(T, np.float32)
    dx = np.zeros(T, np.float32)
    dy = np.zeros(T, np.float32)
    divg = np.zeros(T, np.float32)
    resid = np.zeros(T, np.float32)
    clipped = np.zeros(T, bool)
    trackable = np.zeros(T, np.float32)

    lh = luma_hists(L)
    for i in range(1, T):
        a, b = Lm[i - 1], Lm[i]
        delta[i] = np.abs(L[i] - L[i - 1]).mean() / 255.0
        ndelta[i] = np.abs(Lnorm[i] - Lnorm[i - 1]).mean() / 255.0
        area[i] = float((np.abs(L[i] - L[i - 1]) > 12).mean())
        hist_dist[i] = 0.5 * float(np.abs(lh[i] - lh[i - 1]).sum())
        if ndelta[i] <= 0.004 or min(std_luma[i], std_luma[i - 1]) < 4.0:
            resid[i] = ndelta[i]          # static, or too flat to track at all
            continue
        ddx, ddy, res, clipped_i = estimate_translation(a, b)
        clipped[i] = clipped_i
        dx[i], dy[i], resid[i] = ddx, ddy, res / 255.0
        # Zoom shows up as the two halves of the frame moving apart (zoom in) or
        # together (zoom out). Needs texture to measure, so gate on contrast.
        if min(std_luma[i], std_luma[i - 1]) > 8:
            tr = [0.0]
            divg[i], dx[i], dy[i] = estimate_motion_model(a, b, ddx, ddy, tr)
            trackable[i] = tr[0]

    # A camera sensor cannot produce two pixel-identical frames — noise alone
    # forbids it. Measured across real footage the separation is absolute:
    # camera clips bottom out at delta 0.003, screen recordings and renders hit
    # exactly 0. So this tells us whether the words "camera pan" even apply.
    rendered = bool((delta[1:] < 0.0005).mean() > 0.15) if T > 2 else False

    cut_score = 0.5 * np.minimum(delta * 6.0, 1.0) + 0.5 * hist_dist
    return VisualSignals(fps, times, L, mean_luma, std_luma, delta, ndelta, area,
                         hist_dist, cut_score, dx, dy, divg, resid, clipped,
                         trackable, rendered, color_hists(frames))


# --------------------------------------------------------------------------
# shot segmentation
# --------------------------------------------------------------------------

def in_fade(vs: VisualSignals, i: int) -> bool:
    """True if analysis index i sits inside a gradual ramp between darkness and
    a visible image. Such ramps produce a run of moderate change scores that
    would otherwise be carved into several phantom shots — a fade to black is
    one shot boundary, not five."""
    w = max(2, int(vs.fps * 0.75))
    lo, hi = max(0, i - w), min(len(vs.mean_luma), i + w + 1)
    seg = vs.mean_luma[lo:hi]
    if seg.size < 3 or float(seg.min()) >= DARK_LUMA or float(seg.max()) <= 30:
        return False
    # a single-frame jump between black and a picture is a real hard cut
    return float(np.abs(np.diff(seg)).max()) <= 55


def detect_cuts(vs: VisualSignals, sensitivity: float, min_shot: float
                ) -> list[tuple[int, str]]:
    T = len(vs.cut_score)
    if T < 3:
        return []
    win = max(4, int(vs.fps * 2))
    cuts: list[tuple[int, str]] = []
    score = vs.cut_score
    for i in range(1, T):
        lo, hi = max(1, i - win), min(T, i + win + 1)
        local = np.delete(score[lo:hi], min(i - lo, hi - lo - 1))
        if local.size < 2:
            continue
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med))) + 1e-4
        thresh = max(med + (6.0 / sensitivity) * mad, 0.30 / sensitivity)
        # A fast camera move replaces most of the frame in one sample and looks
        # exactly like a cut by every magnitude measure. What separates them is
        # whether a single translation *explains* the change: across a whip-pan
        # most of the difference disappears once the frames are registered, so
        # the leftover residual is a fraction of the raw change. Across a real
        # cut nothing lines up and the residual stays as large as the change.
        # Measured: real cuts sit at residual/change 1.0-1.15, handheld whip-pans
        # at 0.39-0.50.
        # Two conditions, because either alone misfires. Frames can register by
        # coincidence at a genuine cut, and motion can legitimately start right
        # after one. A whip-pan is both explainable *and* ongoing: the camera
        # keeps moving into the following samples. A cut into a fast-moving shot
        # fails the first test (nothing registers at the cut itself); a cut
        # between two still shots fails the second.
        after = float(vs.ndelta[i + 1:i + 3].max()) if i + 1 < T else 0.0
        explained = (vs.ndelta[i] > 1e-6
                     and vs.residual[i] < 0.7 * vs.ndelta[i]
                     and after > 0.25 * vs.ndelta[i])

        hard_cut = (score[i] > thresh and vs.hist_dist[i] > 0.18 / sensitivity
                    and not explained)

        # A slide change, a screen share switching windows, a UI navigating to a
        # new page: a large part of the frame is repainted at once, but the
        # palette barely moves, so the histogram test above never fires. Detect
        # it as a big area change against a locally *static* baseline.
        #
        # The baseline comparison is what makes this safe. During a pan, ~100%
        # of pixels change every single sample, so the local median is also
        # ~100% and nothing triggers; it only fires when a still image is
        # suddenly replaced.
        lo2, hi2 = max(1, i - win), min(T, i + win + 1)
        base = float(np.median(np.delete(vs.area[lo2:hi2],
                                         min(i - lo2, hi2 - lo2 - 1))))
        repaint = (vs.area[i] > 0.06 / sensitivity
                   and vs.area[i] > 8.0 * (base + 0.002)
                   and not explained)

        if hard_cut or repaint:
            if in_fade(vs, i):
                continue
            cuts.append((i, "cut" if hard_cut else "repaint"))

    # non-maximum suppression + minimum shot length, including against the
    # start and end of the video (a 0.25s opening sliver is not a shot)
    keep: list[tuple[int, str]] = []
    gap = max(1, math.ceil(min_shot * vs.fps))
    for c, kind in cuts:
        if c < gap or T - c < gap:
            continue
        if keep and c - keep[-1][0] < gap:
            if score[c] > score[keep[-1][0]]:
                keep[-1] = (c, kind)
            continue
        keep.append((c, kind))
    return keep


def classify_transition(vs: VisualSignals, i: int, kind: str = "cut"
                        ) -> tuple[str, str]:
    """Return (label, evidence) for the transition arriving at analysis index i.

    `kind` is which detector fired. A screen repaint is not a dissolve, and
    saying "dissolve" for a page navigating or a slide advancing borrows film
    vocabulary for something that never happened.
    """
    if kind == "repaint":
        return "repaint", (f"{vs.area[i] * 100:.0f}% of the frame was replaced in one "
                           f"sample against an otherwise still image — a screen "
                           f"redraw (new slide, new page, window switch), not an edit")
    fps = vs.fps
    w = max(2, int(fps * 0.6))
    lo, hi = max(0, i - w), min(len(vs.mean_luma), i + w + 1)
    before = vs.mean_luma[lo:i]
    after = vs.mean_luma[i:hi]
    dark_before = before.size and float(before.min()) < 14
    dark_after = after.size and float(after.min()) < 14
    peak = float(vs.cut_score[i])
    neigh = np.concatenate([vs.cut_score[lo:i], vs.cut_score[i + 1:hi]])
    nmax = float(neigh.max()) if neigh.size else 0.0

    if dark_before and not dark_after:
        return "fade_in", f"luma rises from {float(before.min()):.0f} to {float(after.mean()):.0f}"
    if dark_after and not dark_before:
        return "fade_out", f"luma falls from {float(before.mean()):.0f} to {float(after.min()):.0f}"
    if nmax > 0.55 * peak and peak < 0.75:
        return "dissolve", f"change spread over ~{(hi - lo) / fps:.1f}s rather than one frame"
    return "cut", f"single-frame change score {peak:.2f} vs neighbours {nmax:.2f}"


def describe_camera(vs: VisualSignals, a: int, b: int) -> tuple[str, str]:
    """Camera motion label for analysis range [a, b).

    Everything is converted to units that don't depend on the proxy size or the
    analysis rate: framewidths-per-second for pan, frameheights-per-second for
    tilt, fractional-scale-per-second for zoom. Labels are camera-relative, so
    content drifting left means the camera panned right.
    """
    sl = slice(max(a + 1, 1), max(b, a + 2))
    dxs, dys = vs.dx[sl], vs.dy[sl]
    dv, res = vs.divergence[sl], vs.residual[sl]
    if dxs.size == 0:
        return "unknown", "shot too short to measure"
    if (b - a) / vs.fps < 0.25:
        return "unknown", f"shot is only {(b - a) / vs.fps:.2f}s — too short to judge motion"

    # Only samples where motion was actually measurable count: a frozen frame
    # would otherwise contribute a zero that drags the estimate toward nothing.
    valid = vs.ndelta[sl] > 0.004
    n_valid = int(valid.sum())
    if n_valid < max(2, int(0.2 * dxs.size)):
        return "static", (f"only {n_valid} of {dxs.size} samples had measurable "
                          f"motion once brightness changes are discounted — "
                          f"the frame is essentially still")
    dxs, dys, dv, res = dxs[valid], dys[valid], dv[valid], res[valid]

    H, W = vs.L.shape[1], vs.L.shape[2]
    # Camera-relative rates. Panning right pushes content left (dx < 0), so pan
    # is -dx. Tilting up pushes content *down* the screen (dy > 0) because the
    # y axis points down, so tilt is +dy. The two signs genuinely differ.
    pan = -float(np.median(dxs)) / W * vs.fps
    tilt = float(np.median(dys)) / H * vs.fps
    zoom = float(np.median(dv)) * vs.fps
    jitter = float(np.median(np.hypot(dxs - np.median(dxs),
                                      dys - np.median(dys)))) / W * vs.fps
    mres = float(np.median(res))

    # Coherence of each axis: how big its typical displacement is next to how
    # much that displacement scatters. A camera move is steady, so the median
    # towers over the spread. Content moving inside a locked-off frame produces
    # a median of similar size but with scatter to match, because it is not
    # going anywhere in particular.
    #
    # This is what ranks the axes. Ranking by rate alone picks the wrong one:
    # on a pure horizontal pan across animated content, the content's vertical
    # churn produced a *larger* tilt rate than the real pan, and only coherence
    # tells them apart (pan 7.8, tilt 0.8).
    def coherence(arr: np.ndarray) -> float:
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return abs(med) / (mad + 1e-6)

    k_pan, k_tilt, k_zoom = coherence(dxs), coherence(dys), coherence(dv)

    def show_k(k: float) -> str:
        """Coherence is unbounded above — a perfectly uniform scroll has almost
        no scatter, so the ratio can read 126. That is arithmetically right and
        decision-irrelevant (it is clamped to 5 where it is actually used), but
        printed raw it looks like a bug. Anything past 10 is just "very steady"."""
        return ">10" if k > 10 else f"{k:.1f}"

    ev = (f"pan {pan:+.3f} framewidths/s, tilt {tilt:+.3f} frameheights/s, "
          f"zoom {zoom:+.3f} scale/s, jitter {jitter:.3f}, unexplained motion {mres:.3f}, "
          f"steadiness {show_k(k_pan)}/{show_k(k_tilt)}/{show_k(k_zoom)} (pan/tilt/zoom, "
          f"median over scatter; below 0.5 is not a coherent move, above 10 is "
          f"reported as >10)")

    if vs.rendered:
        ev += (" — this source has pixel-identical frames in places, so it is "
               "screen-recorded or rendered, not camera footage: the displacement "
               "is measured and real, but 'camera' is the wrong frame of "
               "reference. Vertical motion here is usually the view scrolling")

    tr = vs.trackable[sl][valid]
    tr = tr[tr > 0]
    if tr.size and float(np.median(tr)) < 0.85:
        ev += (f" — only {float(np.median(tr)) * 100:.0f}% of the frame had enough "
               f"texture to track; the rest is flat (open sky, a blank wall, a "
               f"page background, motion blur), so this describes the trackable "
               f"part, not necessarily the whole frame")

    clip_frac = float(vs.clipped[sl][valid].mean()) if n_valid else 0.0
    if clip_frac > 0.25:
        ev += (f" — {clip_frac * 100:.0f}% of samples moved further than the "
               f"search range, so these rates are lower bounds")
        if clip_frac > 0.6:
            return "fast_motion", ev

    steady = math.hypot(pan, tilt)
    # "unsteady" is a qualifier, not an alternative: a shot can be a clean pan
    # that also judders, and collapsing that to one word throws away the pan.
    shaky = jitter > 0.025 and jitter > 1.2 * max(steady, abs(zoom))
    qual = " + unsteady" if shaky else ""

    if steady < 0.012 and abs(zoom) < 0.02:
        if shaky:
            return "unsteady", ev
        return ("static" if mres < 0.012 else "static_camera_moving_subject"), ev

    # Score each candidate against its own detection threshold and take the
    # winner, but only among axes whose direction actually held over the shot.
    cands = [(abs(pan) / 0.015, "pan_right" if pan > 0 else "pan_left", k_pan),
             (abs(tilt) / 0.015, "tilt_up" if tilt > 0 else "tilt_down", k_tilt),
             (abs(zoom) / 0.025, "zoom_in" if zoom > 0 else "zoom_out", k_zoom)]
    # The rate threshold decides whether an axis moved enough to mention;
    # coherence decides whether the movement was a camera or just churn, and
    # breaks the tie when more than one axis clears the threshold.
    cands = [(sc * min(k, 5.0), lb) for sc, lb, k in cands if sc >= 1.0 and k >= 0.5]
    if not cands:
        return (("unsteady" if shaky else "static_camera_moving_subject"),
                ev + " — no axis held a consistent direction, so this is movement "
                     "within the frame rather than a camera move")
    _, label = max(cands)
    return label + qual, ev


def motion_phases(vs: VisualSignals, a: int, b: int, window: float = 1.25,
                  min_phase: float = 1.5) -> list[dict]:
    """Split a shot into consecutive runs of homogeneous camera motion.

    One label per shot is a summary, and a summary of a long take is a lie by
    omission: a shot that holds still for ten seconds and then pans is reported
    by its median as barely moving at all. This walks the shot in short windows,
    labels each the same way the whole shot is labelled, and merges neighbours
    that agree.

    Returns [] when the shot has only one phase — the shot's own label already
    says everything in that case — so the timeline stays quiet unless the camera
    actually did more than one thing.
    """
    w = max(2, int(round(window * vs.fps)))
    if b - a < 2 * w:
        return []

    runs: list[list] = []
    for s in range(a, b, w):
        e = min(b, s + w)
        if e - s < max(2, w // 2):        # stub tail: extend the last run
            if runs:
                runs[-1][1] = e
            break
        label, ev = describe_camera(vs, s, e)
        base = label.split(" + ")[0]      # merge "pan_right" with "pan_right + unsteady"
        if runs and runs[-1][2] == base:
            runs[-1][1] = e
        else:
            runs.append([s, e, base, label, ev])

    # Absorb runs too short to be real into whichever neighbour is longer. A
    # one-window blip is usually the boundary between two phases, not a phase.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if (r[1] - r[0]) / vs.fps >= min_phase:
                continue
            prev_len = (runs[i - 1][1] - runs[i - 1][0]) if i > 0 else -1
            next_len = (runs[i + 1][1] - runs[i + 1][0]) if i + 1 < len(runs) else -1
            if prev_len < 0 and next_len < 0:
                break
            if prev_len >= next_len:
                runs[i - 1][1] = r[1]
            else:
                runs[i + 1][0] = r[0]
            runs.pop(i)
            changed = True
            break

    # Absorbing a short run can leave its two neighbours adjacent and agreeing,
    # and re-measuring over a merged span can change that span's label. So keep
    # collapsing equal neighbours and re-measuring until the labels stop moving,
    # otherwise a shot reports two consecutive phases that say the same thing.
    spans = [[r[0], r[1]] for r in runs]
    for _ in range(len(spans) + 1):
        merged: list[list] = []
        for s, e in spans:
            base = describe_camera(vs, s, e)[0].split(" + ")[0]
            if merged and merged[-1][2] == base:
                merged[-1][1] = e
            else:
                merged.append([s, e, base])
        if len(merged) == len(spans):
            break
        spans = [[s, e] for s, e, _ in merged]

    if len(spans) < 2:
        return []
    out = []
    for s, e in spans:
        label, ev = describe_camera(vs, s, e)
        out.append({"start": round(float(vs.times[s]), 3),
                    "end": round(float(vs.times[min(e, len(vs.times) - 1)]), 3),
                    "camera": label, "camera_evidence": ev})
    return out


def edge_fades(vs: VisualSignals, a: int, b: int) -> list[str]:
    """Describe ramps out of / into black at the two ends of a shot."""
    out = []
    n = int(min(b - a, max(3, vs.fps * 2.5)))
    if b - a < 3:
        return out
    head = vs.mean_luma[a:a + n]
    if float(head[0]) < DARK_LUMA and float(head.max()) > 30:
        k = int(np.argmax(head >= 0.9 * float(head.max())))
        out.append(f"opens black and fades up over {k / vs.fps:.1f}s")
    tail = vs.mean_luma[b - n:b]
    if float(tail[-1]) < DARK_LUMA and float(tail.max()) > 30:
        k = n - int(np.argmax(tail[::-1] >= 0.9 * float(tail.max())))
        out.append(f"fades down to black over the last {(n - k) / vs.fps:.1f}s")
    return out


def activity_class(v: float) -> str:
    if v < 0.004:
        return "frozen"
    if v < 0.015:
        return "very low"
    if v < 0.04:
        return "low"
    if v < 0.10:
        return "moderate"
    return "high"


def brightness_class(v: float) -> str:
    if v < 20:
        return "black/near-black"
    if v < 60:
        return "dark"
    if v < 130:
        return "mid"
    if v < 200:
        return "bright"
    return "very bright/blown"


def frozen_spans(vs: VisualSignals, a: int, b: int, min_len: float = 1.0
                 ) -> list[tuple[float, float]]:
    out, start = [], None
    for i in range(a + 1, b):
        if vs.delta[i] < 0.002:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) / vs.fps >= min_len:
                out.append((float(vs.times[start]), float(vs.times[i])))
            start = None
    if start is not None and (b - start) / vs.fps >= min_len:
        out.append((float(vs.times[start]), float(vs.times[b - 1])))
    return out


# --------------------------------------------------------------------------
# keyframe selection
# --------------------------------------------------------------------------

@dataclass
class Keyframe:
    index: int
    t: float
    shot: int
    pos_in_shot: str
    reason: str
    file: str = ""
    ocr: list[dict] = field(default_factory=list)


def pick_keyframes(vs: VisualSignals, shots: list[dict], max_frames: int,
                   coverage: float, max_per_shot: int = 8,
                   min_spacing: float = 0.7) -> list[Keyframe]:
    # The per-shot cap exists to stop one shot eating a budget that other shots
    # need. With few shots there is nobody to starve, so let the cap rise.
    room = max_frames // max(1, len(shots))
    hard_cap = max(max_per_shot, room)

    budget: list[int] = []
    caps: list[int] = []
    for s in shots:
        n = 1 + int(s["duration"] // coverage)
        if s["activity_mean"] > 0.05 and s["duration"] > 2.0:
            n += 1
        n = int(clamp(n, 1, hard_cap))
        budget.append(n)
        # Extra frames beyond the coverage floor are only worth it in proportion
        # to how much time the shot spends actually changing — roughly one frame
        # per 1.5 s of active time. Otherwise a long static shot soaks up the
        # whole budget producing near-identical images.
        active_time = s["duration"] * s.get("active_fraction", 0.0)
        caps.append(int(clamp(max(n, math.ceil(active_time / 1.5)), 1, hard_cap)))

    while sum(budget) > max_frames and max(budget) > 1:
        budget[budget.index(max(budget))] -= 1

    # Spend whatever is left of the budget on the shots where an extra frame
    # buys the most — long shots with a lot of change. An unspent budget just
    # means longer unobserved gaps, which is the exact failure this format is
    # trying to avoid.
    while sum(budget) < max_frames:
        best, best_gain = -1, 0.0
        for i, s in enumerate(shots):
            if budget[i] >= caps[i]:
                continue
            if s["duration"] / (budget[i] + 1) < min_spacing:
                continue
            # How *often* the image changes, not how much on average. A screen
            # recording is still for most of its length and then changes
            # abruptly; judging it by mean change calls it frozen and starves it
            # of frames, which is precisely when the changes get missed.
            if s["active_fraction"] < 0.02:
                continue          # genuinely never changes — another frame is a duplicate
            gain = s["duration"] * s["active_fraction"] / (budget[i] ** 2)
            if gain > best_gain:
                best, best_gain = i, gain
        if best < 0:
            break
        budget[best] += 1

    kfs: list[Keyframe] = []
    settle = max(1, int(0.3 * vs.fps))
    for si, (s, n) in enumerate(zip(shots, budget)):
        a, b = s["_a"], s["_b"]
        span = max(1, b - a)
        for j in range(n):
            lo = a + int(span * j / n)
            hi = max(lo + 1, a + int(span * (j + 1) / n))
            pos = f"{j + 1}/{n}"

            # If something actually happens in this window, show the state just
            # after it — that is the informative frame. Otherwise fall back to
            # the most representative frame, which avoids landing on a
            # transition or a motion-blurred one.
            seg_d = vs.delta[lo:hi]
            peak = float(seg_d.max()) if seg_d.size else 0.0
            if peak > max(4 * float(np.median(seg_d)), 0.01):
                kp = lo + int(np.argmax(seg_d))
                k = min(b - 1, kp + settle)
                why = (f"the state just after the biggest change in this window "
                       f"(change {peak:.3f} at {fmt_ts(float(vs.times[kp]))})")
            else:
                seg = vs.chists[lo:hi]
                med = np.median(seg, axis=0)
                k = lo + int(np.argmin(np.abs(seg - med).sum(axis=1)))
                if n == 1:
                    why = "single representative frame for the shot"
                elif s["duration"] > coverage:
                    why = (f"coverage sample {j + 1} of {n} — nothing much changes "
                           f"here, but the shot is longer than the {coverage:.0f}s floor")
                else:
                    why = (f"representative frame for window {j + 1} of {n} "
                           f"({activity_class(s['activity_mean'])} activity)")
            if n == 1 and "just after" not in why:
                pos = "1/1"
            kfs.append((Keyframe(index=0, t=float(vs.times[k]), shot=si + 1,
                                 pos_in_shot=pos, reason=why), k))

    # Drop frames that are indistinguishable from the one before them. A still
    # image with one animated corner — a music upload with a waveform strip, a
    # dashboard with a ticking clock — reads as "changing" on every sample and
    # would otherwise soak up the whole budget on identical pictures. Frames
    # still land at least every `coverage` seconds regardless.
    #
    # The test is *how much of the frame* changed, not a colour histogram: when
    # a table of data repaints, the palette is identical and the histogram says
    # nothing, but a tenth of the picture is new. Measured on real files, area
    # changed separates the two cases cleanly (waveform 2-5%, repainted UI
    # 9-12%) where histogram similarity does not (0.98-0.99 for both).
    def new_content(i: int, j: int) -> float:
        """Fraction of the frame that visibly differs between two samples."""
        return float((np.abs(vs.L[j] - vs.L[i]) > 12).mean())

    out: list[Keyframe] = []
    prev_k, prev_shot = None, None
    for kf, k in kfs:
        # No time condition. An earlier version only de-duplicated frames closer
        # together than the coverage floor, on the theory that a distant frame is
        # owed regardless. That was wrong: it put four pixel-identical copies of
        # one slide in a bundle, 20-30s apart, each showing the reader nothing.
        # If the picture has not changed, a later copy of it is not coverage.
        if prev_k is not None and kf.shot == prev_shot and new_content(prev_k, k) < 0.07:
            continue
        out.append((kf, k))
        prev_k, prev_shot = k, kf.shot

    # The coverage floor is advertised as a guarantee, so enforce it rather than
    # hope the allocation happened to satisfy it. Selection and de-duplication
    # both work per-window and can leave a gap wider than the floor; this fills
    # any that survived, so "no stretch longer than `coverage` goes unobserved"
    # is actually true of the bundle.
    def already_shown(last_kept: int | None, j: int) -> bool:
        """Would a frame at j tell the reader anything the last one did not?

        This replaced a `max delta < 0.002` test for "provably frozen". That
        threshold was stricter than the de-duplication threshold, so slides
        carrying faint compression noise counted as *changing* and kept earning
        fill frames that were visually identical. Asking the same question the
        same way in both places is what makes the two agree.
        """
        return last_kept is not None and new_content(last_kept, j) < 0.07

    final: list[tuple[Keyframe, int]] = []
    for si, s in enumerate(shots):
        a, b = s["_a"], s["_b"]
        mine = [(kf, k) for kf, k in out if kf.shot == si + 1]
        marks = sorted(mine, key=lambda p: p[0].t)
        cursor = a
        last_kept: int | None = None
        filled: list[tuple[Keyframe, int]] = []

        def fill(mid: int) -> Keyframe:
            return Keyframe(index=0, t=float(vs.times[mid]), shot=si + 1,
                            pos_in_shot="fill",
                            reason=f"coverage floor — nothing else marked this stretch, "
                                   f"and {coverage:.0f}s had passed unobserved")

        for kf, k in marks:
            while (k - cursor) / vs.fps > coverage:
                mid = cursor + int(coverage * vs.fps)
                # A stretch showing nothing new is not unobserved: the frame
                # before it depicts the screen for its whole length, and the shot
                # record names the frozen span outright. Filling it would emit
                # identical images — a slide held for 80s produced four copies of
                # itself before this check compared against the last kept frame.
                if already_shown(last_kept, mid):
                    cursor = mid
                    continue
                filled.append((fill(mid), mid))
                cursor = last_kept = mid
            filled.append((kf, k))
            cursor = last_kept = k
        while (b - 1 - cursor) / vs.fps > coverage:
            mid = cursor + int(coverage * vs.fps)
            if already_shown(last_kept, mid):
                cursor = mid
                continue
            filled.append((fill(mid), mid))
            cursor = last_kept = mid
        final.extend(filled)

    result: list[Keyframe] = []
    for kf, _ in final:
        kf.index = len(result) + 1
        result.append(kf)
    return result


# --------------------------------------------------------------------------
# frame extraction + context burn-in
# --------------------------------------------------------------------------

def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def extract_frame(src_path: str, t: float, dest: Path, width: int | None) -> bool:
    vf = f"scale={width}:-2" if width else "scale=iw:ih"
    p = run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", src_path,
             "-frames:v", "1", "-vf", vf, "-q:v", "2", str(dest)])
    return dest.exists() and dest.stat().st_size > 0


def label_frame(path: Path, lines: list[str]) -> None:
    """Prepend a header bar so the image is self-describing out of context."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    fs = max(11, int(w / 52))
    pad = max(4, fs // 2)
    bar = pad * 2 + fs * len(lines) + (len(lines) - 1) * (pad // 2)
    out = Image.new("RGB", (w, h + bar), (14, 14, 16))
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    f = _font(fs)
    y = pad
    for i, ln in enumerate(lines):
        d.text((pad, y), ln, font=f, fill=(255, 214, 92) if i == 0 else (198, 200, 208))
        y += fs + pad // 2
    d.line([(0, bar - 1), (w, bar - 1)], fill=(90, 90, 100), width=1)
    out.save(path, quality=90)


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

def ocr_image(path: Path, min_conf: float) -> list[dict]:
    if not have("tesseract"):
        return []
    p = run(["tesseract", str(path), "stdout", "--psm", "11", "-l", "eng", "tsv"])
    if p.returncode != 0:
        return []
    rows = p.stdout.splitlines()
    if not rows:
        return []
    groups: dict[tuple, list[tuple[str, float]]] = {}
    for row in rows[1:]:
        c = row.split("\t")
        if len(c) < 12:
            continue
        try:
            conf = float(c[10])
        except ValueError:
            continue
        word = c[11].strip()
        if conf < min_conf or not word:
            continue
        if not re.search(r"[A-Za-z0-9]", word):
            continue
        groups.setdefault((c[2], c[3], c[4]), []).append((word, conf))
    out = []
    for words in groups.values():
        text = " ".join(w for w, _ in words).strip()
        if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 2:
            continue
        out.append({"text": text,
                    "confidence": round(sum(c for _, c in words) / len(words), 1)})
    return out


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def build_text_events(samples: list[tuple[float, list[dict]]], probe_step: float
                      ) -> list[dict]:
    """Collapse per-frame OCR lines into events with a first/last-seen span."""
    samples = sorted(samples, key=lambda x: x[0])
    open_ev: dict[str, dict] = {}
    done: list[dict] = []
    tol = probe_step * 2.5 + 0.5
    for t, lines in samples:
        seen = set()
        for ln in lines:
            key = norm_text(ln["text"])
            if not key:
                continue
            seen.add(key)
            ev = open_ev.get(key)
            if ev and t - ev["last_seen"] <= tol:
                ev["last_seen"] = t
                ev["count"] += 1
                ev["_conf"].append(ln["confidence"])
                if ln["confidence"] > ev["best_conf"]:
                    ev["text"], ev["best_conf"] = ln["text"], ln["confidence"]
            else:
                if ev:
                    done.append(ev)
                open_ev[key] = {"text": ln["text"], "first_seen": t, "last_seen": t,
                                "count": 1, "best_conf": ln["confidence"],
                                "_conf": [ln["confidence"]]}
    done.extend(open_ev.values())
    out = []
    for ev in done:
        ev["mean_conf"] = round(sum(ev["_conf"]) / len(ev["_conf"]), 1)
        ev.pop("_conf")
        ev["first_seen"] = round(ev["first_seen"], 3)
        ev["last_seen"] = round(ev["last_seen"], 3)
        out.append(ev)
    return sorted(out, key=lambda e: (e["first_seen"], -e["count"]))


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def analyze_audio(src: Source, ss: float | None, dur: float | None, t0: float
                  ) -> dict | None:
    if not src.has_audio:
        return None
    cmd = ["ffmpeg", "-v", "error"]
    if ss:
        cmd += ["-ss", f"{ss}"]
    cmd += ["-i", src.path]
    if dur:
        cmd += ["-t", f"{dur}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, _ = proc.communicate()
    if len(raw) < AUDIO_SR:
        return None
    x = np.frombuffer(raw[: (len(raw) // 2) * 2], np.int16).astype(np.float32) / 32768.0

    hop = int(AUDIO_SR * AUDIO_HOP)
    n = len(x) // hop
    if n < 2:
        return None
    fr = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    times = t0 + np.arange(n) * AUDIO_HOP

    win = np.hanning(hop).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr * win, axis=1))
    freqs = np.fft.rfftfreq(hop, 1 / AUDIO_SR)
    total = spec.sum(axis=1) + 1e-9
    speech_band = spec[:, (freqs >= 300) & (freqs <= 3400)].sum(axis=1) / total
    high_band = spec[:, freqs > 4000].sum(axis=1) / total
    centroid = (spec * freqs).sum(axis=1) / total

    peak_db = float(db.max())
    floor_db = float(np.percentile(db, 5))
    gate = max(floor_db + 8.0, peak_db - 35.0)
    active = db > gate

    spans, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if (i - start) * AUDIO_HOP >= 0.25:
                spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, n))

    segs = []
    for a, b in spans:
        sb = float(speech_band[a:b].mean())
        hb = float(high_band[a:b].mean())
        lvl = float(db[a:b].mean())
        if sb > 0.55 and hb < 0.25:
            kind = "speech-like"
        elif hb > 0.35 or float(centroid[a:b].mean()) > 4000:
            kind = "bright/broadband (music, effects or noise)"
        else:
            kind = "tonal/low (music or hum)"
        segs.append({"start": round(t0 + a * AUDIO_HOP, 2),
                     "end": round(t0 + b * AUDIO_HOP, 2),
                     "mean_dbfs": round(lvl, 1),
                     "peak_dbfs": round(float(db[a:b].max()), 1),
                     "speech_band_ratio": round(sb, 2),
                     "heuristic_kind": kind})

    onsets = []
    for i in range(2, n):
        if db[i] > gate + 4 and db[i] - db[i - 2] > 12:
            if not onsets or times[i] - onsets[-1]["t"] > 0.4:
                onsets.append({"t": round(float(times[i]), 2),
                               "jump_db": round(min(float(db[i] - db[i - 2]), 60.0), 1)})

    return {
        "duration": round(n * AUDIO_HOP, 2),
        "peak_dbfs": round(peak_db, 1),
        "noise_floor_dbfs": round(floor_db, 1),
        "active_fraction": round(float(active.mean()), 3),
        "silent_fraction": round(float(1 - active.mean()), 3),
        "gate_dbfs": round(gate, 1),
        "segments": segs,
        "onsets": onsets[:200],
        "_track": {"t": [round(float(v), 2) for v in times[::4]],
                   "dbfs": [round(float(v), 1) for v in db[::4]]},
    }


# --------------------------------------------------------------------------
# transcript (optional, only if an engine happens to be installed)
# --------------------------------------------------------------------------

def transcribe(src: Source, workdir: Path, mode: str, ss, dur) -> dict:
    if mode == "off" or not src.has_audio:
        return {"available": False,
                "reason": "disabled by --asr off" if mode == "off" else "source has no audio stream",
                "segments": []}

    wav = workdir / "audio.wav"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if ss:
        cmd += ["-ss", f"{ss}"]
    cmd += ["-i", src.path]
    if dur:
        cmd += ["-t", f"{dur}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", str(wav)]
    run(cmd)
    if not wav.exists():
        return {"available": False, "reason": "could not extract audio", "segments": []}

    engine = None
    if mode not in ("auto",):
        engine = mode
    elif have("whisper-cli"):
        engine = "whisper-cli"
    elif have("whisper"):
        engine = "whisper"
    else:
        for mod in ("mlx_whisper", "faster_whisper", "whisper"):
            if run([sys.executable, "-c", f"import {mod}"]).returncode == 0:
                engine = f"py:{mod}"
                break

    if engine is None:
        return {"available": False, "segments": [],
                "reason": ("no speech-recognition engine found on this machine "
                           "(looked for: whisper-cli, whisper, mlx_whisper, "
                           "faster_whisper). Install one and re-run to add a "
                           "transcript; the rest of the bundle is unaffected.")}

    try:
        if engine == "whisper":
            run(["whisper", str(wav), "--model", "base", "--output_format", "json",
                 "--output_dir", str(workdir), "--fp16", "False"], timeout=3600)
            j = workdir / (wav.stem + ".json")
            if j.exists():
                data = json.loads(j.read_text())
                return {"available": True, "engine": "openai-whisper (base)",
                        "segments": [{"start": round(s["start"], 2),
                                      "end": round(s["end"], 2),
                                      "text": s["text"].strip()}
                                     for s in data.get("segments", [])]}
        elif engine == "whisper-cli":
            run(["whisper-cli", "-f", str(wav), "-oj", "-of", str(workdir / "asr")],
                timeout=3600)
            j = workdir / "asr.json"
            if j.exists():
                data = json.loads(j.read_text())
                segs = []
                for s in data.get("transcription", []):
                    off = s.get("offsets", {})
                    segs.append({"start": round(off.get("from", 0) / 1000, 2),
                                 "end": round(off.get("to", 0) / 1000, 2),
                                 "text": s.get("text", "").strip()})
                return {"available": True, "engine": "whisper.cpp", "segments": segs}
        elif engine.startswith("py:"):
            mod = engine[3:]
            code = textwrap.dedent(f"""
                import json, sys
                p = sys.argv[1]
                if "{mod}" == "mlx_whisper":
                    import mlx_whisper as W; r = W.transcribe(p)
                    segs = r["segments"]
                elif "{mod}" == "faster_whisper":
                    from faster_whisper import WhisperModel
                    m = WhisperModel("base"); it, _ = m.transcribe(p)
                    segs = [{{"start": s.start, "end": s.end, "text": s.text}} for s in it]
                else:
                    import whisper as W; r = W.load_model("base").transcribe(p)
                    segs = r["segments"]
                print(json.dumps([{{"start": round(s["start"],2), "end": round(s["end"],2),
                                    "text": s["text"].strip()}} for s in segs]))
            """)
            p = run([sys.executable, "-c", code, str(wav)], timeout=3600)
            if p.returncode == 0 and p.stdout.strip():
                return {"available": True, "engine": mod,
                        "segments": json.loads(p.stdout.strip().splitlines()[-1])}
    except Exception as e:  # engine present but failed — say so, don't pretend
        return {"available": False, "segments": [],
                "reason": f"engine {engine!r} failed: {e}"}

    return {"available": False, "segments": [],
            "reason": f"engine {engine!r} produced no output"}


# --------------------------------------------------------------------------
# scene grouping
# --------------------------------------------------------------------------

def hist_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def group_scenes(shots: list[dict], threshold: float = 0.72) -> list[dict]:
    scenes: list[dict] = []
    for i, s in enumerate(shots):
        h = np.array(s["_hist"])
        if scenes:
            recent = scenes[-1]["_members"][-4:]
            best = max(hist_sim(h, np.array(shots[j]["_hist"])) for j in recent)
            if best >= threshold:
                scenes[-1]["_members"].append(i)
                scenes[-1]["end"] = s["end"]
                scenes[-1]["_sims"].append(round(best, 3))
                continue
        scenes.append({"index": len(scenes) + 1, "start": s["start"], "end": s["end"],
                       "_members": [i], "_sims": []})
    for sc in scenes:
        sc["shots"] = [m + 1 for m in sc["_members"]]
        sc["duration"] = round(sc["end"] - sc["start"], 3)
        sc["internal_similarity"] = sc.pop("_sims")
        sc.pop("_members")
    return scenes


def find_resemblances(shots: list[dict], threshold: float = 0.80) -> None:
    H = [np.array(s["_hist"]) for s in shots]
    for i, s in enumerate(shots):
        matches = []
        for j in range(len(shots)):
            if abs(i - j) < 2:
                continue
            sim = hist_sim(H[i], H[j])
            if sim >= threshold:
                matches.append({"shot": j + 1, "similarity": round(sim, 3)})
        matches.sort(key=lambda m: -m["similarity"])
        s["resembles"] = matches[:3]


# --------------------------------------------------------------------------
# markdown rendering
# --------------------------------------------------------------------------

def scenes_are_informative(b: dict) -> bool:
    """Does the scene layer actually partition the shots?

    Scenes group adjacent shots by colour histogram. On camera footage that
    tracks real settings. On screen content it collapses: a slide deck is one
    scene for every slide because they share a white background, and a
    fast-cutting sequence is one scene per shot. Both are the layer saying
    nothing while looking like structure.

    It earns a place in the document only when it is a genuine middle level —
    more than one scene, and fewer scenes than shots. Otherwise `resembles`
    already carries the relationship, and carries it better: it names which
    shots match and how strongly.
    """
    n_sc, n_sh = len(b.get("scenes") or []), len(b.get("shots") or [])
    return 1 < n_sc < n_sh


def render_markdown(b: dict) -> str:
    src, summ = b["source"], b["summary"]
    show_scenes = scenes_are_informative(b)
    o: list[str] = []
    A = o.append

    A(f"# VIDEO TIMELINE — {src['filename']}")
    A("")
    A(f"`{FORMAT_VERSION}` · generated by vtl_convert {CONVERTER_VERSION} · "
      f"{summ['duration_human']} · {summ['shot_count']} shots"
      + (f" in {summ['scene_count']} scenes" if show_scenes else "") + " "
      f"· {summ['keyframe_count']} extracted frames")
    A("")

    A("## How to read this")
    A("")
    A("This is a video rendered as a measured timeline. It exists because sampled")
    A("screenshots leave you guessing about the intervals between them, and guesses")
    A("about intervals are where descriptions of video go wrong.")
    A("")
    A("- Every line is anchored to a timestamp in `HH:MM:SS.mmm` from the video start.")
    A("- **Measured** values (pixel change, luma, displacement, dBFS, OCR confidence)")
    A("  are stated with their numbers. Trust them.")
    A("- **Inferred** labels (`camera:`, `heuristic_kind`, `resembles`) are conclusions")
    A("  drawn from those numbers by simple rules. Each carries the evidence that")
    A("  produced it. Treat them as strong hints, not observations.")
    A("- Images in `frames/` are illustrations of the timeline, not the primary")
    A("  record. Each one has its timestamp and shot number burned into the header,")
    A("  so it stays interpretable if separated from this file.")
    A("- Where a stretch of video has no extracted frame, the timeline says so and")
    A("  gives the measurements for that stretch. Do not assume; read the numbers.")
    A("")
    miss = summ.get("missing_signals", [])
    if miss:
        A("**Signals not available for this file:**")
        for m in miss:
            A(f"- {m}")
        A("")

    A("## Source")
    A("")
    A("| field | value |")
    A("|---|---|")
    A(f"| file | `{src['filename']}` |")
    A(f"| duration | {summ['duration_human']} ({summ['duration']:.3f}s) |")
    A(f"| frame size | {src['width']}×{src['height']} (as decoded, rotation applied) |")
    A(f"| frame rate | {src['fps']} fps |")
    A(f"| video codec | {src['video_codec']} |")
    A(f"| audio | {(src['audio_codec'] + ', ' + str(src['audio_rate']) + ' Hz, ' + str(src['audio_channels'])) if src['has_audio'] else 'none'} |")
    A(f"| rotation tag | {src['rotation']}° |")
    A(f"| size | {src['size_bytes'] / 1e6:.1f} MB |")
    if b["analysis"].get("rendered_source"):
        A("| source character | **screen-recorded or rendered** — some consecutive "
          "frames are pixel-identical, which a camera sensor cannot do. Motion "
          "labels below describe how the *frame* moved, not a camera |")
    if src.get("sha256"):
        A(f"| sha256 | `{src['sha256'][:16]}…` |")
    r = b["analysis"].get("range")
    if r and (r[0] > 0.01 or (src.get("container_duration") or 0) - r[1] > 0.25):
        A(f"| analysed range | {fmt_ts(r[0])} → {fmt_ts(r[1])} "
          f"— **a subset of the file**, the rest was not looked at |")
    A("")

    A("## Summary")
    A("")
    A(f"- **{summ['shot_count']} shots**, mean length {fmt_dur(summ['mean_shot_length'])}, "
      f"shortest {fmt_dur(summ['min_shot_length'])}, longest {fmt_dur(summ['max_shot_length'])}.")
    if show_scenes:
        A(f"- **{summ['scene_count']} scenes** (runs of shots sharing a colour signature).")
    else:
        A(f"- Scene grouping is not reported: colour histograms put "
          f"{summ['scene_count']} scene(s) across {summ['shot_count']} shots, which "
          f"partitions nothing. Use `resembles` on each shot instead — it names "
          f"which other shots match and how strongly.")
    A(f"- Camera profile: {summ['camera_profile']}.")
    A(f"- Visual activity: {summ['activity_profile']}.")
    if summ.get("audio_profile"):
        A(f"- Audio: {summ['audio_profile']}")
    if summ.get("text_profile"):
        A(f"- On-screen text: {summ['text_profile']}")
    A(f"- Coverage: {summ['keyframe_count']} frames extracted; longest unobserved gap "
      f"{fmt_dur(summ['max_frame_gap'])} (at {fmt_ts(summ['max_frame_gap_at'])}).")
    if summ.get("budget_note"):
        A(f"- {summ['budget_note']}")
    A("")

    A("## Timeline")
    A("")
    scene_of = {}
    for sc in b["scenes"]:
        for sh in sc["shots"]:
            scene_of[sh] = sc

    prev_scene = None
    prev_kf_t = None
    for s in b["shots"]:
        sc = scene_of.get(s["index"])
        if show_scenes and sc and sc is not prev_scene:
            A(f"### ▬ SCENE {sc['index']} — {fmt_ts(sc['start'])} → {fmt_ts(sc['end'])} "
              f"({fmt_dur(sc['duration'])}, shots {sc['shots'][0]}–{sc['shots'][-1]})")
            A("")
            prev_scene = sc

        A(f"#### [{fmt_ts(s['start'])} → {fmt_ts(s['end'])}] Shot {s['index']}/{summ['shot_count']} "
          f"· {fmt_dur(s['duration'])}")
        A("")
        A(f"- **enters by** {s['transition_in']} — {s['transition_evidence']}")
        cam_row = "frame motion" if b["analysis"].get("rendered_source") else "camera"
        parts = s["camera_evidence"].split(" — ")
        A(f"- **{cam_row}** {s['camera']} *(inferred: {parts[0]})*")
        for c in parts[1:]:
            A(f"    - *caveat:* {c}")
        ph = s.get("motion_phases") or []
        if ph:
            A(f"- **the camera does more than one thing here** — the label above is "
              f"the median over the whole shot, which averages these together:")
            for x in ph[:12]:
                e = x["camera_evidence"]
                head = e.split(", jitter")[0]
                # keep any caveat: trimming the evidence for readability must
                # never trim the part that says the measurement is unreliable
                caveat = e[e.index("—"):] if "—" in e else ""
                A(f"    - `{fmt_ts(x['start'])}–{fmt_ts(x['end'])}` **{x['camera']}** "
                  f"*({head}{' ' + caveat if caveat else ''})*")
            if len(ph) > 12:
                A(f"    - … {len(ph) - 12} more phases in `timeline.json`")
        A(f"- **activity** {s['activity_class']} (mean frame-to-frame change "
          f"{s['activity_mean']:.4f}, peak {s['activity_peak']:.4f}; the image "
          f"changes at all during {s.get('active_fraction', 0) * 100:.0f}% of the shot)")
        A(f"- **brightness** {s['brightness_class']} (mean luma {s['mean_luma']:.0f}"
          + (f", trending {s['luma_trend']}" if s.get("luma_trend") else "") + ")")
        for f in s.get("fades", []):
            A(f"- **{f}**")
        if s["frozen_spans"]:
            fs = ", ".join(f"{fmt_ts(a)}–{fmt_ts(b_)}" for a, b_ in s["frozen_spans"])
            A(f"- **image frozen** during {fs} — nothing on screen changed at all")
        if s["resembles"]:
            r = ", ".join(f"shot {m['shot']} ({m['similarity']:.2f})" for m in s["resembles"])
            A(f"- **resembles** {r} *(inferred from colour histogram — likely the same "
              f"setting or a repeated framing)*")
        if s.get("audio_summary"):
            A(f"- **audio** {s['audio_summary']}")
        if s.get("speech"):
            A("- **speech**")
            for seg in s["speech"]:
                A(f"    - `{fmt_ts(seg['start'])}` {seg['text']}")
        if s.get("text_events"):
            A("- **on-screen text**")
            for ev in s["text_events"]:
                span = (f"{fmt_ts(ev['first_seen'])}–{fmt_ts(ev['last_seen'])}"
                        if ev["last_seen"] - ev["first_seen"] > 0.05
                        else fmt_ts(ev["first_seen"]))
                A(f"    - `{span}` \"{ev['text']}\" (OCR conf {ev['mean_conf']:.0f})")

        kfs = [k for k in b["keyframes"] if k["shot"] == s["index"]]
        if kfs:
            A("- **frames**")
            for k in kfs:
                gap = "" if prev_kf_t is None else f", {k['t'] - prev_kf_t:.1f}s after the previous frame"
                A(f"    - `{fmt_ts(k['t'])}` → [{k['file']}]({k['file']}) — {k['reason']}{gap}")
                if k.get("ocr"):
                    txt = "; ".join(f"\"{x['text']}\"" for x in k["ocr"][:6])
                    A(f"      text in this frame: {txt}")
                prev_kf_t = k["t"]
        else:
            A(f"- **frames** none extracted in this shot — it is {fmt_dur(s['duration'])} long "
              f"with {s['activity_class']} change; see the numbers above for what happens here")
        A("")

    A("## Track: transcript")
    A("")
    tr = b["transcript"]
    if tr.get("available"):
        A(f"Engine: {tr.get('engine')}")
        A("")
        for seg in tr["segments"]:
            A(f"- `{fmt_ts(seg['start'])} → {fmt_ts(seg['end'])}` {seg['text']}")
    else:
        A(f"*No transcript.* {tr.get('reason', 'unavailable')}")
        A("")
        A("Do not infer dialogue from the images. The audio-activity track below is")
        A("what is actually known about the sound.")
    A("")

    A("## Track: on-screen text")
    A("")
    if b["text_events"]:
        A("Deduplicated OCR events. `conf` is tesseract's confidence — under 70, treat")
        A("the exact wording as uncertain.")
        A("")
        A("| first seen | last seen | text | conf |")
        A("|---|---|---|---|")
        for ev in b["text_events"]:
            safe = ev["text"].replace("|", "\\|")
            A(f"| {fmt_ts(ev['first_seen'])} | {fmt_ts(ev['last_seen'])} | {safe} | {ev['mean_conf']:.0f} |")
    else:
        A("*No text detected.* " + b["ocr_status"])
    A("")

    A("## Track: audio activity")
    A("")
    au = b["audio"]
    if au:
        A(f"Peak {au['peak_dbfs']} dBFS, noise floor {au['noise_floor_dbfs']} dBFS, "
          f"active {au['active_fraction'] * 100:.0f}% of the runtime "
          f"(gate at {au['gate_dbfs']} dBFS).")
        A("")
        A("| start | end | level | speech-band | heuristic |")
        A("|---|---|---|---|---|")
        for sg in au["segments"][:120]:
            A(f"| {fmt_ts(sg['start'])} | {fmt_ts(sg['end'])} | {sg['mean_dbfs']} dBFS "
              f"| {sg['speech_band_ratio']:.2f} | {sg['heuristic_kind']} |")
        if len(au["segments"]) > 120:
            A(f"| … | | {len(au['segments']) - 120} more segments in `timeline.json` | | |")
        A("")
        if au["onsets"]:
            A("Sudden level jumps (often mark a hit, a cut, or something starting):")
            A("")
            A(", ".join(f"`{fmt_ts(x['t'])}` (+{x['jump_db']} dB)" for x in au["onsets"][:40]))
    else:
        A("*No audio track in this file.*" if not src["has_audio"]
          else "*Audio present but could not be analysed.*")
    A("")

    A("## Files in this bundle")
    A("")
    A("- `TIMELINE.md` — this document")
    A("- `timeline.json` — everything here plus the raw per-sample signal tracks")
    A("- `manifest.json` — provenance and tool versions")
    A(f"- `frames/` — {summ['keyframe_count']} extracted frames with burned-in context headers")
    if b.get("sheets"):
        n_sh = len(b["sheets"])
        A(f"- `sheets/` — {n_sh} contact sheet{'' if n_sh == 1 else 's'}, "
          f"{'frames in time order' if not show_scenes else 'one per scene, in time order'}")
    A("")
    return "\n".join(o)


def contact_sheets(bundle: Path, b: dict, cols: int = 4) -> list[str]:
    out_dir = bundle / "sheets"
    out_dir.mkdir(exist_ok=True)
    made = []
    by_shot: dict[int, list[dict]] = {}
    for k in b["keyframes"]:
        by_shot.setdefault(k["shot"], []).append(k)

    for sc in b["scenes"]:
        kfs = [k for sh in sc["shots"] for k in by_shot.get(sh, [])]
        if not kfs:
            continue
        kfs = kfs[:12]
        thumbs = []
        for k in kfs:
            p = bundle / k["file"]
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((420, 420))
                thumbs.append(im)
        if not thumbs:
            continue
        tw = max(t.width for t in thumbs)
        th = max(t.height for t in thumbs)
        rows = math.ceil(len(thumbs) / cols)
        pad, header = 8, 34
        sheet = Image.new("RGB", (cols * (tw + pad) + pad,
                                  header + rows * (th + pad) + pad), (20, 20, 24))
        d = ImageDraw.Draw(sheet)
        d.text((pad, 9), f"SCENE {sc['index']}  {fmt_ts(sc['start'])} - {fmt_ts(sc['end'])}"
                         f"   shots {sc['shots'][0]}-{sc['shots'][-1]}   (frames in time order, left to right)",
               font=_font(17), fill=(255, 214, 92))
        for i, im in enumerate(thumbs):
            x = pad + (i % cols) * (tw + pad)
            y = header + (i // cols) * (th + pad)
            sheet.paste(im, (x, y))
        name = f"sheets/scene{sc['index']:02d}.jpg"
        sheet.save(bundle / name, quality=88)
        made.append(name)
    return made


# --------------------------------------------------------------------------
# main conversion
# --------------------------------------------------------------------------

def convert(args) -> Path:
    src_path = Path(args.input).expanduser()
    if not src_path.exists():
        sys.exit(f"error: no such file: {src_path}")

    bundle = Path(args.output).expanduser() if args.output else \
        src_path.with_suffix("").with_name(src_path.stem + ".vtl")
    if bundle.exists():
        if not args.force:
            sys.exit(f"error: {bundle} already exists (use --force to overwrite)")
        shutil.rmtree(bundle)
    (bundle / "frames").mkdir(parents=True)

    log(f"probing {src_path.name}")
    src = probe(src_path, do_hash=not args.no_hash)
    t0 = args.ss or 0.0

    aw, ah = analysis_dims(src)
    log(f"decoding analysis proxy at {args.fps} Hz, {aw}x{ah}")
    frames = decode_proxy(src, args.fps, aw, ah, args.ss, args.duration)
    log(f"{frames.shape[0]} analysis samples")

    log("computing motion / change signals")
    vs = visual_signals(frames, args.fps, t0)
    T = frames.shape[0]
    total_dur = T / args.fps

    log("segmenting shots")
    cut_list = detect_cuts(vs, args.sensitivity, args.min_shot)
    cut_kind = {i: k for i, k in cut_list}
    bounds = [0] + [i for i, _ in cut_list] + [T]

    shots: list[dict] = []
    for i in range(len(bounds) - 1):
        a, b_ = bounds[i], bounds[i + 1]
        if b_ - a < 1:
            continue
        seg_delta = vs.delta[a + 1:b_] if b_ - a > 1 else vs.delta[a:a + 1]
        cam, cam_ev = describe_camera(vs, a, b_)
        if i == 0:
            trans, tev = "start", "beginning of the analysed range"
        else:
            trans, tev = classify_transition(vs, a, cut_kind.get(a, "cut"))
        lum = vs.mean_luma[a:b_]
        trend = None
        if lum.size > 3:
            d = float(lum[-1] - lum[0])
            if abs(d) > 25:
                trend = f"{'brighter' if d > 0 else 'darker'} by {abs(d):.0f} luma across the shot"
        shots.append({
            "index": len(shots) + 1,
            "start": round(float(vs.times[a]), 3),
            "end": round(float(vs.times[b_ - 1] + 1 / args.fps), 3),
            "duration": round((b_ - a) / args.fps, 3),
            "transition_in": trans, "transition_evidence": tev,
            "camera": cam, "camera_evidence": cam_ev,
            "activity_mean": round(float(seg_delta.mean()), 5),
            "activity_peak": round(float(seg_delta.max()), 5),
            "activity_class": activity_class(float(seg_delta.mean())),
            "active_fraction": round(float((seg_delta > 0.004).mean()), 3),
            "mean_luma": round(float(lum.mean()), 1),
            "brightness_class": brightness_class(float(lum.mean())),
            "luma_trend": trend,
            "frozen_spans": frozen_spans(vs, a, b_),
            "motion_phases": motion_phases(vs, a, b_),
            "fades": edge_fades(vs, a, b_),
            "_a": a, "_b": b_,
            "_hist": vs.chists[a:b_].mean(axis=0).tolist(),
        })
    log(f"{len(shots)} shots")

    find_resemblances(shots)
    scenes = group_scenes(shots)
    log(f"{len(scenes)} scenes")

    log("selecting keyframes")
    kfs = pick_keyframes(vs, shots, args.max_frames, args.coverage,
                         args.max_per_shot, args.min_spacing)

    log(f"extracting {len(kfs)} frames")
    kept: list[Keyframe] = []
    for k in kfs:
        name = f"k{k.index:03d}_t{fmt_ts(k.t).replace(':', '-')}_shot{k.shot:02d}.jpg"
        dest = bundle / "frames" / name
        if extract_frame(src.path, k.t, dest, args.frame_width):
            k.file = f"frames/{name}"
            kept.append(k)
    # NB: the context headers are burned in *after* OCR runs, further down. They
    # are text drawn onto the frame by this tool, and OCR reading them back would
    # report the converter's own annotations as text found in the video — the
    # bundle inventing observations. Do not move the labelling above the OCR.

    # ---- OCR -------------------------------------------------------------
    ocr_samples: list[tuple[float, list[dict]]] = []
    ocr_status = ""
    probe_step = 0.0
    if args.ocr != "off" and have("tesseract"):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            probe_step = args.text_probe or max(2.0, total_dur / args.max_probes)
            n_probes = int(total_dur / probe_step)
            if n_probes > 0:
                log(f"OCR: sampling {n_probes} probe frames every {probe_step:.1f}s")
                cmd = ["ffmpeg", "-v", "error", "-y"]
                if args.ss:
                    cmd += ["-ss", f"{args.ss}"]
                cmd += ["-i", src.path]
                if args.duration:
                    cmd += ["-t", f"{args.duration}"]
                cmd += ["-vf", f"fps=1/{probe_step},scale=1280:-2", "-q:v", "3",
                        str(tdp / "p%05d.jpg")]
                run(cmd)
                probes = sorted(tdp.glob("p*.jpg"))
                with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
                    results = list(ex.map(lambda p: ocr_image(p, args.ocr_conf), probes))
                for i, r in enumerate(results):
                    ocr_samples.append((t0 + (i + 0.5) * probe_step, r))
        log(f"OCR: {len(kept)} keyframes")
        with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
            kres = list(ex.map(lambda k: ocr_image(bundle / k.file, args.ocr_conf), kept))
        for k, r in zip(kept, kres):
            k.ocr = r
            ocr_samples.append((k.t, r))
        ocr_status = "tesseract ran and found no text above the confidence threshold."
    elif args.ocr == "off":
        ocr_status = "OCR disabled by --ocr off."
    else:
        ocr_status = "tesseract is not installed, so on-screen text was not read."

    text_events = build_text_events(ocr_samples, probe_step or 3.0) if ocr_samples else []
    text_events = [e for e in text_events if e["mean_conf"] >= args.ocr_conf]

    # Only now that every frame has been read is it safe to draw on them.
    log("labelling frames with their context headers")
    for i, k in enumerate(kept):
        prev = f"{k.t - kept[i - 1].t:.1f}s after prev frame" if i else "first frame"
        nxt = f"{kept[i + 1].t - k.t:.1f}s to next" if i + 1 < len(kept) else "last frame"
        label_frame(bundle / k.file, [
            f"t={fmt_ts(k.t)}   shot {k.shot}/{len(shots)} ({k.pos_in_shot})   frame {i + 1}/{len(kept)}",
            f"{src.filename}  |  {prev}  |  {nxt}  |  {k.reason}",
        ])

    # ---- audio -----------------------------------------------------------
    log("analysing audio")
    audio = analyze_audio(src, args.ss, args.duration, t0)

    with tempfile.TemporaryDirectory() as td:
        log("transcript")
        transcript = transcribe(src, Path(td), args.asr, args.ss, args.duration)

    # ---- attach tracks to shots -----------------------------------------
    for s in shots:
        if audio:
            ov = [g for g in audio["segments"] if g["end"] > s["start"] and g["start"] < s["end"]]
            if ov:
                lvl = sum(g["mean_dbfs"] for g in ov) / len(ov)
                kinds = sorted({g["heuristic_kind"] for g in ov})
                covered = sum(min(g["end"], s["end"]) - max(g["start"], s["start"]) for g in ov)
                s["audio_summary"] = (f"active {covered / max(s['duration'], .001) * 100:.0f}% of the shot, "
                                      f"mean {lvl:.0f} dBFS — {', '.join(kinds)} *(heuristic)*")
            else:
                s["audio_summary"] = "silent (below the activity gate for the whole shot)"
        if transcript.get("available"):
            s["speech"] = [g for g in transcript["segments"]
                           if g["end"] > s["start"] and g["start"] < s["end"]]
            s["speech"] = [g for g in s["speech"] if g["end"] - s["start"] > 0.05]
        s["text_events"] = [e for e in text_events
                            if e["last_seen"] > s["start"] and e["first_seen"] < s["end"]]

    # ---- summary ---------------------------------------------------------
    durs = [s["duration"] for s in shots]
    cams: dict[str, float] = {}
    for s in shots:
        spans = s.get("motion_phases") or [
            {"camera": s["camera"], "start": s["start"], "end": s["end"]}]
        for x in spans:
            cams[x["camera"]] = cams.get(x["camera"], 0) + (x["end"] - x["start"])
    cam_profile = ", ".join(f"{k} {v / max(total_dur, .001) * 100:.0f}%"
                            for k, v in sorted(cams.items(), key=lambda x: -x[1])[:4])
    gaps = [(kept[i + 1].t - kept[i].t, kept[i].t) for i in range(len(kept) - 1)]
    if kept:
        gaps.append((kept[0].t - t0, t0))
        gaps.append((t0 + total_dur - kept[-1].t, kept[-1].t))
    max_gap, max_gap_at = max(gaps) if gaps else (total_dur, t0)

    missing = []
    if not transcript.get("available"):
        missing.append(f"**Speech transcript** — {transcript.get('reason')}")
    if not have("tesseract") and args.ocr != "off":
        missing.append("**On-screen text (OCR)** — tesseract is not installed.")
    if not src.has_audio:
        missing.append("**Audio** — the file contains no audio stream.")

    mean_act = float(np.mean([s["activity_mean"] for s in shots])) if shots else 0.0
    summary = {
        "duration": round(total_dur, 3),
        "duration_human": fmt_dur(total_dur),
        "shot_count": len(shots),
        "scene_count": len(scenes),
        "keyframe_count": len(kept),
        "mean_shot_length": round(sum(durs) / max(len(durs), 1), 2),
        "min_shot_length": round(min(durs), 2) if durs else 0,
        "max_shot_length": round(max(durs), 2) if durs else 0,
        "camera_profile": cam_profile or "unknown",
        "activity_profile": f"{activity_class(mean_act)} overall (mean frame-to-frame change {mean_act:.4f})",
        "audio_profile": (f"{audio['active_fraction'] * 100:.0f}% of the runtime has sound above the gate, "
                          f"peak {audio['peak_dbfs']} dBFS, {len(audio['segments'])} active spans"
                          if audio else None),
        "text_profile": (f"{len(text_events)} distinct text events detected by OCR"
                         if text_events else None),
        "budget_note": (
            f"{len(kept)} frames were extracted against a budget of "
            f"{args.max_frames}: every shot gets at least one frame, and this "
            f"video has {len(shots)} shots." if len(kept) > args.max_frames else None),
        "max_frame_gap": round(max_gap, 2),
        "max_frame_gap_at": round(max_gap_at, 2),
        "missing_signals": missing,
    }

    bundle_data = {
        "format": FORMAT_VERSION,
        "converter": CONVERTER_VERSION,
        "source": {k: v for k, v in asdict(src).items() if k != "raw_probe"},
        "analysis": {
            "analysis_fps": args.fps,
            "proxy_size": [aw, ah],
            "samples": T,
            "sensitivity": args.sensitivity,
            "min_shot": args.min_shot,
            "coverage_floor": args.coverage,
            "ocr_probe_step": round(probe_step, 2) if probe_step else None,
            "rendered_source": vs.rendered,
            "range": [t0, t0 + total_dur],
        },
        "summary": summary,
        "scenes": scenes,
        "shots": [{k: v for k, v in s.items() if not k.startswith("_")} for s in shots],
        "keyframes": [asdict(k) for k in kept],
        "transcript": transcript,
        "text_events": text_events,
        "ocr_status": ocr_status,
        "audio": audio,
        "signals": {
            "t": [round(float(x), 3) for x in vs.times],
            "delta": [round(float(x), 5) for x in vs.delta],
            "cut_score": [round(float(x), 4) for x in vs.cut_score],
            "mean_luma": [round(float(x), 1) for x in vs.mean_luma],
            "dx": [round(float(x), 2) for x in vs.dx],
            "dy": [round(float(x), 2) for x in vs.dy],
        },
    }

    if not args.no_sheets:
        log("building contact sheets")
        bundle_data["sheets"] = contact_sheets(bundle, bundle_data)

    (bundle / "timeline.json").write_text(json.dumps(bundle_data, indent=1))
    (bundle / "TIMELINE.md").write_text(render_markdown(bundle_data))
    (bundle / "manifest.json").write_text(json.dumps({
        "format": FORMAT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "entry_point": "TIMELINE.md",
        "source_file": src.filename,
        "source_sha256": src.sha256,
        "tools": {
            "ffmpeg": (run(["ffmpeg", "-version"]).stdout.splitlines() or [""])[0],
            "tesseract": ((run(["tesseract", "--version"]).stdout or
                           run(["tesseract", "--version"]).stderr).splitlines() or [""])[0]
                         if have("tesseract") else None,
            "asr": transcript.get("engine") if transcript.get("available") else None,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "signals_present": {
            "visual": True,
            "shots": True,
            "audio_activity": audio is not None,
            "transcript": bool(transcript.get("available")),
            "ocr": bool(text_events),
        },
    }, indent=1))

    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an MP4/MOV into a VTL video-timeline bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The bundle's TIMELINE.md is the entry point — read that first.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output .vtl directory (default: alongside input)")
    ap.add_argument("-f", "--force", action="store_true", help="overwrite an existing bundle")
    ap.add_argument("--fps", type=float, default=8.0, help="analysis sample rate (default 8)")
    ap.add_argument("--max-frames", type=int, default=48, help="keyframe budget (default 48)")
    ap.add_argument("--max-per-shot", type=int, default=8,
                    help="most frames to extract from any one shot (default 8)")
    ap.add_argument("--min-spacing", type=float, default=0.7,
                    help="minimum seconds between extracted frames (default 0.7)")
    ap.add_argument("--coverage", type=float, default=15.0,
                    help="max seconds of a shot without an extracted frame (default 15)")
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help="cut detection sensitivity; >1 finds more cuts (default 1.0)")
    ap.add_argument("--min-shot", type=float, default=0.4,
                    help="minimum shot length in seconds (default 0.4)")
    ap.add_argument("--frame-width", type=int, default=1280,
                    help="extracted frame width, 0 for native (default 1280)")
    ap.add_argument("--ocr", choices=["auto", "off"], default="auto")
    ap.add_argument("--ocr-conf", type=float, default=55.0, help="min OCR confidence (default 55)")
    ap.add_argument("--text-probe", type=float, default=0.0,
                    help="seconds between OCR probe frames (default: auto)")
    ap.add_argument("--max-probes", type=int, default=80, help="cap on OCR probes (default 80)")
    ap.add_argument("--asr", default="auto",
                    help="'auto' (use a whisper if installed), 'off', or an engine name")
    ap.add_argument("--no-sheets", action="store_true", help="skip contact sheets")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of the source")
    ap.add_argument("--ss", type=float, default=0.0, help="start offset in seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="analyse only N seconds")
    args = ap.parse_args()
    args.ss = args.ss or None
    args.duration = args.duration or None
    if args.frame_width == 0:
        args.frame_width = None

    if not have("ffmpeg"):
        sys.exit("error: ffmpeg is required and was not found on PATH")

    print(f"vtl_convert {CONVERTER_VERSION} → {Path(args.input).name}", file=sys.stderr)
    out = convert(args)
    print(f"\nwrote {out}", file=sys.stderr)
    print(f"  read {out / 'TIMELINE.md'} first", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
