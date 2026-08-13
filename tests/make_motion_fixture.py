#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seth Wheeler
"""
Build a camera-motion fixture with exactly known ground truth.

Every frame is rendered here, in Python, by sampling a fixed still image through
a viewport whose position and zoom are given by closed-form functions of time.
Nothing in the scene moves, so any motion the converter reports is either the
camera or an error, and the true rate is known analytically rather than assumed.

An earlier version built this with ffmpeg's scale+crop filters and turned out to
have an unintended horizontal drift, which read as a converter bug for a while.
A fixture you have to debug is not a fixture, hence rendering frames directly.

Segments (20 s at 30 fps, 1280x720):
    0– 4 s  still                       pan 0        tilt 0        zoom 0
    4– 8 s  centred zoom in  1.0 -> 1.6                            zoom +0.115 /s
    8–12 s  centred zoom out 1.6 -> 1.0                            zoom -0.115 /s
   12–16 s  viewport slides right 250 px/s   pan  +0.195 framewidths/s
   16–20 s  viewport slides up    100 px/s   tilt +0.139 frameheights/s
   20–23 s  whip-pan right 1000 px/s          pan  +0.781 framewidths/s

The zoom rates are the median of d(zoom)/dt / zoom over each segment.

The viewport path is continuous throughout: position never jumps, only its
velocity changes. So this fixture contains **no cuts at all**, and any shot
boundary the converter reports in it is a false positive. That is the whole
assertion — camera movement, however fast, is not a cut.

The whip-pan is there to defend a specific bug. It moves further per sample
than the motion search can follow, so it replaces most of the frame at once and
looks exactly like a hard cut by every magnitude measure. It is continuous with
the segment before it, so any shot boundary detected inside it is a false
positive. Real handheld footage does this constantly.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

W, H, FPS, DUR = 1280, 720, 30, 23
SRC_W, SRC_H = 7000, 2400
CX0, CY0 = 2000.0, 1400.0


def make_source() -> Image.Image:
    """A big still with plenty of trackable, non-repeating detail."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, (SRC_H // 8, SRC_W // 8, 3))
    img = Image.fromarray(
        np.clip((a - a.min()) / (a.max() - a.min()) * 255, 0, 255).astype(np.uint8)
    ).resize((SRC_W, SRC_H), Image.BICUBIC)
    px = np.asarray(img).astype(np.float32)
    # add hard-edged shapes: block matching needs corners, not just soft gradients
    for _ in range(400):
        w, h = rng.integers(40, 260, 2)
        x, y = rng.integers(0, SRC_W - w), rng.integers(0, SRC_H - h)
        col = rng.integers(0, 256, 3).astype(np.float32)
        px[y:y + h, x:x + w] = px[y:y + h, x:x + w] * 0.25 + col * 0.75
    return Image.fromarray(px.astype(np.uint8))


def viewport(t: float) -> tuple[float, float, float]:
    """(centre_x, centre_y, zoom) of the camera at time t."""
    if t < 4:
        return CX0, CY0, 1.0
    if t < 8:
        return CX0, CY0, 1.0 + 0.15 * (t - 4)
    if t < 12:
        return CX0, CY0, 1.6 - 0.15 * (t - 8)
    if t < 16:
        return CX0 + 250.0 * (t - 12), CY0, 1.0
    if t < 20:
        # continues from where the pan stopped — no teleport back to centre
        return CX0 + 1000.0, CY0 - 100.0 * (t - 16), 1.0
    # continuous with the end of the tilt segment — no jump, just fast
    return CX0 + 1000.0 + 1000.0 * (t - 20), CY0 - 400.0, 1.0


def main() -> None:
    out = Path(__file__).parent / "fixture_motion.mp4"
    src = make_source()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for n in range(FPS * DUR):
            t = n / FPS
            cx, cy, z = viewport(t)
            vw, vh = W / z, H / z
            src.resize((W, H), Image.BICUBIC,
                       box=(cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)) \
               .save(d / f"f{n:05d}.png")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS),
             "-i", str(d / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "16", "-preset", "veryfast", str(out)], check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
