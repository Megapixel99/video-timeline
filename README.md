# VTL — hand a video to a language model without the guesswork

[![tests](https://github.com/Megapixel99/video-timeline/actions/workflows/tests.yml/badge.svg)](https://github.com/Megapixel99/video-timeline/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![requires](https://img.shields.io/badge/requires-ffmpeg-orange)](https://ffmpeg.org/)

When you give me an MP4 or MOV, the usual move is to pull N screenshots and
reason over them. That fails in a specific way: a screenshot says nothing about
*when* it happened, how long after the last one, whether the camera moved or the
subject did, or what happened in the gap between samples. So the gap gets filled
with a plausible story. The errors aren't in reading the frames — they're in
reading the intervals between frames.

`vtl_convert.py` turns a video into a **VTL bundle**: a measured timeline with
frames attached to it, rather than frames with a timeline guessed around them.

```bash
python3 vtl_convert.py myclip.mov
# -> myclip.vtl/   (read TIMELINE.md first)
```

Then point the model at `myclip.vtl/TIMELINE.md`.

## Quick start

Needs `ffmpeg` on PATH, plus two Python packages. `tesseract` is optional but
worth having — it is what reads slides, titles and UI text.

```bash
pip install numpy pillow          # ffmpeg and tesseract come from your package manager
bash tests/make_fixture.sh        # builds a 25s sample video with known content
python3 vtl_convert.py tests/fixture.mp4
cat tests/fixture.vtl/TIMELINE.md
```

No sample video ships with the repo — the test fixtures are generated, so the
first command above builds one whose contents are known in advance (title cards,
hard cuts, a fade from black, a pan, frozen frames, tone/silence/noise audio).
That makes it easy to check the output against what is actually in the file.

## What you get

```
myclip.vtl/
├── TIMELINE.md      read this first — the whole video, in time order
├── timeline.json    same thing machine-readable, plus raw per-sample signals
├── manifest.json    provenance: source hash, tool versions, which signals exist
├── frames/          keyframes with their timestamp burned into the header
└── sheets/          one contact sheet per scene, frames left-to-right in time
```

A slice of `TIMELINE.md`:

```
#### [00:00:04.000 → 00:00:09.000] Shot 2/6 · 5.00s

- **enters by** cut — single-frame change score 0.99 vs neighbours 0.17
- **camera** pan_right + unsteady *(inferred: camera pan +0.208 framewidths/s,
  tilt +0.023 frameheights/s, zoom -0.003 scale/s, jitter 0.247)*
- **activity** moderate (mean frame-to-frame change 0.0443, peak 0.0509)
- **brightness** mid (mean luma 128, trending darker by 41 luma across the shot)
- **resembles** shot 4 (0.86) *(inferred from colour histogram — likely the
  same setting or a repeated framing)*
- **audio** active 81% of the shot, mean -21 dBFS — speech-like *(heuristic)*
- **frames**
    - `00:00:04.250` → frames/k006_… — the state just after the biggest change
      in this window (change 0.051 at 00:00:04.125), 1.1s after the previous frame
```

Five design choices do the real work:

- **Frames land where the content changes**, not on a fixed clock — each one
  shows the state *just after* a change, and says so. Near-duplicate frames are
  dropped, and the coverage floor is then enforced as a hard guarantee, so no
  stretch longer than it goes unobserved. The worst gap is stated up front.
- **Every frame is self-describing.** Its timestamp, shot number, position in the
  shot, and the elapsed time since the previous frame are burned into a header
  bar. A VTL frame still knows where it came from if it gets separated from the
  timeline.
- **Measured and inferred stay separate.** `pan +0.208 framewidths/s` is
  measured. `pan_right` is a rule applied to that number, labelled as inferred,
  and always shown next to the measurement that produced it.
- **A shot that changes gets broken down.** When the camera does more than one
  thing in a single take, the timeline lists each phase rather than reporting a
  median that averages them away. Shots with one steady behaviour say nothing
  extra.
- **Absent signals are declared.** No speech recognizer installed means the
  bundle says so in the summary and in the transcript section, rather than
  leaving a hole that gets filled with invention.

See [SPEC.md](SPEC.md) for the format definition and its limits, and
[BENCHMARKS.md](BENCHMARKS.md) for measured results against uniform sampling.

## Requirements

| | |
|---|---|
| required | `ffmpeg`, `numpy`, `Pillow`, Python 3.10+ (CI runs 3.10 / 3.12 / 3.13) |
| optional | `tesseract` — on-screen text (titles, UI, slides, captions) |
| optional | a whisper CLI or python module — speech transcript |

`ffprobe` is *not* required; metadata is parsed from `ffmpeg -i`. Without
tesseract or a whisper you still get the full visual and audio-activity
timeline, and the bundle records which signals were unavailable and why.

## Options

```
-o, --output PATH      output bundle (default: alongside the input)
    --max-frames N     keyframe budget (default 48; every shot still gets one)
    --coverage SEC     longest a shot may run without a frame (default 15)
    --sensitivity F    cut detection; >1 finds more cuts (default 1.0)
    --max-per-shot N   most frames from any one shot (default 8; rises
                       automatically when there are few shots)
    --min-spacing SEC  closest two extracted frames may be (default 0.7)
    --min-shot SEC     shortest allowed shot (default 0.4)
    --frame-width PX   extracted frame width, 0 for native (default 1280)
    --ocr auto|off     on-screen text (default auto)
    --asr auto|off|CMD speech recognition (default auto: use one if installed)
    --ss / --duration  analyse only part of the file
```

Long, slow videos: raise `--max-frames`. Fast-cut videos where shots are being
merged: raise `--sensitivity`. A 5-minute video takes about 30 s.

## Tests

```bash
python3 tests/test_vtl.py
```

73 checks against two fixtures the suite builds itself. The point of the second
fixture is that its camera motion is known *analytically* — every frame is
rendered in Python by moving a viewport over a still image along a closed-form
path — so the camera rates can be asserted numerically rather than eyeballed:

```
segment      quantity   true      measured
 4.3– 7.7s   zoom      +0.115      +0.080
 8.3–11.7s   zoom      -0.115      -0.071
12.3–15.7s   pan       +0.195      +0.208
16.3–19.7s   tilt      +0.139      +0.147
```

The second fixture's viewport path is continuous from end to end, so it contains
no cuts at all and ends with a whip-pan faster than the matcher can follow. Any
boundary reported inside it is a false positive — that is the regression test for
mistaking fast camera movement for an edit.

The first fixture covers what the second deliberately excludes: hard cuts,
a fade from black, frozen frames, repeated settings, on-screen text, and audio
with known tone/silence/noise spans.

### Motion phases

A real 16-second handheld take that holds still, then pans:

```
- **camera** unsteady   (the median over the whole shot)
- **the camera does more than one thing here** — the label above is the median
  over the whole shot, which averages these together:
    - `00:00.000–00:02.500` tilt_up + unsteady
    - `00:02.500–00:07.500` unsteady
    - `00:07.500–00:15.750` pan_right   (pan +0.037 framewidths/s)
```

The pan across the last eight seconds is invisible in the shot-level median.
Phase correlation independently measures that span at +0.057.

### Benchmarks

```bash
python3 tests/benchmark.py
```

Compares frame selection against uniform sampling — N evenly spaced screenshots,
the baseline this format replaces — at an equal frame budget. Across 28 shots of
real footage, uniform sampling never looks at 9 of them (32%) and spends 12 of
its 40 frames on pictures it had already seen; VTL misses no shot and wastes 2.

On the *size of the largest unobserved gap* the two come out level, which is the
honest result — the advantage is in never missing a shot and not wasting budget,
not in smaller gaps. Full numbers, caveats and conversion cost in
[BENCHMARKS.md](BENCHMARKS.md).

### Cross-checking on real footage

Real video has no ground truth, so `tests/crosscheck_motion.py` measures the same
clips with an independent algorithm — FFT phase correlation at 480 px, sharing no
code or assumptions with the converter's spatial block matching on a 96 px proxy —
and compares. It calibrates itself on the synthetic fixture first.

```bash
python3 tests/crosscheck_motion.py myclip.mp4
```

On CC0 stock footage from pexels.com the two methods agree closely:

```
clip (pexels, CC0)          converter        phase correlation
pan shot of a grass field   pan_left -0.088   -0.090
trains in a railroad yard   pan_right +0.083  +0.082
blurred passing cars        no camera move    ~0 (peak 0.04)
```

## Known limits

- Camera motion comes from block matching on a 96 px proxy. It separates
  static / pan / tilt / zoom / unsteady reliably; it is not optical flow and
  does not track individual objects.
- Pan and tilt *rates* land within about 10% of truth. Zoom rate reads roughly
  30% low (see the table above) because block displacement under zoom is only a
  fraction of a proxy pixel near the frame centre. The direction and the
  in/out call are right; treat the zoom magnitude as a lower bound.
- Cut detection handles hard cuts and screen repaints — slide changes, window
  switches, page navigations, which keep the palette and so are invisible to
  histogram-based detection. Dissolves are reported as gradual transitions
  rather than exact boundaries, and a cut between two near-identical shots will
  be missed, as it would be by eye.
- OCR is tesseract on video frames: strong on titles, slides and screen
  recordings, weak on small text over motion. Per-event confidence is reported;
  under ~70 treat the wording as a guess.
- Speech-like vs music-like is a band-energy heuristic, labelled as such
  everywhere it appears. A sustained pure tone reads as "speech-like".
- A scene with strong depth parallax — shot from a moving vehicle, say — has no
  single camera motion: the foreground streaks past while the horizon barely
  moves. The converter reports whichever plane dominates the frame, which for a
  sky-heavy shot can read as no camera movement at all. When a large part of the
  frame is too flat to track, the evidence line says so.
- Motion phases resolve changes to about 1.25 s, so a phase boundary marks
  roughly where the camera changed behaviour, not the exact frame.
- Very long videos produce a long `TIMELINE.md` (~18 KB per minute of video).
  `timeline.json` carries the same content if you would rather query it.

## TODO

### Speech transcript — deliberately not installed

Every video run so far has come out with *"No transcript — no
speech-recognition engine found"*. That is the biggest remaining hole in the
output: a narrated 7-minute presentation currently yields slides and timings but
not a word of what was said over them. Deferred on purpose, not forgotten —
nothing here needs voice recognition yet.

To add it later, on this machine:

```bash
brew install whisper-cpp
```

That is the whole job. **No code change is needed** — the converter probes for
`whisper-cli`, `whisper`, `mlx_whisper` and `faster_whisper` and uses whichever
it finds, with no flags. Re-run any video and a timestamped transcript appears
per shot, alongside the OCR.

Two findings from when this was investigated, so they don't need rediscovering:

- **A standalone binary (`whisper-cpp`) beats a Python package on a mismatched
  toolchain.** On Apple silicon it is common to end up with an Intel x86_64
  Python under Rosetta (check with
  `python3 -c 'import platform; print(platform.machine())'`). A standalone binary
  keeps the emulated interpreter out of the loop entirely.
- **`mlx-whisper` needs *native* arm64 Python**, so it is unavailable on a
  Rosetta toolchain even on Apple silicon. It is the fastest option on that
  hardware, so it is worth revisiting once the toolchain is native.

To confirm it took effect, check that the bundle's `manifest.json` shows
`"asr"` populated under `tools`, and that `TIMELINE.md` no longer lists
**Speech transcript** under *Signals not available for this file*.

## On how this was built

Written with heavy AI assistance (Claude Code), and the process is worth being
straight about, because it is the reason the numbers in this README are
trustworthy rather than plausible.

Nearly every correctness claim here came from testing against ground truth and
finding the code wrong. A sample of what the fixtures and the cross-check caught,
each of which shipped as a bug first:

- pan direction was inverted — labelled by content motion instead of camera motion
- a fade from black was split into three phantom shots
- SAD matching is brightness-sensitive, so a fade fabricated a `zoom_in` on a
  static title card
- aliasing on a downscaled proxy turned a true +0.94 px/frame drift into a
  *confident* −3.94; one blur pass fixed it
- block matchers returned their own search boundary as if it were a measurement
- a handheld whip-pan read as a hard cut (real footage does this constantly)
- OCR read the tool's own burned-in frame headers back out and reported them as
  text found in the video

Two of those were found only because a fixture I had written was itself wrong,
which is why `tests/make_motion_fixture.py` renders every frame in Python from a
closed-form viewport path instead of leaning on video filters — a fixture you
have to debug is not a fixture.

The honest summary: the design judgement and the "is this actually true?" loop
were the work; the typing was assisted.
