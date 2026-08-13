# Benchmarks

```bash
python3 tests/benchmark.py                    # the generated fixtures
python3 tests/benchmark.py a.mp4 b.mov ...    # your own files
```

Two things are measured: whether content-aware frame selection actually sees more
of a video than uniform sampling at the same budget, and what conversion costs.

## Frame selection vs uniform sampling

Uniform sampling — N evenly spaced screenshots — is the baseline this format
exists to replace, so it is the thing worth measuring against. Both methods get
**the same number of frames**; the question is which frames.

Three measures, all computed from pixels rather than from opinion:

| Measure | What it means |
|---|---|
| **shots never seen** | Shots that receive no frame at all, and so are invisible in the output. VTL allocates at least one frame per shot by construction, so this is its central claim. |
| **redundant frames** | Frames where under 7% of the picture differs from the previous frame taken — the same threshold the converter's own de-duplication uses. Budget spent to learn nothing. |
| **worst blind interval** | Across every gap between consecutive frames, the largest visible difference between the two frames bracketing it. How much had changed while nobody was looking. |

### Results

Five real videos, 18 minutes total: a narrated slide presentation, two screen
recordings, a short scroll capture, and handheld phone footage.

```
video                      frames   shots never seen   redundant frames   worst blind interval
                                       VTL / uniform      VTL / uniform          VTL / uniform
-----------------------------------------------------------------------------------------------
slide presentation (7m22s)      8        0 / 1              0 / 1             0.20 / 0.21
screen recording (8m48s)       11        0 / 5              0 / 5             0.16 / 0.16
dashboard capture (1m22s)       7        0 / 2              1 / 5             0.14 / 0.07
scroll capture (7s)             3        0 / 1              1 / 1             0.54 / 0.55
handheld phone (16s)           11        0 / 0              0 / 0             0.80 / 0.77
-----------------------------------------------------------------------------------------------
TOTAL                          40        0 / 9              2 / 12
```

Across 28 shots, uniform sampling never looks at **9 of them (32%)** and spends
**12 of its 40 frames** on pictures it had already seen. VTL misses no shot and
wastes 2 frames.

### Where it does *not* win

On **worst blind interval the two are level**, and on the dashboard capture
uniform sampling is better (0.07 vs 0.14). That is the honest result and it
follows from the design: VTL concentrates frames at moments of change and
deliberately declines to re-photograph a motionless slide, which can leave a
wider gap than even spacing would. The difference is that VTL's gaps are
measured and named in the timeline as frozen spans, where uniform sampling's are
simply unexplained.

So the claim this benchmark supports is narrow and specific: **at an equal
budget, content-aware selection sees every shot and wastes almost nothing.** It
is not "smaller gaps everywhere".

### What the measures cannot see

- The blind-interval measure compares the frames bracketing a gap, so it misses
  anything that appears and vanishes entirely inside one gap. Summing change
  across the gap would catch that, but a sum accumulates faint compression noise
  until 80 seconds of a motionless slide looks eventful. Neither is complete;
  this one is harder to mislead with.
- "Shots never seen" depends on VTL's own shot boundaries, which is the thing
  under test. It is fair for counting *coverage* — a shot boundary is a
  measured discontinuity either way — but it is not an independent oracle.
- Uniform sampling is given the frame count VTL chose. On a video where VTL
  selects very few frames (a static image with an animated corner collapses to
  2), uniform sampling is handed a budget too small to do well with. The
  comparison is fairest on ordinary content, which is what the table above is.

## Cost

```
video                        resolution    length   convert    ratio
--------------------------------------------------------------------
slide presentation            1728x1116    441.9s     23.3s    0.05x
screen recording              1728x1116    528.2s     24.4s    0.05x
dashboard capture              1440x900     82.5s      9.6s    0.12x
scroll capture                3430x1880      7.1s      3.2s    0.45x
handheld phone                1080x1920     15.9s      6.5s    0.41x
--------------------------------------------------------------------
TOTAL                                     1075.6s     66.9s    0.06x
```

`ratio` is processing seconds per second of video, so lower is faster and
anything under 1.00 beats real time. 18 minutes of video converts in 67 seconds.

The ratio is much worse on short clips because fixed costs — probing, process
startup, OCR model load — do not amortise. It is close to flat per *frame
extracted* rather than per second of runtime, which is why a 7-second 4K capture
costs about as much as 80 seconds of 1440p.

Measured on an M1 Max, with an x86_64 Python and ffmpeg under Rosetta; a native
arm64 toolchain would be faster.

## Accuracy against known ground truth

Camera-motion rates, asserted numerically in `tests/test_vtl.py` against a
fixture whose every frame is rendered from a closed-form viewport path:

```
segment      quantity   true      measured
 4.3– 7.7s   zoom      +0.115      +0.080
 8.3–11.7s   zoom      -0.115      -0.071
12.3–15.7s   pan       +0.195      +0.208
16.3–19.7s   tilt      +0.139      +0.147
```

Pan and tilt land within ~10%. Zoom reads ~30% low, because block displacement
under zoom is a fraction of a proxy pixel near the frame centre; direction is
reliable, magnitude is a lower bound. Both facts are stated in the bundle itself.

On real footage there is no ground truth, so `tests/crosscheck_motion.py`
measures the same clips with an independent algorithm — FFT phase correlation at
480 px, sharing no code or assumptions with the converter's block matching on a
96 px proxy:

```
clip                        converter        phase correlation
pan shot of a grass field   pan_left -0.088   -0.090
trains in a railroad yard   pan_right +0.083  +0.082
blurred passing cars        no camera move    ~0 (peak 0.04)
handheld pan (last 8s)      pan_right +0.037  +0.057
```

It calibrates itself on the synthetic fixture before being trusted on anything
else.
