# VTL — Video Timeline Bundle, version 1.0

A container format for handing a video to a language model.

## The problem it solves

The naive way to show an LLM a video is to sample N screenshots and paste them
in. That fails for a specific, diagnosable reason: **a screenshot carries no
temporal context.** Given a frame, the model cannot tell

- *when* it happened, or how long after the previous frame,
- whether the previous frame was 0.2 s earlier (same shot, continuous action) or
  40 s earlier (unrelated scene),
- whether the camera moved between them or the subject did,
- whether the two frames are the same place at different times, or different
  places,
- what was said, heard, or written on screen in the gap between samples,
- what it *didn't* see — a 30 s stretch that fell between two samples.

So the model invents a bridge. That's the hallucination: not misreading a
frame, but misreading the *interval between* frames.

VTL fixes this by making the intervals explicit and measured. Frames stop being
the primary evidence and become illustrations attached to a timeline that is
described quantitatively.

## Core design rules

1. **Everything is timestamped, in one monotonic timeline.** Every fact in the
   bundle is anchored to `t` seconds from the start of the video.
2. **Gaps are named.** If nothing was sampled between 00:12 and 00:41, the
   timeline says so, with what the measurements imply about that gap.
3. **Frames are chosen by content, not by clock.** A frame is extracted for the
   state *just after* each significant change, because that is the informative
   moment. Frames that are visually indistinguishable from the previous one are
   dropped. The coverage floor is then enforced: no stretch longer than it goes
   unobserved. A stretch *measured* as frozen counts as observed — the frame
   before it shows the screen for its whole length, and the shot record names
   the frozen span — so the floor does not emit copies of a still image.
4. **Frames are self-describing.** Each extracted image has a burned-in header
   with its timestamp, shot number, position within the shot, and the elapsed
   time since the previous extracted frame. A VTL frame viewed in isolation
   still knows where it came from.
5. **Measured and inferred are separated.** Pixel differences, luma, RMS, OCR
   confidences are measured. "Camera pans left", "likely a slide", "same
   setting as shot 3" are inferred, are labelled as such, and carry the
   measurement that produced them.
6. **Absent signals are declared, not silently skipped.** If no speech
   recognizer was available, the bundle says "no transcript: no ASR engine
   found" rather than leaving a hole the reader fills with assumption.
7. **The bundle is a directory of plain files.** Markdown to read, JSON to
   parse, JPEGs to look at. No custom binary parser to implement.

## Layout

```
name.vtl/
├── TIMELINE.md        Primary document. Read this first.
├── timeline.json      Same content, machine-readable, plus raw signal tracks.
├── manifest.json      Format version, source identity, provenance, tool versions.
├── frames/            Keyframes, full quality, with burned-in context headers.
│   └── k004_t00-01-23.400_shot07.jpg
└── sheets/            Contact sheets: one strip per scene, in time order.
    └── scene02.jpg
```

Extension `.vtl`, a directory bundle. Zip it for transport; the layout is
unchanged inside.

## The timeline model

Four nested levels, from coarse to fine:

| Level | What it is | How it's derived |
|---|---|---|
| **Scene** | Consecutive shots that share a setting | Color-histogram similarity between adjacent shots, agglomerated |
| **Shot** | An unbroken run of camera, or one screen state | Two detectors: a hard cut (frame delta + luma histogram distance, adaptive threshold), or a *repaint* — a large area of the frame replaced at once against a locally static baseline |
| **Keyframe** | An extracted, viewable image | The state just after each significant change, plus a coverage floor |
| **Motion phase** | A run within a shot where the camera does one thing | The shot is walked in short windows, each labelled like a shot, and agreeing neighbours merged |
| **Sample** | One row of measurements | Fixed analysis rate, default 8 Hz |

Plus three time-aligned tracks that run underneath all of it:

- **Audio activity** — RMS envelope, speech-band ratio, onsets, silence spans.
- **Transcript** — if an ASR engine is present. Segment-level, timestamped.
- **On-screen text** — OCR with confidence, deduplicated into text *events* with
  a `first_seen`/`last_seen` span rather than repeated per frame.

## Per-shot record

Every shot carries:

```
start, end, duration
transition_in         cut | fade_in | fade_out | dissolve | start
camera                static | static_camera_moving_subject | pan_left |
                      pan_right | tilt_up | tilt_down | zoom_in | zoom_out |
                      unsteady | drifting | fast_motion, optionally suffixed
                      "+ unsteady" (+ the measured rates that produced the
                      label, in framewidths/s, frameheights/s and scale/s)
activity              mean inter-frame delta, 0..1, a coarse class, and the
                      fraction of the shot during which the image changes at
                      all — a screen recording is near-zero on the mean and
                      still full of information
brightness            mean luma, plus trend if it changes across the shot
motion_phases         when the camera does more than one thing in a shot, the
                      sequence of things it does, each with its own rates.
                      Empty when one label already covers the shot — a per-shot
                      median describes "still for 10s, then pans" as barely
                      moving, which is the summary erasing the event
frozen_spans          intervals where the image stops changing at all
resembles             other shots whose color histogram matches, with score
keyframes             which images belong to this shot, and why each was picked
audio                 activity summary over the shot's span
text                  OCR events overlapping the shot
```

`resembles` is the piece that makes cutting patterns legible: it's how a reader
can tell that shot 7 is a return to the setting of shot 3 (A-B dialogue cutting,
a recurring UI screen) rather than somewhere new.

## Reading order

`TIMELINE.md` is written to be read top to bottom:

1. **How to read this** — what's measured, what's inferred, what's missing.
2. **Source** — duration, dimensions, fps, codecs, rotation, hash.
3. **Summary** — shot/scene counts, motion profile, audio profile, coverage.
4. **Timeline** — every shot in order, with frames referenced inline at the
   point in time where they occur, each with a one-line reason for its
   selection.
5. **Tracks** — transcript, text events, and audio events as flat time-ordered
   lists, for when you want one signal without the interleave.

## Honest limits

- Camera motion is estimated from a 96-px proxy by sub-pixel block matching,
  fitting translation and scale jointly over a 3x3 grid. It distinguishes
  static / pan / tilt / zoom / unsteady reliably and reports pan and tilt rates
  within about 10%; zoom magnitude reads ~30% low and should be treated as a
  lower bound. It is not optical flow and won't track individual objects.
- A camera label is only assigned when the movement is coherent: the typical
  displacement has to stand clear of its own scatter. Content moving inside a
  locked-off frame is reported as such rather than as a slow drift of camera.
- Fast camera movement is not a cut. A whip-pan replaces most of the frame in
  one sample and matches a cut on every magnitude measure; the two are told
  apart by whether a single translation *explains* the change, and by whether
  the movement continues into the following samples.
- A scene with strong depth parallax has no single camera motion. The converter
  describes the dominant plane and flags how much of the frame was trackable.
- Motion phase boundaries are accurate to about a window (1.25 s); they mark
  where the camera changed behaviour, not the exact frame. On footage whose
  motion cannot be measured reliably in the first place — a sky-dominated shot
  from a moving vehicle — phases inherit that unreliability, and each one
  carries the caveat that says so.
- Shot detection is tuned for hard cuts. Slow dissolves are detected as
  gradual-transition spans rather than precise boundaries.
- OCR is tesseract on video frames: strong on screen recordings and titles,
  weak on small text over motion. Confidence is reported per event; treat
  anything under ~70 as a guess.
- Audio classification (speech-like / music-like) is a band-energy heuristic,
  not a classifier. It is labelled as such everywhere it appears.
- Without an ASR engine there is no transcript. The bundle will tell you.

The format's job is to make the model's uncertainty match the video's actual
ambiguity — not to eliminate it.
