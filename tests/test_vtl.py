#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seth Wheeler
"""
Ground-truth tests for vtl_convert.

Run:  python3 tests/test_vtl.py
Needs the two fixtures; it builds them if they are missing.

The camera tests call describe_camera on explicit time ranges rather than going
through shot segmentation, because fixture_motion.mp4 deliberately contains no
cuts — the camera behaviour is what is under test, not the cut detector.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import vtl_convert as V  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}   {detail}")
        FAILURES.append(f"{label} {detail}")


def near(got: float, want: float, tol: float) -> bool:
    return abs(got - want) <= tol


def ensure_fixtures() -> tuple[Path, Path]:
    main, motion = HERE / "fixture.mp4", HERE / "fixture_motion.mp4"
    if not main.exists():
        subprocess.run(["bash", str(HERE / "make_fixture.sh")], check=True)
    if not motion.exists():
        subprocess.run([sys.executable, str(HERE / "make_motion_fixture.py")], check=True)
    return main, motion


# ---------------------------------------------------------------------------
# camera motion, against analytically known rates
# ---------------------------------------------------------------------------

def test_camera(motion: Path) -> None:
    print("\ncamera motion (fixture_motion.mp4, rates known in closed form)")
    src = V.probe(motion, do_hash=False)
    aw, ah = V.analysis_dims(src)
    fps = 8.0
    vs = V.visual_signals(V.decode_proxy(src, fps, aw, ah, None, None), fps, 0.0)

    # (start, end, expected label, field, expected rate, tolerance)
    cases = [
        (0.3, 3.7, "static", None, 0.0, 0.0),
        (4.3, 7.7, "zoom_in", "zoom", +0.115, 0.045),
        (8.3, 11.7, "zoom_out", "zoom", -0.115, 0.045),
        (12.3, 15.7, "pan_right", "pan", +0.195, 0.030),
        (16.3, 19.7, "tilt_up", "tilt", +0.139, 0.030),
    ]
    for t0, t1, want_label, field, want_rate, tol in cases:
        a, b = int(t0 * fps), int(t1 * fps)
        label, ev = V.describe_camera(vs, a, b)
        check(label.split(" +")[0] == want_label,
              f"{t0:>4.1f}-{t1:<4.1f}s labelled {want_label}", f"got {label!r}")
        if field:
            got = float(ev.split(f"{field} ")[1].split()[0])
            check(near(got, want_rate, tol),
                  f"{t0:>4.1f}-{t1:<4.1f}s {field} rate {want_rate:+.3f}",
                  f"got {got:+.3f} (tol {tol})")


def test_whip_pan(motion: Path) -> None:
    """A camera move too fast for the matcher must not be read as a cut."""
    print("\nfast camera motion is not a cut (fixture_motion.mp4)")
    import json
    out = Path("/tmp/_vtl_whip.vtl")
    subprocess.run([sys.executable, str(HERE.parent / "vtl_convert.py"), str(motion),
                    "-o", str(out), "--force", "--ocr", "off", "--no-sheets",
                    "--no-hash"], check=True, capture_output=True)
    d = json.loads((out / "timeline.json").read_text())
    # the viewport path is continuous end to end, so there is nothing to cut on
    check(len(d["shots"]) == 1,
          "continuous camera movement yields exactly one shot, no false cuts",
          f"got {len(d['shots'])} shots starting "
          f"{[round(s['start'], 1) for s in d['shots']]}")
    check(all(s["camera"] != "static" for s in d["shots"][:1]),
          "the moving fixture is not called static overall",
          f"got {d['shots'][0]['camera']}")


def test_motion_phases(motion: Path) -> None:
    """A single shot containing several camera moves must report each of them.

    fixture_motion.mp4 is one uncut shot with six known movements in sequence,
    which makes it the exact case a single per-shot label cannot describe.
    """
    print("\nmotion phases within one shot (fixture_motion.mp4)")
    src = V.probe(motion, do_hash=False)
    aw, ah = V.analysis_dims(src)
    fps = 8.0
    vs = V.visual_signals(V.decode_proxy(src, fps, aw, ah, None, None), fps, 0.0)
    ph = V.motion_phases(vs, 0, len(vs.times))

    want = [(0, 4, "static"), (4, 8, "zoom_in"), (8, 12, "zoom_out"),
            (12, 16, "pan_right"), (16, 20, "tilt_up"), (20, 23, "fast_motion")]
    check(len(ph) == len(want), f"{len(want)} phases recovered", f"got {len(ph)}: "
          f"{[(round(p['start'], 1), p['camera']) for p in ph]}")
    for i, (t0, t1, label) in enumerate(want):
        if i >= len(ph):
            break
        got = ph[i]["camera"].split(" + ")[0]
        check(got == label, f"phase {i + 1} ({t0}-{t1}s) is {label}", f"got {got!r}")
        check(near(ph[i]["start"], t0, 1.3),
              f"phase {i + 1} starts near {t0}s", f"got {ph[i]['start']:.1f}")

    # no two neighbours may carry the same label — that is a merge failure, and
    # it makes the timeline claim a change happened where none did
    dup = [i for i in range(len(ph) - 1)
           if ph[i]["camera"].split(" + ")[0] == ph[i + 1]["camera"].split(" + ")[0]]
    check(not dup, "no two adjacent phases share a label", f"at indices {dup}")


def test_no_spurious_phases() -> None:
    """A shot with one steady behaviour must report no phases at all."""
    print("\nphases stay silent when the camera does only one thing")
    import json
    for name, path in (("static title cards / cuts", HERE / "fixture.mp4"),):
        out = Path("/tmp/_vtl_nophase.vtl")
        subprocess.run([sys.executable, str(HERE.parent / "vtl_convert.py"), str(path),
                        "-o", str(out), "--force", "--ocr", "off", "--no-sheets",
                        "--no-hash"], check=True, capture_output=True)
        d = json.loads((out / "timeline.json").read_text())
        n = sum(len(s.get("motion_phases") or []) for s in d["shots"])
        check(n == 0, f"{name}: no phases emitted", f"got {n}")


def test_rendered_source(motion: Path) -> None:
    """Synthetic and screen-recorded sources must be recognised as not-a-camera.

    A camera sensor cannot emit two pixel-identical frames; screen recordings and
    renders do it constantly. Getting this wrong means the timeline says "the
    camera tilted up" when a page scrolled.
    """
    print("\nsource character (camera vs rendered)")
    src = V.probe(motion, do_hash=False)
    aw, ah = V.analysis_dims(src)
    vs = V.visual_signals(V.decode_proxy(src, 8.0, aw, ah, None, None), 8.0, 0.0)
    check(vs.rendered, "a rendered fixture is detected as not camera footage")
    label, ev = V.describe_camera(vs, int(12.3 * 8), int(15.7 * 8))
    check("wrong frame of reference" in ev,
          "a rendered source carries the frame-of-reference caveat")


def test_presentation_layers() -> None:
    """Reporting rules that keep the document from claiming structure it lacks."""
    print("\nreporting hygiene")
    # scene grouping is only a real middle level when it partitions the shots
    cases = [(1, 8, False), (6, 6, False), (1, 1, False), (3, 12, True), (2, 3, True)]
    for n_sc, n_sh, want in cases:
        got = V.scenes_are_informative({"scenes": [{}] * n_sc, "shots": [{}] * n_sh})
        check(got == want,
              f"{n_sc} scene(s) over {n_sh} shot(s) -> "
              f"{'reported' if want else 'suppressed'}", f"got {got}")

    # coherence is unbounded above; printed raw it reads like a bug
    import re
    src = (HERE.parent / "vtl_convert.py").read_text()
    check("return \">10\" if k > 10" in src,
          "coherence above 10 is displayed as >10, not as a bare large number")


def test_estimator_unit() -> None:
    """The block matcher itself, on synthetic shifts with no video in the loop."""
    print("\nmotion estimator (synthetic)")
    rng = np.random.default_rng(0)
    big = rng.normal(128, 40, (200, 300)).astype(np.float32)
    for _ in range(3):
        big = (big + np.roll(big, 1, 0) + np.roll(big, -1, 0)
               + np.roll(big, 1, 1) + np.roll(big, -1, 1)) / 5
    big = (big - big.mean()) / big.std() * 45 + 128

    def crop(cx, cy, scale=1.0, h=54, w=96):
        ys = (np.arange(h) - h / 2) / scale + cy + 100
        xs = (np.arange(w) - w / 2) / scale + cx + 150
        return big[np.ix_(np.clip(np.round(ys).astype(int), 0, 199),
                          np.clip(np.round(xs).astype(int), 0, 299))]

    for vx, vy in [(0, 0), (2, 0), (-2, 0), (0, 2), (3, -1.5)]:
        dx, dy, _, edge = V.estimate_translation(crop(0, 0), crop(vx, vy))
        check(near(dx, -vx, 0.25) and near(dy, -vy, 0.25) and not edge,
              f"shift ({vx},{vy}) -> content ({-vx},{-vy})",
              f"got ({dx:+.2f},{dy:+.2f}) edge={edge}")
    for sc in [1.05, 0.95]:
        a, b = crop(0, 0), crop(0, 0, sc)
        g = V.estimate_translation(a, b)
        s, tx, ty = V.estimate_motion_model(a, b, g[0], g[1])
        check(near(s, sc - 1, 0.02) and near(tx, 0, 0.6) and near(ty, 0, 0.6),
              f"scale {sc} -> s {sc - 1:+.3f}", f"got s={s:+.3f} t=({tx:+.2f},{ty:+.2f})")


# ---------------------------------------------------------------------------
# structure, transitions, OCR, audio
# ---------------------------------------------------------------------------

def test_structure(main: Path) -> None:
    print("\nshots, text and audio (fixture.mp4)")
    import json
    out = Path("/tmp/_vtl_test.vtl")
    subprocess.run([sys.executable, str(HERE.parent / "vtl_convert.py"), str(main),
                    "-o", str(out), "--force", "--no-hash"],
                   check=True, capture_output=True)
    d = json.loads((out / "timeline.json").read_text())

    shots = d["shots"]
    check(len(shots) == 6, "6 shots found", f"got {len(shots)}")
    # cuts at 4, 9, 13, 17, 21 s; the 0-4 s fade must not be split
    for i, want in enumerate([0.0, 4.0, 9.0, 13.0, 17.0, 21.0]):
        if i < len(shots):
            check(near(shots[i]["start"], want, 0.3),
                  f"shot {i + 1} starts at {want}s", f"got {shots[i]['start']}")

    if shots:
        check("fades up" in " ".join(shots[0].get("fades", [])),
              "opening fade from black is detected")
        check(shots[0]["camera"] == "static",
              "a fading-in title card is not mistaken for camera motion",
              f"got {shots[0]['camera']}")
    if len(shots) >= 5:
        check(shots[1]["camera"].startswith("pan_right"),
              "shot 2 (viewport slides right over a test pattern) reads as pan_right",
              f"got {shots[1]['camera']}")
        # shot 4 is a locked-off frame full of animated content: the camera is
        # not moving, and no coherent pan/tilt/zoom should be invented for it
        c4 = shots[3]["camera"]
        check(not any(c4.startswith(x) for x in ("pan_", "tilt_", "zoom_")),
              "shot 4 (static camera, animated content) invents no camera move",
              f"got {c4}")
    if len(shots) >= 3:
        res = [m["shot"] for m in shots[2].get("resembles", [])]
        check(1 in res, "shot 3 (CHAPTER TWO) resembles shot 1 (CHAPTER ONE)",
              f"got {res}")
    # --min-shot is a floor, not an aspiration: int() truncation once let a
    # 0.375s shot through a stated 0.4s minimum
    short = [s["duration"] for s in shots if s["duration"] < 0.4 - 1e-6]
    check(not short, "no shot is shorter than the 0.4s --min-shot floor",
          f"got {short}")
    # these are genuine hard cuts; they must not be reported as dissolves or as
    # screen repaints, which would attribute them to the wrong mechanism
    kinds = {s["transition_in"] for s in shots}
    check(kinds <= {"start", "cut"},
          "genuine hard cuts are labelled 'cut', not 'dissolve' or 'repaint'",
          f"got {sorted(kinds)}")

    frozen = [s for s in shots if s["activity_class"] == "frozen"]
    check(len(frozen) >= 2, "frozen shots detected", f"got {len(frozen)}")

    texts = {e["text"].lower(): e for e in d["text_events"]}
    for want in ["chapter one", "chapter two", "end of tape", "a quiet beginning"]:
        check(any(want in t for t in texts), f"OCR found {want!r}")
    for want, lo, hi in [("chapter one", 0.5, 4.5), ("end of tape", 20.5, 25.5)]:
        hits = [e for t, e in texts.items() if want in t]
        if hits:
            check(lo <= hits[0]["first_seen"] <= hi,
                  f"{want!r} is timestamped in {lo}-{hi}s",
                  f"got {hits[0]['first_seen']}")

    # The converter burns context headers into the frames it extracts. If OCR
    # runs after that, it reads those headers back and reports the tool's own
    # annotations as text found in the video.
    mine = [e["text"] for e in d["text_events"]
            if "t=00:" in e["text"] or "shot 1/" in e["text"]
            or "after prev frame" in e["text"] or "fixture.mp4" in e["text"]]
    check(not mine, "no text event is the converter's own burned-in frame header",
          f"leaked {mine[:2]}")

    au = d["audio"]
    check(au is not None, "audio track analysed")
    if au:
        segs = au["segments"]
        check(any(near(s["start"], 2.0, 0.4) and near(s["end"], 8.0, 0.4) for s in segs),
              "tone at 2-8s found", f"got {[(s['start'], s['end']) for s in segs]}")
        check(any(near(s["start"], 11.0, 0.4) for s in segs), "noise from 11s found")

    kfs = d["keyframes"]
    check(all((out / k["file"]).exists() for k in kfs), "every keyframe file exists")
    check(d["summary"]["max_frame_gap"] < 6.0,
          "no unobserved gap longer than 6s", f"got {d['summary']['max_frame_gap']}")
    # the coverage floor is advertised as a guarantee, so assert it holds
    # the floor applies to stretches where something could have happened; a
    # stretch measured as frozen is covered by the frame before it
    frozen = [(a, b) for s in d["shots"] for a, b in s["frozen_spans"]]
    gaps = []
    for i in range(len(kfs) - 1):
        t0, t1 = kfs[i]["t"], kfs[i + 1]["t"]
        if any(a <= t0 + 0.2 and b >= t1 - 0.2 for a, b in frozen):
            continue
        gaps.append(t1 - t0)
    check(all(g <= 15.0 + 0.5 for g in gaps),
          "no non-frozen gap between frames exceeds the 15s coverage floor",
          f"worst {max(gaps) if gaps else 0:.1f}s")
    check(not d["transcript"]["available"] and d["transcript"]["reason"],
          "missing transcript is declared with a reason")
    check((out / "TIMELINE.md").exists() and (out / "manifest.json").exists(),
          "bundle has TIMELINE.md and manifest.json")


def test_containers(main: Path) -> None:
    """MP4 and MOV of identical content must describe the same timeline."""
    print("\nMP4 vs MOV")
    import json
    mov = main.with_suffix(".mov")
    if not mov.exists():
        print("  SKIP  no .mov fixture")
        return
    res = {}
    for f in (main, mov):
        o = Path(f"/tmp/_vtl_{f.suffix[1:]}.vtl")
        subprocess.run([sys.executable, str(HERE.parent / "vtl_convert.py"), str(f),
                        "-o", str(o), "--force", "--ocr", "off", "--no-sheets",
                        "--no-hash"], check=True, capture_output=True)
        res[f.suffix] = json.loads((o / "timeline.json").read_text())
    a, b = res[".mp4"], res[".mov"]
    check(len(a["shots"]) == len(b["shots"]),
          "same shot count from both containers",
          f"mp4 {len(a['shots'])} vs mov {len(b['shots'])}")
    check(all(near(x["start"], y["start"], 0.2)
              for x, y in zip(a["shots"], b["shots"])),
          "same shot boundaries from both containers")


if __name__ == "__main__":
    main_fx, motion_fx = ensure_fixtures()
    test_estimator_unit()
    test_camera(motion_fx)
    test_whip_pan(motion_fx)
    test_motion_phases(motion_fx)
    test_no_spurious_phases()
    test_rendered_source(motion_fx)
    test_presentation_layers()
    test_structure(main_fx)
    test_containers(main_fx)
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("\nfailures:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
