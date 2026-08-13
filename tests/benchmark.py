#!/usr/bin/env python3
"""
Benchmark VTL's frame selection against uniform sampling, at an equal budget.

Uniform sampling — "take N screenshots, evenly spaced" — is the baseline this
project exists to replace. The interesting question is not whether VTL is
prettier but whether, *given the same number of frames*, it sees more of what
happened. Three measures, all derived from pixels rather than from opinion:

  shots never sampled   Uniform sampling lands zero frames in some shots, so
                        those shots are invisible in the output. VTL allocates at
                        least one frame per shot by construction, so this number
                        is its central claim and should always be 0.

  redundant frames      Frames whose visible content is nearly identical to the
                        previous frame taken (<7% of the picture changed — the
                        same threshold the converter's own de-duplication uses).
                        Budget spent to learn nothing.

  worst blind interval  Across every gap between consecutive sampled frames, the
                        largest visible difference between the frames bracketing
                        that gap. It answers "how much had changed while nobody
                        was looking" — the quantity a reader has to guess about,
                        and guessing about it is where descriptions of video go
                        wrong.

                        Measured as difference between the two bracketing frames
                        rather than as change summed over the gap, because a sum
                        accumulates faint compression noise: 80 seconds of a
                        motionless slide adds up to a number that looks like
                        something happened. The trade-off is that this cannot see
                        something that appears and vanishes entirely inside one
                        gap; the summed version cannot tell noise from content.
                        Neither is complete, and this one is harder to mislead
                        with.

Also reports wall-clock conversion cost.

Usage:
    python3 tests/benchmark.py                      # the generated fixtures
    python3 tests/benchmark.py a.mp4 b.mov ...      # plus/instead, real files
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import vtl_convert as V  # noqa: E402

FPS = 8.0
DUP_AREA = 0.07          # same "visibly different" threshold the converter uses


def convert(path: Path, out: Path) -> tuple[dict, float]:
    """Run the real converter; return its bundle data and how long it took."""
    t = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(HERE.parent / "vtl_convert.py"), str(path),
         "-o", str(out), "--force", "--no-hash", "--no-sheets"],
        capture_output=True, text=True)
    elapsed = time.perf_counter() - t
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return json.loads((out / "timeline.json").read_text()), elapsed


def measure(sample_t: list[float], shots: list[dict], L: np.ndarray,
            delta: np.ndarray, times: np.ndarray) -> dict:
    """Score one set of sample times against the same pixel evidence."""
    idx = [int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
           for t in sorted(sample_t)]

    missed = sum(1 for s in shots
                 if not any(s["start"] <= t < s["end"] for t in sample_t))

    redundant = 0
    for a, b in zip(idx, idx[1:]):
        if a == b:
            redundant += 1
            continue
        if float((np.abs(L[b] - L[a]) > 12).mean()) < DUP_AREA:
            redundant += 1

    # how much had changed across each unobserved gap, endpoints included
    edges = [0] + idx + [len(times) - 1]
    blind = [float((np.abs(L[b] - L[a]) > 12).mean())
             for a, b in zip(edges, edges[1:]) if b > a]

    return {"n": len(sample_t), "missed": missed, "redundant": redundant,
            "worst_blind": max(blind) if blind else 0.0}


def uniform_times(t0: float, dur: float, n: int) -> list[float]:
    """Centre of each of n equal intervals — the standard naive sampling."""
    return [t0 + dur * (i + 0.5) / n for i in range(n)]


def run(path: Path) -> dict | None:
    out = Path("/tmp/_vtl_bench.vtl")
    try:
        d, secs = convert(path, out)
    except RuntimeError as e:
        print(f"  {path.name}: FAILED — {e}")
        return None

    src = V.probe(path, do_hash=False)
    aw, ah = V.analysis_dims(src)
    frames = V.decode_proxy(src, FPS, aw, ah, None, None)
    L = V.luma(frames)
    T = frames.shape[0]
    times = np.arange(T) / FPS
    delta = np.zeros(T, np.float32)
    for i in range(1, T):
        delta[i] = np.abs(L[i] - L[i - 1]).mean() / 255.0

    shots = d["shots"]
    vtl_t = [k["t"] for k in d["keyframes"]]
    n = len(vtl_t)
    dur = d["summary"]["duration"]

    return {
        "name": path.name,
        "dur": dur,
        "res": f"{src.width}x{src.height}",
        "shots": len(shots),
        "secs": secs,
        "rendered": bool(d["analysis"].get("rendered_source")),
        "vtl": measure(vtl_t, shots, L, delta, times),
        "uni": measure(uniform_times(0.0, dur, n), shots, L, delta, times),
    }


def main() -> None:
    args = [Path(a).expanduser() for a in sys.argv[1:]]
    if not args:
        main_fx, motion_fx = HERE / "fixture.mp4", HERE / "fixture_motion.mp4"
        for p, gen in ((main_fx, ["bash", str(HERE / "make_fixture.sh")]),
                       (motion_fx, [sys.executable, str(HERE / "make_motion_fixture.py")])):
            if not p.exists():
                subprocess.run(gen, check=True)
        args = [main_fx, motion_fx]

    rows = [r for r in (run(p) for p in args if p.exists()) if r]
    if not rows:
        sys.exit("no videos to benchmark")

    print("\nFRAME SELECTION — equal budget, VTL vs uniform sampling")
    print(f"\n  {'video':<26} {'frames':>6}  {'shots never seen':>17}  "
          f"{'redundant frames':>17}  {'worst blind interval':>21}")
    print(f"  {'':<26} {'':>6}  {'VTL / uniform':>17}  {'VTL / uniform':>17}  "
          f"{'VTL / uniform':>21}")
    print("  " + "-" * 94)
    for r in rows:
        v, u = r["vtl"], r["uni"]
        print(f"  {r['name'][:26]:<26} {v['n']:>6}  "
              f"{v['missed']:>7} / {u['missed']:<7}  "
              f"{v['redundant']:>7} / {u['redundant']:<7}  "
              f"{v['worst_blind']:>9.2f} / {u['worst_blind']:<9.2f}")

    tm, um = sum(r["vtl"]["missed"] for r in rows), sum(r["uni"]["missed"] for r in rows)
    tr, ur = sum(r["vtl"]["redundant"] for r in rows), sum(r["uni"]["redundant"] for r in rows)
    tot_shots = sum(r["shots"] for r in rows)
    print("  " + "-" * 94)
    print(f"  {'TOTAL':<26} {sum(r['vtl']['n'] for r in rows):>6}  "
          f"{tm:>7} / {um:<7}  {tr:>7} / {ur:<7}")
    print(f"\n  Across {len(rows)} videos and {tot_shots} shots: uniform sampling never "
          f"looks at {um} of them\n  ({um / max(tot_shots, 1) * 100:.0f}%), and spends "
          f"{ur} of its frames on pictures it had already seen.")

    print("\nCOST")
    print(f"\n  {'video':<26} {'resolution':>12} {'length':>9} {'convert':>9} {'ratio':>8}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['name'][:26]:<26} {r['res']:>12} {r['dur']:>8.1f}s "
              f"{r['secs']:>8.1f}s {r['secs'] / max(r['dur'], .01):>7.2f}x")
    tot_d = sum(r["dur"] for r in rows)
    tot_s = sum(r["secs"] for r in rows)
    print("  " + "-" * 70)
    print(f"  {'TOTAL':<26} {'':>12} {tot_d:>8.1f}s {tot_s:>8.1f}s "
          f"{tot_s / max(tot_d, .01):>7.2f}x")
    print(f"\n  'ratio' is processing seconds per second of video — under 1.00 is "
          f"faster than\n  real time. Includes decode, motion analysis, OCR and frame "
          f"extraction.\n")


if __name__ == "__main__":
    main()
