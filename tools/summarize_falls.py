"""Summarize tools/fall_breakdown.py outputs: python tools/summarize_falls.py <folder of .jsonl files>"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

def a_physics_varies(rows):
    return np.std([r["stall"] for r in rows]) > 1e-6


d = Path(sys.argv[1])
for f in sorted(d.glob("*.jsonl")):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        continue
    c = Counter(r["outcome"] for r in rows)
    n = len(rows)
    print(f"\n=== {f.stem}  ({n} episodes, 0.4 m/s) ===")
    print(f"survived {c['survived']}/{n}   fell {c['fell']}/{n}   broke (hardware limit) {c['broke']}/{n}"
          f"   mean speed {np.mean([r['speed'] for r in rows]):.2f} m/s")
    bad = [r for r in rows if r["outcome"] != "survived"]
    if not bad:
        continue
    near = [r for r in bad if r.get("since_last_push", 99) < 1.5]
    print(f"  ended within 1.5 s of a push: {len(near)}/{len(bad)}")
    dirs = Counter()
    for r in near:
        fwd, left = r["last_push_fwd"], r["last_push_left"]
        dirs["forward" if abs(fwd) >= abs(left) and fwd > 0 else
             "backward" if abs(fwd) >= abs(left) else
             "left" if left > 0 else "right"] += 1
    if dirs:
        print("  direction of that push (robot frame):", dict(dirs))
        print(f"  its magnitude: median {np.median([r['last_push_mag'] for r in near]):.2f} m/s")
    broke = [r for r in bad if r["outcome"] == "broke"]
    if broke:
        imp = sum(r["end_impact_bw"] > 3.0 for r in broke)
        print(f"  'broke' cause: impact>3xBW {imp}, joint back-driven {len(broke) - imp}")
    # physics of failing vs surviving episodes
    ok = [r for r in rows if r["outcome"] == "survived"]
    if ok and a_physics_varies(rows):
        for k in ("stall", "latency_ms", "friction_scale", "gravity_tilt_deg"):
            print(f"  {k:>16}: failed {np.mean([r[k] for r in bad]):6.2f}  vs survived {np.mean([r[k] for r in ok]):6.2f}")
    print("  peak tilt before end (deg), failed eps: median", f"{np.median([r['peak_tilt_deg'] for r in bad]):.0f}")
    if ok:
        print("  peak tilt over whole ep (deg), survived eps: median",
              f"{np.median([r['peak_tilt_deg'] for r in ok]):.0f}, max {np.max([r['peak_tilt_deg'] for r in ok]):.0f}")

