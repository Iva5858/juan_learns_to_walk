"""Gates of scripts/overnight_walk.sh: does a stage's result pass, and which run to continue from. Prints the
reasons, then a last line "PASS <run>" or "FAIL".

    python scripts/overnight_gate.py getup <before.json> <after.json> <run>          (getup_handover --json)
    python scripts/overnight_gate.py moves|servos|final <report.json> <moves.json> <from> <run> [<run>_best]
                                                                         (finetune.sh's walk_test --json files)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
from walk_test import verdict  # noqa: E402

MOVES = ("back 0.15", "left 0.10", "right 0.10", "turn L 0.5", "turn R 0.5")   # each >= MOVE_SHARE of the command
MOVE_SHARE = 0.4
WEAR_CUT = 0.85        # servo care: calm load and reversals at most this share of where it started
MOVES_SLACK = 0.10     # servo care / final: the moves' mean share may drop at most this much


def getup(before, after, run):
    b, a = (json.loads(Path(p).read_text()) for p in (before, after))
    up = a["got_up"] >= b["got_up"] - 2
    wear = a["load"] <= WEAR_CUT * b["load"]
    print(f"getup: got up {b['got_up']} -> {a['got_up']} of {a['seeds']} ({'ok' if up else 'fewer'}), kept {b['kept']} -> "
          f"{a['kept']}; load {b['load']:.3f} -> {a['load']:.3f} ({'ok' if wear else f'not -{100 * (1 - WEAR_CUT):.0f}%'}), "
          f"peak torque {b['peak_torque']:.2f} -> {a['peak_torque']:.2f}x stall, reversals {b['reversals_per_s']:.0f} -> "
          f"{a['reversals_per_s']:.0f}/s, peak impact {b['peak_impact']:.1f} -> {a['peak_impact']:.1f}x body weight")
    print(f"PASS {run}" if up and wear else "FAIL")


def walker(kind, report, moves, base, *runs):
    R, M = (json.loads(Path(p).read_text()) for p in (report, moves))
    passed = []
    for run in runs:
        if run not in R or run not in M:
            print(f"{run}: not tested (export failed?)")
            continue
        v = verdict(R[base], R[run])
        ok = not v.startswith("NOT BETTER (worse)")
        why = [f"verdict {v.split(' (')[0]}"]
        shares = {m: M[run][m]["share"] for m in MOVES}
        moves_mean, base_mean = M[run]["summary"]["share"], M[base]["summary"]["share"]
        score = min(shares.values())
        if kind == "moves":
            ok &= score >= MOVE_SHARE
            why.append("moves " + " ".join(f"{m} {100 * s:.0f}%" for m, s in shares.items()))
        else:
            ok &= moves_mean >= base_mean - MOVES_SLACK
            why.append(f"moves mean {100 * base_mean:.0f} -> {100 * moves_mean:.0f}%")
        if kind == "servos":
            b, c = R[base]["calm"]["summary"], R[run]["calm"]["summary"]
            ok &= c["load"] <= WEAR_CUT * b["load"] and c["reversals_per_s"] <= WEAR_CUT * b["reversals_per_s"]
            why.append(f"load {b['load']:.3f} -> {c['load']:.3f}, reversals {b['reversals_per_s']:.0f} -> "
                       f"{c['reversals_per_s']:.0f}/s")
            score = -c["load"]
        if kind == "final":
            score = -(R[run]["calm"]["summary"]["falls"] + R[run]["shoves"]["summary"]["falls"])
        print(f"{Path(run).name}: {'pass' if ok else 'fail'} ({'; '.join(why)})")
        if ok:
            passed.append((score, run))
    print(f"PASS {max(passed)[1]}" if passed else "FAIL")


if __name__ == "__main__":
    kind, *args = sys.argv[1:]
    getup(*args) if kind == "getup" else walker(kind, *args)
