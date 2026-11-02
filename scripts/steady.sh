#!/usr/bin/env bash
# juan_v7_21-23: v7_13 (the policy that walks best on the real robot) made steadier, straighter and harder
# to knock over, in three fine-tunes (scripts/finetune.sh, the 0.5 rad action range and v7_13's gait kept:
# v7_14-v7_20 widened it to 0.8 rad and lifted the feet higher, and on the floor they went diagonally and
# toppled by themselves).
#
#   1 steady    speed tracking 3 -> 5 with more reward for partial speed (broad share 0.15 -> 0.3), feet
#               kept apart (v7_18's feet_distance 10, min 9 cm): it loses its speed stumbling, not walking
#   2 straight  heading 2 -> 3, turn-rate tracking 1 -> 2, larger joint offsets (0.03 -> 0.05 rad, each joint
#               its own, so the legs differ like the real ones and it learns to hold a line anyway)
#   3 shoves    the shove curriculum continues past 0.3 up to 0.5 m/s
#
# Each stage is judged with scripts/analysis/walk_test.py (standing at home, then walking, like the robot),
# its final or best checkpoint, whichever is better. It is kept if it doesn't fall more than the one before
# (calm +2, with shoves +3 of 48) and improves its own target; otherwise the next stage starts from the
# one before again. Report: runs/logs/v7/steady_report.txt.
#
#   caffeinate -i ./scripts/steady.sh 2>&1 | tee runs/logs/v7/steady.log      # ~2.5 h on an 18-core Mac
#   FROM=runs/v7/other ./scripts/steady.sh                                   # another start
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs/v7
FROM=${FROM:-runs/v7/juan_v7_13_straight}
REPORT=runs/logs/v7/steady_report.txt
L=runs/logs/v7

STEADY=(--reward tracking_vel=5.0 tracking_broad_share=0.3 feet_distance=10.0 min_feet_distance=0.09)
STRAIGHT=(--reward heading=3.0 tracking_lat_yaw=2.0 --randomization joint_offset=0.05)
SHOVES=(--randomization push_velocity=0.5)

walk_test() {  # run, json
    python scripts/analysis/walk_test.py "$1" --json "$2" | tee -a "$REPORT"
}

better_of() {  # run, run_best -> whichever falls less (calm + shoves), then the faster
    python - "$L/$(basename "$1")_walk.json" "$L/$(basename "$2")_walk.json" "$1" "$2" <<'EOF'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
score = lambda r: (r["calm"]["summary"]["falls"] + r["shoves"]["summary"]["falls"], -r["calm"]["summary"]["share"])
print(sys.argv[3] if score(a) <= score(b) else sys.argv[4])
EOF
}

pick() {  # previous json, candidate json, target (share|weave|shoves) -> keep / revert, with the reason
    python - "$1" "$2" "$3" <<'EOF'
import json, sys
p, c, target = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
P, C = ({m: r[m]["summary"] for m in ("calm", "shoves")} for r in (p, c))
safe = C["calm"]["falls"] <= P["calm"]["falls"] + 2 and C["shoves"]["falls"] <= P["shoves"]["falls"] + 3
goal = {"share": C["calm"]["share"] >= P["calm"]["share"] + 0.05,
        "weave": C["calm"]["weave_deg"] <= P["calm"]["weave_deg"] - 2 and C["calm"]["turn_deg"] <= P["calm"]["turn_deg"] + 1,
        "shoves": C["shoves"]["falls"] <= P["shoves"]["falls"] - 3}[target]
print(("keep" if safe and goal else "revert") + f" (falls calm {P['calm']['falls']}->{C['calm']['falls']}, "
      f"shoves {P['shoves']['falls']}->{C['shoves']['falls']}; share {100*P['calm']['share']:.0f}->{100*C['calm']['share']:.0f}%; "
      f"weave {P['calm']['weave_deg']:.1f}->{C['calm']['weave_deg']:.1f} deg; turn {P['calm']['turn_deg']:.1f}->{C['calm']['turn_deg']:.1f} deg)")
EOF
}

stage() {  # name, steps, target, extra train.py options...
    local name=$1 steps=$2 target=$3; shift 3
    echo "=== $(date +%H:%M) stage $name from $FROM ($steps steps, target: $target)" | tee -a "$REPORT"
    REPORTS=0 FROM="$FROM" STEPS="$steps" ./scripts/finetune.sh "v7/$name" "$@"
    walk_test "runs/v7/$name" "$L/${name}_walk.json"
    walk_test "runs/v7/${name}_best" "$L/${name}_best_walk.json"
    local cand; cand=$(better_of "runs/v7/$name" "runs/v7/${name}_best")
    local verdict; verdict=$(pick "$PREV_JSON" "$L/$(basename "$cand")_walk.json" "$target")
    echo "=== $(basename "$cand"): $verdict" | tee -a "$REPORT"
    python scripts/runs_index.py --note "v7/$(basename "$cand")" "steady.sh stage $target from $(basename "$FROM"): $verdict"
    if [[ $verdict == keep* ]]; then
        FROM=$cand; PREV_JSON="$L/$(basename "$cand")_walk.json"
    fi
}

echo "steady.sh from $FROM ($(date))" > "$REPORT"
PREV_JSON="$L/$(basename "$FROM")_walk.json"
walk_test "$FROM" "$PREV_JSON"
stage juan_v7_21_steady 30000000 share "${STEADY[@]}"
stage juan_v7_22_straight 20000000 weave "${STEADY[@]}" "${STRAIGHT[@]}"
stage juan_v7_23_shoves 30000000 shoves "${STEADY[@]}" "${STRAIGHT[@]}" "${SHOVES[@]}"
python scripts/runs_index.py
echo "=== $(date +%H:%M) done. Best policy: $FROM (exported in robot/policies/)" | tee -a "$REPORT"
