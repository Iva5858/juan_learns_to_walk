#!/usr/bin/env bash
# juan_v7_24_speed: v7_21_steady (18 cm/s on the robot at a 0.4 m/s command) made faster, one fine-tune
# (scripts/finetune.sh) with v7_21's settings kept: the 0.5 rad action range, its gait, feet apart, shoves.
#
# v7_21 was never asked for more than 0.4 m/s (commands drawn evenly from -0.15..0.4, the speed curriculum
# never promoted it) and takes ~4 steps/s of ~5 cm: speed has to come from longer steps, not a quicker rhythm.
#   commands up to 0.6 m/s, 30% of them fast (0.45-0.6)
#   a step-length reward: each lifted step pays for how far it lands ahead, up to 10 cm
#
# Judged with scripts/analysis/walk_test.py (standing at home, then walking, 0.1-0.6 m/s), the final or best
# checkpoint, whichever falls less. Kept if at commands of 0.4 m/s and up it is >= 3 cm/s faster than v7_21,
# without falling more (calm +2, with shoves +3 of 72) or weaving more than 3 deg more. Report:
# runs/logs/v7/speed_report.txt.
#
#   caffeinate -i ./scripts/speed.sh 2>&1 | tee runs/logs/v7/speed.log      # ~1.5 h on an 18-core Mac
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs/v7
FROM=${FROM:-runs/v7/juan_v7_21_steady}
NAME=${NAME:-juan_v7_24_speed}
STEPS=${STEPS:-30000000}
REPORT=runs/logs/v7/speed_report.txt
L=runs/logs/v7

STEADY=(--reward tracking_vel=5.0 tracking_broad_share=0.3 feet_distance=10.0 min_feet_distance=0.09)
SPEED=(--command speed_range=0.0,0.6 fast_prob=0.3 --reward step_length=2.0 step_length_target=0.10)

walk_test() {  # run, json
    python scripts/analysis/walk_test.py "$1" --json "$2" | tee -a "$REPORT"
}

echo "speed.sh from $FROM ($(date))" > "$REPORT"
PREV_JSON="$L/$(basename "$FROM")_walk6.json"
walk_test "$FROM" "$PREV_JSON"

echo "=== $(date +%H:%M) train $NAME from $FROM ($STEPS steps)" | tee -a "$REPORT"
REPORTS=0 FROM="$FROM" STEPS="$STEPS" ./scripts/finetune.sh "v7/$NAME" "${STEADY[@]}" "${SPEED[@]}"
walk_test "runs/v7/$NAME" "$L/${NAME}_walk6.json"
walk_test "runs/v7/${NAME}_best" "$L/${NAME}_best_walk6.json"

python - "$PREV_JSON" "$L/${NAME}_walk6.json" "$L/${NAME}_best_walk6.json" "$NAME" <<'EOF' | tee -a "$REPORT"
import json, sys
p, a, b = (json.load(open(f)) for f in sys.argv[1:4])
name = sys.argv[4]
S = lambda r: {m: r[m]["summary"] for m in ("calm", "shoves")}
falls = lambda r: r["calm"]["summary"]["falls"] + r["shoves"]["summary"]["falls"]
cand, c = (name, a) if (falls(a), -a["calm"]["summary"]["fast_speed"]) <= (falls(b), -b["calm"]["summary"]["fast_speed"]) else (name + "_best", b)
P, C = S(p), S(c)
safe = (C["calm"]["falls"] <= P["calm"]["falls"] + 2 and C["shoves"]["falls"] <= P["shoves"]["falls"] + 3
        and C["calm"]["weave_deg"] <= P["calm"]["weave_deg"] + 3)
faster = C["calm"]["fast_speed"] >= P["calm"]["fast_speed"] + 0.03
print(f"=== {cand}: {'KEEP' if safe and faster else 'NOT BETTER'} "
      f"(speed at 0.4+ {P['calm']['fast_speed']:.3f} -> {C['calm']['fast_speed']:.3f} m/s; "
      f"falls calm {P['calm']['falls']} -> {C['calm']['falls']}, shoves {P['shoves']['falls']} -> {C['shoves']['falls']}; "
      f"weave {P['calm']['weave_deg']:.1f} -> {C['calm']['weave_deg']:.1f} deg; share {100*P['calm']['share']:.0f} -> {100*C['calm']['share']:.0f}%)")
EOF
python scripts/runs_index.py --note "v7/$NAME" "speed.sh from $(basename "$FROM"): $(tail -1 "$REPORT" | sed 's/^=== //')"
python scripts/runs_index.py
echo "=== $(date +%H:%M) done (policies exported in robot/policies/)" | tee -a "$REPORT"
