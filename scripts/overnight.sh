#!/usr/bin/env bash
# Three fine-tunes in a row, for the two goals (walk backward, walk faster), then robustness:
#   v7_16  backward: more and faster backward commands
#   v7_17  speed:    more high-speed commands, longer step target, quicker rhythm
#   v7_18  shoves:   pushes back on, less sway, wider foot spacing
# Each stage starts from the previous stage's best policy, unless that stage made forward or backward
# walking worse (scripts/analysis/speeds.py); then the next stage starts from the last good one.
# All stages share v7_15d's stepping recipe. Results: runs/overnight_report.txt.
#
#   ./scripts/overnight.sh 2>&1 | tee runs/overnight.log        # ~4 h on an 18-core Mac
#   FROM=runs/other STEPS=20000000 ./scripts/overnight.sh      # another start or stage length
set -euo pipefail
cd "$(dirname "$0")/.."
FROM=${FROM:-runs/juan_v7_15d_best}
STEPS=${STEPS:-30000000}
REPORT=runs/overnight_report.txt
TOLERANCE=${TOLERANCE:-0.05}   # a stage may lose at most 5 points of forward or backward share

BASE=(--randomization push_velocity=0 action_delay_steps=2,4
      --command swing_ratio=0.48 swing_height=0.035
      --reward tracking_vel=5.0 step_length=4.0 feet_air_time=2.0 swing_lift=2.0 swing_reach=1.0 double_support=1.0
               lin_vel_z=0.2 footstep=0.5 gait_mismatch=0.3 foot_slip=4.0 recover_tilt_deg=20)
BACKWARD=(--command backward_prob=0.25 backward_speed=0.25)
SPEED=(--command fast_prob=0.25 fast_speed=0.65 --reward step_length_target=0.15)
SHOVES=(--randomization push_velocity=0.3 --reward orientation=4.0 feet_distance=10.0 min_feet_distance=0.09)

speeds() {  # run, json -> prints the table, writes the json
    python scripts/analysis/speeds.py "$1" --json "$2" | tee -a "$REPORT"
}

pick() {  # previous json, candidate json -> "keep" if the candidate is no worse than the previous
    python - "$1" "$2" "$TOLERANCE" <<'EOF'
import json, sys
prev, cand, tol = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), float(sys.argv[3])
p, c = next(iter(prev.values()))["summary"], next(iter(cand.values()))["summary"]
ok = c["forward_share"] >= p["forward_share"] - tol and c["backward_share"] >= p["backward_share"] - tol
print("keep" if ok else "revert")
EOF
}

stage() {  # name, extra train.py options...
    local name=$1; shift
    echo "=== $(date +%H:%M) stage $name from $FROM" | tee -a "$REPORT"
    FROM="$FROM" STEPS="$STEPS" ./scripts/finetune.sh "$name" "${BASE[@]}" "$@"
    cat "runs/${name}_report.txt" >> "$REPORT"
    speeds "runs/${name}_best" "runs/${name}_speeds.json"
    if [[ $(pick "$PREV_JSON" "runs/${name}_speeds.json") == keep ]]; then
        echo "=== $name kept: the next stage starts from it" | tee -a "$REPORT"
        FROM="runs/${name}_best"; PREV_JSON="runs/${name}_speeds.json"
    else
        echo "=== $name made forward or backward walking worse: the next stage starts from $FROM again" | tee -a "$REPORT"
    fi
}

echo "Overnight run from $FROM, $STEPS steps per stage ($(date))" > "$REPORT"
PREV_JSON=runs/overnight_start_speeds.json
speeds "$FROM" "$PREV_JSON"
stage juan_v7_16_backward "${BACKWARD[@]}"
stage juan_v7_17_speed "${BACKWARD[@]}" "${SPEED[@]}"
stage juan_v7_18_shoves "${BACKWARD[@]}" "${SPEED[@]}" "${SHOVES[@]}"
echo "=== $(date +%H:%M) done. Best policy: $FROM (exported in robot/policies/)" | tee -a "$REPORT"
