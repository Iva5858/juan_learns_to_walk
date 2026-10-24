#!/usr/bin/env bash
# juan_v7_11: stand upright. v7_10 stood ~7 deg off home with the torso leaning back ~9 deg, and
# jumped ~10 deg when it took over from home. Fine-tune of v7_10b best (most push-robust) with
#   * orientation 2 -> 8 (torso tilt, walking too: a 10 deg lean now costs ~21%, Microduck-style)
#   * stand_pose 20 (new: commanded to stand still, stay at home)
# on the same measured servos, floors and pushes; then export and compare with v7_10.
#
#   caffeinate -i ./scripts/train_v7_11.sh 2>&1 | tee runs/juan_v7_11.log      # ~1.5 h on the Mac
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FROM=${FROM:-runs/juan_v7_10b_real_servos_pushes_best}
RUN=juan_v7_11_upright
STEPS=${STEPS:-30000000}
REPORT=runs/juan_v7_11_report.txt

echo "=== $(date +%H:%M) 1/3 pushes: $RUN from $FROM"
$PY scripts/train.py --run-name $RUN --stage pushes --resume "$FROM" --timesteps "$STEPS" \
    --reward orientation=8 stand_pose=20 --randomization push_velocity=0.3

echo "=== $(date +%H:%M) 2/3 export for the robot"
mkdir -p runs/${RUN}_best
cp runs/$RUN/best_model.zip runs/${RUN}_best/final_model.zip
cp runs/$RUN/best_vecnormalize.pkl runs/${RUN}_best/vecnormalize.pkl
cp runs/$RUN/config.json runs/$RUN/curriculum.json runs/${RUN}_best/
exported=()
for run in $RUN ${RUN}_best; do
    if $PY scripts/export_policy.py runs/$run; then exported+=("$run"); else echo "   ($run not exported: it fell in the check)"; fi
done

echo "=== $(date +%H:%M) 3/3 compare with v7_10 (on the measured servos) -> $REPORT"
{
    echo "Standing and straight walking (robot controller, measured servos):"
    $PY scripts/analysis/standing.py juan_v7_10b_real_servos_pushes_best juan_v7_10a_real_servos ${exported[@]+"${exported[@]}"}
    echo
    echo "Fall test (30 episodes at 0.4 m/s, measured servos, with / without 0.3 m/s pushes):"
    mkdir -p out/v7_11_falls
    for run in juan_v7_10b_real_servos_pushes_best ${exported[@]+"${exported[@]}"}; do
        for mode in full nopush; do
            $PY scripts/analysis/fall_breakdown.py runs/$run $mode -n 30 --physics-from runs/juan_v7_10a_real_servos \
                --out out/v7_11_falls/${run}_$mode.jsonl > /dev/null
        done
    done
    $PY scripts/analysis/summarize_falls.py out/v7_11_falls/ | grep -A1 "^===" | grep -v "^--"
} 2>&1 | tee $REPORT
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
