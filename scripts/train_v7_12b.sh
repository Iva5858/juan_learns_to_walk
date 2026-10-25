#!/usr/bin/env bash
# juan_v7_12b: the pushes stage of v7_12 (scripts/train_v7_12.sh), resumed from its straightest
# physics-stage policy (juan_v7_12a_delay_vibration_more1: within 3 deg over 10 s at 0.1-0.3 m/s),
# with the lower learning rate for steadier checkpoints. Then export and compare with v7_8_best and
# the starting policy on the corrected robot model: falls (with / without pushes) and straightness.
#
#   caffeinate -i ./scripts/train_v7_12b.sh 2>&1 | tee runs/juan_v7_12b.log      # ~1.5 h on the Mac
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FROM=${FROM:-runs/juan_v7_12a_delay_vibration_more1}
B=juan_v7_12b_pushes
STEPS=${STEPS:-30000000}
REPORT=runs/juan_v7_12b_report.txt

echo "=== $(date +%H:%M) 1/3 pushes: $B from $FROM"
$PY scripts/train.py --run-name $B --stage pushes --resume "$FROM" --timesteps "$STEPS" --lr 1e-4 \
    --randomization push_velocity=0.3

echo "=== $(date +%H:%M) 2/3 export for the robot"
mkdir -p runs/${B}_best
cp runs/$B/best_model.zip runs/${B}_best/final_model.zip
cp runs/$B/best_vecnormalize.pkl runs/${B}_best/vecnormalize.pkl
cp runs/$B/config.json runs/$B/curriculum.json runs/${B}_best/
exported=()
for run in $B ${B}_best; do
    if $PY scripts/export_policy.py runs/$run; then exported+=("runs/$run"); else echo "   ($run not exported: it fell in the check)"; fi
done

echo "=== $(date +%H:%M) 3/3 compare on the corrected robot model -> $REPORT"
CANDS=(runs/juan_v7_8_best "$FROM" ${exported[@]+"${exported[@]}"})
{
    echo "Fall test (30 episodes at 0.4 m/s; the robot of $B; with / without 0.3 m/s pushes):"
    mkdir -p out/v7_12b_falls
    for run in "${CANDS[@]}"; do
        for mode in full nopush; do
            $PY scripts/analysis/fall_breakdown.py $run $mode -n 30 --physics-from runs/$B \
                --out out/v7_12b_falls/$(basename $run)_$mode.jsonl > /dev/null
        done
    done
    $PY scripts/analysis/summarize_falls.py out/v7_12b_falls/ | grep -A1 "^===" | grep -v "^--"
    echo
    echo "Straight walking (10 s, no pushes):"
    $PY scripts/analysis/straight.py "${CANDS[@]}" --physics-from runs/$B
} 2>&1 | tee $REPORT
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
