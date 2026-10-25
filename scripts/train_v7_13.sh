#!/usr/bin/env bash
# juan_v7_13: walk straight. Fine-tune of v7_12b_pushes_best on the servo model fitted from the robot's
# own walking logs (scripts/analysis/replay.py: kp 25-40, 70-90 ms total delay). v7_12b trained with
# kp 15 and 45-79 ms; on the fitted model it weaves like the real robot (heading swings ~46 deg peak to
# peak with a ~5 s period, vs ~22 deg / 1.5 s on its training model), so the weave is the policy
# steering too late for the real servos. Here it trains on the real servos, with yaw rate and heading
# weighted up and more straight commands.
#
#   caffeinate -i ./scripts/train_v7_13.sh 2>&1 | tee runs/juan_v7_13.log      # ~1 h on the Mac
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FROM=${FROM:-runs/juan_v7_12b_pushes_best}
B=juan_v7_13_straight
STEPS=${STEPS:-30000000}
REPORT=runs/juan_v7_13_report.txt

echo "=== $(date +%H:%M) 1/3 straight: $B from $FROM"
$PY scripts/train.py --run-name $B --stage pushes --resume "$FROM" --timesteps "$STEPS" --lr 1e-4 \
    --robot kp=30 \
    --randomization push_velocity=0.3 action_delay_steps=3,4 \
    --command straight_prob=0.5 \
    --reward heading=2.0 tracking_lat_yaw=1.0 path_offset=4.0

echo "=== $(date +%H:%M) 2/3 export for the robot"
mkdir -p runs/${B}_best
cp runs/$B/best_model.zip runs/${B}_best/final_model.zip
cp runs/$B/best_vecnormalize.pkl runs/${B}_best/vecnormalize.pkl
cp runs/$B/config.json runs/$B/curriculum.json runs/${B}_best/
exported=()
for run in $B ${B}_best; do
    if $PY scripts/export_policy.py runs/$run; then exported+=("runs/$run"); else echo "   ($run not exported: it fell in the check)"; fi
done

echo "=== $(date +%H:%M) 3/3 compare on the fitted robot model -> $REPORT"
CANDS=("$FROM" ${exported[@]+"${exported[@]}"})
{
    echo "Straight walking (10 s, no pushes; swing = heading weave peak to peak, drift removed):"
    $PY scripts/analysis/straight.py "${CANDS[@]}" --physics-from runs/$B --speeds 0.1 0.2 0.3
    echo
    echo "Fall test (30 episodes at 0.4 m/s; the robot of $B; with / without 0.3 m/s pushes):"
    mkdir -p out/v7_13_falls
    for run in "${CANDS[@]}"; do
        for mode in full nopush; do
            $PY scripts/analysis/fall_breakdown.py $run $mode -n 30 --physics-from runs/$B \
                --out out/v7_13_falls/$(basename $run)_$mode.jsonl > /dev/null
        done
    done
    $PY scripts/analysis/summarize_falls.py out/v7_13_falls/ | grep -A1 "^===" | grep -v "^--"
} 2>&1 | tee $REPORT
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
