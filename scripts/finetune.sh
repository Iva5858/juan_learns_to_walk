#!/usr/bin/env bash
# Fine-tune a walking policy, export it for the robot and compare it with where it started.
# The defaults are the recipe of juan_v7_13 (walk straight): train on the servo model measured on the
# real robot (stiffer, 70-90 ms delay), with heading and turn-rate rewards up and more straight commands.
#
#   ./scripts/finetune.sh my_run                                  # ~50 min for 30M steps on an 18-core Mac
#   FROM=runs/other STEPS=10000000 ./scripts/finetune.sh my_run --reward heading=3.0   # extra train.py options
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/finetune.sh <run name> [train.py options]}
shift
FROM=${FROM:-checkpoints/juan_v7_12b_pushes_best}
STEPS=${STEPS:-30000000}
REPORT=runs/${NAME}_report.txt

echo "=== $(date +%H:%M) 1/3 train runs/$NAME from $FROM"
python scripts/train.py --run-name "$NAME" --stage pushes --resume "$FROM" --timesteps "$STEPS" --lr 1e-4 \
    --robot kp=30 \
    --randomization push_velocity=0.3 action_delay_steps=3,4 \
    --command straight_prob=0.5 \
    --reward heading=2.0 tracking_lat_yaw=1.0 path_offset=4.0 \
    "$@"

echo "=== $(date +%H:%M) 2/3 export for the robot (final and best model)"
mkdir -p "runs/${NAME}_best"
cp "runs/$NAME/best_model.zip" "runs/${NAME}_best/final_model.zip"
cp "runs/$NAME/best_vecnormalize.pkl" "runs/${NAME}_best/vecnormalize.pkl"
cp "runs/$NAME/config.json" "runs/$NAME/curriculum.json" "runs/${NAME}_best/"
exported=()
for run in "runs/$NAME" "runs/${NAME}_best"; do
    if python scripts/export_policy.py "$run"; then exported+=("$run"); else echo "   ($run not exported: it failed the check)"; fi
done

echo "=== $(date +%H:%M) 3/3 compare on the measured servos -> $REPORT"
{
    echo "Straight walking (10 s, no pushes; swing = heading weave, peak to peak):"
    python scripts/analysis/straight.py "$FROM" ${exported[@]+"${exported[@]}"}
    echo
    echo "Falls (30 episodes at 0.4 m/s, with and without 0.3 m/s shoves):"
    python scripts/analysis/falls.py "$FROM" ${exported[@]+"${exported[@]}"}
} 2>&1 | tee "$REPORT"
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
