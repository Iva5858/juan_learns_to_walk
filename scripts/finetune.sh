#!/usr/bin/env bash
# Fine-tune a walking policy, export it for the robot and judge it against where it started.
#   1. train: the base recipe below (the servo model measured on the robot, 0.3 m/s shoves, straighter
#      walking, hardware limits these servos meet) plus any train.py options given after the name;
#   2. export the final and the best model to robot/policies/ (each checked against the simulator);
#   3. walk test (scripts/analysis/walk_test.py) of FROM, the final and the best model, with a KEEP / NOT
#      BETTER verdict for each against FROM, the other moves (backward, sideways, turning: walk_test --moves),
#      then the standing shove test. Report: runs/logs/<name>_report.txt.
# The recipes of every fine-tune so far are in docs/training.md.
#
#   caffeinate -i ./scripts/finetune.sh v7/juan_v7_25 2>&1 | tee runs/logs/v7/juan_v7_25.log      # ~1.5 h for 30M steps
#   FROM=runs/v7/juan_v7_24_speed STEPS=60000000 ./scripts/finetune.sh v7/juan_v7_25 --reward heading=3.0
#   QUICK=1 ./scripts/finetune.sh ...      # step 3 without the standing shove test (~15 min less)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/finetune.sh <generation>/<run name> [train.py options]}
shift
FROM=${FROM:-checkpoints/juan_v7_13_straight}
STEPS=${STEPS:-30000000}
REPORT=runs/logs/${NAME}_report.txt
mkdir -p "$(dirname "$REPORT")"

echo "=== $(date +%H:%M) 1/3 train runs/$NAME from $FROM ($STEPS steps)"
python scripts/train.py --run-name "$NAME" --stage pushes --resume "$FROM" --timesteps "$STEPS" --lr 1e-4 \
    --robot kp=30 \
    --randomization push_velocity=0.3 action_delay_steps=3,4 \
    --command straight_prob=0.5 \
    --reward heading=2.0 tracking_lat_yaw=1.0 path_offset=4.0 \
             break_impact=4.0 safe_torque=1.0 safe_joint_speed=1.3 safe_impact=3.0 \
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

echo "=== $(date +%H:%M) 3/3 judge against $FROM -> $REPORT"
{
    python scripts/analysis/walk_test.py "$FROM" ${exported[@]+"${exported[@]}"} --verdict --json "${REPORT%.txt}.json"
    echo
    python scripts/analysis/walk_test.py "$FROM" ${exported[@]+"${exported[@]}"} --moves --json "${REPORT%.txt}_moves.json"
    if [[ -z ${QUICK:-} ]]; then
        echo
        python scripts/analysis/stand_pushes.py "$FROM" ${exported[@]+"${exported[@]}"}
    fi
} 2>&1 | tee "$REPORT"
python scripts/runs_index.py --note "$NAME" "from $(basename "$FROM"): $(grep "^=== $(basename "$NAME") vs" "$REPORT" | sed 's/^=== [^:]*: //')"
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
