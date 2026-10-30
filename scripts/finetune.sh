#!/usr/bin/env bash
# Fine-tune a walking policy, export it for the robot and compare it with where it started.
# The current recipe: train on the servo model measured on the real robot (stiffer, 70-90 ms delay),
# heading and turn-rate rewards up and more straight commands (juan_v7_13), a 4x body-weight landing
# limit, and speed-curriculum safety limits these servos can actually meet (juan_v7_14).
#
#   ./scripts/finetune.sh my_run                                  # ~50 min for 30M steps on an 18-core Mac
#   FROM=runs/other STEPS=10000000 ./scripts/finetune.sh my_run --reward heading=3.0   # extra train.py options
#
# juan_v7_14 (faster, larger joint range): first give v7_13 the 0.8 rad range, then fine-tune it:
#   python scripts/rescale_actions.py runs/juan_v7_13_straight runs/juan_v7_13_scale08 --scale 0.8
#   FROM=runs/juan_v7_13_scale08 ./scripts/finetune.sh juan_v7_14
#
# juan_v7_15 (longer strides, quicker rhythm, no shoves while it learns to go faster):
#   FROM=runs/juan_v7_13_scale08 ./scripts/finetune.sh juan_v7_15 \
#     --randomization push_velocity=0 action_delay_steps=2,4 \
#     --command fast_speed=0.5 swing_ratio=0.48 swing_height=0.035 \
#     --reward tracking_vel=5.0 step_length=4.0 feet_air_time=2.0 lin_vel_z=0.2 footstep=0.5 gait_mismatch=0.3
# juan_v7_15b: the same, with real steps required (feet lifted >= 1 cm), more foot lift and less sliding:
#   ... (as v7_15) ... feet_clearance=2.0 foot_slip=4.0
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

echo "=== $(date +%H:%M) 3/3 compare on the measured servos -> $REPORT"
{
    echo "Straight walking (10 s, no pushes; swing = heading weave, peak to peak):"
    python scripts/analysis/straight.py "$FROM" ${exported[@]+"${exported[@]}"}
    echo
    echo "Falls (30 episodes at 0.4 m/s, with and without 0.3 m/s shoves):"
    python scripts/analysis/falls.py "$FROM" ${exported[@]+"${exported[@]}"}
} 2>&1 | tee "$REPORT"
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
