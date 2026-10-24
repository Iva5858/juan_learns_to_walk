#!/usr/bin/env bash
# juan_v7_10: v7_8_best re-trained on the servos measured on the robot (bench: ~3.0 rad/s no-load,
# kp ~6, 24-28 ms before a servo moves), with bare-PLA-to-rubber floor friction, more foot lift and a
# steadier torso. Then the policies are exported for the robot (robot/policies/).
#
#   caffeinate -i ./scripts/train_v7_10.sh 2>&1 | tee runs/juan_v7_10.log     # ~3-4.5 h on the Mac
#
# 1. physics: no pushes, forgiving speed tracking: re-learn the gait for slower, softer servos
# 2. pushes:  pushes up to 0.3 m/s (Microduck: pushes faster than the walk make a nervous gait)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FROM=${FROM:-runs/juan_v7_8_best}
A=juan_v7_10a_real_servos
B=juan_v7_10b_real_servos_pushes
STEPS_A=${STEPS_A:-25000000}
STEPS_B=${STEPS_B:-40000000}

ROBOT=(--robot no_load_speed=3.0 kp=6)
RAND=(--randomization action_delay_steps=1,2 friction=0.3,1.2)
REWARD=(--reward feet_clearance=0.8 ang_vel_xy=0.15)

echo "=== $(date +%H:%M) 1/3 physics: $A from $FROM"
$PY scripts/train.py --run-name $A --stage physics --resume "$FROM" --timesteps "$STEPS_A" --min-std 0.15 \
    "${ROBOT[@]}" "${RAND[@]}" "${REWARD[@]}"

# The physics curriculum eases the realism while the robot falls a lot; continue until it is back at
# (nearly) full difficulty, so the pushes stage starts on the realistic robot. Up to 2 extensions.
prev=$A
for n in 1 2; do
    level=$($PY -c "import json; print(json.load(open('runs/$prev/curriculum.json'))['difficulty'])")
    echo "=== $(date +%H:%M) physics difficulty reached: $level"
    $PY -c "import sys; sys.exit(0 if $level >= 0.95 else 1)" && break
    echo "=== $(date +%H:%M) extending physics -> runs/${A}_more$n"
    $PY scripts/train.py --run-name "${A}_more$n" --stage physics --resume "runs/$prev" --timesteps 15000000
    prev="${A}_more$n"
done

echo "=== $(date +%H:%M) 2/3 pushes: $B from runs/$prev"
$PY scripts/train.py --run-name $B --stage pushes --resume "runs/$prev" --timesteps "$STEPS_B" \
    --randomization push_velocity=0.3

echo "=== $(date +%H:%M) 3/3 export for the robot"
# The checkpoint the training evaluation liked best, as its own run folder
mkdir -p runs/${B}_best
cp runs/$B/best_model.zip runs/${B}_best/final_model.zip
cp runs/$B/best_vecnormalize.pkl runs/${B}_best/vecnormalize.pkl
cp runs/$B/config.json runs/$B/curriculum.json runs/${B}_best/
for run in runs/$B runs/${B}_best "runs/$prev"; do
    $PY scripts/export_policy.py "$run" || echo "   ($run not exported: it fell in the check)"
done
echo "=== $(date +%H:%M) done: robot/policies/"
ls -la robot/policies/
