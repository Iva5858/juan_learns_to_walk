#!/usr/bin/env bash
# juan_v7_12: back to v7_8_best (the policy that works best on the real robot), fine-tuned for what the
# robot logs showed (2026-10-24):
#   * servos: the original model (5.8 rad/s, kp 15) responds like the real ones once they move (~50-80 ms);
#     the bench-fit 3.0 rad/s / kp 6 model of v7_10/v7_11 was too soft, and those policies overcorrect.
#   * dead time: the real joints start moving 40-60 ms after a command (training had ~13): action delay
#     2-3 steps + the within-step latency = 45-79 ms.
#   * gyro: servo buzz shakes the torso (0.5-2.7 rad/s at 15-20 Hz); trained with that buzz and with the
#     robot's 4 Hz gyro low-pass.
#   * upright (from v7_11): torso tilt 2 -> 8, stand at home when commanded to stand.
#   * floors bare PLA to rubber (friction 0.3-1.2), pushes up to 0.3 m/s.
#
#   caffeinate -i ./scripts/train_v7_12.sh 2>&1 | tee runs/juan_v7_12.log      # ~2.5 h on the Mac
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FROM=${FROM:-runs/juan_v7_8_best}
A=juan_v7_12a_delay_vibration
B=juan_v7_12b_pushes
STEPS_A=${STEPS_A:-20000000}
STEPS_B=${STEPS_B:-30000000}
REPORT=runs/juan_v7_12_report.txt

ROBOT=(--robot no_load_speed=5.8 kp=15)
RAND=(--randomization action_delay_steps=2,3 friction=0.3,1.2 gyro_vibration=0,1.2 gyro_lowpass_hz=4)
REWARD=(--reward orientation=8 stand_pose=20 feet_clearance=0.8 ang_vel_xy=0.15)

echo "=== $(date +%H:%M) 1/4 physics: $A from $FROM"
$PY scripts/train.py --run-name $A --stage physics --resume "$FROM" --timesteps "$STEPS_A" --min-std 0.15 \
    "${ROBOT[@]}" "${RAND[@]}" "${REWARD[@]}"

prev=$A
for n in 1 2; do
    level=$($PY -c "import json; print(json.load(open('runs/$prev/curriculum.json'))['difficulty'])")
    echo "=== $(date +%H:%M) physics difficulty reached: $level"
    $PY -c "import sys; sys.exit(0 if $level >= 0.95 else 1)" && break
    echo "=== $(date +%H:%M) extending physics -> runs/${A}_more$n"
    $PY scripts/train.py --run-name "${A}_more$n" --stage physics --resume "runs/$prev" --timesteps 15000000
    prev="${A}_more$n"
done

echo "=== $(date +%H:%M) 2/4 pushes: $B from runs/$prev (lower learning rate: steadier checkpoints)"
$PY scripts/train.py --run-name $B --stage pushes --resume "runs/$prev" --timesteps "$STEPS_B" --lr 1e-4 \
    --randomization push_velocity=0.3

echo "=== $(date +%H:%M) 3/4 export for the robot"
mkdir -p runs/${B}_best
cp runs/$B/best_model.zip runs/${B}_best/final_model.zip
cp runs/$B/best_vecnormalize.pkl runs/${B}_best/vecnormalize.pkl
cp runs/$B/config.json runs/$B/curriculum.json runs/${B}_best/
exported=()
for run in $B ${B}_best $prev; do
    if $PY scripts/export_policy.py runs/$run; then exported+=("$run"); else echo "   ($run not exported: it fell in the check)"; fi
done

echo "=== $(date +%H:%M) 4/4 compare with v7_8_best on the corrected robot model -> $REPORT"
{
    echo "Fall test (30 episodes at 0.4 m/s; servo model, delays and gyro buzz of $B; with / without 0.3 m/s pushes):"
    mkdir -p out/v7_12_falls
    for run in juan_v7_8_best ${exported[@]+"${exported[@]}"}; do
        for mode in full nopush; do
            $PY scripts/analysis/fall_breakdown.py runs/$run $mode -n 30 --physics-from runs/$B \
                --out out/v7_12_falls/${run}_$mode.jsonl > /dev/null
        done
    done
    $PY scripts/analysis/summarize_falls.py out/v7_12_falls/ | grep -A1 "^===" | grep -v "^--"
} 2>&1 | tee $REPORT
echo "=== $(date +%H:%M) done: robot/policies/ and $REPORT"
