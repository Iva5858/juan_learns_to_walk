#!/usr/bin/env bash
# One night of free-run training (v11), robustness first while keeping the speed, each stage one change:
#   A. kp ladder        servo stiffness 20 -> 25 -> 30 (the robot's measured kp), delay 1-4, nominal robot   3 x STEPS_KP
#   B. physics          the measured servos (kp 30, delay 3-4) on randomized robots, phased in by the curriculum  STEPS_B
#   C. pushes           full randomization + shoves from 8 directions, each growing to 0.4 m/s as it copes       STEPS_C
#   E. steering         a turn rate while going (+-0.6 rad/s, the dashboard's range), redrawn every few seconds  STEPS_E
#   D. servo protection v7's servo-wear terms, light (jerk, load, energy, reversals), steering kept            STEPS_D
# Every stage continues the previous one's pick (free_run_test --pick on that stage's robot) with exploration kept up
# (--min-std-rad 0.03: at 0.0156 training collapsed, juan_v11h) at lr 1e-4. A stage that fails its check (> 30% falls on
# its robot) runs once more for half its steps; the night goes on either way, and the report says so.
# Report: runs/logs/<name>_night.txt (every stage's test), exports of C, E and D for the robot. On the robot these start
# from straight legs (Straight, not Home); forward = go, the turn passes through, they never learned to stand: stop = Esc.
#
#   FROM=runs/v11/juan_v11k_knee_speed caffeinate -i ./scripts/overnight_v11.sh v11/juan_v11n 2>&1 | tee runs/logs/v11/night.log
# Smoke test: STEPS_KP=40000 STEPS_B=40000 STEPS_C=40000 STEPS_E=40000 STEPS_D=40000 EXTRA="--n-envs 4 --eval-every 36000
#   --eval-episodes 1 --video-every 0" FROM=... ./scripts/overnight_v11.sh test/night   (then delete runs/test/night_*)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: FROM=<free-run run> scripts/overnight_v11.sh <generation>/<name prefix>}
P=${FROM:?FROM=<the free-run run to start from>}
STEPS_KP=${STEPS_KP:-10000000}
STEPS_B=${STEPS_B:-80000000}
STEPS_C=${STEPS_C:-80000000}
STEPS_E=${STEPS_E:-30000000}
STEPS_D=${STEPS_D:-30000000}
REPORT=runs/logs/${NAME}_night.txt
mkdir -p "$(dirname "$REPORT")"
: > "$REPORT"
# shellcheck disable=SC2206
EXTRA=(${EXTRA:-})
COMMON=(--min-std 0.01 --min-std-rad 0.03 --lr 1e-4 --reward hip_roll=100 hip_roll_free_deg=12)
MEASURED=(--robot kp=30 --randomization action_delay_steps=3,4)

say() { echo "=== $(date +%H:%M) $*" | tee -a "$REPORT"; }
best_copy() {
    mkdir -p "runs/${1}_best"
    cp "runs/$1/best_model.zip" "runs/${1}_best/final_model.zip"
    cp "runs/$1/best_vecnormalize.pkl" "runs/${1}_best/vecnormalize.pkl"
    cp "runs/$1/config.json" "runs/$1/curriculum.json" "runs/${1}_best/"
}
# train <run> <stage> <steps> <test robot> <train.py options...>: train from $P, test final and best, P = the pick;
# returns 1 if the pick fails the check on <test robot> (> 30% falls)
train() {
    local run=$1 stage=$2 steps=$3 robot=$4
    shift 4
    say "$run: $stage from $P ($steps steps) $*"
    python scripts/train.py --run-name "$run" --stage "$stage" --resume "$P" --timesteps "$steps" "${COMMON[@]}" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
    best_copy "$run"
    P=$(python scripts/analysis/free_run_test.py "runs/$run" "runs/${run}_best" --pick "$robot" --json "runs/logs/${run}_test.json" \
        | tee -a "$REPORT" | tail -1 | sed 's/^pick ([a-z]*): //')
    local falls
    falls=$(python -c "import json,sys; r=json.load(open('runs/logs/${run}_test.json'))[sys.argv[1]][sys.argv[2]]; print(r['falls']/r['episodes'])" \
        "$P" "$robot")
    python -c "import sys; sys.exit(float(sys.argv[1]) > 0.3)" "$falls" && { say "$run passed on $robot: $P"; return 0; }
    say "$run: $P falls in $(python -c "print(round(100*$falls))")% on $robot"
    return 1
}
# stage <run> <stage> <steps> <test robot> <options...>: train, and once more for half the steps if it failed its check
stage() {
    local run=$1 stage=$2 steps=$3 robot=$4
    shift 4
    train "$run" "$stage" "$steps" "$robot" "$@" || train "${run}b" "$stage" $((steps / 2)) "$robot" "$@" || say "${run}b: going on anyway"
}

say "start from $P"
for kp in 20 25 30; do   # A
    stage "${NAME}_kp$kp" speed "$STEPS_KP" nominal --robot kp=$kp --randomization nominal_delay_range=1,4
done
stage "${NAME}_physics" speed_physics "$STEPS_B" randomized "${MEASURED[@]}"   # B
stage "${NAME}_pushes" speed_physics "$STEPS_C" shoved "${MEASURED[@]}" \
    --randomization physics_curriculum=0 push_curriculum=1 push_velocity=0.4   # C
C=$P
PUSHES=(--randomization physics_curriculum=0 push_curriculum=1 push_velocity=0.4)
stage "${NAME}_steer" speed_physics "$STEPS_E" turning "${MEASURED[@]}" "${PUSHES[@]}" --command go_turn_range=-0.6,0.6   # E
E=$P
stage "${NAME}_servos" speed_physics "$STEPS_D" shoved "${MEASURED[@]}" "${PUSHES[@]}" --command go_turn_range=-0.6,0.6 \
    --reward action_rate=0.1 action_acc=0.05 servo_load=0.1 energy=0.002 reversals=0.02   # D

say "all stages, every robot; export of the pushes, steering and servo-protection results"
python scripts/analysis/free_run_test.py "$C" "$E" "$P" 2>&1 | tee -a "$REPORT"
for run in "$C" "$E" "$P"; do
    python scripts/export_policy.py "$run" 2>&1 | tee -a "$REPORT" || say "($run not exported: it failed the check)"
done
say "done: $REPORT"
