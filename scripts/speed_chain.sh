#!/usr/bin/env bash
# The free run, chained (biped_rl/training/stages.py apply_speed_stage): a walker trained from scratch for speed
# only, from straight legs, then on randomized robots, then on the servo model measured on the robot. Each stage
# resumes the better of the previous stage's final and best model (free_run_test --pick); a gate after stage 1
# stops the chain if it doesn't travel (no night spent on a policy that stands, shuffles or glides).
#   1. speed          nominal physics; fresh network, or FROM: continue that run              STEPS1
#      gate: free_run_test --gate GATE (net of sliding, nominal robot)
#   2. speed_physics  the randomized robot, phased in by the physics curriculum                 STEPS2
#   3. speed_physics + the measured servos (kp 30, 3-4 steps of delay: scripts/finetune.sh's base) STEPS3
#   then free_run_test of every stage on nominal / measured / randomized robots and export for the robot (checked).
# Resumed stages learn at LR (juan_v11_speed at 3e-4 swung from walking to falling between updates after 10M).
# Report: runs/logs/<name>_report.txt. Smoothness, servo wear and commands come after, as fine-tunes.
# Smoke test: STEPS1=40000 STEPS2=40000 STEPS3=40000 GATE=-1 ./scripts/speed_chain.sh test/sc --n-envs 4
#   --eval-every 36000 --eval-episodes 1 --video-every 0   (then delete runs/test/sc_* and runs/logs/test/sc_*)
#
#   STAGES=1 STEPS1=20000000 caffeinate -i ./scripts/speed_chain.sh v11/juan_v11 2>&1 | tee runs/logs/v11/pilot.log
#   FROM=runs/v11/juan_v11_speed_best caffeinate -i ./scripts/speed_chain.sh v11/juan_v11b 2>&1 | tee runs/logs/v11/chain.log
#   STEPS1=0 FROM=runs/v11/juan_v11d_speed_best ./scripts/speed_chain.sh v11/juan_v11e ...   (stages 2-3 only)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/speed_chain.sh <generation>/<name prefix> [train.py options for every stage]}
shift
STEPS1=${STEPS1:-20000000}
STEPS2=${STEPS2:-80000000}
STEPS3=${STEPS3:-80000000}
GATE=${GATE:-0.15}
LR=${LR:-1e-4}
STAGES=${STAGES:-3}
REPORT=runs/logs/${NAME}_report.txt
mkdir -p "$(dirname "$REPORT")"

best_copy() {  # runs/<x>_best: the best evaluation as a run of its own
    mkdir -p "runs/${1}_best"
    cp "runs/$1/best_model.zip" "runs/${1}_best/final_model.zip"
    cp "runs/$1/best_vecnormalize.pkl" "runs/${1}_best/vecnormalize.pkl"
    cp "runs/$1/config.json" "runs/$1/curriculum.json" "runs/${1}_best/"
}
pick() {  # the run to continue from: test final and best of stage $1 on robot $2, append to the report
    python scripts/analysis/free_run_test.py "runs/$1" "runs/${1}_best" --pick "$2" --json "runs/logs/${1}_test.json" \
        | tee -a "$REPORT" | tail -1 | sed 's/^pick ([a-z]*): //'
}

S1=${NAME}_speed
if [[ $STEPS1 == 0 ]]; then  # skip stage 1: FROM is already a speed-stage result that passed the gate
    P1=${FROM:?STEPS1=0 needs FROM}
    : > "$REPORT"
    echo "=== $(date +%H:%M) 1/3 speed: skipped, continuing from $P1"
elif [[ -n ${FROM:-} ]]; then
    echo "=== $(date +%H:%M) 1/3 speed: continue $FROM -> runs/$S1 ($STEPS1 more steps, lr $LR)"
    python scripts/train.py --run-name "$S1" --stage speed --resume "$FROM" --min-std 0.01 --lr "$LR" --timesteps "$STEPS1" "$@"
else
    echo "=== $(date +%H:%M) 1/3 speed: from scratch -> runs/$S1 ($STEPS1 steps)"
    python scripts/train.py --run-name "$S1" --stage speed --init-std-rad 0.05 --timesteps "$STEPS1" "$@"
fi
if [[ $STEPS1 != 0 ]]; then
    best_copy "$S1"
    : > "$REPORT"
    P1=$(pick "$S1" nominal)
    if [[ $STAGES == 1 ]]; then
        echo "=== $(date +%H:%M) done (stage 1 only): $REPORT"; exit 0
    fi
fi
if ! python scripts/analysis/free_run_test.py "$P1" --gate "$GATE" | tail -1 | tee -a "$REPORT" | grep -q PASS; then
    echo "=== $(date +%H:%M) STOP: $P1 doesn't reach $GATE m/s net of sliding with <= 30% falls" | tee -a "$REPORT"
    exit 1
fi

S2=${NAME}_physics
echo "=== $(date +%H:%M) 2/3 speed_physics: $P1 -> runs/$S2 ($STEPS2 steps)"
python scripts/train.py --run-name "$S2" --stage speed_physics --resume "$P1" --min-std 0.01 --lr "$LR" --timesteps "$STEPS2" "$@"
best_copy "$S2"
P2=$(pick "$S2" randomized)
if [[ $STAGES == 2 ]]; then
    echo "=== $(date +%H:%M) done (stages 1-2): $REPORT"; exit 0
fi

S3=${NAME}_servos
echo "=== $(date +%H:%M) 3/3 measured servos: $P2 -> runs/$S3 ($STEPS3 steps)"
python scripts/train.py --run-name "$S3" --stage speed_physics --resume "$P2" --min-std 0.01 --lr "$LR" --timesteps "$STEPS3" \
    --robot kp=30 --randomization action_delay_steps=3,4 "$@"
best_copy "$S3"
P3=$(pick "$S3" randomized)

echo "=== $(date +%H:%M) every stage on every robot, export -> $REPORT"
{
    echo
    python scripts/analysis/free_run_test.py "$P1" "$P2" "$P3"
    for run in "runs/$S3" "runs/${S3}_best"; do
        python scripts/export_policy.py "$run" || echo "   ($run not exported: it failed the check)"
    done
} 2>&1 | tee -a "$REPORT"
echo "=== $(date +%H:%M) done: $REPORT (continue from $P3)"
