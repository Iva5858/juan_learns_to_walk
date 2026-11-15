#!/usr/bin/env bash
# Second night for speed (2026-11-12 plan): the free-run line only (W1/W2 on Nov 11 showed the default walker gains
# nothing from the fitted servos). Same recipe as juan_v11p_fitted (scripts/overnight_speed.sh S1), one change per run:
#   S2  juan_v11p_fitted (final; 0.290 m/s measured, 0/6 falls, randomized 1/6) continued, nothing changed   STEPS (~4.2 h)
#   S3  from the better of S2 / S2_best / v11p_fitted: net_speed 10 -> 15 (more pay per m/s)                  STEPS (~4.2 h)
# "Better": fastest on the measured robot among those with 0 falls there and randomized falls no more than
# v11p_fitted's; none qualifies -> v11p_fitted. Tests use 12 episodes per robot (6 left 0.01 m/s inside the noise).
# Goal for the robot: >= 0.30 m/s measured with 0 falls, randomized falls <= 2/12; margin for sim-to-real wanted.
#
#   caffeinate -i ./scripts/overnight_speed2.sh 2>&1 | tee runs/logs/overnight_speed2.log
# Smoke test: TEST=1 ./scripts/overnight_speed2.sh   (then delete runs/test/os2_*, robot/policies/os2_*, runs/logs/test/os2*)
# Report: runs/logs/overnight_speed2.txt (smoke test: runs/logs/test/os2.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-120000000}
EPISODES=12
START=runs/v11/juan_v11p_fitted
if [[ -n ${TEST:-} ]]; then
    STEPS=40000 EPISODES=1
    S2=test/os2_s2 S3=test/os2_s3 REPORT=runs/logs/test/os2.txt
    EXTRA=(--n-envs 4 --eval-every 36000 --eval-episodes 1 --video-every 0)
else
    S2=v11/juan_v11q_more S3=v11/juan_v11r_pay15 REPORT=runs/logs/overnight_speed2.txt
    EXTRA=()
fi
mkdir -p "$(dirname "$REPORT")" runs/logs/v11
: > "$REPORT"
say() { echo "=== $(date +%H:%M) $*" | tee -a "$REPORT"; }

# v11p_fitted's command line (a resumed run takes its settings from here); $1 = run, $2 = from, rest = the change
train() {
    local run=$1 from=$2; shift 2
    python scripts/train.py --run-name "$run" --stage speed_physics --resume "$from" \
        --timesteps "$STEPS" --min-std 0.01 --min-std-rad 0.03 --lr 1e-4 --reward hip_roll=100 hip_roll_free_deg=12 \
        --robot kp=30 --randomization action_delay_steps=3,4 --randomization physics_curriculum=0 push_curriculum=1 \
        push_velocity=0.4 --robot no_load_speed=4.3 stall_torque=3.2 overspeed_brake=1 "$@" \
        ${EXTRA[@]+"${EXTRA[@]}"} > "runs/logs/${run}.log" 2>&1 || return 1
    mkdir -p "runs/${run}_best"
    cp "runs/$run/best_model.zip" "runs/${run}_best/final_model.zip"
    cp "runs/$run/best_vecnormalize.pkl" "runs/${run}_best/vecnormalize.pkl"
    cp "runs/$run/config.json" "runs/$run/curriculum.json" "runs/${run}_best/"
}
test_export() {   # $1 = run: free_run_test of START, run, run_best; exports both
    local run=$1
    python scripts/analysis/free_run_test.py "$START" "runs/$run" "runs/${run}_best" --episodes "$EPISODES" --pick measured \
        --json "runs/logs/${run}_test.json" 2>&1 | grep -v -i warn | tee -a "$REPORT"
    for r in "runs/$run" "runs/${run}_best"; do
        python scripts/export_policy.py "$r" >> "runs/logs/${run}_export.log" 2>&1 && say "exported $r" || say "$r not exported (failed the check)"
    done
}

say "S2 runs/$S2 from $START ($STEPS steps): same recipe, more steps"
if train "$S2" "$START"; then test_export "$S2"; else say "S2 training failed: runs/logs/${S2}.log"; fi

# S3's start: the fastest on the measured robot with 0 falls there and randomized falls <= START's
S3_FROM=$(python - "$START" "runs/logs/${S2}_test.json" <<'EOF'
import json, sys
start, path = sys.argv[1], sys.argv[2]
try:
    rep = json.load(open(path))
except OSError:
    print(start); sys.exit()
limit = rep[start]["randomized"]["falls"]
ok = [r for r, v in rep.items() if v["measured"]["falls"] == 0 and v["randomized"]["falls"] <= limit]
print(max(ok, key=lambda r: rep[r]["measured"]["speed"]) if ok else start)
EOF
)
say "S3 runs/$S3 from $S3_FROM ($STEPS steps): net_speed 10 -> 15"
if train "$S3" "$S3_FROM" --reward net_speed=15; then test_export "$S3"; else say "S3 training failed: runs/logs/${S3}.log"; fi
say "done: $REPORT"
