#!/usr/bin/env bash
# One night for speed (2026-11-10 plan): every run on the servo model fitted on Nov 8 (4.3 rad/s with braking past it,
# 3.2 Nm stall; until now no policy trained on it), one change per run, in order of what matters most:
#   S1  speed policy: juan_v11n_pushesb_best (0.25 m/s on the fitted servos, 0.33 on the old model) continued on the
#       fitted servos, its own recipe otherwise                                                        STEPS_S (~4 h)
#       check: free_run_test, measured servos >= 0.30 m/s with 0/6 falls; randomized <= 2/6 falls
#   W1  default walker: juan_v7_31_takeover36_best, its own recipe (finetune.sh) on the fitted servos   STEPS_W (~2.5 h)
#   W2  from W1 if KEEP (else v7_31): longer strides, step_length_target 0.10 -> 0.13                   STEPS_W (~2.5 h)
#       check: walk_test speed at 0.4+ >= 0.27 m/s, falls no more than where it started
# Each step goes on whatever the one before did; exports land in robot/policies/; the report collects every test.
#
#   caffeinate -i ./scripts/overnight_speed.sh 2>&1 | tee runs/logs/overnight_speed.log
# Smoke test: TEST=1 ./scripts/overnight_speed.sh   (then delete runs/test/os_*, robot/policies/os_*, runs/logs/test/os_*)
# Report: runs/logs/overnight_speed.txt (smoke test: runs/logs/test/os.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS_S=${STEPS_S:-120000000}
STEPS_W=${STEPS_W:-60000000}
FITTED=(--robot no_load_speed=4.3 stall_torque=3.2 overspeed_brake=1)
if [[ -n ${TEST:-} ]]; then
    STEPS_S=40000 STEPS_W=40000
    S1=test/os_s1 W1=test/os_w1 W2=test/os_w2 REPORT=runs/logs/test/os.txt
    EXTRA=(--n-envs 4 --eval-every 36000 --eval-episodes 1 --video-every 0)
else
    S1=v11/juan_v11p_fitted W1=v7/juan_v7_35_fitted W2=v7/juan_v7_36_stride REPORT=runs/logs/overnight_speed.txt
    EXTRA=()
fi
mkdir -p "$(dirname "$REPORT")" runs/logs/v11 runs/logs/v7
: > "$REPORT"
say() { echo "=== $(date +%H:%M) $*" | tee -a "$REPORT"; }

# S1: juan_v11n_pushesb's command line (a resumed run takes its settings from here) + the fitted servos
say "S1 runs/$S1: speed_physics from runs/v11/juan_v11n_pushesb_best ($STEPS_S steps) on the fitted servos"
if python scripts/train.py --run-name "$S1" --stage speed_physics --resume runs/v11/juan_v11n_pushesb_best \
        --timesteps "$STEPS_S" --min-std 0.01 --min-std-rad 0.03 --lr 1e-4 --reward hip_roll=100 hip_roll_free_deg=12 \
        --robot kp=30 --randomization action_delay_steps=3,4 --randomization physics_curriculum=0 push_curriculum=1 \
        push_velocity=0.4 "${FITTED[@]}" ${EXTRA[@]+"${EXTRA[@]}"} > "runs/logs/${S1}.log" 2>&1; then
    mkdir -p "runs/${S1}_best"
    cp "runs/$S1/best_model.zip" "runs/${S1}_best/final_model.zip"
    cp "runs/$S1/best_vecnormalize.pkl" "runs/${S1}_best/vecnormalize.pkl"
    cp "runs/$S1/config.json" "runs/$S1/curriculum.json" "runs/${S1}_best/"
    python scripts/analysis/free_run_test.py runs/v11/juan_v11n_pushesb_best "runs/$S1" "runs/${S1}_best" --pick measured \
        --json "runs/logs/${S1}_test.json" 2>&1 | grep -v -i warn | tee -a "$REPORT"
    for run in "runs/$S1" "runs/${S1}_best"; do
        python scripts/export_policy.py "$run" > "runs/logs/${S1}_export.log" 2>&1 && say "exported $run" || say "$run not exported (failed the check)"
    done
else
    say "S1 training failed: runs/logs/${S1}.log"
fi

# W1, W2: v7_31's command line (finetune.sh's base + these) + the fitted servos
V7_31=(--reward tracking_vel=5.0 tracking_broad_share=0.3 feet_distance=10.0 min_feet_distance=0.09 step_length=2.0
       step_length_target=0.10 recover_tilt_deg=20 --command speed_range=0.0,0.6 fast_prob=0.3
       --randomization start_pool=out/getup_pool_36.npz start_pool_prob=0.25)
say "W1 runs/$W1 from runs/v7/juan_v7_31_takeover36_best ($STEPS_W steps) on the fitted servos"
FROM=runs/v7/juan_v7_31_takeover36_best STEPS=$STEPS_W QUICK=1 ./scripts/finetune.sh "$W1" "${V7_31[@]}" "${FITTED[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"} > "runs/logs/${W1}.log" 2>&1 || say "W1 failed: runs/logs/${W1}.log"
grep "^=== " "runs/logs/${W1}_report.txt" 2>/dev/null | tee -a "$REPORT" || true
W2_FROM=runs/v7/juan_v7_31_takeover36_best
for r in "${W1}_best" "$W1"; do
    if grep -q "^=== $(basename "$r") vs .*: KEEP" "runs/logs/${W1}_report.txt" 2>/dev/null; then W2_FROM=runs/$r; break; fi
done

say "W2 runs/$W2 from $W2_FROM ($STEPS_W steps): step_length_target 0.13"
FROM=$W2_FROM STEPS=$STEPS_W QUICK=1 ./scripts/finetune.sh "$W2" "${V7_31[@]}" "${FITTED[@]}" --reward step_length_target=0.13 \
    ${EXTRA[@]+"${EXTRA[@]}"} > "runs/logs/${W2}.log" 2>&1 || say "W2 failed: runs/logs/${W2}.log"
grep "^=== " "runs/logs/${W2}_report.txt" 2>/dev/null | tee -a "$REPORT" || true
say "done: $REPORT (full walk tests: runs/logs/${W1}_report.txt, runs/logs/${W2}_report.txt)"
