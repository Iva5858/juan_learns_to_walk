#!/usr/bin/env bash
# One night for the walker and the getup (2026-11-07 plan), each stage one change, each judged by tests:
#   G. getup servo care   juan_getup_36_rand + getup_23's servo terms (servo_load=4 servo_speed=4 impact=2),
#                         randomized robots                                                        STEPS_G
#      gate: getup_handover (randomized) gets up as often (-2 of 20 at most), servo load -15 %
#   1. moves              juan_v7_31_takeover36_best + v7_26's moves recipe (turning on the spot came back
#                         with it; sideways and backward did not) + an exploration floor (--min-std-rad 0.03:
#                         the walker's noise had decayed to 0.03, too little to find a new gait)       STEPS_1
#      gate: walk_test --moves: backward 0.15, left, right, turn L and R each >= 40 % of the command,
#            and walk_test not NOT BETTER (worse) against where it started
#   2. servo care         the walker's wear terms doubled or more: servo_load 0.5 -> 1, reversals 0.1 -> 0.3,
#                         action_acc 0.2 -> 0.4 (the moves commands kept)                              STEPS_2
#      gate: calm servo load and reversals -15 %, not NOT BETTER (worse), moves no more than 10 points worse
#   3. consolidation      the same rewards at half the learning rate, no exploration floor               STEPS_3
#      gate: not NOT BETTER (worse)
# A walker stage continues the last walker that passed its gate (final or _best, the better on its gate);
# a failed stage is reported and skipped. At the end: every walker that passed against the start
# (walk_test --verdict, --moves, wear), the getup result, and a summary at the top of the report.
# Everything is exported to robot/policies/ by finetune.sh; roles are left as they are (scripts/policies.py).
#
#   caffeinate -i ./scripts/overnight_walk.sh 2>&1 | tee runs/logs/v7/overnight_walk.log
# Smoke test: STEPS_G=40000 STEPS_1=40000 STEPS_2=40000 STEPS_3=40000 SEEDS=2 TEST=1 ./scripts/overnight_walk.sh
#   (names under test/; then delete runs/test/ow_*, runs/logs/test/ow_* and robot/policies/ow_*)
# Report: runs/logs/v7/overnight_walk.txt (smoke test: runs/logs/test/ow.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS_G=${STEPS_G:-30000000}
STEPS_1=${STEPS_1:-60000000}
STEPS_2=${STEPS_2:-40000000}
STEPS_3=${STEPS_3:-40000000}
SEEDS=${SEEDS:-20}
W0=${W0:-runs/v7/juan_v7_31_takeover36_best}
G0=${G0:-runs/getup/juan_getup_36_rand}
PHYSICS=runs/v7/juan_v7_13_straight
if [[ -n ${TEST:-} ]]; then
    GEN=test; NAMES=(test/ow_getup test/ow_moves test/ow_servos test/ow_final); REPORT=runs/logs/test/ow.txt
    SMOKE=(--n-envs 4 --eval-every 36000 --eval-episodes 1 --video-every 0); GSMOKE=(--n-envs 4 --video-every 0)
else
    GEN=v7; NAMES=(getup/juan_getup_37_servos v7/juan_v7_32_moves v7/juan_v7_33_servos v7/juan_v7_34_final)
    REPORT=runs/logs/v7/overnight_walk.txt; SMOKE=(); GSMOKE=()
fi
mkdir -p "$(dirname "$REPORT")" "runs/logs/$GEN"
: > "$REPORT"
say() { echo "=== $(date +%H:%M) $*" | tee -a "$REPORT"; }
gate() { python scripts/overnight_gate.py "$@" 2>&1 | tee -a "$REPORT" | tail -1; }   # prints PASS <run> | FAIL

# v7_31's recipe (finetune.sh's base + these), the moves commands of v7_26
RECIPE=(--reward tracking_vel=5.0 tracking_broad_share=0.3 feet_distance=10.0 min_feet_distance=0.09 step_length=2.0
        step_length_target=0.10 recover_tilt_deg=20 --command speed_range=0.0,0.6 fast_prob=0.3
        --randomization start_pool=out/getup_pool_36.npz start_pool_prob=0.25)
MOVES=(--reward stand_still_moves=1.0 tracking_lat_yaw=2.0 --command turn_in_place_prob=0.25 sideways_prob=0.15 backward_prob=0.15)
SERVOS=(--reward servo_load=1.0 reversals=0.3 action_acc=0.4)
GETUP_W=(--weights upright_on_feet=1 feet_only=1.0 height_right_side_up=1 standing=1 settled=8 home_pose=8 idle_home_deg=15)

# G. getup servo care
say "G. getup servo care: $G0 -> runs/${NAMES[0]} ($STEPS_G steps)"
handover() { python scripts/analysis/getup_handover.py "$1" "$W0" --seeds "$SEEDS" --physics-from "$PHYSICS" --side-deg 45 \
                 --joint-rad 0.25 --json "$2" 2>&1 | tail -2 | tee -a "$REPORT"; }
handover "$G0" "runs/logs/$GEN/ow_getup_before.json"
if python scripts/train_getup.py --run-name "${NAMES[0]}" --resume "$G0" --timesteps "$STEPS_G" --seconds 8 --lr 1e-4 \
        --min-std 0.15 --physics-from "$PHYSICS" --start side_deg=45 joint_rad=0.25 any_heading=1 knee_min_deg=5 \
        "${GETUP_W[@]}" servo_load=4 servo_speed=4 impact=2 ${GSMOKE[@]+"${GSMOKE[@]}"}; then
    handover "runs/${NAMES[0]}" "runs/logs/$GEN/ow_getup_after.json"
    gate getup "runs/logs/$GEN/ow_getup_before.json" "runs/logs/$GEN/ow_getup_after.json" "runs/${NAMES[0]}" >/dev/null || true
    python scripts/export_getup.py "runs/${NAMES[0]}" 2>&1 | tail -2 | tee -a "$REPORT" || say "(not exported: it failed the check)"
else
    say "G. training failed: going on with the walker"
fi

# walker stages: stage <index> <steps> <gate kind> <train.py options...>
P=$W0
stage() {
    local i=$1 steps=$2 kind=$3 name=${NAMES[$1]}
    shift 3
    say "$i. $name: from $P ($steps steps) $*"
    if ! QUICK=1 FROM=$P STEPS=$steps ./scripts/finetune.sh "$name" "${RECIPE[@]}" "$@" ${SMOKE[@]+"${SMOKE[@]}"} \
            > "runs/logs/${name}_finetune.log" 2>&1; then
        say "$i. $name: finetune.sh failed (runs/logs/${name}_finetune.log); going on from $P"; return
    fi
    grep -E "^=== .* vs |summary|wear:" "runs/logs/${name}_report.txt" | tee -a "$REPORT" >/dev/null
    local verdict
    verdict=$(gate "$kind" "runs/logs/${name}_report.json" "runs/logs/${name}_report_moves.json" "$P" "runs/$name" "runs/${name}_best")
    if [[ $verdict == PASS* ]]; then P=${verdict#PASS }; say "$i. passed: continuing from $P"; else say "$i. failed its gate: going on from $P"; fi
}
stage 1 "$STEPS_1" moves "${MOVES[@]}" --min-std-rad 0.03
stage 2 "$STEPS_2" servos "${MOVES[@]}" "${SERVOS[@]}"
stage 3 "$STEPS_3" final "${MOVES[@]}" "${SERVOS[@]}" --lr 5e-5

say "summary: the night's walker is $P (started from $W0); getup: see G above"
if [[ $P != "$W0" ]]; then
    { python scripts/analysis/walk_test.py "$W0" "$P" --seeds "$SEEDS" --verdict | grep -E "^=== |summary|wear:"
      python scripts/analysis/walk_test.py "$W0" "$P" --seeds "$SEEDS" --moves | grep -E "summary|back|left|right|turn [LR]"; } 2>&1 | tee -a "$REPORT"
fi
say "done: $REPORT"
