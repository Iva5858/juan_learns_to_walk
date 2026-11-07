#!/usr/bin/env bash
# Getting up on the corrected model (2026-11-07: weighed 1627 g, box legs) at the robot's servo delay, one change
# per rung. juan_getup_20 got up 10/10 only with the servos 2 steps late (the delay buffer capped its configured 3),
# the old masses and capsule legs; the robot measured 3-4 steps. The free run learned 1-4 steps only as a ladder
# (juan_v11i; 1-4 at once never stabilized), so the delay widens a step per rung:
#   1. nominal robot, delay 1-2 (the corrected model is the only change from juan_getup_20)   STEPS each
#   2. delay 1-3
#   3. delay 1-4
#   4. randomized robots (juan_v7_13_straight's physics: delay 3-4, kp +-30 %, masses, friction)   STEPS_R
#   5. the same, starts rolled up to 90 deg onto a side (the robot was tested lying on its side)   STEPS_R
# Each rung continues the previous one with juan_getup_20's rewards; after each, getup_handover (nominal robot at
# 3 steps, and randomized robots at 3-4) appended to the report. Judge by those, and by the videos.
#
#   FROM=runs/getup/juan_getup_20 caffeinate -i ./scripts/getup_ladder.sh getup/juan_getup_30 2>&1 | tee runs/logs/getup/ladder.log
#   RUNGS="4 5" FROM=runs/getup/juan_getup_30_d1_4 ./scripts/getup_ladder.sh getup/juan_getup_30    (later rungs only)
# Smoke test: STEPS=40000 STEPS_R=40000 SEEDS=2 EXTRA="--n-envs 4 --video-every 0" FROM=runs/getup/juan_getup_20
#   ./scripts/getup_ladder.sh test/gl   (then delete runs/test/gl_* and runs/logs/test/gl_*)
# Report: runs/logs/<name>_ladder.txt
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: FROM=<getup run> scripts/getup_ladder.sh <generation>/<name prefix>}
P=${FROM:?FROM=<the getup run to start from>}
STEPS=${STEPS:-20000000}
STEPS_R=${STEPS_R:-30000000}
SEEDS=${SEEDS:-20}
RUNGS=${RUNGS:-"1 2 3 4 5"}
WALKER=${WALKER:-runs/v7/juan_v7_28_takeover_best}
PHYSICS=runs/v7/juan_v7_13_straight
REPORT=runs/logs/${NAME}_ladder.txt
mkdir -p "$(dirname "$REPORT")"
# shellcheck disable=SC2206
EXTRA=(${EXTRA:-})
WEIGHTS=(--weights upright_on_feet=1 feet_only=1.0 height_right_side_up=1 standing=3 settled=4 strict_standing=1 speed=2
         flat_feet=1 sole_flat=2 idle_finish=10 action_rate=1.0 joint_speed=0.1 two_legs=3 home_pose=8 idle_home_deg=15)
START=(side_deg=45 joint_rad=0.25 any_heading=1 knee_min_deg=5)

say() { echo "=== $(date +%H:%M) $*" | tee -a "$REPORT"; }
# rung <run> <steps> <side deg> <train_getup.py options...>: train from $P, test, P = the new run
rung() {
    local run=$1 steps=$2 side=$3
    shift 3
    say "$run: from $P ($steps steps) $*"
    python scripts/train_getup.py --run-name "$run" --resume "$P" --timesteps "$steps" --seconds 8 \
        --start "${START[@]/side_deg=45/side_deg=$side}" "${WEIGHTS[@]}" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
    P=runs/$run
    for robot in nominal randomized; do
        python scripts/analysis/getup_handover.py "$P" "$WALKER" --seeds "$SEEDS" --side-deg "$side" --joint-rad 0.25 \
            --physics-from "$([[ $robot == nominal ]] && echo '' || echo $PHYSICS)" 2>&1 | tail -1 | sed "s/^/   $robot: /" | tee -a "$REPORT"
    done
}

: > "$REPORT"
say "start from $P"
for r in $RUNGS; do
    case $r in
        1) rung "${NAME}_d1_2" "$STEPS" 45 --randomization nominal_delay_range=1,2 ;;
        2) rung "${NAME}_d1_3" "$STEPS" 45 --randomization nominal_delay_range=1,3 ;;
        3) rung "${NAME}_d1_4" "$STEPS" 45 --randomization nominal_delay_range=1,4 ;;
        4) rung "${NAME}_rand" "$STEPS_R" 45 --physics-from "$PHYSICS" ;;
        5) rung "${NAME}_side" "$STEPS_R" 90 --physics-from "$PHYSICS" ;;
    esac
done
say "done: $REPORT (last $P)"
