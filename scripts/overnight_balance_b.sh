#!/usr/bin/env bash
# scripts/overnight_balance.sh, picked up after stopping its from-scratch run early (a copy: that script was running):
# step 1 continues juan_getup_39_scratch to SCRATCH_TOTAL steps (100M, as juan_getup_36/38 had ~94M before their last
# stage), with its own learning rate and noise, as juan_getup_39_scratch100; the report is appended to. The rest as there.
#
# With BASE=<run> (chosen by hand, no automatic gate): step 1 is skipped, the arms start from it as juan_getup_40_*, a fifth
# arm has all four changes at once, and step 3's automatic combination is off.
#
# One night for a getup that stands, balanced, on its own (2026-11-08 plan). Every run: the fitted servo model
# (4.3 rad/s, braking, 3.2 Nm), 5 s episodes (one quick try), standing only within 15 deg of upright (the robot's
# "up", robot/walker.py; juan_getup_38 leaned 22-33 deg), juan_getup_35's other rewards, randomized robots.
#   1. two candidate bases: from scratch (STEPS_SCRATCH), and juan_getup_38_servos retargeted (STEPS)
#      gate: scripts/analysis/getup_balance.py; the better is the base
#   2. from the base, one change each (STEPS):  weak servos (speed and torque to 0.7x)   friction 0.2-1.4
#      centre of mass +-3 cm, torso to 1.35x   servo protection (juan_getup_37's terms)
#      an arm "improves" if it ranks above the base
#   3. if two or more arms improved: the best one continued with the others' changes added (STEPS)
#   4. every run tested (getup_balance), exported (robot/policies/getup/), ranked: the report's end lists what to try
#
#   caffeinate -i ./scripts/overnight_balance_b.sh 2>&1 | tee runs/logs/getup/overnight_balance_b.log
# Smoke test: STEPS=20000 SEEDS=1 TEST=1 SCRATCH=<a finished getup run> SCRATCH_TOTAL=<its total_steps + 20000> ./scripts/overnight_balance_b.sh
#   (names under test/; then delete runs/test/ob_*, robot/policies/getup/ob_* and runs/logs/test/ob*)
# Report: runs/logs/getup/overnight_balance.txt (smoke test: runs/logs/test/ob.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-30000000}
HAND=${BASE:+1}   # the base was chosen by hand
SCRATCH=${SCRATCH:-runs/getup/juan_getup_39_scratch}
SCRATCH_TOTAL=${SCRATCH_TOTAL:-100000000}
SEEDS=${SEEDS:-20}
B0=${B0:-runs/getup/juan_getup_38_servos}
PHYSICS=runs/v7/juan_v7_13_straight
if [[ -n ${TEST:-} ]]; then
    PRE=test/ob; REPORT=runs/logs/test/ob.txt; EXTRA=(--n-envs 4 --video-every 0); STRESS=1
else
    PRE=getup/juan_getup_${HAND:+40}; [[ -z $HAND ]] && PRE=getup/juan_getup_39; REPORT=runs/logs/getup/overnight_balance.txt; EXTRA=(); STRESS=10
fi
mkdir -p "$(dirname "$REPORT")"
say() { echo "=== $(date +%H:%M) $*" | tee -a "$REPORT"; }

WEIGHTS=(upright_on_feet=1 feet_only=1.0 height_right_side_up=1 standing=1 settled=8 home_pose=8 idle_home_deg=15 standing_tilt_deg=15)
START=(--start side_deg=45 joint_rad=0.25 any_heading=1 knee_min_deg=5)
RES=(--lr 1e-4 --min-std 0.15)   # how a resumed run starts
COMMON=(--seconds 5 --physics-from "$PHYSICS" "${START[@]}")
# train <name> <steps> <from or ""> <extra weights...> [-- <train_getup.py options...>]
train() {
    local name=$1 steps=$2 from=$3 w=() opts=()
    shift 3
    while [[ $# -gt 0 && $1 != -- ]]; do w+=("$1"); shift; done
    [[ ${1:-} == -- ]] && shift && opts=("$@")
    local res=()
    [[ -n $from ]] && res=(--resume "$from" "${RES[@]}")
    say "train runs/$name ($steps steps) from ${from:-scratch} ${w[*]-} ${opts[*]-}"
    python scripts/train_getup.py --run-name "$name" --timesteps "$steps" ${res[@]+"${res[@]}"} "${COMMON[@]}" \
        --weights "${WEIGHTS[@]}" ${w[@]+"${w[@]}"} ${opts[@]+"${opts[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
        > "runs/logs/${name}.log" 2>&1 || { say "training failed: runs/logs/${name}.log"; return 1; }
    if python scripts/export_getup.py "runs/$name" > "runs/logs/${name}_export.log" 2>&1; then
        tail -1 "runs/logs/${name}_export.log" | tee -a "$REPORT"
    else
        say "not exported (did not get up, or failed the check): runs/logs/${name}_export.log"
    fi
}
# balance <json> <runs...>: test, append to the report, print the best run (by getup_balance's ranking)
balance() {
    local json=$1
    shift
    python scripts/analysis/getup_balance.py "$@" --seeds "$SEEDS" --stress-seeds "$STRESS" --json "$json" 2>&1 \
        | grep -v -i warn | tee -a "$REPORT" >&2
    python -c "import json,sys; r=json.load(open(sys.argv[1])); print(max(r, key=lambda k: r[k]['score']))" "$json"
}
better() {   # better <json> <a> <b>: is a ranked above b?
    python -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r[sys.argv[2]]['score'] > r[sys.argv[3]]['score'] else 1)" "$@"
}
mkdir -p "runs/logs/$(dirname "$PRE")"

if [[ -n ${BASE:-} ]]; then
    say "base (chosen by hand): $BASE"
else
# 1. two candidate bases
[[ -f $SCRATCH/final_model.zip ]] || { say "$SCRATCH has no saved policy: stop its training first"; exit 1; }
DONE_STEPS=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['total_steps'])" "$SCRATCH/run.json")
say "continuing $SCRATCH from $DONE_STEPS to $SCRATCH_TOTAL steps"
RES=(--min-std 0.01)   # its own learning rate (3e-4) and noise (0.06): the same training, longer
train "${PRE}_scratch100" "$((SCRATCH_TOTAL - DONE_STEPS))" "$SCRATCH" || true
RES=(--lr 1e-4 --min-std 0.15)
train "${PRE}_retarget" "$STEPS" "$B0" || true
CAND=()
for r in "runs/${PRE}_scratch100" "runs/${PRE}_retarget"; do [[ -f $r/final_model.zip ]] && CAND+=("$r"); done
[[ ${#CAND[@]} -gt 0 ]] || { say "no base trained: stopping"; exit 1; }
BASE=$(balance "runs/logs/${PRE}_bases.json" "$B0" "${CAND[@]}")
[[ $BASE == "$B0" ]] && BASE=${CAND[0]}   # juan_getup_38 itself leans: continue a retargeted run, never it
say "base: $BASE"
fi

# 2. one change each, from the base
arm() {   # an arm's change: extra weights, or "-- --randomization ..." (bash 3.2: no associative arrays)
    case $1 in
        weak) echo "-- --randomization servo_strength=0.7,1.13" ;;
        friction) echo "-- --randomization friction=0.2,1.4" ;;
        com) echo "-- --randomization com_offset=0.03 torso_mass_scale=1.0,1.35" ;;
        servos) echo "servo_load=4 servo_speed=4 impact=2" ;;
    esac
}
ARMS=(weak friction com servos)
for a in "${ARMS[@]}"; do
    # shellcheck disable=SC2046
    train "${PRE}_$a" "$STEPS" "$BASE" $(arm "$a") || true
done
if [[ -z $HAND ]]; then
DONE=()
for a in "${ARMS[@]}"; do [[ -f runs/${PRE}_$a/final_model.zip ]] && DONE+=("runs/${PRE}_$a"); done
BEST=$(balance "runs/logs/${PRE}_arms.json" "$BASE" ${DONE[@]+"${DONE[@]}"})
UP=()
for r in ${DONE[@]+"${DONE[@]}"}; do better "runs/logs/${PRE}_arms.json" "$r" "$BASE" && UP+=("$r"); done
say "improved on the base: ${UP[*]:-none}"

# 3. the improving arms combined, from the best of them
if [[ ${#UP[@]} -ge 2 ]]; then
    W=() O=()   # every improving arm's change, the best's included: a resumed run takes its settings from here
    for r in "${UP[@]}"; do
        a=${r##*_}
        if [[ $a == servos ]]; then W+=(servo_load=4 servo_speed=4 impact=2); else x=$(arm "$a"); read -ra x <<< "${x#-- --randomization }"; O+=("${x[@]}"); fi
    done
    train "${PRE}_combined" "$STEPS" "$BEST" ${W[@]+"${W[@]}"} -- ${O[@]+--randomization "${O[@]}"} || true
else
    say "3. skipped: fewer than two arms improved"
fi
else
    train "${PRE}_all" "$STEPS" "$BASE" servo_load=4 servo_speed=4 impact=2 -- --randomization servo_strength=0.7,1.13 \
        friction=0.2,1.4 com_offset=0.03 torso_mass_scale=1.0,1.35 || true
fi

# 4. everything, tested together and ranked
ALL=("$B0")
[[ -n $HAND ]] && ALL+=("$BASE")
for r in scratch100 retarget weak friction com servos all combined; do [[ -f runs/${PRE}_$r/final_model.zip ]] && ALL+=("runs/${PRE}_$r"); done
say "final test of ${#ALL[@]} getups (juan_getup_38_servos = where it started)"
balance "runs/logs/${PRE}_final.json" "${ALL[@]}" > /dev/null
say "done: $REPORT; to try on the robot, in order: the PASS runs at the top of the ranking above (exported to robot/policies/getup/)"
