#!/bin/bash
# juan_v7: training stages, each resuming the previous one (see biped_rl/training/stages.py).
#
# Default: warm start. Learning to walk from zero on Juan's model got stuck standing still, so
# juan_v3's walking is copied into the v7 network (scripts/warm_start.py -> runs/juan_v7_0_warm) and
# stages 2-4 run on top: commands, physics, pushes. ~65M steps (+10M per physics extension), ~3 h.
# The physics stage phases in the realistic robot (physics curriculum) and is extended until it
# reaches full difficulty; stage 4 then trains on the full-difficulty robot with pushes.
# caffeinate keeps the Mac awake; the log is runs/juan_v7.log.
#
#   ./scripts/train_v7.sh                       # warm start from runs/juan_v3, then commands, physics, pushes
#   ./scripts/train_v7.sh physics               # start at a later stage (the earlier ones must exist)
#   FROM=runs/juan_v6 ./scripts/train_v7.sh     # warm start from another forward-only run
#   FROM=scratch ./scripts/train_v7.sh          # all four stages from zero, starting with "walk" (gets stuck
#                                       # standing on Juan's model so far)
#   SMOKE=1 ./scripts/train_v7.sh               # quick check of the whole chain: 1M steps per stage,
#                                       # runs/smoke_v7_*, a video clip every 250k steps
#
# Stop with Ctrl+C at any time: the current stage saves its policy, and the script stops there.
set -euo pipefail
cd "$(dirname "$0")/.."   # the project folder
PY=.venv/bin/python
NAME=juan_v7
FROM=${FROM:-runs/juan_v3}

stages=(walk commands physics pushes)
steps=(20000000 15000000 25000000 25000000)
more_steps=10000000   # each extension of the physics stage, if it hasn't reached full difficulty
extra=()
if [[ ${SMOKE:-0} == 1 ]]; then
    NAME=smoke_v7
    steps=(1000000 1000000 1000000 1000000)
    more_steps=500000
    extra=(--video-every 250000)
fi

prev=""
if [[ $FROM == scratch ]]; then
    start=${1:-walk}
else
    start=${1:-commands}
    prev="${NAME}_0_warm"
    if [[ ! -d runs/$prev ]]; then
        echo "=== $(date '+%H:%M') warm start: $FROM -> runs/$prev ===" | tee -a "runs/$NAME.log"
        $PY scripts/warm_start.py "$FROM" "runs/$prev" 2>&1 | grep -v -i warn | tee -a "runs/$NAME.log"
    fi
fi

# Before pushes, the physics stage must have reached full difficulty: extend it (up to 3 times) until
# it does. Done here, at the start of stage 4, so "./train_v7.sh pushes" also resumes correctly.
physics_gate() {
    local base="${NAME}_3_physics" n level
    for n in 1 2 3 4; do
        level=$($PY -c "import json; print(json.load(open('runs/$prev/curriculum.json'))['difficulty'])")
        echo "=== $(date '+%H:%M') physics difficulty reached: $level (runs/$prev) ===" | tee -a "runs/$NAME.log"
        $PY -c "import sys; sys.exit(0 if $level >= 0.99 else 1)" && return
        if [[ $n == 4 ]]; then
            echo "=== WARNING: physics difficulty still $level after 3 extensions; stage 4 trains at full" \
                 "difficulty anyway ===" | tee -a "runs/$NAME.log"
            return
        fi
        [[ -d runs/${base}_more$n ]] && { prev="${base}_more$n"; continue; }  # already done earlier
        echo "=== $(date '+%H:%M') extending physics -> runs/${base}_more$n ===" | tee -a "runs/$NAME.log"
        caffeinate -i $PY scripts/train.py --run-name "${base}_more$n" --stage physics --timesteps "$more_steps" \
            --resume "runs/$prev" ${extra[@]+"${extra[@]}"} >> "runs/$NAME.log" 2>&1
        prev="${base}_more$n"
    done
}

started=false
for i in "${!stages[@]}"; do
    stage=${stages[$i]}
    run="${NAME}_$((i + 1))_${stage}"
    [[ $stage == "$start" ]] && started=true
    if $started; then
        [[ $stage == pushes ]] && physics_gate
        resume=()
        [[ -n $prev ]] && resume=(--resume "runs/$prev")
        echo "=== $(date '+%H:%M') stage $((i + 1))/4: $stage -> runs/$run ===" | tee -a "runs/$NAME.log"
        caffeinate -i $PY scripts/train.py --run-name "$run" --stage "$stage" --timesteps "${steps[$i]}" \
            ${resume[@]+"${resume[@]}"} ${extra[@]+"${extra[@]}"} >> "runs/$NAME.log" 2>&1
        prev=$run
    elif [[ -d runs/$run ]]; then
        prev=$run  # an earlier stage that already exists: resume from it
    fi
done
echo "=== $(date '+%H:%M') done: final policy in runs/$prev ===" | tee -a "runs/$NAME.log"
