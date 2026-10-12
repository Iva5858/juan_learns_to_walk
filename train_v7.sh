#!/bin/bash
# juan_v7: training stages, each resuming the previous one (see apply_stage in train.py).
#
# Default: warm start. Learning to walk from zero on Juan's model got stuck standing still, so
# juan_v3's walking is copied into the v7 network (tools/warm_start.py -> runs/juan_v7_0_warm) and
# stages 2-4 run on top: commands, physics, pushes. ~65M steps (+10M per physics extension), ~3 h.
# The physics stage phases in the realistic robot (physics curriculum) and is extended until it
# reaches full difficulty; stage 4 then trains on the full-difficulty robot with pushes.
# caffeinate keeps the Mac awake; the log is runs/juan_v7.log.
#
#   ./train_v7.sh                       # warm start from runs/juan_v3, then commands, physics, pushes
#   ./train_v7.sh physics               # start at a later stage (the earlier ones must exist)
#   FROM=runs/juan_v6 ./train_v7.sh     # warm start from another forward-only run
#   FROM=scratch ./train_v7.sh          # all four stages from zero, starting with "walk" (gets stuck
#                                       # standing on Juan's model so far)
#   SMOKE=1 ./train_v7.sh               # quick check of the whole chain: 1M steps per stage,
#                                       # runs/smoke_v7_*, a video clip every 250k steps
#
# Stop with Ctrl+C at any time: the current stage saves its policy, and the script stops there.
set -euo pipefail
cd "$(dirname "$0")"
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
        $PY tools/warm_start.py "$FROM" "runs/$prev" 2>&1 | grep -v -i warn | tee -a "runs/$NAME.log"
    fi
fi

started=false
for i in "${!stages[@]}"; do
    stage=${stages[$i]}
    run="${NAME}_$((i + 1))_${stage}"
    [[ $stage == "$start" ]] && started=true
    if $started; then
        resume=()
        [[ -n $prev ]] && resume=(--resume "runs/$prev")
        echo "=== $(date '+%H:%M') stage $((i + 1))/4: $stage -> runs/$run ===" | tee -a "runs/$NAME.log"
        caffeinate -i $PY train.py --run-name "$run" --stage "$stage" --timesteps "${steps[$i]}" \
            ${resume[@]+"${resume[@]}"} ${extra[@]+"${extra[@]}"} >> "runs/$NAME.log" 2>&1
        prev=$run
        # The physics stage must reach full difficulty before pushes: extend it until it does.
        if [[ $stage == physics ]]; then
            for more in 1 2 3; do
                level=$($PY -c "import json; print(json.load(open('runs/$prev/curriculum.json'))['_difficulty'])")
                echo "=== $(date '+%H:%M') physics difficulty reached: $level ===" | tee -a "runs/$NAME.log"
                $PY -c "import sys; sys.exit(0 if $level >= 0.99 else 1)" && break
                [[ $more == 3 ]] && { echo "=== WARNING: physics difficulty still $level after 3 extensions;" \
                    "stage 4 trains at full difficulty anyway ===" | tee -a "runs/$NAME.log"; break; }
                ext="${run}_more$more"
                echo "=== $(date '+%H:%M') extending physics -> runs/$ext ===" | tee -a "runs/$NAME.log"
                caffeinate -i $PY train.py --run-name "$ext" --stage physics --timesteps "${more_steps}" \
                    --resume "runs/$prev" ${extra[@]+"${extra[@]}"} >> "runs/$NAME.log" 2>&1
                prev=$ext
            done
        fi
    elif [[ -d runs/$run ]]; then
        prev=$run  # an earlier stage that already exists: resume from it (or its latest extension)
        for ext in runs/${run}_more1 runs/${run}_more2 runs/${run}_more3; do
            [[ -d $ext ]] && prev=${ext#runs/}
        done
    fi
done
echo "=== $(date '+%H:%M') done: final policy in runs/$prev ===" | tee -a "runs/$NAME.log"
