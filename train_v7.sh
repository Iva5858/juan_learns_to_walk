#!/bin/bash
# juan_v7: training stages, each resuming the previous one (see apply_stage in train.py).
#
# Default: warm start. Learning to walk from zero on Juan's model got stuck standing still, so
# juan_v3's walking is copied into the v7 network (tools/warm_start.py -> runs/juan_v7_0_warm) and
# stages 2-4 run on top: commands, physics, pushes. ~65M steps, ~4-5 h on this Mac.
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
extra=()
if [[ ${SMOKE:-0} == 1 ]]; then
    NAME=smoke_v7
    steps=(1000000 1000000 1000000 1000000)
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
    elif [[ -d runs/$run ]]; then
        prev=$run  # an earlier stage that already exists: resume from it
    fi
done
echo "=== $(date '+%H:%M') done: final policy in runs/$prev ===" | tee -a "runs/$NAME.log"
