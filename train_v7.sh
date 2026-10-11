#!/bin/bash
# juan_v7: four training stages, each resuming the previous one (see apply_stage in train.py).
# About 85M steps, ~9 h on this Mac. caffeinate keeps it from sleeping; the log is runs/juan_v7.log.
#
#   ./train_v7.sh                 # all four stages
#   ./train_v7.sh physics         # start at a later stage (the earlier ones must exist)
#   SMOKE=1 ./train_v7.sh         # quick check of the whole chain: 1M steps per stage, ~30 min,
#                                 # runs/smoke_v7_*, a video clip every 250k steps
#
# Stop with Ctrl+C at any time: the current stage saves its policy, and the script stops there.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
NAME=juan_v7

stages=(walk commands physics pushes)
steps=(20000000 15000000 25000000 25000000)
extra=()
if [[ ${SMOKE:-0} == 1 ]]; then
    NAME=smoke_v7
    steps=(1000000 1000000 1000000 1000000)
    extra=(--video-every 250000)
fi
start=${1:-walk}

prev=""
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
    fi
    prev=$run
done
echo "=== $(date '+%H:%M') done: final policy in runs/$prev ===" | tee -a "runs/$NAME.log"
