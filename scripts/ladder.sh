#!/usr/bin/env bash
# One setting raised step by step on the free run (scripts/speed_chain.sh stage 1): each rung continues the previous
# rung's pick with the next value, and the ladder stops when a rung doesn't pass free_run_test --gate GATE on its own
# robot (falls <= 30%). Sudden jumps stalled or collapsed every time (servo delay 1 -> 1-4: juan_v11f/g); rungs of
# one step passed (juan_v11i). Exploration must stay up: --min-std-rad 0.03 (at 0.0156 it collapsed, juan_v11h).
#
#   SETTING="robot kp" VALUES="20 25 30" FROM=runs/v11/juan_v11i_d1_4_speed caffeinate -i ./scripts/ladder.sh v11/juan_v11j \
#       --min-std-rad 0.03 --randomization nominal_delay_range=1,4 --reward hip_roll=100 hip_roll_free_deg=12
#   (scripts/delay_ladder.sh is the same for RandomizationConfig.nominal_delay_range)
# Reports: runs/logs/<name>_<value>_report.txt, one per rung.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: SETTING="<group> <name>" VALUES="..." scripts/ladder.sh <generation>/<name prefix> [train.py options]}
shift
read -r GROUP KEY <<< "${SETTING:?SETTING=\"<reward|command|robot|randomization> <name>\"}"
VALUES=${VALUES:?VALUES=\"v1 v2 ...\"}
FROM=${FROM:?FROM=<a free-run run to start from>}
STEPS=${STEPS:-8000000}
GATE=${GATE:-0.1}

for v in $VALUES; do
    rung=${NAME}_${KEY}${v//,/_}
    echo "=== $(date +%H:%M) $GROUP $KEY=$v: $FROM -> runs/${rung}_speed ($STEPS steps)"
    STAGES=1 STEPS1=$STEPS FROM=$FROM ./scripts/speed_chain.sh "$rung" "$@" --"$GROUP" "$KEY=$v"
    FROM=$(grep '^pick' "runs/logs/${rung}_report.txt" | tail -1 | sed 's/^pick ([a-z]*): //')
    if ! python scripts/analysis/free_run_test.py "$FROM" --gate "$GATE" | tail -1 | grep -q PASS; then
        echo "=== $(date +%H:%M) STOP at $KEY=$v: $FROM falls in more than 30% of runs or is slower than $GATE m/s"
        exit 1
    fi
    echo "=== $(date +%H:%M) $KEY=$v passed: $FROM"
done
echo "=== $(date +%H:%M) done: every rung passed, last $FROM"
