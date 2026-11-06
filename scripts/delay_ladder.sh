#!/usr/bin/env bash
# The servo delay, widened step by step on the free run (scripts/speed_chain.sh stage 1, nominal robot): each rung
# continues the previous rung's pick with RandomizationConfig.nominal_delay_range one step wider, and the ladder stops
# when a rung doesn't pass free_run_test --gate GATE on its own robot (falls <= 30%). juan_v11e learned a 1-step delay
# in ~2.5M steps; 1-4 steps at once (juan_v11f, juan_v11g) never stabilized. The robot measured 3-4 steps.
#
#   FROM=runs/v11/juan_v11e_physics caffeinate -i ./scripts/delay_ladder.sh v11/juan_v11h --reward hip_roll=100 hip_roll_free_deg=12
#   RANGES="1,3 1,4" STEPS=12000000 FROM=... ./scripts/delay_ladder.sh ...      (other rungs, longer)
# Reports: runs/logs/<name>_d<lo>_<hi>_report.txt, one per rung.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/delay_ladder.sh <generation>/<name prefix> [train.py options for every rung]}
shift
FROM=${FROM:?FROM=<a free-run run to start from>}
STEPS=${STEPS:-8000000}
RANGES=${RANGES:-"1,2 1,3 1,4"}
GATE=${GATE:-0.1}

for r in $RANGES; do
    rung=${NAME}_d${r/,/_}
    echo "=== $(date +%H:%M) delay $r steps: $FROM -> runs/${rung}_speed ($STEPS steps)"
    STAGES=1 STEPS1=$STEPS FROM=$FROM ./scripts/speed_chain.sh "$rung" --randomization nominal_delay_range="$r" "$@"
    FROM=$(grep '^pick' "runs/logs/${rung}_report.txt" | tail -1 | sed 's/^pick ([a-z]*): //')
    if ! python scripts/analysis/free_run_test.py "$FROM" --gate "$GATE" | tail -1 | grep -q PASS; then
        echo "=== $(date +%H:%M) STOP at delay $r: $FROM falls in more than 30% of runs or is slower than $GATE m/s"
        exit 1
    fi
    echo "=== $(date +%H:%M) delay $r passed: $FROM"
done
echo "=== $(date +%H:%M) done: every rung passed, last $FROM"
