#!/usr/bin/env bash
# A fourth overnight stage, to find Juan's top speed: waits until scripts/overnight.sh has finished, then
# fine-tunes its best policy while the top speed command ramps from 0.4 to 1.2 m/s over the run
# (--speed-ramp), and every 2M steps measures the speed actually reached at 0.2-1.2 m/s commands
# (--speed-probe-every, runs/juan_v7_19_topspeed/speed_probe.csv). Where the reached speed stops rising
# while the ramp keeps going is the limit. Report: runs/topspeed_report.txt.
#
#   ./scripts/overnight_topspeed.sh 2>&1 | tee runs/topspeed.log       # start any time: it waits
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-40000000}
NAME=juan_v7_19_topspeed
REPORT=runs/topspeed_report.txt

while pgrep -f "scripts/overnight.sh" >/dev/null; do
    sleep 60
done
FROM=${FROM:-$(sed -n 's/.*Best policy: \([^ ]*\).*/\1/p' runs/overnight_report.txt | tail -1)}
[[ -d "$FROM" ]] || { echo "no best policy found in runs/overnight_report.txt (set FROM=...)" >&2; exit 1; }

echo "=== $(date +%H:%M) top speed: $NAME from $FROM ($STEPS steps)" | tee "$REPORT"
FROM="$FROM" STEPS="$STEPS" ./scripts/finetune.sh "$NAME" \
    --randomization push_velocity=0 action_delay_steps=2,4 \
    --command swing_ratio=0.48 swing_height=0.035 backward_prob=0.25 backward_speed=0.25 \
              fast_prob=0.4 fast_speed=1.0 speed_limit=1.2 \
    --reward tracking_vel=5.0 step_length=4.0 step_length_target=0.18 feet_air_time=2.0 swing_lift=2.0 \
             swing_reach=1.0 double_support=1.0 lin_vel_z=0.2 footstep=0.5 gait_mismatch=0.3 foot_slip=4.0 \
             recover_tilt_deg=20 \
    --speed-ramp 0.4,1.2 --speed-probe-every 2000000
cat "runs/${NAME}_report.txt" >> "$REPORT"
echo "Speed reached during training (top command ramping 0.4 -> 1.2 m/s; randomized robots, no shoves):" | tee -a "$REPORT"
python - "runs/$NAME/speed_probe.csv" <<'EOF' | tee -a "$REPORT"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cmds = [k.split("_")[-1] for k in rows[0] if k.startswith("reached_at_")]
print(f"{'steps':>8s} {'top cmd':>8s} | " + " ".join(f"{'@' + c:>7s}" for c in cmds) + " | best   (m/s reached; * = fell)")
for r in rows:
    cells = " ".join(f"{float(r['reached_at_' + c]):6.2f}{'*' if int(r['falls_at_' + c]) else ' '}" for c in cmds)
    best = max(float(r["reached_at_" + c]) for c in cmds)
    print(f"{int(r['steps']) / 1e6:7.0f}M {float(r['ceiling_mps']):8.2f} | {cells} | {best:.2f}")
EOF
echo "Speed per command, up to 1.2 m/s (measured servos):" | tee -a "$REPORT"
python scripts/analysis/speeds.py "$FROM" "runs/${NAME}_best" --fast | tee -a "$REPORT"
python - "runs/$NAME/curriculum.json" <<'EOF' | tee -a "$REPORT"
import json, sys
c = json.load(open(sys.argv[1]))
print(f"speed curriculum reached a top command of {c['speed_max']:.2f} m/s")
EOF
echo "=== $(date +%H:%M) done" | tee -a "$REPORT"
