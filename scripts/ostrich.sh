#!/usr/bin/env bash
# The ostrich gait (biped_rl/tasks/locomotion/mdp/reference.py): a policy pre-trained to walk the reference
# (scripts/pretrain_reference.py), then reinforcement learning in three stages, each resuming the last:
#
#   walk     every command (forward, backward with the knees forward, sideways, turning), the nominal robot
#            with the measured servos; the critic warms up first so the pre-trained walk isn't disturbed
#   physics  the randomized robot, eased in by the physics curriculum (difficulty 0 = nominal)
#   pushes   the full-difficulty robot and the push curriculum
#
# The reward: following the speed command (strict), imitating the reference (its weight fading 3 -> 2 ->
# 1.5 as the robot takes over), the clock-timed footfall and stepping terms, penalties ramping 0.3 -> 0.6
# -> 1.0. Penalties that would fight this gait are off: bent-back knees, a low body, the old standing pose.
# Imitation within 5 deg, and 30% of the stepping terms paid whatever the speed (gait_floor): with 10 deg and
# stepping paid only at the right speed, standing in the crouch scored most of the imitation (juan_v9).
# Then it marched in place (imitation and the stepping floor paid for the leg motion, speed only ~0.16):
# speed tracking 5, stepping floor 0.1, sliding feet cost more (foot_slip 10). With imitation within 5 deg it
# then drifted off the reference (no pull once >10 deg off) and flipped its knees forward: imitation within
# 8 deg at 3 -> 2 -> 1.5, and an episode ends (as a fall) beyond 25 deg rms from the reference.
#
#   ./scripts/ostrich.sh juan_v9                        # runs/v9/juan_v9_{bc,walk,physics,pushes}
#   STEPS="20000000 20000000 20000000" ./scripts/ostrich.sh juan_v9_short
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/v9 runs/logs/v9
NAME=${1:?usage: scripts/ostrich.sh <name>}
read -r -a steps <<< "${STEPS:-60000000 60000000 50000000}"   # walk, physics, pushes
STAGES=${STAGES:-walk physics pushes}
imitation=(3.0 2.0 1.5)
penalty=(0.3 0.6 1.0)
D=runs/v9

if [[ ! -f "$D/${NAME}_bc/final_model.zip" ]]; then
    echo "=== $(date +%H:%M) ${NAME}_bc (pre-training on the reference)"
    python scripts/pretrain_reference.py "$D/${NAME}_bc"
fi

REWARD=(action_rate=0.5 action_acc=0.2 step_length=4.0 feet_air_time=2.0 swing_lift=2.0 swing_reach=1.0
        double_support=1.0 foot_slip=10.0 lin_vel_z=0.2 footstep=0.5 gait_mismatch=0.3 recover_tilt_deg=20
        break_impact=4.0 safe_torque=1.0 safe_joint_speed=1.3 safe_impact=3.0
        knee_hyperextension=0 hip_roll_inward=0 base_height=0 stand_pose=0 knee_bend=0 short_step=0
        forward_speed=0 termination=-10 tracking_sigma2=0.008 tracking_broad_share=0.15 imitation_sigma_deg=8 imitation_terminate_deg=25 gait_floor=0.1 tracking_vel=5.0
        tracking_lat_yaw=0.5 heading=1.0 path_offset=2.0 stand_still=1.0 gait_contact=1.0 feet_clearance=0.3
        symmetry=3.0)
# No backward commands tonight (juan_v9): flipping the knees forward for a backward command caused 23 of 31
# falls at 46M (mostly hardware-limit stops mid-flip) and held the whole stage back. Forward, sideways and
# turning first; the knees-forward backward walk gets its own fine-tune from a solid forward gait.
COMMAND=(reference_gait=1 swing_ratio=0.4 swing_height=0.035 backward_prob=0 backward_speed=0 fixed_forward=0)

prev="$D/${NAME}_bc"
i=0
for stage in walk physics pushes; do
    run="${NAME}_${stage}"
    preset=$stage; [[ $stage == walk ]] && preset=commands
    if [[ " $STAGES " == *" $stage "* && ! -f "$D/$run/final_model.zip" ]]; then
        echo "=== $(date +%H:%M) $run (${steps[$i]} steps)"
        extra=(); [[ $stage == walk ]] && extra=(--critic-warmup 3000000)
        if [[ $stage != walk ]] && ! python scripts/check_handoff.py "$prev" "$preset" --keep-commands; then
            echo "=== stopped: $prev can't stay up in the $stage stage's setup"
            exit 1
        fi
        python scripts/train.py --run-name "v9/$run" --stage "$preset" --resume "$prev" --timesteps "${steps[$i]}" \
            --robot kp=30 --randomization nominal_delay_steps=3 action_delay_steps=3,4 \
            --command "${COMMAND[@]}" --reward "${REWARD[@]}" "imitation=${imitation[$i]}" "penalty_scale=${penalty[$i]}" \
            --video-every 4000000 ${extra[@]+"${extra[@]}"}
    fi
    [[ -f "$D/$run/final_model.zip" ]] || { echo "=== $run did not finish"; exit 1; }
    prev="$D/$run"
    i=$((i + 1))
done

echo "=== $(date +%H:%M) training done: $prev"
REPORT=runs/logs/v9/${NAME}_report.txt
candidates=("$prev")
if [[ -f "$prev/best_model.zip" ]]; then
    mkdir -p "${prev}_best"
    cp "$prev/best_model.zip" "${prev}_best/final_model.zip"
    cp "$prev/best_vecnormalize.pkl" "${prev}_best/vecnormalize.pkl"
    cp "$prev/config.json" "$prev/curriculum.json" "${prev}_best/"
    candidates+=("${prev}_best")
fi
exported=()
for run in "${candidates[@]}"; do
    if python scripts/export_policy.py "$run"; then exported+=("$run"); else echo "   ($run not exported: it failed the check)"; fi
done
{
    echo "Ostrich gait: $NAME (last stage: $prev)"
    python scripts/analysis/speeds.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
    echo
    python scripts/analysis/falls.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
    echo
    python scripts/analysis/stand_pushes.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
} 2>&1 | tee "$REPORT"
echo "=== $(date +%H:%M) done: $REPORT"
