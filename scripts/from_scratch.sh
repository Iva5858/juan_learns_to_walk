#!/usr/bin/env bash
# Train a new generation from zero on today's joint ranges (scripts/model/sync_joint_limits.py), through all
# five stages (walk -> stride -> commands -> physics -> pushes, docs/training.md), each resuming the last. It doesn't
# start from any earlier policy, so it can find a different way of walking: a wider action range (each joint
# can move ACTION_SCALE rad from home), knees allowed to bend backwards, inward hip roll no longer
# penalized, and the stepping rewards (v7_15d) on. Stiff measured servos throughout (kp 30); delays and the
# rest of the real robot come in with the physics stage's curriculum.
#
# Explore first, then constrain. The walk stage is a free run: perfect physics but the real servos' timing
# (stiffness and their 60-80 ms delay: with instant servos the fastest gait was a 9.5 steps/s shuffle), and
# the reward is how fast it goes along its start heading (forward_speed, whatever command it is shown) plus
# staying up, with every other penalty at 10% (servo protection stays at full weight) and a small fall
# penalty. No rhythm or pose to imitate: whatever way of walking is fastest. Each later stage brings the
# commands, the tracking and gait terms back and the penalties closer to full weight (PENALTIES).
#
#   STAGES="walk stride" ./scripts/from_scratch.sh juan_v8                 # trial: the free run only
#   WALK_STEPS=40000000 STEPS=40000000 ./scripts/from_scratch.sh juan_v8   # all five; a finished stage is skipped
#   ACTOR=512,256,128 ./scripts/from_scratch.sh juan_v8_big                # a bigger policy network
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs
NAME=${1:?usage: scripts/from_scratch.sh <generation>/<name> (runs/v10/juan_v10_walk, _commands, _physics, _pushes)}
STEPS=${STEPS:-30000000}
WALK_STEPS=${WALK_STEPS:-$STEPS}   # the free run can get longer: learning from zero is the slow part
STAGES=${STAGES:-walk stride commands physics pushes}
ACTION_SCALE=${ACTION_SCALE:-limits_straight}
START_POSES=${START_POSES:-home,knees_back,straight}   # crouch with the knees forward, backward, or straight legs
KNEE_BEND=${KNEE_BEND:-1.0}   # pay for knees bent backwards, in full from the crouch's 46 deg on (0 = off; the free run
                              # alone found nearly straight legs, knees 5 deg back)
SHORT_STEP=${SHORT_STEP:-8.0} # stride only: cost of a step shorter than the gait design's fastest swing (0.2 s): speed
                              # alone found 10 steps/s shuffles, cadence was its cheapest speed
ACTOR=${ACTOR:-}       # policy network widths for the first stage, e.g. 512,256,128 (default 256,128,64)
FROM=${FROM:-}         # a run in runs/ the first stage in STAGES resumes from instead of the previous stage,
                       # e.g. FROM=juan_v8j_cmd104 STAGES="commands physics pushes" (a checkpoint that still walked)
# Neither knee direction is the default: action 0 means straight legs ("limits_straight"), and episodes start
# from the home crouch, the same crouch with the knees bent backwards, or straight legs (START_POSES).
# The action range is a reach, not a resolution. "limits_straight" (default) / "limits": the policy's output
# -1..+1 spans each joint's whole range (-1 = lower limit, 0 = straight legs / home, +1 = upper limit), the
# starting exploration noise is ~11 deg on every joint and the smoothness penalties count joint motion in
# radians (ppo.new_model, rewards.action_rate). A number: every joint reaches that many rad from home; the
# noise and smoothness weights are scaled here to stay the same in radians as at the classic 0.5.
if [[ $ACTION_SCALE == limits* ]]; then
    LOG_STD="" SMOOTH=1
else
    read -r LOG_STD SMOOTH <<< "$(python -c "import math; s=$ACTION_SCALE/0.5; print(round(-1.0 - math.log(s), 3), round(s * s, 3))")"
fi
PENALTIES=${PENALTIES:-0.1 0.3 0.6 1.0}    # penalty_scale for walk, commands, physics, pushes
read -r -a scale <<< "$PENALTIES"
REWARD=(action_rate=$(python -c "print(0.5 * $SMOOTH)") action_acc=$(python -c "print(0.2 * $SMOOTH)")
        step_length=4.0 feet_air_time=2.0 swing_lift=2.0 swing_reach=1.0 double_support=1.0 foot_slip=4.0
        lin_vel_z=0.2 footstep=0.5 gait_mismatch=0.3 recover_tilt_deg=20 knee_hyperextension=0 hip_roll_inward=0
        knee_bend=$KNEE_BEND knee_bend_dir=-1
        break_impact=4.0 safe_torque=1.0 safe_joint_speed=1.3 safe_impact=3.0)

prev=""
i=0
for stage in walk stride commands physics pushes; do
    if [[ $stage == walk || $stage == stride ]]; then
        # free run: the reward is speed along the start heading, nothing else to follow or imitate. It is
        # shown the usual changing commands, and ignores them: shown one constant command (juan_v8d), it
        # fell as soon as the next stage showed it any other. Short steps cost only from the second half
        # (stride), once it can move: costing every clumsy step from the start, it learned to dive forward
        # once and fall without ever completing a step (juan_v8i)
        short=0; [[ $stage == stride ]] && short=$SHORT_STEP
        explore=(--reward "penalty_scale=${scale[0]}" termination=-3 forward_speed=2.0 short_step=$short
                          tracking_vel=0 tracking_lat_yaw=0 heading=0 path_offset=0 stand_still=0
                          gait_contact=0 feet_clearance=0 feet_air_time=0 swing_lift=0 step_length=0 swing_reach=0
                          gait_mismatch=0 footstep=0 double_support=0 symmetry=0
                 --command fixed_forward=0)
    else
        # following commands again: the task terms and gait penalties back (REWARD above restores the rest).
        # The gait clock shapes the steps from here (feet_air_time, swing_lift, step_length), not short_step:
        # with it at 8, every footfall of a slow commanded step was fined and it settled for tapping in
        # place at 0.05 m/s whatever the command (juan_v8j). Strict speed tracking (the default kernel, not the
        # stages' forgiving one): with the forgiving one, standing still at every command scored 0.8 of the
        # tracking reward (most commands are slow, stops or turns) and a 0.3 m/s gait no more, so it stood
        # (juan_v8j). Air time and foot lift stay at REWARD's weight: doubled, they did slow the cadence (8 ->
        # 6 steps/s) but drove ever higher, bolder steps until 42% of episodes fell (juan_v8j).
        explore=(--reward "penalty_scale=${scale[$i]}" termination=-10 forward_speed=0 short_step=0
                          tracking_sigma2=0.008 tracking_broad_share=0.15
                          tracking_lat_yaw=0.5 heading=1.0 path_offset=2.0 stand_still=1.0 gait_contact=1.0
                          feet_clearance=0.3 symmetry=3.0
                 --command fixed_forward=0)
    fi
    [[ $stage == walk ]] || i=$((i + 1))   # walk and stride share the first penalty scale
    run="${NAME}_${stage}"
    if [[ " $STAGES " == *" $stage "* ]]; then
        if [[ -f "runs/$run/final_model.zip" ]]; then
            echo "=== $run already trained: skipped"
        else
            steps=$STEPS; [[ $stage == walk ]] && steps=$WALK_STEPS
            echo "=== $(date +%H:%M) $run ($steps steps)"
            if [[ -n "$FROM" ]]; then prev=$FROM; FROM=""; echo "    (from runs/$prev)"; fi
            if [[ -z "$prev" ]]; then
                from=(--env "action_scale=$ACTION_SCALE" "start_poses=$START_POSES")
                if [[ -n "$LOG_STD" ]]; then from+=(--log-std-init "$LOG_STD"); fi
                if [[ -n "$ACTOR" ]]; then from+=(--actor "$ACTOR"); fi
            else
                from=(--resume "runs/$prev")
            fi
            if [[ $stage == commands && -n "$prev" ]] && ! python scripts/check_handoff.py "runs/$prev" commands; then
                echo "=== stopped: runs/$prev can't stay up in the commands stage's setup, which would spend it falling"
                exit 1
            fi
            # the free run (walk, stride) takes the commands stage's settings with its own rewards: it sees every
            # kind of command, changing every few seconds, before it must follow them. Shown only a steady forward
            # speed per episode (the walk stage's), it fell at the first turn, stop or change (juan_v8j)
            preset=$stage; [[ $stage == walk || $stage == stride ]] && preset=commands
            python scripts/train.py --run-name "$run" --stage "$preset" "${from[@]}" --timesteps "$steps" \
                --robot kp=30 --randomization nominal_delay_steps=3 action_delay_steps=3,4 \
                --reward "${REWARD[@]}" "${explore[@]}" --video-every 2000000
        fi
    fi
    if [[ ! -f "runs/$run/final_model.zip" ]]; then
        [[ " $STAGES " == *" $stage "* ]] && break   # it didn't finish: the later stages need it
        continue                                       # not asked for and not there (a FROM start)
    fi
    prev=$run
done
echo "=== $(date +%H:%M) training done: last stage runs/$prev"
[[ -n "$prev" ]] || exit 1

# Morning report: export the last stage (final and best) for the robot and compare it with v7_13 on the
# measured servos. Analysis tools load today's model, so v7_13 gets the new ranges too (it never uses them).
REPORT=runs/logs/${NAME}_report.txt
mkdir -p "$(dirname "$REPORT")"
candidates=("runs/$prev")
if [[ -f "runs/$prev/best_model.zip" ]]; then
    mkdir -p "runs/${prev}_best"
    cp "runs/$prev/best_model.zip" "runs/${prev}_best/final_model.zip"
    cp "runs/$prev/best_vecnormalize.pkl" "runs/${prev}_best/vecnormalize.pkl"
    cp "runs/$prev/config.json" "runs/$prev/curriculum.json" "runs/${prev}_best/"
    candidates+=("runs/${prev}_best")
fi
exported=()
for run in "${candidates[@]}"; do
    if python scripts/export_policy.py "$run"; then exported+=("$run"); else echo "   ($run not exported: it failed the check)"; fi
done
{
    echo "From scratch: $NAME, stages $STAGES, $STEPS steps each, action scale $ACTION_SCALE, actor ${ACTOR:-256,128,64} (last stage: $prev)"
    python scripts/analysis/speeds.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
    echo
    python scripts/analysis/straight.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
    echo
    python scripts/analysis/falls.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
    echo
    python scripts/analysis/stand_pushes.py runs/v7/juan_v7_13_straight ${exported[@]+"${exported[@]}"}
} 2>&1 | tee "$REPORT"
echo "=== $(date +%H:%M) done: $REPORT"
