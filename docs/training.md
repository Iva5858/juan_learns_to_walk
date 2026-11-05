# Training and the simulator

How the simulated Juan, the task and the training work, and the tools to evaluate a policy. Start with the [README](../README.md).

Experiments that didn't become the main line (wider joint range pilot, speed from scratch): [experiments-2026-11.md](experiments-2026-11.md).

## The simulated robot
- **Hobby servos, not ideal motors.** Each joint is a PD controller with a linear torque-speed curve, command delay, gear slack and a zero-calibration error ([env.py](../biped_rl/tasks/locomotion/env.py)). The numbers are in [robots/juan.py](../biped_rl/robots/juan.py). The real servos were fitted from walking logs ([scripts/analysis/replay.py](../scripts/analysis/replay.py)): stiffer and slower to react than the training default (kp ~30 vs 15, 70–90 ms vs 25–59 ms). `EnvConfig.use_measured_servos()` switches to them, and the analysis scripts test on them by default.
- **Actions** are servo target offsets from a crouched home pose (0.5 rad per unit), smoothed (`filtered = 0.5·action + 0.5·filtered`), at 50 Hz.
- **The policy only sees what the robot measures:** IMU gravity direction and gyro (low-passed at 4 Hz, as on the robot), servo angles and speeds, its last actions (3 frames of history), the command, a gait clock and its heading error. True velocity, contacts and friction go only to the critic (asymmetric actor-critic, [policy.py](../biped_rl/policy.py); layout in [mdp/observations.py](../biped_rl/tasks/locomotion/mdp/observations.py)).
- **Left-right symmetric by construction:** the actor averages its action with the mirrored action for the mirrored situation ([symmetry.py](../biped_rl/symmetry.py)), so it cannot favor a leg. Rewarding symmetry instead bent the gait into a skip.

## Commands
Forward speed (−0.15 m/s up to a speed curriculum's top), sideways speed (±0.15 m/s), turn rate (±0.6 rad/s), or stand still: the same three a joystick gives. A new command every 2–8 s. Of the walking commands, 20% only turn, 10% only walk backward and 10% only side-step; 30% of the rest walk straight.

## Reward
`reward = (alive + tracking × task terms) × exp(−sum of penalties)`, plus a one-off penalty on falling. Task terms only pay while the speed command is met; penalties scale the reward down instead of making it negative. Each term is a small function in [mdp/rewards.py](../biped_rl/tasks/locomotion/mdp/rewards.py), named like its weight in `RewardWeights` ([env_cfg.py](../biped_rl/tasks/locomotion/env_cfg.py)).

| Goal | Terms |
|---|---|
| Follow the command | `tracking_vel`, `tracking_lat_yaw`, `heading`, `path_offset`, `stand_still` (`stand_still_moves`: sideways, turning), `stand_pose` |
| Don't fall | `alive`, `termination`, `orientation`, `ang_vel_xy`, `lin_vel_z`, `base_height` |
| Protect the servos | `servo_load`, `servo_overload`, `servo_saturation`, `backdrive`, `impact`, `joint_limits`, `energy` |
| Smooth, low-wear motion | `action_rate`, `action_acc`, `joint_acc`, `reversals`, `footstep` |
| A real walking gait | `gait_contact`, `gait_mismatch`, `feet_clearance`, `foot_slip`, `feet_distance`, `knee_hyperextension`, `hip_roll`, `hip_roll_inward`, `symmetry` |
| Longer strides with real steps (off by default; on from v7_15) | `step_length` (a lifted foot lands ahead of the other in the walking direction, up to 12 cm), `feet_air_time` (foot stays lifted for the swing the clock plans), `swing_lift` and `swing_reach` (dense: the swing foot's height, and its speed in the commanded direction while lifted), `double_support` (penalty: both feet down when one should swing) |

Speed is measured along the target heading (the start heading, turned only as commanded), so walking in circles earns nothing. A landing above 3× body weight, or a joint forced past 1.3× servo speed while the servo pushes back hard, ends the episode like a fall ("broke").

## Randomized every episode
Servo strength and speed (one battery-voltage draw, ±5% per servo), stiffness and damping ±30%, 0–1° gear slack, ±1.7° zero error; 25–59 ms of action delay and sensor readings up to a step old; 1.65–2.25 kg with extra torso mass, CoM shift; IMU tilt bias ±2.5°, gyro bias, servo buzz on the gyro; floor friction 0.5–1.2 and slopes up to 3°; a perturbed, already-moving start; and shoves.

## Training stages
Each stage is its own run that resumes the previous one (`--stage` and `--resume`); curriculum levels carry over in `curriculum.json`.

| Stage | Physics | Commands | Pushes |
|---|---|---|---|
| `walk` | nominal | one forward speed per episode | none |
| `commands` | nominal | all, changing every 2–8 s | none |
| `physics` | randomized, phased in by a difficulty curriculum (0 → 1) | all | none |
| `pushes` | randomized, full difficulty | all | per-direction curriculum, up to 0.4 m/s |

Fine-tunes start from a `pushes`-stage policy with a lower learning rate (`--lr 1e-4`); [scripts/finetune.sh](../scripts/finetune.sh) is the whole loop: train with the base recipe, export, and judge against the starting policy (walk test with a KEEP / NOT BETTER verdict, standing shove test). `--reward`, `--command`, `--robot` and `--randomization NAME=VALUE` override any setting for a run. Learning to walk from zero on Juan's model has so far ended in standing still (v8, v9: removed in November 2026, see git history); every policy here descends from one first trained on a simpler placeholder biped.

### The free run (v11): speed from scratch
Stages `speed` and `speed_physics` ([stages.py](../biped_rl/training/stages.py) `apply_speed_stage`) train a new
walker for speed only, with nothing that suggests a gait: legs straight at the start and as the zero of the
actions (`neutral_pose = "straight"`), the full joint range, one "go" command (`go_command`, no speed target, no
stand), no gait clock. Reward: net centre-of-mass speed over 1 s, no cap (`net_speed`); a fall takes back the
last second of it (`fall_refund_s`: a dive pays nothing); lifted steps that carry the body (`free_air_time`). Costs:
only the robot's physical limits (foot slip, overload, back-drive, landings, end stops; the hardware stops stay).
Style and wear terms are off, for later fine-tunes. Each choice answers an earlier failure: see the docstring.
[scripts/speed_chain.sh](../scripts/speed_chain.sh) runs `speed`, a gate (net speed on the nominal robot), then
`speed_physics`; [free_run_test.py](../scripts/analysis/free_run_test.py) judges the results (net speed, falls,
cadence, lift, landings, torque, knee direction). On the robot: start from straight legs (Straight, not Home);
any forward command is "go", zero stops (the policy never learned to stand still).

## Fine-tune recipes

Each one is `FROM=<start> STEPS=<n> ./scripts/finetune.sh v7/<name> <options>`, on top of finetune.sh's base recipe. A run's exact command line is in its `run.json` (from v7_21 on).

| Run | From | Steps | Options | Outcome |
|---|---|---|---|---|
| v7_13_straight | v7_12b_pushes_best | 30M | (the base recipe) | **safe on the robot**; ~25% of the command |
| v7_13_scale08 | v7_13_straight | – | `python scripts/rescale_actions.py runs/v7/juan_v7_13_straight runs/v7/juan_v7_13_scale08 --scale 0.8` | same walk, 0.8 rad action range |
| v7_15d | v7_13_scale08 | 30M | `--randomization push_velocity=0 action_delay_steps=2,4 --command swing_ratio=0.48 swing_height=0.035 --reward tracking_vel=5.0 step_length=4.0 feet_air_time=2.0 swing_lift=2.0 swing_reach=1.0 double_support=1.0 lin_vel_z=0.2 footstep=0.5 gait_mismatch=0.3 foot_slip=4.0 recover_tilt_deg=20` (STEP) | longer steps with real lift |
| v7_16_backward | v7_15d_best | 30M | STEP + `--command backward_prob=0.25 backward_speed=0.25` (BACK) | forward or backward walking worse |
| v7_17_speed | v7_15d_best | 30M | STEP + BACK + `--command fast_prob=0.25 fast_speed=0.65 --reward step_length_target=0.15` | kept |
| v7_18_shoves | v7_17_speed_best | 30M | v7_17's + `--randomization push_velocity=0.3 --reward orientation=4.0 feet_distance=10.0 min_feet_distance=0.09` | forward or backward walking worse |
| v7_19_topspeed | v7_17_speed_best | 40M | STEP + `--command backward_prob=0.25 backward_speed=0.25 fast_prob=0.4 fast_speed=1.0 speed_limit=1.2 --reward step_length_target=0.18 --speed-ramp 0.4,1.2 --speed-probe-every 2000000` | faster in sim; **on the robot it went diagonal and toppled** |
| v7_20_stand | v7_19_topspeed_best | 20M | STEP + `--randomization push_velocity=0.8 --command stand_prob=0.35 backward_prob=0.25 backward_speed=0.25 fast_prob=0.15 fast_speed=1.0 speed_limit=1.2 --reward step_length_target=0.18 stand_settle_s=1.0 stand_feet=1.0 --video stand --video-every 2000000` | stands up to shoves; same as v7_19 on the robot |
| v7_21_steady | v7_13_straight | 30M | `--reward tracking_vel=5.0 tracking_broad_share=0.3 feet_distance=10.0 min_feet_distance=0.09` (STEADY) | **good on the robot**; 18 cm/s at 0.4 m/s |
| v7_22_straight | v7_21_steady | 20M | STEADY + `--reward heading=3.0 tracking_lat_yaw=2.0 --randomization joint_offset=0.05` | not better (more falls) |
| v7_23_shoves | v7_21_steady | 30M | v7_22's + `--randomization push_velocity=0.5` | not better |
| v7_24_speed | v7_21_steady | 150M | STEADY + `--command speed_range=0.0,0.6 fast_prob=0.3 --reward step_length=2.0 step_length_target=0.10` | sim: 0.22 m/s at 0.4+ (v7_21: 0.12), fewer falls; works on the robot, leans while standing |
| v7_25_stand | v7_24_speed | 30M | v7_24's (SPEED) + `--reward recover_tilt_deg=20` | KEEP: standing tilt 9.5° → 5.1°, same speed, no calm falls; lost turning in place |
| v7_26_moves | v7_25_stand | 100M | SPEED + `--reward recover_tilt_deg=20 stand_still_moves=1.0 tracking_lat_yaw=2.0 --command turn_in_place_prob=0.25 sideways_prob=0.15 backward_prob=0.15` | not better (3 calm falls); `_best`: turning in place back (~100%), same speed, tilt 2.9°; still no sideways or backward |
| v7_25_range08 | v7_25_stand | – | `python scripts/rescale_actions.py runs/v7/juan_v7_25_stand runs/v7/juan_v7_25_range08 --scale 0.8 --joints hip_pitch knee ankle` | same walk, hip pitch, knee and ankle reach ±46° |
| v7_27_range | v7_25_range08 | 100M | v7_25's | no faster (uses the range only at the knee, for lift); final falls more; `_best` KEEP by fewer shove falls |

The 0.8 rad runs (v7_14–v7_20) did well in the simulator and badly on the robot; the 0.5 rad line (v7_13, v7_21, v7_24) is the one that walks.

## Reading TensorBoard

| Group | Answers | Healthy |
|---|---|---|
| `1_overview` | Is it learning? | falls and hardware-limit stops toward 0, episode length toward 20 s, push strength rising |
| `2_commands` | Does it do what it's told? | forward, sideways, turn-rate and heading errors shrinking |
| `3_hardware` | Is it gentle on the servos? | torque and joint speed well under 1 (× stall, × max speed), landings under 2 × body weight |
| `4_gait` | What does the walk look like? | double support well below 90% (near 100% = standing still) |
| `5_rewards`, `6_penalties` | Why? | grouped sums; `--log-details` adds every single penalty |

Walking clips land in `runs/<generation>/<name>/videos/` (layout: [runs.md](runs.md)) every 1M steps. `best_model.zip` is the best evaluation during training (randomized robot, 0.3 m/s shoves).

## Tools

| Script | What it tells you |
|---|---|
| `scripts/evaluate.py --run R [--robust -n 50] [--sweep]` | speed, gait, impacts, saturation, smoothness, drift; fall rate under randomization; top speed |
| `scripts/play.py --run R [--test]` | drive it with the keyboard after the robot's boot sequence; `--test` checks it without a window |
| `scripts/analysis/walk_test.py R ...` | the robot's way of walking (stand at home, take over, walk 8 s) at 0.1–0.6 m/s, calm and with shoves: speed reached, falls and hardware-limit stops, drift, weave, cadence, step length, foot lift, standing tilt and crouch; `--verdict` judges runs against the first; `--moves`: backward, sideways, turning |
| `scripts/analysis/stand_pushes.py R ...` | commanded to stand: shoves from 8 directions at growing strength |
| `scripts/analysis/standing.py NAME` | an exported policy through the robot's own controller: standing lean, offset from home, the jump at take-over |
| `scripts/analysis/replay.py LOG` | a robot log replayed through the simulator: which servo model matches the real joints |
| `scripts/analysis/replay_video.py LOG` | a robot log rendered in 3D with its telemetry |
| `scripts/check_gestures.py` | every gesture on the simulated robot before it may run on the real one |
| `scripts/check_robot_joints.py` | the robot's joint table still matches the model |

Analysis scripts test on the measured real servos (`stand_pushes.py --as-trained`: each run's own).

## Updating the robot model from Onshape

1. In Onshape: mate connectors named `dof_<joint>` on each servo axis.
2. API keys from https://dev-portal.onshape.com/keys go in `.env` (never commit it). `juan/config.json` holds the export settings.
3. `onshape-to-robot juan`, then `python scripts/model/postprocess_onshape.py juan`: writes `juan/rl_robot.xml` and `juan/rl_scene.xml` with part masses from the BoM in that script, the joint sign conventions, hip roll limits, box feet with rubber pads, the IMU site and the home pose.
4. `python scripts/check_robot_joints.py`, then fine-tune from the latest policy.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Stands still, never walks (double support → 100%) | Standing is a local optimum: fine-tune from a walking policy instead of training from zero |
| Falls under randomization but not on nominal physics | Train longer, or phase in the hardest settings (delay, torso mass, push strength) |
| Weaves or drifts on the robot but not in the sim | The servo model: check with `walk_test.py` (measured servos) and fit it again from new logs with `replay.py` |
| Jittery | Raise `action_rate` / `action_acc`; on the robot, check the vibration gauge on the IMU card |
| `mirror check` fails before training | A sign in `MIRROR_PAIRS` is wrong, or the model is no longer symmetric |
| Training or loading hangs with no CPU use | iCloud placeholders: move the project out of iCloud |
| `launch_passive` error on macOS | Use `mjpython` instead of `python` |
