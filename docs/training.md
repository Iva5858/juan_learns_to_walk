# Training and the simulator

How the simulated Juan, the task and the training work, and the tools to evaluate a policy. Start with the [README](../README.md).

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
| Follow the command | `tracking_vel`, `tracking_lat_yaw`, `heading`, `path_offset`, `stand_still`, `stand_pose` |
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

Fine-tunes start from a `pushes`-stage policy with a lower learning rate (`--lr 1e-4`); [scripts/finetune.sh](../scripts/finetune.sh) is the whole loop (train, export, compare) with the current recipe. `--reward`, `--command`, `--robot` and `--randomization NAME=VALUE` override any setting for a run. Learning to walk from zero on Juan's model has so far ended in standing still; every policy here descends from one first trained on a simpler placeholder biped.

## Reading TensorBoard

| Group | Answers | Healthy |
|---|---|---|
| `1_overview` | Is it learning? | falls and hardware-limit stops toward 0, episode length toward 20 s, push strength rising |
| `2_commands` | Does it do what it's told? | forward, sideways, turn-rate and heading errors shrinking |
| `3_hardware` | Is it gentle on the servos? | torque and joint speed well under 1 (× stall, × max speed), landings under 2 × body weight |
| `4_gait` | What does the walk look like? | double support well below 90% (near 100% = standing still) |
| `5_rewards`, `6_penalties` | Why? | grouped sums; `--log-details` adds every single penalty |

Walking clips land in `runs/<name>/videos/` every 1M steps. `best_model.zip` is the best evaluation during training (randomized robot, 0.3 m/s shoves).

## Tools

| Script | What it tells you |
|---|---|
| `scripts/evaluate.py --run R [--robust -n 50] [--sweep]` | speed, gait, impacts, saturation, smoothness, drift; fall rate under randomization; top speed |
| `scripts/play.py --run R [--test]` | drive it with the keyboard after the robot's boot sequence; `--test` checks it without a window |
| `scripts/analysis/straight.py R ...` | forward distance, sideways drift, turning and heading weave at fixed speeds |
| `scripts/analysis/falls.py R ...` | falls vs hardware-limit stops, with and without shoves, and the direction of the shove before each fall |
| `scripts/analysis/standing.py NAME` | an exported policy through the robot's own controller: standing lean, offset from home, the jump at take-over |
| `scripts/analysis/replay.py LOG` | a robot log replayed through the simulator: which servo model matches the real joints |
| `scripts/analysis/replay_video.py LOG` | a robot log rendered in 3D with its telemetry |
| `scripts/check_gestures.py` | every gesture on the simulated robot before it may run on the real one |
| `scripts/check_robot_joints.py` | the robot's joint table still matches the model |

Analysis scripts test on the measured real servos; `--as-trained` uses each run's own.

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
| Weaves or drifts on the robot but not in the sim | The servo model: check with `straight.py` (measured servos) and fit it again from new logs with `replay.py` |
| Jittery | Raise `action_rate` / `action_acc`; on the robot, check the vibration gauge on the IMU card |
| `mirror check` fails before training | A sign in `MIRROR_PAIRS` is wrong, or the model is no longer symmetric |
| Training or loading hangs with no CPU use | iCloud placeholders: move the project out of iCloud |
| `launch_passive` error on macOS | Use `mjpython` instead of `python` |
