# Juan learns to walk

Reinforcement learning for **Juan**, a 43 cm, 8-servo biped, trained in MuJoCo and meant to run on the real robot. The goal: walk at a decent speed, don't fall, and follow steering commands from a phone, joystick or computer.

**Status (October 2026):** Juan walks in simulation at about 0.4 m/s (juan_v3). The current recipe, **v7**, adds steering commands, left-right symmetry and much more realistic physics on top of that walk. The real robot is being built in parallel.

---

## The robot

| | |
|---|---|
| Size | 43 cm tall, center of mass ~25 cm up |
| Weight | 1.75 kg in the CAD model; expected closer to 2 kg built (PLA, wiring) |
| Legs | 4 joints each: hip roll, hip pitch, knee, ankle |
| Servos | 8 × Hiwonder HTD-35H serial bus servos (3.4 Nm, 5.8 rad/s at 11.1 V) via a BusLinker USB board |
| Computer | Raspberry Pi 4, policy at 50 Hz |
| Sensors | Adafruit BNO055 IMU; servo angle readback; Pi Camera v2.1 (later) |
| Power | 3S 2200 mAh LiPo for servos, 3S 650 mAh for the Pi |

Specs, measurements to make and recommended hardware changes are in [reports/Juan robot electronics specs.md](reports/Juan%20robot%20electronics%20specs.md).

---

## Quick start

```bash
# one-time setup (Python 3.11; uv: https://docs.astral.sh/uv/)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .   # the biped_rl package and its dependencies
source .venv/bin/activate

# train the current recipe (~65M steps, ~4-5 h on an 18-core Mac; Ctrl+C saves)
SMOKE=1 ./scripts/train_v7.sh      # 15-minute check of the whole pipeline first
./scripts/train_v7.sh

# watch it
tensorboard --logdir runs  # http://localhost:6006
tail -f runs/juan_v7.log

# evaluate a run
python scripts/evaluate.py --run runs/juan_v7_4_pushes                  # nominal physics, writes eval.mp4
python scripts/evaluate.py --run runs/juan_v7_4_pushes --robust -n 50   # randomized physics + pushes: fall rate
mjpython scripts/evaluate.py --run runs/juan_v7_4_pushes --viewer       # live 3D viewer (macOS needs mjpython)
mjpython scripts/play.py --run runs/juan_v7_4_pushes [--robust|--pushes]  # drive it with the arrow keys, after the boot
                                                                # sequence (straight legs -> crouch -> policy)
```

Run everything from the project folder. The project should not live in an iCloud-synced folder: iCloud turns checkpoints and meshes into placeholders that hang training until they download, and it hides the editable-install file in `.venv`, so `import biped_rl` stops working.

---

## How it works

The code is organized like Berkeley Humanoid Lite [[1]](#references): a robot description, a task made of small term modules (commands, observations, rewards, terminations, events, curricula), a training package, and thin scripts. Its training recipe follows Li et al. [[2]](#references).

```
Onshape CAD ─onshape-to-robot─▶ juan/robot.xml ─scripts/model/postprocess_onshape.py─▶ juan/rl_scene.xml
                                                                                            │
biped_rl/robots/juan.py ───▶ biped_rl/tasks/locomotion/ ◀───────────────────────────────────┘
(joints, servo numbers)      env.py       simulation loop, servo model
                             mdp/         commands, observations, rewards, terminations,
                                          events (randomization, pushes), curriculums
                             env_cfg.py   every setting
                                   │
                                   ▼
            scripts/train.py + biped_rl/training/ (PPO, stages, dashboard)
                                   │
                                   ▼
                        runs/<name>/final_model ──▶ scripts/evaluate.py, play.py, analysis/
                                                ──▶ the real robot (to do)
```

### The simulated robot
- **Servo model, not ideal motors.** Each joint is driven like a hobby servo: a PD controller, a linear torque-speed curve (no torque left at no-load speed), command delay, gear slack, and a zero-calibration error. The servo model is in [env.py](biped_rl/tasks/locomotion/env.py); its numbers in [robots/juan.py](biped_rl/robots/juan.py) and `RandomizationConfig` ([env_cfg.py](biped_rl/tasks/locomotion/env_cfg.py)).
- **Actions** are servo target offsets from a crouched home pose (0.5 rad per unit), smoothed by a filter (`filtered = 0.5·action + 0.5·filtered`), at 50 Hz.
- **The policy only sees what the robot can measure:** IMU gravity direction and gyro, servo angles and speeds, its own last actions (3 frames of history), the command, a gait clock, and its heading error. True velocity, foot contacts and friction go only to the critic (asymmetric actor-critic, [policy.py](biped_rl/policy.py); the layout is documented in [mdp/observations.py](biped_rl/tasks/locomotion/mdp/observations.py)).

### Commands
The policy follows three commands, the same three a joystick will give: **forward speed** (−0.15 to 0.4 m/s, raised by a speed curriculum), **sideways speed** (±0.15 m/s) and **turn rate** (±0.6 rad/s), plus **stand still**. Of the walking commands, 20% only turn, 10% only walk backward and 10% only walk sideways, so each of these gets practised on its own. During training a new command is drawn every 2–8 s. Training on many tasks also makes the policy more robust: it learns more ways to move, and reuses them to recover (Li et al. 2024).

### Left-right symmetry
Juan's model is symmetric (left and right part positions agree within 1 mm), so the policy is made symmetric **by construction**: it computes its action for the situation and for its mirror image and averages them ([symmetry.py](biped_rl/symmetry.py), [policy.py](biped_rl/policy.py)). It cannot favor a leg or drift to one side. Earlier versions tried to *reward* symmetry instead, and that bent the gait (see the history below). Every new training run first checks the mirror map on the model.

### Reward
`reward = (alive + tracking × task terms) × exp(−sum of penalties)`, with a one-off penalty on falling. Task terms only pay while the command is being followed, and penalties scale the reward down instead of making it negative. Each term is a small function in [mdp/rewards.py](biped_rl/tasks/locomotion/mdp/rewards.py), named like its weight in `RewardWeights` ([env_cfg.py](biped_rl/tasks/locomotion/env_cfg.py)).

| Goal | Terms | Why |
|---|---|---|
| Follow the command | `tracking_vel`, `tracking_lat_yaw`, `heading`, `path_offset`, `stand_still` | Speed is measured along the *target* heading (the start heading, turned only as commanded), so walking in circles earns nothing; drifting off the commanded path costs. |
| Don't fall | `alive`, `termination`, `orientation`, `ang_vel_xy`, `lin_vel_z`, `base_height` | A fall is any non-foot contact with the ground, more than 53° of tilt, or the body sinking to half its height. |
| Protect the hardware | `servo_load`, `servo_overload`, `servo_saturation`, `backdrive`, `impact`, `joint_limits`, `energy` | Hobby servos strip gears and overheat near stall; hard landings break brackets. A landing above 3× body weight, or a joint forced past 1.3× servo speed while the servo is pushing back hard, ends the episode like a fall. |
| Smooth, low-wear motion | `action_rate`, `action_acc`, `joint_acc`, `reversals`, `footstep` | Fewer servo direction changes and fewer, rhythm-paced steps mean less gear and pot wear. |
| A real walking gait | `gait_contact`, `gait_mismatch`, `feet_clearance`, `foot_slip`, `feet_distance`, `knee_hyperextension`, `hip_roll`, `hip_roll_inward`, `symmetry` | A gait clock says when each foot should be in the air; feet must lift 2.5 cm (no shuffling), not slide, and not collide. |

Speed tracking uses a forgiving kernel while the policy learns to walk (creeping must beat standing) and a sharp one afterwards (so it can't trade speed for small penalty savings).

### Robustness: what changes every episode
- **Servos:** one battery-voltage draw scales torque and speed together (−15%/+13%), ±5% per servo, stiffness and damping ±30%, 0–1° gear slack, ±1.7° zero error, 0.24° angle resolution
- **Delay:** 5–59 ms from sensing to the servos moving, and readings up to one step old. Reading 8 bus servos over USB likely takes 20–30 ms: measure it and narrow this range
- **Body:** 1.65–2.25 kg, extra weight mostly in the torso; torso center of mass ±1.5 cm; rotor inertia, gearbox friction
- **IMU:** a fixed tilt bias of up to ±2.5° and gyro offset of up to ±0.02 rad/s per episode, plus noise
- **Start:** from stage 3 on, each episode starts from a perturbed crouch (joint angles 70–130% of home) with the body already moving, as in Berkeley Humanoid Lite [[1]](#references)
- **World:** floor friction 0.5–1.2, slopes up to 3°, pushes (see below)

### Training recipe (v7)
Following Li et al. [[2]](#references) (Cassie): learn the skill, then the variety of tasks, then robustness, with pushes last.

| Stage | Physics | Commands | Pushes | Steps |
|---|---|---|---|---|
| 0 warm start | | | | copies juan_v3's walk into the v7 network ([scripts/warm_start.py](scripts/warm_start.py)) |
| 1 `walk` | nominal | one forward speed per episode | none | skipped by default |
| 2 `commands` | nominal | all, changing every 2–8 s | none | 15M |
| 3 `physics` | randomized, the v7 realism phased in by the physics curriculum | all | none | 25M, extended by 10M (up to 3×) until full difficulty |
| 4 `pushes` | randomized, **full difficulty** | all | per-direction curriculum | 25M |

- **Why the warm start:** learning to walk from zero on Juan's model has so far ended with the policy standing still. Every Juan policy before v7 also inherited its walk, from a policy trained on a simpler placeholder biped. The v7 network has the same inputs as v3 plus the two new commands, so v3's weights copy over directly and the copy walks from the first step. `FROM=scratch ./scripts/train_v7.sh` runs all four stages from zero.
- **Physics curriculum:** switching the whole realistic robot on at once (v7's first stage-3 attempt) made it fall in 60–95% of episodes and degraded the walk it had. Instead, one difficulty level (0 → 1) scales the realism added in v7: extra delay, extra and torso mass, gear slack, servo zero error, IMU bias. Each training environment raises it by 0.05 once the robot stayed up in 80% of its last 20 episodes. Stage 3 is extended until the level reaches 1, and stage 4 always trains at full difficulty, so the final policy is trained on the full robot. Evaluation (and so `best_model`) always uses full difficulty.
- **Between stages:** curriculum levels (top speed, difficulty, push strengths) carry over to the next run (`curriculum.json`), exploration noise is raised back to at least 0.1, and speed tracking stays forgiving until stage 4.
- **Push curriculum:** 8 push directions relative to the robot, each with its own strength, starting at 0.05 m/s. A direction gets 0.02 m/s stronger once the robot stays up through 9 of 10 pushes from it, weaker otherwise, up to 0.4 m/s. The policy trains at the edge of what it can take from each side.
- Each stage is its own run (`runs/juan_v7_<n>_<stage>`) that resumes the previous one; `./scripts/train_v7.sh physics` restarts from a later stage. In stage 4, evaluation keeps the fixed standard test (random shoves up to 0.3 m/s per axis), so results compare with older runs.

---

## Reading TensorBoard

Charts are grouped and numbered in reading order:

| Group | Answers | Healthy |
|---|---|---|
| `1_overview` | Is it learning? | falls % and hardware-limit stops % toward 0, `episode_seconds` toward 20, speed error falling, push strength rising |
| `2_commands` | Does it do what it's told? | forward, sideways and turn-rate errors, heading error and path drift shrinking |
| `3_hardware` | Is it gentle on the servos? | torque and joint speed well under 1 (× stall, × servo max), landing force under 2 × body weight |
| `4_gait` | What does the walk look like? | double support well below ~90% (near 100% = standing still) |
| `5_rewards`, `6_penalties` | Why? | grouped sums; `scripts/train.py --log-details` adds every single penalty |

Walking clips land in `runs/<name>/videos/` every 1M steps, named with the distance walked in 8 s at the top speed command (0.4 m/s → ~3 m).

---

## Evaluating and comparing policies

| Tool | What it tells you |
|---|---|
| `scripts/evaluate.py --run R [--robust -n 50] [--sweep]` | speed, flight/double support, impacts, saturation, smoothness, drift; fall rate under randomization; top speed |
| `scripts/analysis/fall_breakdown.py R full\|nopush\|pushonly` + `scripts/analysis/summarize_falls.py` | falls vs hardware-limit stops, and the push direction before each failure (same seeds in every mode) |
| `scripts/analysis/joint_breakdown.py R full` | which joint trips the speed limit, when in the stride, and whether the robot would have walked on |
| `scripts/analysis/gait_symmetry.py R`, `scripts/analysis/step_rhythm.py R` | left vs right step timing, length, foot lift, posture |
| `scripts/analysis/wear.py R1 R2 ...` | servo reversals, footsteps and joint range per meter: compare wear between runs |
| `scripts/analysis/hardware_loads.py R` | per-joint torque and speed, foot impacts, estimated battery current |
| `scripts/analysis/push_video.py R` | videos of the robot being shoved, including falls |

---

## Project layout

| Path | What |
|---|---|
| [biped_rl/robots/](biped_rl/robots/) | Robot descriptions: [juan.py](biped_rl/robots/juan.py) (model, feet, joint conventions, mirror pairs, servo numbers), and model paths that survive moving the project |
| [biped_rl/tasks/locomotion/](biped_rl/tasks/locomotion/) | The walking task: [env.py](biped_rl/tasks/locomotion/env.py) (simulation loop, servo model), [env_cfg.py](biped_rl/tasks/locomotion/env_cfg.py) (**every setting**; `LEGACY` makes older run configs behave exactly as trained), and [mdp/](biped_rl/tasks/locomotion/mdp/): one module per kind of term (commands, observations, rewards, terminations, events, curriculums) |
| [biped_rl/training/](biped_rl/training/) | [ppo.py](biped_rl/training/ppo.py) (PPO hyperparameters, a fresh model), [stages.py](biped_rl/training/stages.py) (training stages, curriculum carry-over), [callbacks.py](biped_rl/training/callbacks.py) (TensorBoard dashboard, clips) |
| [biped_rl/policy.py](biped_rl/policy.py), [symmetry.py](biped_rl/symmetry.py), [motions.py](biped_rl/motions.py) | Asymmetric actor-critic; mirror maps, the model check and symmetric normalization; the boot sequence |
| [scripts/](scripts/) | Entry points: `train.py`, `train_v7.sh` (the v7 recipe), `evaluate.py`, `play.py` (keyboard driving), `warm_start.py`; [analysis/](scripts/analysis/) (the tools above); [model/](scripts/model/) (`postprocess_onshape.py`: CAD export → RL model, `inspect_model.py`: mass/limits check and stand test, `cad_to_mjcf.py`: URDF route) |
| [robot/](robot/) | Runs on the Pi: [joints.py](robot/joints.py) (joint map and calibration), [hiwonder.py](robot/hiwonder.py) (servo bus), [imu.py](robot/imu.py) (LSM9DS1 + gravity filter, in the simulator's conventions; [bno055.py](robot/bno055.py) as a fallback), [policy.py](robot/policy.py) (numpy policy and the simulator's observation pipeline), [walker.py](robot/walker.py) (50 Hz loop, safety, logs), [camera.py](robot/camera.py), [dashboard.py](robot/dashboard.py), [selftest.py](robot/selftest.py), [BRINGUP.md](robot/BRINGUP.md) |
| [juan/](juan/) | Juan's model from Onshape (`robot.xml`) and the RL-ready version (`rl_robot.xml`, `rl_scene.xml`) |
| [models/simple_biped.xml](models/simple_biped.xml) | The placeholder biped used before Juan's CAD existed |
| [reports/](reports/), [research_notes/](research_notes/) | Research: fall recovery best practices; Juan's electronics |
| `runs/` | Training outputs (not in git, except each run's `config.json` and `report.txt`) |

Old run configs load and evaluate exactly as trained: settings added later fall back to their `LEGACY` values, and absolute model paths from another computer are resolved inside the project.

---

## History: what each version taught us

| Runs | Robot | What changed | What we learned |
|---|---|---|---|
| `old_*` | placeholder | first attempts (one with 9 rad/s servos and a 1.2 kg body) | policies walked in circles (→ speed measured along a fixed heading, heading and path terms) and settled for standing still (→ later the `stand_still` penalty) |
| walk_v3–v5 | placeholder | hardware limits, servo protection and impacts; then symmetry and `stand_still`; then footstep, reversal and gait-mismatch penalties | a robust walk on the placeholder |
| juan_v1–v2 | **Juan** | continued from the placeholder policy, on Juan's CAD model | the walk transferred to Juan's geometry and mass |
| **juan_v3** | Juan | sharper speed tracking | **the preferred walk.** 0% falls on nominal physics; 52% failures under randomized physics + 0.3 m/s pushes, half of them hardware-limit stops (a right-knee flick exceeding servo speed) rather than falls |
| juan_v4 | Juan | strong posture-symmetry penalty | the gait turned into a skip: rewarding symmetry bends the gait |
| juan_v5–v6 | Juan | step-timing penalty | longer steps and less wear, but still uneven and veering left |
| **v7** | Juan | steering commands, mirror-symmetric policy, realistic physics (2 kg, real delays, servo slack, IMU bias), staged training with a push curriculum, warm start from v3 | in progress. From scratch on Juan, the policy got stuck standing; the warm start walks from step 0 |

Next experiments: a 2-second history encoder (the dual-history architecture of Li et al. [[2]](#references)) as an A/B test against v7; a servo model with separate driving and braking torque limits and load-dependent backlash, as on PhoneBot [[3]](#references), a biped of Juan's size; and a from-scratch recipe (a feet-air-time reward, more exploration, no hardware stops at first, as in [[1]](#references) and [[3]](#references)), so a policy can be retrained from zero after big CAD changes.

---

## Deployment on the robot

The robot side lives in [robot/](robot/) and runs on the Pi; it needs only `pyserial`, `smbus2` and `numpy` (plus `python3-picamera2`), not the training stack. Bring-up, step by step: [robot/BRINGUP.md](robot/BRINGUP.md).

### Robot dashboard

```bash
# on the computer: export a policy (numpy only), checked step by step against the simulator
python scripts/export_policy.py runs/juan_v7_8_best      # -> robot/policies/juan_v7_8_best.npz

# on the Pi, once
sudo apt install python3-serial python3-smbus2 python3-numpy python3-picamera2 i2c-tools alsa-utils espeak-ng
# on the Pi, from the project folder
python3 -m robot.selftest     # reads every sensor, moves nothing
python3 -m robot.dashboard    # then open http://<pi-name>.local:8000
```

All 8 servos (one BusLinker per leg: IDs 1–4 left, 5–8 right, in the order hip roll, hip pitch, knee, ankle), the IMU, the camera, the battery and the Pi's health. It shows angles in the simulator's convention (legs straight = 0°, signs below) and walks you through calibration: zero with the legs straight, check each joint's direction, set physical limits by hand, and store them in the servos' memory. Every move is clamped to the limits; the calibration is saved in `robot/calibration.json`. Servos relax at 60 °C, everything relaxes if the Pi reaches 75 °C. The bus cards show how long reading one leg's 4 servos takes, the number that sets the control loop's delay. The **walking policy** runs from the same page ([robot/walker.py](robot/walker.py)): start it at Home and steer with the arrow keys or the on-screen pad. It stops by itself if a sensor goes quiet or the page disconnects, relaxes everything past 40° of tilt, and logs every run to `robot/logs/`. The IMU is an LSM9DS1 (accelerometer + gyro, no on-chip fusion): [robot/imu.py](robot/imu.py) estimates gravity with a complementary filter (0.5 s), checked in closed loop in the simulator (2–3° mean error, the policy walks as with the true gravity). **Personality** ([robot/personality.py](robot/personality.py)): gestures (happy bounce, wake-up stretch, sad, toe tap, bow, sway, dance to a BPM) from [robot/gestures.py](robot/gestures.py), each checked on the simulated robot with the measured servos by `python scripts/check_gestures.py [--video]` before it may run; a procedurally generated chirping voice per emotion ([robot/sound.py](robot/sound.py); your own `.wav` files go in `robot/sounds/`, text-to-speech with `espeak-ng`); a playful mode with idle animations; and sounds when it starts walking, stops, is bumped, falls or runs low on battery. After changing the model, `python scripts/check_robot_joints.py` checks that [robot/joints.py](robot/joints.py) still matches it.

### Calibration before deployment

| Item | Why it matters | Reference |
|---|---|---|
| Servo direction and zero offset, per joint | Mirrored servo mounting can reverse a joint on one side; the policy assumes the simulation's sign conventions (below) | `scripts/model/postprocess_onshape.py` |
| Control-loop delay (read 8 servos + write 8 targets) | Training randomizes 5–59 ms; the measured value narrows `latency` and `action_delay_steps` | `RandomizationConfig` |
| Servo voltage-limit register | A 12.0 V limit would unload the servos on a full 12.6 V pack | electronics report |
| Part masses | Training assumes 1.65–2.25 kg | `RandomizationConfig`, BoM in `postprocess_onshape.py` |

Joint sign conventions (torso frame: x forward, y left, z up):

| Joint | Positive rotation |
|---|---|
| Hip roll | left leg outward, right leg inward |
| Hip pitch | leg swings back |
| Knee | shin swings back (knee bends) |
| Ankle | toes down |

### Control loop (50 Hz)

1. **Observation:** built exactly as `read_sensors()`, `history_frames()` and `compute()` in [mdp/observations.py](biped_rl/tasks/locomotion/mdp/observations.py): 3 history frames of [gravity in torso frame, gyro, joint angles − home, 0.1 × joint speeds, previous action], then the command (forward, sideways, turn rate), the gait clock and the heading error. The privileged part is zero-filled.
2. **Normalization:** the run's `vecnormalize.pkl` statistics, clipped to ±10.
3. **Policy:** the full forward pass of `AsymmetricActorCritic`, including the mirror averaging (export the whole pass, e.g. to ONNX, not a single network). The action is clipped to [−1, 1].
4. **Servo targets:** `filtered = 0.5·action + 0.5·filtered`, then `target = home + 0.5·filtered`, clipped to the joint limits, then mapped through the per-servo sign and zero offset.

### Safety layer

Independent of the policy: a tilt watchdog that unloads or freezes the servos on an unrecoverable fall, battery-voltage and servo-temperature monitoring, and a hardware kill switch. Initial tests run on a harness.

---

## Updating the robot model from Onshape

1. In Onshape: mate connectors named `dof_<joint>` on each servo axis; wires can stay in the CAD (they are ignored in the export).
2. API keys from https://dev-portal.onshape.com/keys go in `.env` (never commit it). `juan/config.json` holds the export settings.
3. `onshape-to-robot juan`, then `python scripts/model/postprocess_onshape.py juan`. It writes `juan/rl_robot.xml` and `juan/rl_scene.xml`: part masses from the BoM (not Onshape's materials), the training's joint sign conventions (hip pitch and ankle negative = leg forward / toes up, knee positive = bends back), hip roll limits, box feet with rubber pads, leg capsules, the IMU site and the home pose. Update the BoM masses in that script when parts change.
4. `python scripts/model/inspect_model.py juan/rl_scene.xml --view`, and compare part masses with the real parts on a scale.
5. Retrain: for small changes, warm start from the latest Juan policy (`FROM=runs/<latest> ./scripts/train_v7.sh`, or `scripts/train.py --resume`); for big ones, from scratch.

---

## Running (flight phase)

Rewarding air time directly produces hopping in place. Instead, fine-tune a walking policy with a flight phase built into the gait clock: `scripts/train.py --resume <walk run> --swing-ratio 0.6 --max-speed 0.5` gives 80 ms of flight per step at a 0.8 s period. With 3.4 Nm, 5.8 rad/s servos and ~2 kg, real flight is near the servos' limit: watch saturation and landing force.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Stands still, never walks (double support → 100%, clips walk 0.00 m) | Standing is a local optimum. Warm start from a walking policy (`scripts/warm_start.py`); from scratch, use the forgiving speed tracking and forward-only commands of stage `walk` |
| Falls a lot under randomization but not on nominal physics | Train longer; or phase in the hardest settings (delay, torso mass, push strength) |
| Many "hardware limit" stops | Check `scripts/analysis/joint_breakdown.py`: which joint, and whether it is a real overload |
| Jittery | Raise `action_rate`/`action_acc`, lower `action_filter` (e.g. 0.3) |
| Shuffles feet | Raise `feet_clearance` and `gait_contact` |
| `mirror check` fails before training | A sign in `RobotConfig.mirror_pairs` is wrong, or the model is no longer symmetric |
| Training or loading hangs with no CPU use | Files are iCloud placeholders: download them (Finder → Download Now) or move the project out of iCloud |
| Model explodes | Zero or tiny masses/inertias, or overlapping collision shapes |
| `launch_passive` error on macOS | Use `mjpython` instead of `python` |
| `ModuleNotFoundError: biped_rl` | Install the project (`uv pip install --python .venv/bin/python -e .`). In an iCloud folder macOS hides the install file and Python skips it: move the project out of iCloud (or, as a stopgap, `export PYTHONPATH=$PWD`) |

---

## References

1. Y. Chi, Q. Liao, J. Long, X. Huang, S. Shao, B. Nikolic, Z. Li and K. Sreenath, "Demonstrating Berkeley Humanoid Lite: An Open-source, Accessible, and Customizable 3D-printed Humanoid Robot," arXiv:2504.17249, 2025. [Paper](https://arxiv.org/abs/2504.17249) · [Code](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite) · [Website](https://lite.berkeley-humanoid.org/). The model for this project's code structure (robot description, task with term modules, training package, scripts) and for its varied starting states.
2. Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth and K. Sreenath, "Reinforcement learning for versatile, dynamic, and robust bipedal locomotion control," *The International Journal of Robotics Research*, vol. 44, no. 5, pp. 840–888, 2025. [arXiv:2401.16889](https://arxiv.org/abs/2401.16889). The staged training, the variety of commands as a source of robustness, and the planned 2-second-history encoder.
3. R. Hou, Q. Wang, D. Koh and D. W. Hong, "PhoneBot: A Low-Cost Open Humanoid Robot Platform Reusing Smartphones," arXiv:2610.08737, 2026. [Paper](https://arxiv.org/abs/2610.08737) · [Website](https://phonebot.dev). A biped of Juan's size: the planned torque-aware servo model, load-dependent backlash and from-scratch recipe.
