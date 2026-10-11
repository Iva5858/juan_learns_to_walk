# Bipedal robot RL with MuJoCo

Train a walking/running policy for Juan (428 mm, 1.75 kg, 8 × Hiwonder HTD-35H) in simulation, with sim-to-real in mind.

```
Onshape assembly ──onshape-to-robot──▶ MJCF ──inspect_model.py──▶ checked model
                                                       │
            eval.mp4 / viewer ◀──evaluate.py◀── policy ◀──train.py (PPO, asymmetric actor-critic)
```

| File | Purpose |
|---|---|
| [biped_rl/config.py](biped_rl/config.py) | **Every knob**: servo model, gait, reward weights, randomization |
| [biped_rl/env.py](biped_rl/env.py) | Gymnasium env: servo simulation, observations, rewards, falls, pushes |
| [biped_rl/policy.py](biped_rl/policy.py) | Actor sees only robot sensors; critic also sees simulator state |
| [models/simple_biped.xml](models/simple_biped.xml) | Placeholder built from the spec slide and BoM (428 mm, 1.74 kg, CoM 246 mm) until the Onshape export is in |
| `train.py` / `evaluate.py` | Training (with TensorBoard per-term logging) and gait/robustness report |
| `scripts/` | `inspect_model.py` (mass/limits check + stand test), `cad_to_mjcf.py` (URDF route) |

## 1. Train and evaluate

```bash
source .venv/bin/activate
python train.py --run-name walk_v1                   # 30M steps, ~1.5-2 h on this Mac; Ctrl+C saves
tensorboard --logdir runs                            # http://localhost:6006, charts in groups 1_ to 6_
python evaluate.py --run runs/walk_v1                # nominal physics + runs/walk_v1/eval.mp4
python evaluate.py --run runs/walk_v1 --robust -n 50 # randomized physics + pushes -> fall rate
mjpython evaluate.py --run runs/walk_v1 --viewer     # live viewer (macOS needs mjpython)
```

`--best` evaluates the checkpoint with the best eval score instead of the last one.

What to watch in TensorBoard (groups are numbered in reading order):
- `1_overview`: `falls_pct_of_episodes` and `hardware_limit_stops_pct_of_episodes` falling toward 0, `episode_seconds` rising to 20, `speed_error_mps` falling, `top_speed_command_mps` climbing (speed curriculum).
- `2_commands`: error per command axis (forward, sideways, turn rate), heading error and path drift, all shrinking.
- `3_hardware`: torque, joint speed and landing force staying well under 1 (x stall, x servo max) and 2 (x body weight).
- `4_gait`, `5_rewards`, `6_penalties` (grouped sums): for diagnosing *why*. `python train.py ... --log-details` adds every single penalty term under `7_penalty_details`.

## 2. The reward, and why each term is there

The weights live in `RewardWeights` in [config.py](biped_rl/config.py). Positive terms are bounded rewards, negative terms are penalties.

| Goal | Terms | Notes |
|---|---|---|
| **1. Straight-line speed** | `tracking_vel`, `tracking_lat_yaw`, `heading`, `path_offset` | Speed is measured along the direction the robot faced at the start, not its current facing, so walking in circles earns nothing. `heading` rewards holding that direction; `path_offset` penalizes drifting off the line. The policy sees its heading error (IMU yaw − start yaw). The speed is a *commanded* one (random per episode in `speed_range`), not "as fast as possible": maximizing speed directly makes policies dive forward and fall. |
| **2. Don't fall** | `alive`, `termination`, `orientation`, `ang_vel_xy`, `lin_vel_z` | A fall is any non-foot part touching the ground or >53° tilt. It ends the episode with a one-off penalty. |
| **3. Smooth / safe for parts** | `action_rate`, `action_acc`, `joint_acc`, `servo_load`, `servo_saturation`, `impact`, `foot_slip`, `joint_limits`, `feet_distance`, `energy` | Plus two hard limits that aren't rewards: an action low-pass filter (`action_filter`) and the servo model itself, which can't exceed real torque or speed. |
| **4. Running / flight** | `gait_contact`, `feet_clearance`, `flight` | A gait clock tells each foot when to be in the air. `swing_ratio` > 0.5 builds a flight phase into every step. |

Terms I added beyond your list, and why:
- **Servo protection (`servo_load`, `servo_saturation`)**: hobby servos strip gears and overheat when held near stall. This keeps the gait inside their comfortable range.
- **`impact`**: penalizes landings harder than 2× body weight. Hard landings are what break brackets and gears, especially once flight phases appear.
- **`foot_slip`**: a sliding stance foot means the sim is exploiting friction, which won't transfer to your floor.
- **`feet_clearance`**: lifts swing feet 2.5 cm. Without it, small robots learn to shuffle, which works in sim but trips on any real carpet edge.
- **`feet_distance`, `joint_limits`**: keep the legs from colliding and the servos off their end stops.

### About "maximize the time both legs are off the ground"

Rewarding flight time directly tends to produce hopping in place. A more reliable route is a curriculum:

1. Learn a robust walk first (`swing_ratio` 0.45, the default).
2. Fine-tune from it with a flight phase and a higher speed:
   ```bash
   python train.py --run-name run_v1 --resume runs/walk_v1 --swing-ratio 0.6 --max-speed 0.5 --timesteps 20000000
   ```
   `swing_ratio` 0.6 with a 0.8 s period = 80 ms of flight per step. Optionally also set `flight` to a small positive weight (e.g. 0.3); it only pays when moving at ≥ 50% of the commanded speed.

Physics check: with 3.4 Nm, 5.8 rad/s servos and 1.75 kg, a real flight phase is near the limit of what the servos can deliver. Watch `sat%` and `impact×BW` in `evaluate.py`. If they climb, the running gait would be hard on the hardware.

## 3. Robustness (sim-to-real)

Everything below is randomized every episode (`RandomizationConfig`):
- **Servos**: one battery-voltage draw scales torque and speed together (−15%/+13%: 3S LiPo from full to near-empty, rated at 11.1 V), ±5% per servo, stiffness, damping, 0–1° gear slack, ±1.7° zero-calibration error, 0.24° position resolution
- **Delay**: 5–59 ms from reading the sensors to the servos moving (sub-step latency plus 0–2 whole control steps), and IMU/joint readings up to one step old. Reading 8 bus servos over USB likely takes 20–30 ms: measure it on the robot and narrow this range
- **Body**: 1.65–2.25 kg (each part ±10%, the whole robot up to +15%, the torso up to +25% more: a heavy, high torso is the hard case), torso center of mass ±1.5 cm, reflected rotor inertia, gearbox friction
- **IMU**: a tilt bias of up to ±2.5° and a gyro offset of up to ±0.02 rad/s that stay fixed through the episode (BNO055 accelerometer offset and mounting), plus noise
- **Environment**: floor friction 0.5–1.2, slopes up to 3°, random shoves every 2–4 s
- **Sensors**: noise on the IMU and joint readings

Runs saved before a setting existed load with it off (`LEGACY` in config.py), so old runs evaluate exactly as they were trained.

The servo itself is simulated, not idealized: PD control + linear torque–speed curve + latency.

**The policy only sees what the robot can measure**: gravity direction and angular rate (IMU), servo angles and speeds if available, the last 3 frames of these, its previous actions, the speed command and the gait clock. True body velocity, contacts and friction go only to the critic.

**Control rate vs. the servo bus.** The policy runs at 50 Hz (20 ms per cycle). Within each cycle the Pi must read 8 servo positions and send 8 commands through the debug board, at 115200 baud, one servo at a time. Time that loop on the real robot before deploying. If it takes more than ~19 ms, raise `latency` in `RandomizationConfig` to match, or lower `control_hz`, and retrain.

Deploying on the robot, at 50 Hz:
1. Build the actor observation exactly as `_sensor_frame()` + `_get_obs()` do (same order and scaling, 3-frame history). Fill the privileged part with zeros.
2. Normalize with the saved `vecnormalize.pkl` statistics, run the policy, and clip the action to [−1, 1].
3. Apply the same filter, `filtered = 0.5·action + 0.5·filtered`, and send `servo_angle = home + 0.5·filtered`, clipped to the joint limits.

## 4. Bring in your model (Onshape)

1. In Onshape, add mate connectors named `dof_<joint>` on each servo axis, give every part a material, and keep wires in the CAD but ignored in the export (see below).
2. `pip install onshape-to-robot`. Put your API keys (from https://dev-portal.onshape.com/keys) in `.env`, and never share or commit that file.
3. Create `my_robot/config.json`:
   ```json
   {
       "url": "<Onshape assembly URL>",
       "output_format": "mujoco",
       "ignore": {"wire*": "all", "cable*": "all"}
   }
   ```
   then run `onshape-to-robot my_robot`. Ignored parts keep their mass, but their geometry is dropped.
4. Add a `home` keyframe (slightly crouched, feet flat) and replace foot collision meshes with boxes, like in `simple_biped.xml`.
5. `python scripts/inspect_model.py my_robot/robot.xml --view`, and compare the masses with the real robot on a scale.
6. `python train.py --run-name mine_v1 --model my_robot/robot.xml --feet <left_foot_link> <right_foot_link>`

Then set the servo numbers in `RobotConfig`: `kp`/`kd` from your servo's internal controller if you know them, and `no_load_speed` from the datasheet (seconds per 60° → rad/s = 1.047 / s).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Stands still, never walks | Standing is a local optimum: raise `tracking_vel`, or lower `alive`/smoothness weights for the first run, then fine-tune with `--resume` |
| Falls a lot in `--robust` but not nominal | Train longer, or reduce `push_velocity` / `servo_strength` range at first |
| Jittery | Raise `action_rate`/`action_acc`, lower `action_filter` (e.g. 0.3) |
| Shuffles feet | Raise `feet_clearance` and `gait_contact` |
| High `sat%` | Gait demands more than the servos can give: lower speed, raise `servo_saturation` |
| Model explodes | zero or tiny masses/inertias, overlapping collision shapes |
| `launch_passive` error on macOS | run with `mjpython` instead of `python` |
