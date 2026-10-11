# Case Studies: Small / Low-Cost RL-Controlled Bipeds on Hardware (Robustness & Fall Handling)

Context: user's robot is 8-DOF legs-only (hip roll, hip pitch, knee, ankle per leg), 43 cm, 1.75 kg, Hiwonder HTD-35H bus servos, BNO055 IMU, Raspberry Pi 4, PPO in MuJoCo, 50 Hz. Research date: 2026-10-10. Sources fetched directly from arXiv HTML, GitHub READMEs, and raw config files (15 fetches). Where a fetched source did not cover a detail, it is listed under Gaps rather than filled in from memory.

## Q1: Per-project architecture, training setup, robustness tricks, and results

### Takeaway
The most transferable projects for this hobby robot are Open Duck Mini v2 (about 42 cm, hobby servos, Pi Zero 2W, MuJoCo Playground, 50 Hz, under $400) and ToddlerBot (56 cm, 3.4 kg, Dynamixel, MJX/Brax PPO, 50 Hz). Both put most of their effort into actuator system identification and a single walking policy with imitation reward. Neither ships a learned fall-recovery policy. The larger labs (Disney BDX, DeepMind OP3, Cassie) add structured disturbance curricula, action and observation history, and action filtering.

### Cited Findings

**Open Duck Mini v2 (apirrone + collaborators, sponsored by HuggingFace/Pollen; active Jan–Apr 2025)**
- About 42 cm tall with legs extended. BOM under $400. Onboard computer is a Raspberry Pi Zero 2W. Training moved to MuJoCo Playground (separate Open_Duck_Playground repo). Actuators were identified with Rhoban's BAM tool. Reference motions come from a separate Open_Duck_reference_motion_generator repo. The README notes "Big progress on sim2real" (07/02/2025) and links a "real_back_push" test video. — [Open_Duck_Mini README](https://github.com/apirrone/Open_Duck_Mini)
- Training config: `ctrl_dt=0.02` (50 Hz), `sim_dt=0.002`, episode length 1000, `action_scale=0.25`, and `max_motor_velocity=5.24` rad/s. With `USE_MOTOR_SPEED_LIMITS=True`, each target is clipped to ±max_vel·dt of the previous target, which acts as a rate limiter. `soft_joint_pos_limit_factor=0.95`. — [Open_Duck_Playground joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- Push disturbances: enabled. Each push is a random-direction x/y base-velocity impulse of 0.1–1.0 (velocity units), every 5–10 s. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- Observation noise and delay:
  - Joint position noise is 0.03 rad (hip), 0.05 rad (knee), and 0.08 rad (ankle). Joint velocity noise is 2.5 rad/s. Gyro noise is 0.1 and accelerometer noise 0.05.
  - Action delay and IMU delay are each 0–2 control steps (randint with an exclusive upper bound of 3).
  - Backlash is modeled in the actuator, and the backlash joint positions are added to the observed joint angles. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- Observation vector:
  - Raw gyro and accelerometer (no projected gravity; that term is commented out), commands, joint positions minus the default pose, and joint velocities scaled by 0.05.
  - The last three actions (an explicit short action history), current motor targets, two foot contacts, and imitation phase as a cos/sin pair. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- Rewards: tracking_lin_vel 2.5, tracking_ang_vel 6.0, torques -1e-3, action_rate -0.5, stand_still -0.2, alive 20.0, imitation 1.0. A zero ("stand") command is sampled with probability 0.1, so the one walking policy also learns to stand still. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- Reset randomization: base x/y ±0.05 m, full random yaw, initial joint positions scaled by U(0.5, 1.5), and base velocities ±0.05. The training termination for a fall is the gravity z-component going negative (torso past horizontal). — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- The example training run uses the task `flat_terrain_backlash` for 300M timesteps. Flat and rough terrain scenes exist. The imitation reward needs `polynomial_coefficients.pkl` from the reference-motion generator. — [Open_Duck_Playground README](https://github.com/apirrone/Open_Duck_Playground)
- Runtime (Pi Zero 2W):
  - The ONNX policy (`BEST_WALK_ONNX_2.onnx`) runs at `control_freq=50`, and joystick commands are polled at 20 Hz.
  - Position-only gains are kp=30, kd=0, with head joints lowered to kp=8.
  - An optional `LowPassActionFilter` is used only if a cutoff frequency is given (default None, so it is off by default) and only after a 1 s warmup. The 5.24 rad/s velocity clip is commented out at runtime.
  - There is no fall detection and no standing controller. Pressing "A" pauses the policy. — [v2_rl_walk_mujoco.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/scripts/v2_rl_walk_mujoco.py)
- Latency tweak: a udev rule sets the FTDI USB-serial latency timer to 1 ms. The IMU frame is checked with `imu_server.py`/`imu_client.py`, and an "upside down" IMU flag sits in `duck_config.json`. Joint offsets are found with `find_soft_offsets.py`. — [Open_Duck_Mini_Runtime README](https://github.com/apirrone/Open_Duck_Mini_Runtime)

**ToddlerBot (Stanford, arXiv 2502.00893, Feb 2025)**
- Hardware: 0.56 m, 3.4 kg, and 30 active DoF (6 per leg). Dynamixel XC330/XC430/XM430 servos, with XM430 on the knee and ankle pitch. Jetson Orin NX 16GB, chest IMU, cost under $6,000. Full-state motor feedback runs at 50 Hz. — [ToddlerBot paper](https://arxiv.org/html/2502.00893)
- Training: MJX + Brax PPO, 3e8 steps, 1024 environments, episode length 1000.
  - Unroll length 20, batch 256, 4 minibatches, discount γ=0.97, learning rate 1e-4, clip 0.2. Policy and value networks are (512, 256, 128) MLPs.
  - The policy runs at 50 Hz on the Jetson CPU.
  - Observations: phase, commands, joint offset from neutral, joint velocity, previous action, torso orientation, torso angular velocity.
  - Reward combines imitation, regularization, and survival (weight 10.0). — [ToddlerBot paper](https://arxiv.org/html/2502.00893)
- Motor system identification:
  - Damping and friction come from a constant-speed backdrive test, and armature from a spin-down test.
  - The actuation model is fit to a chirp signal (9 parameters, including torque limits and brake torque). Average simulated tracking error is 1.3°.
  - Example (XC330): damping 0.0036 Nms/rad, armature 0.0040 kg·m², friction loss 0.036 Nm. — [ToddlerBot paper](https://arxiv.org/html/2502.00893)
- Hardware results:
  - Walking position error was 0.133±0.018 m on hardware vs 0.082 m in simulation. Linear-velocity error was 0.032 vs 0.016 m/s.
  - The longest walking streak without falling was 19 min. Motor heating eventually pushed the robot out of the training distribution and falls became more frequent. — [ToddlerBot paper](https://arxiv.org/html/2502.00893)

**Disney Research BDX ("Design and Control of a Bipedal Robotic Character", arXiv 2501.05204; RSS 2024)**
- Hardware: 15.4 kg, 0.66 m, 5 DoF per leg plus 4 neck DoF. There is no ankle-roll actuator; passive roll comes from rounded urethane soles. Quasi-direct-drive actuators (34 N·m and 24 N·m peak). Onboard PC with a 600 Hz microcontroller comms board. — [BDX paper](https://arxiv.org/html/2501.05204)
- Control path:
  - The policy runs at 50 Hz and actuators at 600 Hz, bridged by a first-order hold (linear interpolation between consecutive actions) followed by a 37.5 Hz low-pass filter.
  - Actions are PD setpoints, clipped to a maximum deviation from the measured joint position.
  - Observations include the previous two actions. The critic gets noise-free state plus friction (asymmetric actor-critic).
  - Networks: 3×512 ELU layers. PPO ran 100k iterations (about 22 days on an RTX 4090); about 1,500 iterations (~30 min) already shows nominal behavior. — [BDX paper](https://arxiv.org/html/2501.05204)
- Domain randomization:
  - Actuator model parameters within experimentally observed ranges, state noise, mass and friction, reflected inertia up to 20%, encoder offset ±0.02 rad per episode, and terrain (walking policy only).
  - The actuator model includes tanh static friction plus viscous friction, velocity-dependent torque limits, backlash randomized per episode, and position noise σq = σq,0 + σq,1·|q̇|. — [BDX paper](https://arxiv.org/html/2501.05204)
- Disturbance schedule (Table V):
  - Short/small (hips, feet): up to 5 N and 0.25 N·m, on for 0.25–2.0 s, off for 1–3 s.
  - Long/small (pelvis, head): same magnitudes, on for 2–10 s, off for 1–3 s.
  - Short/large (pelvis): 90–150 N in XY plus up to 15 N·m, on for 0.1 s, off for 12–15 s.
  - Disturbances ramp in along a linear curriculum over the first 1,500 iterations. — [BDX paper](https://arxiv.org/html/2501.05204)
- Results: the robot stabilizes under pushes, and the policy "deviates from the reference trajectory and contact schedule to recover." Joint tracking error was 0.035 rad standing and 0.123 rad walking. About 10 h of deployment across up to 3 robots "without a single fall." — [BDX paper](https://arxiv.org/html/2501.05204)

**DeepMind OP3 soccer (Haarnoja et al., arXiv 2304.13653; Science Robotics 2024)**
- Hardware: Robotis OP3, 51 cm, 3.5 kg, 20 Dynamixel XM430-350-R servos, Intel i3 NUC with CPU-only inference. Hardware modifications: 3D-printed torso bumpers, replaced forearms, modified hips. — [OP3 paper](https://arxiv.org/html/2304.13653)
- Control:
  - 40 Hz nominal (25 ms step), which "the agent often missed."
  - Joint position actions are clipped to hand-chosen ranges and smoothed with an exponential filter, u_t = 0.8·u_{t-1} + 0.2·a_t. Servos run P-only position control.
  - Proprioception (joint positions, linear acceleration, angular velocity, gravity direction, filter state) is stacked over 5 timesteps.
  - Trained with Distributional MPO (off-policy, not PPO). — [OP3 paper](https://arxiv.org/html/2304.13653)
- Domain randomization (per episode):
  - Floor friction 0.5–1.0, joint offsets ±2.9°, IMU orientation ±2°, IMU position ±5 mm.
  - Up to 0.5 kg of extra mass at a random torso location.
  - Observation delay 10–50 ms. — [OP3 paper](https://arxiv.org/html/2304.13653)
- Perturbations: an impulse of 5–15 (units as stated) for 0.05–0.15 s at a random torso point, every 1–3 s. Agents trained without randomization and perturbations fell "with every one or two steps" on hardware. — [OP3 paper](https://arxiv.org/html/2304.13653)
- Hardware-protection rewards:
  - A penalty on the time-integral of torque above 5 Nm on targeted joints, mainly to prevent knee gear breakage.
  - An upright reward within 11.5°, added because forward lean caused falls on hardware.
  - Limited action ranges to reduce self-collision. — [OP3 paper](https://arxiv.org/html/2304.13653)
- Learned vs scripted on hardware: walking 0.57 vs 0.20 m/s (+181%), turning 2.85 vs 0.71 rad/s, get-up 0.93 vs 2.52 s (-63%). The get-up-and-shoot set piece scored 29/50 on hardware vs 35/50 in simulation. — [OP3 paper](https://arxiv.org/html/2304.13653)

**Berkeley Humanoid Lite (arXiv 2504.17249, Apr 2025)**
- Hardware: 16 kg, 0.8 m, 3D-printed cycloidal actuators, Intel N95 mini PC, and a cellphone-grade IMU (BNO085). Actuators and IMU run at 250 Hz over 4 CAN buses. Cost under $5k. — [BH-Lite paper](https://arxiv.org/html/2504.17249)
- Policy:
  - PPO trained in Isaac Gym, MLP running onboard at 25 Hz.
  - Observations: angular velocity, projected gravity, joint positions/velocities, command, previous action.
  - Direct sim-to-real with no state estimator. The policy used only about 30% of the actuator torque limit. — [BH-Lite paper](https://arxiv.org/html/2504.17249)

**Booster T1 / Booster Gym (arXiv 2506.15132, Jun 2025)**
- Training: Isaac Gym PPO with an asymmetric actor-critic, policy at 50 Hz, 30 s episodes.
  - Actor observation (47 dims): commands, gait-cycle phase, gravity vector, angular velocity, 12 joint positions, 12 joint velocities, previous action.
  - Critic adds mass, CoM, base linear velocity, base height, and push force/torque.
  - Actions are offsets from the default pose (q_des = q0 + a) into PD control. — [Booster Gym paper](https://arxiv.org/html/2506.15132)
- Randomization covers mass/CoM, joint stiffness, damping, and friction, contact friction, compliance, and restitution, terrain, and kicks/pushes. The only numeric range given is sensor-to-actuator latency of 0–20 ms. MuJoCo sim2sim is used as a proxy before hardware. — [Booster Gym paper](https://arxiv.org/html/2506.15132)
- Hardware: a 10 kg ball dropped from 1 m onto the robot while stepping in place; it "regains stability within a few steps." It walked on a 10° slope and on grass, stone, soil, and asphalt. Inference takes under 1 ms, with 9–12 ms round-trip latency. — [Booster Gym paper](https://arxiv.org/html/2506.15132)

**Unitree H1/G1 (unitree_rl_gym)**
- Pipeline: Train → Play → Sim2Sim (MuJoCo) → Sim2Real. Supports Go2, H1, H1_2, and G1. Play exports `policy_1.pt` (MLP) or `policy_lstm_1.pt`. — [unitree_rl_gym README](https://github.com/unitreerobotics/unitree_rl_gym)
- G1 config:
  - Domain randomization: friction 0.1–1.25, added base mass -1 to +3 kg, `push_robots=True`, push every 5 s, `max_push_vel_xy=1.5` m/s.
  - Control: PD stiffness 100 (hip), 150 (knee), 40 (ankle); damping 2/4/2. Action scale 0.25, decimation 4.
  - Termination on pelvis contact.
  - Rewards: alive 0.15, orientation -1.0, base_height -10, action_rate -0.01, feet_swing_height -20, contact 0.18, hip_pos -1.0.
  - The policy is an LSTM (ActorCriticRecurrent, 64 hidden units, 1 layer). Observations are 47 dims, privileged 50 dims, 12 actions. — [g1_config.py](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py)

**Cassie (Li et al., "RL for Versatile, Dynamic, and Robust Bipedal Locomotion Control", arXiv 2401.16889, IJRR 2024)**
- Architecture:
  - A dual-history policy: a long ~2 s I/O history (66 steps) encoded by a 1D CNN, plus a short 4-step history fed directly to the MLP (2×512 tanh).
  - The policy runs at 33 Hz with 2 kHz PD control and outputs 10 low-pass-filtered motor targets.
  - One skill-specific policy each for walking, running, and jumping. — [Cassie paper](https://arxiv.org/html/2401.16889)
- Domain randomization (Table IV):
  - Friction 0.3–3.0, joint damping 0.3–4.0, link mass 0.5–1.5×, inertia 0.7–1.3×.
  - Pelvis CoM ±0.1 m, other link CoM ±0.05 m, PD gains 0.7–1.3× per joint.
  - Motor position noise ±0.002 rad, communication delay 0–25 ms.
  - Training perturbations: ±20 N and ±5 Nm on the pelvis, over intervals of 0.1–3.0 s (walking) or 0.1–1.0 s (running). Perturbations are not used for jumping or transition-to-standing; the authors note perturbations "may not enhance robustness" for highly dynamic skills. — [Cassie paper](https://arxiv.org/html/2401.16889)
- Training curriculum: (1) single task, (2) task randomization (commands resampled every 1–15 s), (3) dynamics randomization. The authors identify task randomization as a separate source of robustness. — [Cassie paper](https://arxiv.org/html/2401.16889)

### Inferences
- The closest analog to the user's robot (about the same height and mass class, hobby servos, small single-board computer, MuJoCo, 50 Hz) is Open Duck Mini v2. Its Playground config is a concrete template for push magnitudes, noise, delays, rate limiting, and a 3-action history.
- The user's Pi 4 is considerably more capable than the Pi Zero 2W that Open Duck Mini runs a 50 Hz ONNX MLP on, so compute should not be the bottleneck.
- The HTD-35H bus servos most closely resemble Open Duck's servo class: position-only control with a P gain and no torque mode. Sysid of backlash, friction, and velocity limits (BAM-style, or ToddlerBot's backdrive and spin-down tests) is the highest-leverage step reported in these projects.
- The user's 4-DOF leg with no hip yaw cannot turn in place like these robots. Their command ranges for yaw rate are not directly applicable.

### Gaps
- The fetched Open Duck pages did not confirm the exact servo model (believed to be Feetech STS3215) or IMU chip. Not reported.
- ToddlerBot's domain randomization ranges and push settings were not in the paper text fetched; they may be in the repo configs (hshi74/toddlerbot), which were not fetched.
- Robotis OP3/Darwin-OP RL work beyond the DeepMind paper (e.g., other academic papers) was not researched due to budget.
- Original Berkeley Humanoid (2024) and Booster K1 were not fetched. BH-Lite did not report domain randomization, push, or fall details.
- Units for the OP3 perturbation impulse were reported as "5 to 15 Nm" by the extraction; the original wording should be checked before citing it as a force.

## Q2: One policy or several? Switching? Reliance on walking-policy robustness? Fall handling?

### Takeaway
Almost every project relies on a single walking policy that also handles standing (zero command) and push recovery. A separate learned get-up skill appears only in DeepMind OP3, which trained it separately and then distilled it into one policy. Disney uses separate standing, walking, and episodic policies with hand-coded switching at double support. No hobby-scale project (Open Duck, ToddlerBot) ships automatic fall recovery.

### Cited Findings
- **BDX (Disney):**
  - Uses one perpetual standing policy, one periodic walking policy, and several episodic animation policies. They switch on the fly; walk→stand waits for the next double-support phase.
  - Continuity comes from each policy observing the previous two actions.
  - There is no fall-recovery behavior. Training terminates on head/torso ground contact. A runtime "motion stop" mode forces standing and then freezes joints with high gains after 0.5 s. — [BDX paper](https://arxiv.org/html/2501.05204)
- **DeepMind OP3:**
  - The get-up skill is learned (not scripted) by tracking target poses interpolated between 3 key poses extracted from Robotis's scripted get-up. Targets are resampled about every 1.5 s.
  - The soccer and get-up skills are distilled into one policy via KL regularization. Stage-2 episodes start lying on the front, lying on the back, or standing with equal probability, so the single final policy decides by itself when to get up.
  - The final policy also reacts to prevent falls. — [OP3 paper](https://arxiv.org/html/2304.13653)
- **Cassie:**
  - Standing-only policies (with or without perturbation training) lose balance when leaned beyond the support region.
  - The versatile walking+standing policy, even without perturbation training, steps to recover and returns to standing. The running policy side-steps, and the jumping policy makes corrective hops.
  - There is no dedicated fall-recovery routine; falls are terminations (pelvis height under 0.55 m or tarsus ground contact). — [Cassie paper](https://arxiv.org/html/2401.16889)
- **Open Duck Mini:**
  - A single walking policy is trained with 10% zero ("stand") commands. The runtime has no fall detection or standing controller, only a pause button. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py), [v2_rl_walk_mujoco.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/scripts/v2_rl_walk_mujoco.py)
- **ToddlerBot:**
  - There is no automatic fall recovery in the paper. Open-loop keyframe animations (push-ups, pull-ups) exist.
  - Fall handling is mechanical: the robot survived up to 7 falls before breaking, and repair took about 21 min of printing plus 14 min of assembly, including zero-point recalibration. — [ToddlerBot paper](https://arxiv.org/html/2502.00893)
- **Booster Gym:** covers push recovery and stepping but not fall recovery or get-up. — [Booster Gym paper](https://arxiv.org/html/2506.15132)
- **Unitree G1:** the single walking (LSTM) policy terminates on pelvis contact, and the README does not mention fall handling. — [g1_config.py](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py), [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)

### Inferences
- For an 8-DOF robot with no arms, a learned get-up like OP3's (which uses arms) may not be physically feasible. The realistic copyable pattern is:
  - one walk+stand policy trained with stand commands and pushes (Open Duck, Cassie);
  - an IMU-tilt-triggered "safe fall" or freeze mode (BDX motion-stop);
  - durable or replaceable printed parts (ToddlerBot, OP3 bumpers).
- Cassie's finding suggests training standing inside the walking policy, not as a separate standing-only policy. A walking-capable policy can step to recover, whereas a standing-only policy cannot.
- If separate policies are used, BDX's recipe of switching only at double support and feeding previous actions into every policy is the documented way to avoid transition jerks.

### Gaps
- The research found no hobby-scale biped with legs only (no arms) and a learned or scripted get-up on hardware.
- ToddlerBot's GitHub may contain a get-up or crawl policy added after the paper; it was not verified.

## Q3: Push/randomization settings and hardware push-recovery results

### Takeaway
Training pushes fall into two styles. Velocity impulses: Open Duck uses 0.1–1.0 m/s every 5–10 s, and Unitree G1 up to 1.5 m/s every 5 s. Force/torque disturbances: BDX layers small long-duration and large short pulses under a curriculum; Cassie uses ±20 N and ±5 Nm; OP3 applies impulses every 1–3 s. Hardware push results are almost all qualitative. The only quantified hardware tests are Booster's 10 kg ball from 1 m and Disney's 10 h with zero falls.

### Cited Findings
- **Open Duck:** pushes of 0.1–1.0 velocity impulse every 5–10 s, action and IMU delays of 0–2 steps, joint noise of 0.03–0.08 rad. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- **Unitree G1:** push every 5 s at up to 1.5 m/s xy, friction 0.1–1.25, mass -1 to +3 kg. — [g1_config.py](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py)
- **BDX:**
  - Pulses at three scales: up to 5 N on hips/feet; up to 5 N for 2–10 s on pelvis/head; 90–150 N for 0.1 s every 12–15 s.
  - The curriculum ramps in over 1,500 iterations, plus an encoder offset of ±0.02 rad. — [BDX paper](https://arxiv.org/html/2501.05204)
- **OP3:**
  - Impulses for 0.05–0.15 s every 1–3 s; friction 0.5–1.0; joint offsets ±2.9°; IMU tilt ±2°; up to 0.5 kg payload; 10–50 ms observation delay.
  - Without randomization and pushes, the robot fell every 1–2 steps on hardware. — [OP3 paper](https://arxiv.org/html/2304.13653)
- **Cassie:**
  - ±20 N and ±5 Nm on the pelvis; delay 0–25 ms; friction 0.3–3.0; PD gains 0.7–1.3×.
  - Hardware push magnitudes were not given in the sections fetched. — [Cassie paper](https://arxiv.org/html/2401.16889)
- **Booster T1:** a 10 kg ball dropped from 1 m was recovered "within a few steps," and the robot stayed stable under a sustained push while walking (magnitude not reported). — [Booster Gym paper](https://arxiv.org/html/2506.15132)

### Inferences
- Scaled to a 1.75 kg robot, a velocity-impulse push (Open Duck style, 0.1–1.0 m/s) is mass-independent and easy to tune.
- BDX's 90–150 N pulses were for a 15.4 kg robot; scaling by mass gives roughly 10–17 N for 1.75 kg. This is an estimate, not a reported value.
- Starting with small pushes and ramping them up (BDX curriculum) is a documented way to avoid hurting early gait learning.

### Gaps
- No small hobby robot reported quantitative push-recovery thresholds (e.g., maximum survivable impulse) on hardware.

## Q4: Deployment on small onboard computers — control rate, observation history, action filtering

### Takeaway
50 Hz is the norm (Open Duck, ToddlerBot, BDX, Booster), with OP3 at 40 Hz, Cassie at 33 Hz, and BH-Lite at 25 Hz. A small MLP runs fine on a Pi Zero 2W via ONNX. Short action histories (2–3 previous actions) or 5-step proprioception stacks are common. Action smoothing is done by an exponential filter (OP3), a low-pass filter (BDX, Cassie), a per-step rate limit (Open Duck training), or an action_rate penalty (everyone).

### Cited Findings
- **Open Duck (Pi Zero 2W):**
  - ONNX policy at 50 Hz with P-only kp=30.
  - The low-pass filter is optional and off by default. The FTDI latency timer is set to 1 ms.
  - Observations include the last 3 actions. The training sim models 0–2 steps of action/IMU delay. — [v2_rl_walk_mujoco.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/scripts/v2_rl_walk_mujoco.py), [Runtime README](https://github.com/apirrone/Open_Duck_Mini_Runtime), [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- **ToddlerBot (Jetson Orin NX):** 50 Hz policy on the CPU; previous action in the observation; no observation history described. — [ToddlerBot paper](https://arxiv.org/html/2502.00893)
- **BDX:** 50 Hz policy, first-order hold to 600 Hz, then a 37.5 Hz low-pass filter; previous 2 actions in the observation; setpoints clipped relative to measured position. — [BDX paper](https://arxiv.org/html/2501.05204)
- **OP3 (i3 NUC, CPU):** 40 Hz, often missing deadlines; exponential filter 0.8/0.2; 5-step proprioception stack with observation-delay randomization of 10–50 ms. — [OP3 paper](https://arxiv.org/html/2304.13653)
- **Cassie:** 33 Hz; 4-step short history plus a 66-step (~2 s) CNN-encoded long history; low-pass-filtered actions. — [Cassie paper](https://arxiv.org/html/2401.16889)
- **BH-Lite (Intel N95):** MLP at 25 Hz, with actuator/IMU communication at 250 Hz. — [BH-Lite paper](https://arxiv.org/html/2504.17249)
- **Booster T1:** 50 Hz, inference under 1 ms, 9–12 ms round-trip latency, 0–20 ms latency randomization. — [Booster Gym paper](https://arxiv.org/html/2506.15132)
- **Unitree G1:** LSTM policy (64 hidden units) instead of explicit history; action scale 0.25; decimation 4. — [g1_config.py](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py)

### Inferences
- For a Pi 4 driving half-duplex bus servos at 50 Hz, the copyable pattern is:
  - randomize 0–2 steps of action and IMU delay;
  - include 2–3 previous actions in the observation;
  - use action_scale around 0.25 around a default pose;
  - add an action_rate penalty and/or a per-step target rate limit matched to the servo's real maximum speed.
- A filter used at deploy time must also be present in training (OP3 and BDX filter in both). Open Duck's runtime filter is off by default, presumably for this reason.
- Bus servo read latency (reading 8 positions per cycle) may dominate the 20 ms budget. Open Duck's FTDI latency tweak is directly applicable if the HTD-35H bus adapter is FTDI-based (unverified).

### Gaps
- The research found no published per-cycle timing breakdown for Pi-based bus-servo bipeds.
- Whether the BNO055's onboard fusion (which Open Duck may or may not use) adds latency compared with raw gyro/accelerometer was not determined from the sources.
