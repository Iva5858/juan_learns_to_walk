# Sim-to-real for RL legged robots on hobby or serial bus servos (balance and fall robustness)

Context: the user's robot has 8 Hiwonder HTD-35H bus servos (35 kg·cm, 0.18 s/60° at 11.1 V, UART half-duplex at 115200 baud through a debug board), a BNO055 IMU, a Raspberry Pi 4, PPO in MuJoCo, and a 50 Hz policy. The current sim models each servo as PD plus a linear torque-speed curve plus 5-19 ms latency, and randomizes kp, kd, armature, friction loss and strength.

Evidence-quality legend: **[primary]** means a paper, datasheet or repo code. **[anecdotal]** means a forum or user report. **[inference]** means my own arithmetic or reasoning, not stated by any source.

---

## 1. How comparable projects model hobby or Dynamixel servos, and what mattered most

### Takeaway
Every successful small-servo biped (Open Duck Mini, ToddlerBot, Disney BDX, Pollen microduck) says actuator fidelity matters most. Each identifies the real servo on a bench (friction loss, damping or back-EMF, armature, a velocity-dependent torque limit, and an effective kp) instead of using datasheet values. Each also adds delay randomization and, increasingly, explicit backlash. The user's structure (PD, torque-speed curve and latency) matches the field. The gaps are probably (a) parameters that are not identified on the real HTD-35H, (b) no backlash, and (c) a latency range that may be too narrow (see section 3).

### Cited Findings
**Open Duck Mini (Feetech STS3215 bus servos, Pi Zero 2W, the closest analogue to the user's robot)**
- The project's sim2real guide calls motor modeling the most important step. It identifies motors with Rhoban's BAM, exports them with `bam.to_mujoco`, and sets `damping`, `kp`, `frictionloss`, `armature` and `forcerange` in the MJCF. It warns that if the simulated motors behave differently from the real ones, the policy won't work. It describes the STS3215 as cheap, hard to model, and not very powerful. The BAM parameters were identified at 7.4 V. [primary] — [Open_Duck_Mini docs/sim2real.md](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini/v2/docs/sim2real.md)
- The project used Rhoban's BAM for actuator identification and runs its ONNX policy on a Raspberry Pi Zero 2W. [primary] — [Open_Duck_Mini README](https://github.com/apirrone/Open_Duck_Mini)
- Open Duck Playground domain randomization (`randomize.py`) [primary] — [randomize.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/common/randomize.py):
  - Floor friction U(0.5, 1.0). A code comment says it used to be 0.4-1.0.
  - Joint frictionloss ×U(0.9, 1.1), armature ×U(1.0, 1.05), kp ×U(0.9, 1.1). The bias term is rescaled to match kp.
  - Link masses ×U(0.9, 1.1), torso mass +U(-0.1, 0.1) kg, torso CoM ±0.05 m per axis.
  - Default joint position offset ±0.03 rad, which models calibration error.
  - Damping and kd are **not** randomized. Backlash joints are excluded from the friction and armature randomization.
- Open Duck Playground `joystick.py` [primary] — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py):
  - Timing: ctrl_dt 0.02 s (50 Hz), sim_dt 0.002 s.
  - **Action delay 0-3 control steps (0-60 ms)** and **IMU delay 0-3 steps (0-60 ms)**, each sampled from a history buffer.
  - Uniform observation noise: gyro 0.1, accelerometer 0.05, gravity 0.1, hip position 0.03 rad, knee 0.05 rad, ankle 0.08 rad, joint velocity 2.5 rad/s.
  - Pushes every 5-10 s with magnitude 0.1-1.0 m/s, added to the base xy velocity.
  - `max_motor_velocity` = 5.24 rad/s. Each step, targets are rate-limited to 0.105 rad from the previous target.
  - action_scale 0.25.
  - The observation includes the last 3 actions and the current motor targets.
  - A `LowPassActionFilter` at 37.5 Hz is commented out.
  - The README's "current win" task is `flat_terrain_backlash`. [primary] — [Open_Duck_Playground README](https://github.com/apirrone/Open_Duck_Playground)
- Pollen Robotics microduck_rl (a related small biped, about 800 g, Dynamixel XL330):
  - It models a backlash of ±1° (2° total) as an unactuated passive hinge in series with each of 14 servo joints. Because the real encoder sits on the output side of the gear play, the PD emulation and the joint observations read through the backlash.
  - It models the servo with a voltage control law, back-EMF and friction, not an ideal PD.
  - It randomizes battery voltage, voltage sag under load, command delay and friction.
  - The authors argue actuator fidelity is most of the sim2real gap at this scale.
  - [primary repo, seen via search snippet and mirror] — [microduck_rl mirror](https://gitee.com/ncnynl/microduck_rl), [gittrend summary](https://gittrend.io/repo/pollen-robotics/microduck_rl)

**ToddlerBot (Dynamixel, Jetson Orin NX, RL walking at 50 Hz on CPU)**
- Actuation model: τ = kp(q̂ − q) − (kd_min + kd)·q̇. Resistance τ_r = τ_f + d·q̇. The torque limit is constant up to a knee velocity, then tapers linearly to 0 at q̇_max. A separate, larger braking torque is modeled. The passive/active ratio of about 3 implies roughly 58% gearbox efficiency. [primary] — [ToddlerBot arXiv 2502.00893](https://arxiv.org/html/2502.00893v4)
- Identification:
  - Position-tracking data comes from a commanded **chirp**.
  - Friction loss and damping come from back-driving at constant RPM against a torque sensor (intercept = friction, slope = damping).
  - Armature comes from a spin-down test: I = 2E/ω².
  - Nine parameters were jointly optimized, giving 1.3° mean tracking error.
  - Dynamixel unitless kp maps to sim kp by a factor of about 150.
  - Example XC330 values: friction loss 0.036 Nm, armature 0.0040 kg·m², τ_max 0.76 Nm, τ_brake 1.75 Nm, q̇_max 6.5 rad/s, kd_min 0.384.
  - [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4)
- Backlash is about 0.25°, and the encoder has 4096 counts per revolution. The bus is 2 Mbaud with full-state feedback for 30 motors at 50 Hz. [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4)
- Fall-relevant lessons:
  - The longest run without a fall was 19 minutes. **Rising motor temperature pushed the robot out of the training distribution and caused more falls.**
  - The authors say the actuator model ignores temperature and is less accurate near the performance limit, which is exactly where push recovery happens.
  - The robot survived up to 7 falls before breaking. Repairs took 21 minutes of printing plus 14 minutes of assembly.
  - [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4)

**Disney BDX ("Design and Control of a Bipedal Robotic Character", Grandia et al. 2025)**
- Actuator model:
  - PD with an encoder offset, q̃ = q + offset.
  - Friction τ_f = μs·tanh(q̇/q̇s) + μd·q̇.
  - Velocity-dependent torque limits (constant, then a linear ramp to 0 at q̇_max).
  - **Backlash in the measurement: q_meas includes 0.5·b·tanh(τm/τb).**
  - Speed-dependent position noise σq = σq0 + σq1·|q̇|.
  - Armature randomized up to 20%.
  - Parameters were identified on a test bench, including for a Dynamixel XH540-V150.
  - [primary] — [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- Randomization:
  - Encoder offset ±0.02 rad per episode.
  - Backlash per episode: 0.005-0.015 rad (Unitree A1) and 0.002-0.005 rad (Go1, Dynamixel).
  - Pelvis pushes of 90-150 N for 0.1 s every 12-15 s, phased in over the first 1500 iterations.
  - [primary] — [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- No explicit latency model is used. The policy runs at 50 Hz and the actuators at 600 Hz, bridged by a first-order hold plus a 37.5 Hz low-pass filter.
- The low-level control is "identically implemented" in the training and runtime environments.
- The policy sees its previous 2 actions, and smoothness is rewarded.
- Public deployments reportedly ran about 10 h without a fall.
- [primary] — [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)

**Rhoban BAM (Best Actuator Models)**
- BAM identifies friction models from pendulum test-bench trajectories. Its 6 models range from Coulomb-viscous up to more complex load-dependent ones, and they are integrated into MuJoCo and mjlab.
- It ships parameters for MX-64, MX-106, XL-320 and XL-330. It also covers the STS3215 used by Open Duck.
- The authors report more than 50% less simulation error than Coulomb-viscous on 2R arms.
- [primary] — [BAM GitHub](https://github.com/rimim/bam), [BAM docs](https://bam.readthedocs.io/en/latest/usage/actuators.html), [arXiv 2410.08650](https://arxiv.org/abs/2410.08650)

**Actuator networks (Hwangbo et al. 2019, ANYmal)**
- The pipeline is: identify rigid-body parameters, train an "actuator net" on real data, train the policy in sim with that net, then deploy.
- The net exists because the analytical model misses "the actuator dynamics, the delays in control signals introduced by multiple hardware and software layers, the low-level controller dynamics, and compliance/damping at the joints."
- The robot recovered from falls in complex configurations.
- [primary] — [arXiv 1901.08652](https://arxiv.org/abs/1901.08652)
- Recalled from the paper, **not re-verified in this session**:
  - The actuator net is an MLP whose input is the position error and velocity at t, t−0.01 s and t−0.02 s, with 3 hidden layers of 32 softsign units.
  - Policies trained with an ideal actuator model did not transfer well.
  - Treat these specifics as unverified. — [arXiv 1901.08652](https://arxiv.org/abs/1901.08652)

**Older evidence on backlash**
- OpenAI's Rubik's-cube hand randomized action latency in time steps and modeled motor backlash as a custom physics randomization. [primary] — [arXiv 1910.07113](https://arxiv.org/pdf/1910.07113)
- A RoboCup humanoid simulator paper reported that a servo model with backlash behaved very similarly to the real robot's servo. [primary, seen via snippet only] — [DAI-Labor](https://dai-labor.de/?p=7451)

### Inferences
- [inference] HTD-35H servos almost certainly run an internal, closed-firmware P/PD loop on a potentiometer. The 0-1000 command range over 240° gives about 0.24° per count ([openelab listing](https://openelab.io/products/htd-35h-bus-servo)). The LX/Hiwonder protocol does not obviously expose PID gains, so "kp" in sim is an *effective* stiffness that must be fitted from step or chirp data. It cannot be read from a register.
- [inference] Hiwonder `SERVO_MOVE_TIME_WRITE` takes a time parameter alongside the angle ([Purdue-hosted protocol copy](https://engineering.purdue.edu/477grp4/Files/refs/motors_manual.pdf)). If the time is nonzero, the servo interpolates internally, which adds a ramp the sim doesn't model. Send time = 0, or a value equal to the 20 ms control period, and model whatever you choose.
- [inference] Open Duck caps target rate at max_motor_velocity × dt and includes the last 3 actions in the observation. Both are cheap ways to stop the policy from exploiting instantaneous actuation. They are worth copying, and the HTD-35H speed of 0.18 s/60° ≈ 5.8 rad/s no-load is similar to Open Duck's 5.24 rad/s.
- [inference] For fall robustness, backlash plus delay is where hobby servos differ most from ideal PD. Both reduce the phase margin of the ankle and hip balance loop. A policy trained without them tends to learn high-gain corrective motions that oscillate or overshoot on hardware.
- [inference] Two more randomizations are cheap and well supported by BDX and Open Duck: an encoder or calibration offset (±0.02-0.03 rad) and kp ×0.9-1.1. Battery or voltage sag (microduck) matters for 11.1 V LiPo-driven servos, whose torque and speed scale with voltage.

### Gaps
- No HTD-35H-specific identified model (friction, armature, effective kp, backlash) was found anywhere. Hiwonder does not publish deadband or backlash. The user must measure them.
- No public RL biped using Hiwonder or LewanSoul HTD or LX servos with documented sim2real was found.
- The exact backlash implementation constants in Open Duck Playground's base class were not retrieved.

---

## 2. System identification procedures a hobbyist can do

### Takeaway
The proven low-cost recipe has four parts:
1. A pendulum test bench (servo plus a known arm and mass), excited with steps, chirps and random-target trajectories while logging the commanded target and the read-back position.
2. A fit of kp, damping or kd, friction loss, armature and the torque-speed limit, by simulating the same bench in MuJoCo and minimizing the tracking error (BAM does this).
3. A backlash measurement by hand-wiggling a holding servo.
4. Latency measured by timestamping commands against the first detected motion.

### Cited Findings
- BAM identifies servo models from trajectories recorded on a pendulum test bench and exports them directly to MuJoCo. Open Duck used it for the STS3215 bus servos. [primary] — [BAM arXiv 2410.08650](https://arxiv.org/abs/2410.08650), [BAM GitHub](https://github.com/rimim/bam), [Open Duck sim2real.md](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini/v2/docs/sim2real.md)
- ToddlerBot's procedure [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4):
  - Chirp position commands give tracking data.
  - Constant-speed back-driving with a torque sensor gives friction loss (intercept) and damping (slope).
  - A spin-down energy test gives armature.
  - A joint 9-parameter fit reached 1.3° mean error.
  - kp unit conversion is found empirically (about 150× for Dynamixel).
- Disney identified friction, torque limits and backlash on a test bench and used the observed ranges as the randomization ranges. [primary] — [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- To measure read latency on a half-duplex servo bus, time from the last request byte to the last response byte with a logic analyzer or MCU timer. Do it at idle and under motion. [suggested procedure from search synthesis, not a measured result] — [LX-16A manual](https://cdn.robotshop.com/rbm/a5e9ab51-86bc-497b-adac-003de3088fb7/9/92896585-5e83-4c7c-a663-650b81e5978b/67e9729b_lx-16a-serial-bus-servo-user-manual.pdf)

### Inferences (a hobbyist recipe for the HTD-35H) [inference]
1. **Bench:** Clamp one servo. Attach a rigid arm with a known mass, about 0.2-0.5 kg at 10-15 cm, so loads approach realistic hip and knee torques. Measure the arm's mass and CoM. Log at the fastest rate the bus allows for one servo, which is much faster than for 8.
2. **Excitations:**
   - Steps of 5°, 15° and 30° with and without the load, from several starting angles (gravity on and against the motion).
   - Chirps from 0.5 to 8 Hz with ±5-10° amplitude.
   - Random target sequences sent at 50 Hz exactly as the policy will send them.
   - Repeat at a full and a sagged battery voltage (e.g. 12.4 V and 10.8 V).
3. **Fit:** Reproduce the bench in MuJoCo with your actual actuator code (including the latency buffer and torque-speed clip) and optimize kp, kd or damping, frictionloss, armature, τ_max and q̇_max. Use scipy or CMA-ES, or BAM's tooling if you can add a custom model.
4. **Validate:** Check on held-out trajectories. ToddlerBot's 1.3° average error is a good target.
5. **Delay:** Pure delay shows up in step responses as dead time before motion. Separate it from the mechanical lag by fitting delay as a parameter, or by detecting motion onset in read-back (≥2 counts ≈ 0.5°).
6. **Backlash and deadband:**
   - Command a hold position and slowly push the horn by hand while reading the position. The read value lags the output motion within the play only if the pot sits on the motor side. Most hobby servos put the pot on the output shaft, so backlash then appears as the output moving ±b without torque.
   - Measure output play with a dial or a long pointer arm.
   - Measure deadband by commanding 1-count increments and noting the smallest one that produces motion.
7. **Thermal:** Rerun a few step tests after 10 minutes of standing under load. ToddlerBot found heating to be a cause of falls ([ToddlerBot](https://arxiv.org/html/2502.00893v4)).

### Gaps
- No published step-response or chirp data for the HTD-35H or any Hiwonder HTD/LX servo was found.
- No source documents whether HTD-35H firmware exposes PID or compliance tuning registers. The protocol copy found lists command codes, but this session did not confirm any gain-setting command.

---

## 3. Latency: bus timing for 8 servos at 115200 baud, effect on balance, randomization ranges

### Takeaway
At 115200 baud with the Hiwonder protocol, the wire time alone is about 7 ms to write 8 positions and about 10 ms to read 8 positions with individual request and reply pairs. USB-serial adapter latency and per-transaction Python overhead can multiply that, by up to about 16 ms per round trip with a default FTDI latency timer. Real observation-to-actuation latency is therefore plausibly 15-40 ms, which may exceed the user's 5-19 ms range. Open Duck randomizes action and IMU delay independently over 0-60 ms (0-3 steps at 50 Hz). Measure it, then randomize a range around the measured value.

### Cited Findings
- The LX/Hiwonder bus is half-duplex UART at 115200 baud, and reads are one request and reply per servo. [primary] — [LX-16A manual](https://cdn.robotshop.com/rbm/a5e9ab51-86bc-497b-adac-003de3088fb7/9/92896585-5e83-4c7c-a663-650b81e5978b/67e9729b_lx-16a-serial-bus-servo-user-manual.pdf), [HTD-35H listing](https://openelab.io/products/htd-35h-bus-servo)
- Protocol table: `SERVO_MOVE_TIME_WRITE` is command 1 with length 7, and `SERVO_POS_READ` is command 28 with length 3. The servo returns read data immediately, and the host must switch to read mode right after sending. [primary, unofficial host] — [Purdue-hosted Hiwonder protocol](https://engineering.purdue.edu/477grp4/Files/refs/motors_manual.pdf)
- The frame is header 0x55 0x55, ID, length, command, params, checksum. [partially primary; search synthesis] — [HiWonder-BusLinker repo](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)
- A community library notes that servo responses on some setups differ from Hiwonder's documented protocol. [anecdotal] — [HiWonder-BusLinker repo](https://github.com/EtcetFelix/HiWonder-BusLinker-servo-bus)
- The tve/lx16a-servo ESP32 example puts a 10 ms delay between a move and a read. [anecdotal or example code] — [tve/lx16a-servo](https://github.com/tve/lx16a-servo)
- Open Duck's runtime sets the USB-serial `latency_timer` to 1 via a udev rule, which signals that adapter latency matters for bus servos. [primary] — [Open_Duck_Mini_Runtime README](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/README.md)
- Open Duck randomizes action delay over 0-3 steps and IMU delay over 0-3 steps at 50 Hz, i.e. 0-60 ms each. [primary] — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
- An older quadruped sim2real study (Minitaur, Tan et al. 2018) measured system latency and set matching latencies in simulation. [primary] — [arXiv 1804.10332](https://arxiv.org/pdf/1804.10332)
- The Hwangbo actuator net exists partly to capture "delays in control signals introduced by multiple hardware and software layers." [primary] — [arXiv 1901.08652](https://arxiv.org/abs/1901.08652)
- Disney's BDX does not model latency explicitly. It runs the actuators at 600 Hz with a 37.5 Hz low-pass filter on 50 Hz policy actions, on a fast bus. [primary] — [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- ToddlerBot runs 30 servos at 2 Mbaud with full feedback at 50 Hz, roughly 17× faster than 115200 baud. [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4)

### Inferences (bus arithmetic for the user's 8 × HTD-35H) [inference, verify by measurement]
- At 8N1, 115200 baud is 11,520 bytes/s, so **86.8 µs per byte**.
- Frame size = length byte + 3 (two header bytes plus the ID). A move write is 10 bytes. A position-read request is 6 bytes, and its reply is 8 bytes (length 5: command, 2 data bytes, checksum, plus header, ID and length).
- **Write 8 servos:** 80 bytes ≈ **6.9 ms**. The writes can go back to back in one `write()` call because they need no reply.
- **Read 8 servos:** 8 × (6 + 8) = 112 bytes ≈ **9.7 ms** of wire time, plus the servo's response turnaround (unknown, probably 0.1-1 ms each).
- **USB-serial overhead:**
  - Each blocking request-and-wait in Python pays USB polling latency. That is about 1 ms per round trip with a well-configured CH340/CP210x, or up to **16 ms per round trip** with an FTDI chip at its default latency_timer.
  - Eight sequential reads could therefore take about 10-20 ms (good case) or more than 100 ms (bad case).
  - If the debug board echoes TX bytes on RX (common on half-duplex adapters), the reader must discard the echoed bytes. That adds 6 bytes per read to the RX stream.
- **Total loop at 50 Hz (20 ms budget):** about 10 ms read + about 7 ms write + 1-2 ms IMU I2C + under 1 ms inference already uses most of the 20 ms period. Options:
  - Read the servos less often (e.g. half per tick, alternating), or read only some joints.
  - Pipeline the work: write the actions, then read for the next step.
  - Raise the baud rate, if the HTD-35H supports it. That was not confirmed.
  - Move bus I/O to a microcontroller such as an RP2040 or ESP32 that reads continuously and serves the latest state to the Pi.
- **Effective delay the policy sees:** joint observations are up to about 10-20 ms old by the time the action is computed. The command then takes up to about 7 ms to reach the last servo in the write sequence, and the servo has its own internal loop and mechanical lag. An end-to-end observation-to-motion delay of **about 15-40 ms** is plausible. The current 5-19 ms randomization may cover only the optimistic end.
- **Recommended randomization:**
  - Measure the real end-to-end delay. A simple method is to command a step and timestamp the first read-back change, logged over many trials.
  - Randomize action delay over roughly [0.5×, 1.5×] of the measured median, extended to cover the p95. Draw it per episode, with small per-step jitter.
  - Delay joint observations separately from the IMU (the IMU is on a different bus).
  - Open Duck's 0-60 ms independent action and IMU delays are a reasonable upper envelope at 50 Hz.
  - Also model the **staggered write**: servo i receives its command about i × 0.87 ms after servo 0.
- **Effect on balance:** for ankle and hip balance, delay adds phase lag of ω·τ. At a 3 Hz correction bandwidth, 30 ms of delay costs about 32° of phase. Backlash adds a dead zone that produces limit-cycle wobble around upright. Policies trained without them typically oscillate in place or overreact to pushes. Including past actions in the observation (as Open Duck and BDX do) helps the policy infer in-flight commands.

### Gaps
- No public measurement of round-trip read time for Hiwonder or LewanSoul bus servos at 115200 was found. The figures above are wire-time arithmetic only.
- It is unconfirmed whether the HTD-35H supports baud rates above 115200 or any multi-servo read or write command (sync read and write). The listing shows only a default of 115200.
- The adapter chip on the Hiwonder debug board (CH340, CP2102 or other) is unknown. That chip determines USB latency.

---

## 4. BNO055 specifics and IMU use for the policy

### Takeaway
In fusion modes the BNO055 outputs fused orientation, gravity, linear acceleration and gyro at a fixed **100 Hz**, with sensor bandwidths set by the fusion firmware and not configurable. Its magnetometer and auto-calibration cause well-documented heading jumps. For an RL balance policy, use **IMU mode (accelerometer plus gyro fusion, no magnetometer)** for the gravity vector or projected gravity, plus the gyro rates. Avoid NDOF heading. Randomize the IMU delay (0-2 steps), the noise, and a constant tilt bias of a few degrees.

### Cited Findings
- The fusion output data rate is 100 Hz (±1% tolerance) per the Bosch datasheet. Adafruit's forum also states that fusion modes output at 100 Hz. [primary] — [Bosch BNO055 datasheet](https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bno055-ds000.pdf), [Adafruit forum](https://forums.adafruit.com/viewtopic.php?t=194243)
- PJRC forum on fusion rate and bandwidth [anecdotal, but citing the datasheet and a Bosch seminar] — [PJRC forum](https://forum.pjrc.com/index.php?threads/bno055-gyro-update-rate.29541/):
  - A Bosch seminar confirmed the 100 Hz NDOF fused rate.
  - Gyro bandwidth is configurable (12-523 Hz) only in non-fusion modes such as GYROONLY.
  - The fusion engine sets internal sample rates. Non-fusion accelerometer and gyro ODR is about 2 kHz.
  - The datasheet section 3.6.3 lists rates of 20-100 Hz.
- In fusion mode, the fusion algorithm configures sensor range and bandwidth, and they cannot be changed. The outputs include orientation (Euler or quaternion), linear acceleration with gravity removed, the gravity vector in m/s², and calibration status. [primary-ish vendor docs] — [MathWorks BNO055 block](https://www.mathworks.com/help/simulink/supportpkg/arduino_ref/bno055imusensor.html)
- Known glitches [anecdotal]:
  - Heading jitter and drift of up to about ±10° once the magnetometer reaches "3/3" calibration. — [Bosch community](https://community.bosch-sensortec.com/t5/MEMS-sensors-forum/BNO-055-calibrate-using-horizontal-plane-rotations-only-And/td-p/20760)
  - Sudden calibration jumps. — [freesteel blog 2015](https://www.freesteel.co.uk/wpblog/2015/06/05/sudden-calibration-jumps-with-the-bno055/)
  - Sudden heading shifts. — [Adafruit forum](https://forums.adafruit.com/viewtopic.php?t=198775)
  - Calibration loss after many orientation changes. — [Bosch community](https://community.bosch-sensortec.com/mems-sensors-forum-jrmujtaw/post/calibration-loss-on-changing-orientation-many-times-in-bno055-pUwHEqaHT0QXrNM)
  - Calibration lost, and the sensor stops responding, in forward motion. — [Adafruit forum thread](https://forums.adafruit.com/viewtopic.php?f=19&start=105&t=78459)
  - The kbox-firmware issue lists practical BNO055 pitfalls. — [kbox issue #86](https://github.com/sarfata/kbox-firmware/issues/86)
- The BNO055 has no internal non-volatile storage for calibration. Calibration is lost on power cycle, and the workaround is to save and restore offsets. Users report that NDOF may overwrite restored offsets. [vendor guide plus anecdotal] — [Adafruit calibration guide](https://learn.adafruit.com/adafruit-bno055-absolute-orientation-sensor/device-calibration), [Bosch community](https://community.bosch-sensortec.com/mems-sensors-forum-jrmujtaw/post/bno055-does-not-store-heading-value-on-startup-zZOYX3lDySmEebW)
- Magnetic interference from motors is the most common cause of heading errors. The suggested workaround indoors is IMU mode (no magnetometer). [anecdotal] — [Bosch community thread](https://community.bosch-sensortec.com/t5/MEMS-sensors-forum/BNO-055-calibrate-using-horizontal-plane-rotations-only-And/td-p/20760)
- What comparable RL bipeds observe [primary]:
  - Open Duck feeds noisy **raw gyro (3)** and **raw accelerometer (3)**, not fused orientation. Its gravity vector exists only in the privileged critic state, but it is delayed 0-3 steps in sim. — [joystick.py](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)
  - ToddlerBot feeds torso orientation and angular velocity. — [ToddlerBot](https://arxiv.org/html/2502.00893v4)
  - Disney fuses the IMU with actuator measurements in a state estimator. — [arXiv 2501.05204](https://arxiv.org/html/2501.05204v1)
- Open Duck's runtime handles the IMU frame and mounting explicitly (an upside-down flag in `duck_config.json`, plus `raw_imu.py` and `imu_server.py` to verify orientation). [primary] — [Open_Duck_Mini_Runtime README](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/README.md)

### Inferences [inference]
- **Fused vs raw:**
  - Use the BNO055 in **IMU (accelerometer plus gyro) fusion mode**. Read the gravity vector, or the quaternion converted to projected gravity, plus the gyro.
  - The fused gravity vector filters out foot-impact accelerations that would corrupt a raw accelerometer gravity estimate. Raw accelerometer data during a stumble or push includes large linear accelerations, which is exactly when the balance policy needs the correct tilt.
  - The fused output can, however, lag or show small transients under sustained acceleration.
  - Raw accelerometer input (as in Open Duck) forces the policy to learn its own filtering, which is harder with 8 DoF and a short history.
  - A good compromise is fused gravity plus raw gyro, with 2-3 frames of history.
- **Never use magnetometer heading** (NDOF yaw) as a policy input. Yaw is unobservable for balance anyway. Avoid NDOF so magnetometer calibration events can't perturb roll and pitch.
- **Timing:** fusion updates at 100 Hz, so samples are up to 10 ms stale, plus filter group delay (undocumented, likely about 10-30 ms) and the I2C read time.
  - The Pi's I2C hardware has a known clock-stretching bug, and the BNO055 uses clock stretching. Many users switch to UART mode or a reduced or software I2C clock (general community knowledge, not verified this session).
  - Randomize IMU delay over 0-40 ms, i.e. 0-2 policy steps. Open Duck uses 0-3 steps.
- **Noise and bias randomization**, scaled to plausible magnitudes:
  - Gyro: white noise of about 0.05-0.1 rad/s (Open Duck uses 0.1 uniform) plus a per-episode constant bias of about ±0.02-0.05 rad/s.
  - Gravity: noise of about 0.05-0.1 per component plus a **per-episode mounting tilt bias of about ±2-3°**. This covers IMU mounting error and fusion offset.
  - Occasional dropped or repeated samples (repeat the last value) to mimic missed I2C reads.
- **Calibration:** in IMU mode only the accelerometer and gyro need calibration. Store the offsets and reload them at boot. Check the calibration status byte before enabling the policy.

### Gaps
- Exact gyro and accelerometer noise densities and the fusion filter's group delay could not be extracted. The datasheet PDF could not be parsed in this session, so verify them in the Bosch datasheet's electrical characteristics table.
- No measured BNO055 fusion latency was found from any source.
- No source quantifies BNO055 gravity-vector error under sustained linear acceleration. This is a known limitation of complementary or Kalman fusion, but no BNO055-specific number was found.

---

## 5. Running PPO MLP policies on a Raspberry Pi 4 at 50 Hz, and safety for first hardware tests

### Takeaway
Compute is not the bottleneck. A small MLP runs in well under 1 ms on a Pi 4 with ONNX Runtime, or even NumPy, and Open Duck runs its ONNX walking policy on the much weaker Pi Zero 2W. The real timing risks are serial I/O and Python jitter. Structure the control loop around a fixed-period timer, isolate I/O, and test with a harness, a torque limit and a hardware kill switch.

### Cited Findings
- Open Duck Mini runs its ONNX walk policies (`v2_rl_walk_mujoco.py --onnx_model_path ...`) on a **Raspberry Pi Zero 2W** under Raspberry Pi OS Lite 64-bit. [primary] — [Open_Duck_Mini README](https://github.com/apirrone/Open_Duck_Mini), [Open_Duck_Mini_Runtime README](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/README.md)
- ToddlerBot runs its RL walking policy on Jetson CPU at 50 Hz. [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4)
- A TinyML benchmark measured the Pi 4 at about 0.16 ms for a small keyword-spotting model and about 1.76 ms for gesture classification. The runtime was unspecified, so this is only a rough reference. [primary, indirect] — [arXiv 2509.04721](https://arxiv.org/pdf/2509.04721)
- A humanoid standing paper reports about 1.7 ms per policy inference with ONNX on an edge device (not a Pi), and notes that lightweight MLP-style networks make this fast. [primary, indirect] — [arXiv 2508.00355](https://arxiv.org/pdf/2508.00355)
- Open Duck runtime safety and bring-up practice [primary] — [Open_Duck_Mini_Runtime README](https://raw.githubusercontent.com/apirrone/Open_Duck_Mini_Runtime/v2/README.md):
  - Test motors with `check_motors.py`.
  - Find joint offsets with `find_soft_offsets.py`.
  - Verify the IMU orientation before walking.
  - Warns that experimental head control "can break your duck's head".
- ToddlerBot broke after up to 7 falls. Plan for falls: repair took about 35 minutes. [primary] — [ToddlerBot](https://arxiv.org/html/2502.00893v4)

### Inferences (Pi 4 setup and first-test safety) [inference]
- **Inference:**
  - Export to ONNX and use `onnxruntime` with `intra_op_num_threads=1` (multi-threading hurts tiny models). Or skip ONNX entirely and run the MLP in NumPy (a few matmuls).
  - Expect roughly 50-300 µs per call for a 3 × 256 MLP on a Pi 4 Cortex-A72. Benchmark the median and p99 after warm-up.
  - Python call overhead dominates at this size.
- **Loop structure:**
  - Use one control thread on a fixed 20 ms schedule driven by `time.monotonic()` plus `sleep` until the next deadline, never a bare `sleep(0.02)`.
  - Read the IMU in its own thread at 100 Hz and keep the latest sample.
  - Do servo bus I/O in a dedicated thread or process, or offload it to an MCU.
  - Log loop period and jitter. If p99 exceeds about 25 ms, the trained latency range should cover it.
  - Optional: `isolcpus` or `taskset` to pin the control process, `chrt -f` for SCHED_FIFO, the performance CPU governor, and Wi-Fi power save disabled.
- **Matching sim:** apply the same action scale, default pose, joint ordering, signs and offsets, and the same target-rate clip as in training. Log the real observations and replay them through the policy in sim to detect frame and sign errors before powering the legs.
- **First-test safety ladder:**
  1. **Policy-in-the-loop with servos unpowered or torque-off.** Hold the robot by hand and check that the action responses to tilting look sensible (signs and magnitudes).
  2. **Suspended on a gantry or harness** with feet just off the ground. Then lower it until the feet bear partial load. The harness should catch the robot before it hits the ground, not hold it up all the time.
  3. **Conservative limits:**
     - Clip targets to the sim joint limits times about 0.9.
     - Rate-limit targets as in training.
     - Reduce max torque or current if the servo has a torque or load-limit setting. The HTD-35H has stall and over-temperature protection per the listing ([openelab](https://openelab.io/products/htd-35h-bus-servo)), but configurable limits were not confirmed.
  4. **Emergency stop:**
     - A physical switch that cuts servo power, independent of the Pi.
     - A software watchdog that disables torque (the servo unload command) when |roll| or |pitch| exceeds about 30-40°, the loop overruns, or a bus read fails repeatedly.
     - A dead-man button on the gamepad.
  5. **Fall handling:** when a fall is detected, go limp or move to a safe crouch to protect the gears. Hobby servos strip gears when back-driven during impacts.
  6. **Thermal monitoring:** read the servo temperature every few seconds via the bus and stop above a threshold. ToddlerBot saw falls increase as motors heated.
- **Battery:** monitor the 11.1 V pack voltage, since servo strength falls with sag. Randomize strength or voltage in sim to cover the full charge range.

### Gaps
- No direct published benchmark of ONNX Runtime MLP inference latency on a Pi 4 was found. The figures above are an estimate and should be measured.
- No published Hiwonder-specific emergency-stop or torque-limit procedure was found. The torque unload command and any max-torque register should be checked in the official Hiwonder protocol PDF.
