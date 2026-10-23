# Servo bench protocol (HTD-35H)

Measurements that pin down the simulator's servo model, so policies trained in sim move the way the
real servos can. One servo is enough for experiments 1–8; experiment 9 uses one assembled leg (4 servos
on one bus chain).

## What the simulator needs, and which experiment gives it

| Sim parameter | Now (guess) | Experiment |
|---|---|---|
| No-load speed (rad/s) and how fast the servo gets there | 5.8 (datasheet); the 14° bench step suggests ~3.0 | 1 |
| Torque–speed curve (how speed drops under load) | linear, stall → 0 | 2 |
| Stall torque (Nm) | 3.43 (datasheet) | 3 |
| Stiffness kp: torque per degree of error, and where it saturates | 15 Nm/rad (guess); bench fit ~6 | 3 |
| Gear slack and deadband (deg) | 0–1° slack | 4 |
| How the servo follows a target that changes every 20 ms (what the policy actually does) | not measured | 5 |
| Move-time setting used for streaming (0 or 20 ms) | not decided | 5 |
| Speed and torque vs battery voltage | ×0.85–1.13, linear | 6 |
| Torque fade with temperature | not modelled | 7 (optional) |
| Counts-to-degrees scale and linearity | 240° / 1000 counts | 8 |
| Control-loop timing on a full leg at 50 Hz | assumed ~18 ms of 20 | 9 |
| Command-to-motion delay | 24–28 ms (already measured) | 1, 5 (re-checked) |

## Setup

**Hardware**
- One HTD-35H. Use a spare if you have one. Otherwise use the **knee of the assembled leg**: the shin is
  the lever. Take the foot off, or hang weights at the ankle axis. Measure the distance from the knee
  axis to the ankle axis with calipers (about 10–11 cm in the CAD).
- If you use a spare servo, clamp it rigidly (vise or screwed to a board) and fit a stiff lever arm to
  the horn: a 3D-printed bar about 10 cm long, with holes at known radii (e.g. 2.5 / 5 / 7.5 / 10 cm).
- Weights of known mass: water bottles, or bags of coins, weighed on the kitchen scale.
- A kitchen scale (1 g resolution is fine), a phone inclinometer app, and calipers or a ruler.
- The robot's 3S battery. A bench supply at 10.5 / 11.1 / 12.0 / 12.6 V is optional, for experiment 6.

**Bus:** only the servo under test on the bus (except experiment 9), so reads run at the maximum
rate (~3.7 ms each).

**Safety**
- Keep stall holds under 3 s, then relax.
- Stop at 60 °C servo temperature.
- Keep fingers out of the lever's path.
- Have "Relax all" one click away.

**Record for every session:** servo ID, which joint (or "spare"), lever radius (m), each mass (kg),
battery voltage at the start and end, room temperature.

## Experiments

### 1. Large steps: top speed, acceleration, delay
Steps of **15°, 30°, 60° and 90°** (≈62, 125, 250, 375 counts), in **both directions**, each 3 times.
Move time 0 (full speed).
- Spare servo: run it twice, with the bare lever (no load) and with the lever horizontal and a 100 g
  weight at 5 cm.
- Knee: run it with the shin hanging free.

Log every position read with a timestamp, from 100 ms before the command until 500 ms after the
servo settles.

*Gives:* dead time, peak speed for each step size (if peak speed keeps rising with step size, the servo
is accelerating slowly rather than slow), acceleration, overshoot, final error.

### 2. Speed under load: the torque–speed curve
Mount the lever horizontally. Hang mass *m* at radius *r* and command a lift from 30° below
horizontal to 30° above (≈250 counts, move time 0). Then command the move back down, with the weight
pulling in the direction of motion.

Masses: 0, 100, 200, 400, 700 g and 1 kg at 5 cm, or until the servo can't lift it. Do 3 lifts each.

*Gives:* the speed near horizontal, where the gravity torque is about *m·g·r*, at each torque. That is
the torque–speed curve. The lowering moves show how the servo brakes a load that pushes it.

### 3. Stiffness and stall torque: the kitchen-scale test
Lever horizontal, its tip resting on the kitchen scale at radius *r*.
1. Command the angle where the tip just touches. Tare the scale.
2. Command targets further "down", into the scale: **+1, 2, 4, 8, 16, 32, 64 and 120 counts** past the
   touch point.
3. Hold each one for 1.5 s. Type the scale reading (g) into the page, then go back to the touch point.
4. Repeat the sweep once more, then do it in the other direction (flip the servo or the scale side).

The arm is blocked, so the error ≈ (target − measured position) and the torque = grams/1000 · 9.81 · *r*.

*Gives:* torque vs error. The slope is kp (Nm/rad), the flat top is the stall torque, and an offset
near zero error shows the deadband. Also log the voltage during the push: the voltage drop under load
matters too.

### 4. Slack and deadband: push by hand
Torque on, holding a position, lever horizontal, scale or luggage scale at radius *r*:
1. Push the tip gently up with a known force: 50, 100, 200 g-force.
2. Do the same pushing down.
3. Log the read position at each force.

Then command steps of 1, 2, 3, 4 and 5 counts from the same start, 5 times each, approaching from
above and from below, and log the final position.

*Gives:* gear slack (deflection at near-zero force), stiffness at small errors, deadband (smallest
step that moves), and hysteresis (different endpoints from each side).

### 5. Following a streamed target: the most important one
This is what the policy does: a new target every 20 ms. Send a **sine target at 50 Hz**:
- amplitude 5° and 15°;
- frequencies 0.5, 1, 2, 3, 4 and 6 Hz;
- 4 s each.

Run the full set twice, **with move time 0 and with move time 20 ms** in each write. Read the position
right after each write. Spare servo: lever horizontal, 100 g at 5 cm. Knee: shin hanging.

Then send one 30° step as a streamed sequence (the same target written every 20 ms), and compare it
with experiment 1.

*Gives:* gain and lag at each frequency (the real bandwidth for walking), whether rewriting the target
every 20 ms changes the motion, and which move-time setting to use on the robot.

### 6. Voltage (optional; easiest with a bench supply)
Repeat experiment 1 (60° steps, unloaded) and the 400 g lift from experiment 2:
- with a bench supply: at 10.5 / 11.1 / 12.0 / 12.6 V;
- with the battery: fully charged and again near 11.1 V. Log the voltage the servo reports.

*Gives:* how speed and torque scale with voltage (the sim assumes both are proportional).

### 7. Heat (optional)
Lever horizontal, a mass giving about 1 Nm (e.g. 1 kg at 10 cm, or about half the measured stall
torque). Hold for up to 2 minutes, or until 60 °C. Log temperature and position every 2 s, and
repeat a 400 g lift (experiment 2) every 30 s while warm.

*Gives:* how quickly the servo heats up while standing loaded, and whether it gets weaker as it warms.

### 8. Calibration (counts → degrees)
Command 100, 250, 500, 750 and 900 counts. Measure the lever's angle with the phone inclinometer at
each. Log the commanded counts, the read counts and the measured degrees.

*Gives:* the scale (deg/count) and how linear it is. The sim assumes 0.24°/count.

### 9. Control-loop timing on a full leg (4 servos on one chain)
Run the real loop shape at 50 Hz for 60 s:
1. Each cycle: write 4 targets (a slow 0.5 Hz, 5° sine on each), then read 4 positions.
2. Every 10th cycle, also read one voltage and one temperature, round-robin.

Log for each cycle: start time, the time each write and read finished, failed or timed-out reads, and
the total cycle time. Then run it again, with both legs' buses in parallel threads if the second
chain is wired.

*Gives:* the real cycle time and its spread, missed deadlines, and how old the positions are when
the policy sees them. These set the sim's observation and action delays.

## Output files
One folder per session, e.g. `bench/2026-10-16_servo5/`, containing:
- `meta.json`: everything under "Record for every session", plus the servo firmware and protocol
  notes.
- `e1_steps.csv`, `e2_load.csv`, ... one file per experiment, one row per read:
  `t_s, event, target_counts, pos_counts, voltage_v, temp_c, mass_kg, radius_m, scale_g, move_time_ms, trial`
  (leave unused columns empty).
  - `t_s` is `time.perf_counter()` on the Pi.
  - `event` marks command writes (`write`) vs reads (`read`), so the delays can be measured.
- `e9_loop.csv`: one row per cycle, with the timings above.

Copy the folder into `bench/` in this project and I'll fit the servo model to it, compare the fit
with the simulator, and set the training parameters.

---

## Prompt for the dashboard

Paste this into the session that maintains the dashboard on the Pi (`isaacpi.local:8000`):

> I'm measuring my Hiwonder HTD-35H bus servos (serial bus protocol through a BusLinker, 115200 baud)
> so a simulator can model them accurately. Please add a **"Bench lab"** page to the existing servo
> dashboard that runs the experiments below on one selected servo and saves raw data. Keep the existing
> safety rules (temperature cutoff 60 °C, Stop all / Relax all always available), and add:
> - a 3 s cap on any hold where the servo is blocked;
> - an abort if the read position stops changing while the target error is above 20 counts for more
>   than 3 s.
>
> **General:**
> - Time every bus write and read with `time.perf_counter()`.
> - Read position (SERVO_POS_READ) as fast as the bus allows while a move is running. Only the servo
>   under test is on the bus, except experiment 9.
> - Save one CSV per experiment, one row per write or read, with these columns:
>   `t_s, event, target_counts, pos_counts, voltage_v, temp_c, mass_kg, radius_m, scale_g, move_time_ms, trial`
>   (`event` = `write` or `read`; leave unused columns empty).
> - Save a `meta.json` with the servo ID, joint name, lever radius, masses, battery voltage at the start
>   and end, and room temperature. All of these are entered on the page before starting.
> - Write everything to `bench/<date>_servo<id>/`, with a button to download the folder as a zip.
> - Before each experiment, show the setup instructions and wait for me to press Start.
>
> **Experiments (one card each, with Start, Abort and a live position plot):**
> 1. **Steps:** 62, 125, 250 and 375 counts, in both directions, 3 trials each, move time 0. Log from
>    100 ms before the write to 500 ms after settling. Settled = within 2 counts for 100 ms.
> 2. **Load lift:** I enter mass and radius. Command a 250-count move up (the lift), then back down. 3
>    trials per mass, move time 0. I'll run it for several masses.
> 3. **Kitchen-scale stiffness:**
>    - I set the touch point with a slider; save it.
>    - Step through +1, 2, 4, 8, 16, 32, 64 and 120 counts past the touch point. Hold each for 1.5 s
>      (logging position and voltage), then return to the touch point.
>    - After each hold, prompt me to type the scale reading in grams.
>    - Then run the same sweep in the other direction.
> 4. **Slack:**
>    - Part A: hold the current position and log position continuously while I push the lever. Each
>      time I click "mark", record the force I type.
>    - Part B: steps of 1–5 counts, 5 trials each, alternately approaching from above and from below.
>      Log the final position 300 ms after each step.
> 5. **Streaming sine:**
>    - Write a sine target at exactly 50 Hz (scheduled against `perf_counter()`, not `sleep` drift),
>      reading position right after each write.
>    - Amplitudes 21 and 62 counts (5° and 15°); frequencies 0.5, 1, 2, 3, 4 and 6 Hz; 4 s each.
>    - Run the whole set with move time 0, then with move time 20 ms.
>    - Then a 125-count step sent as a stream: the same target written every 20 ms for 1 s.
> 6. **Voltage:** repeat experiment 1 (250-count steps only) and one load lift. Prompt me for the supply
>    voltage and tag the rows with it.
> 7. **Heat (optional):** hold the current target for up to 120 s. Log temperature and position every 2 s.
>    Every 30 s, do one 250-count lift and return. Stop at 60 °C.
> 8. **Calibration:** command 100, 250, 500, 750 and 900 counts. At each, prompt me to type the angle
>    measured with my phone.
> 9. **Loop timing (all servos on the chain):**
>    - Run 60 s at 50 Hz. Each cycle: write 4 targets (a 0.5 Hz, 21-count sine around each servo's
>      home), then read 4 positions.
>    - Every 10th cycle, also read one voltage and one temperature, round-robin.
>    - Log one row per cycle (`e9_loop.csv`): cycle start, end time of each write and read, failed or
>      timed-out reads, and total cycle time.
>    - Show a histogram of cycle times and the count of cycles over 20 ms.
>    - Option to run both legs' buses in parallel threads.
>
> **Don't fit or smooth anything: I need the raw timestamps.** After each experiment, show a quick
> summary on the page:
> - peak speed (deg/s) and dead time (ms) for steps;
> - speed vs mass for lifts;
> - torque vs error for the scale test, with torque = g/1000 · 9.81 · radius;
> - gain and lag vs frequency for the sine.
