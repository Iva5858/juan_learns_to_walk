# Bring-up checklist

Run these in order: each phase depends on the one before. Stop at the first failure and fix it
before going on. "Expect" is what a pass looks like.

IMU: the LSM9DS1 (accelerometer + gyro; the robot estimates gravity itself, see `robot/imu.py`).

Tools: a multimeter, a phone inclinometer app, a set square or protractor, a way to hang the robot
with its feet off the ground (a harness, a stand, or a clamp on the torso), and a kitchen scale for
phase 9.

## 0. On the computer, before the Pi

| # | Test | Expect |
|---|---|---|
| 0.1 | `python scripts/check_robot_joints.py` | "matches the simulator" |
| 0.2 | Push the branch, then `git pull` on the Pi | the Pi has `robot/` |

## 1. Pi alone: no servo power, IMU and camera connected

| # | Test | Expect |
|---|---|---|
| 1.1 | `sudo apt install python3-serial python3-smbus2 python3-numpy python3-picamera2 i2c-tools` | installs |
| 1.2 | `sudo raspi-config` → Interface Options → I2C → enable; in `/boot/firmware/config.txt` add `dtparam=i2c_arm_baudrate=400000`; reboot | `ls /dev/i2c-*` lists `/dev/i2c-1` |
| 1.3 | `i2cdetect -y 1` | `6b` (accelerometer + gyro) and `1e` (magnetometer, unused) in the grid; `6a` / `1c` on some boards. Nothing there: check wiring (3.3 V, GND, SDA pin 3, SCL pin 5) |
| 1.4 | `rpicam-hello --list-cameras` | one camera, `imx219` |
| 1.5 | `vcgencmd get_throttled` | `throttled=0x0`. Anything else: the 5 V supply is weak (measure 5.0–5.25 V at the Pi's GPIO 5 V pin with a multimeter) |
| 1.6 | `lsusb` with the BusLinker plugged in | one line with `1a86:7523` |

## 2. Servo power on, robot hanging, nothing moves

Before switching on: measure the servo battery with a multimeter (11.1–12.6 V), and check the
polarity at the first servo of each cable chain.

| # | Test | Expect |
|---|---|---|
| 2.1 | `python3 -m robot.selftest` | no FAIL. It reports each part below |
| 2.2 | Self-test section 2: IDs on the bus | IDs 1–8, no duplicates. If not, set the IDs one servo at a time with only that servo connected |
| 2.3 | Self-test: voltage limits | WARN "voltage limit 12.0 V < 12.6 V" means a full pack can make a servo go limp. Raise it (cmd 22) to 12.6–13 V before using a full pack |
| 2.4 | Self-test: angle offsets | 0. A nonzero offset is fine, but note it: the dashboard's zero absorbs it |
| 2.5 | Self-test: servo voltages agree | within 0.3 V. Larger: a bad connector or a thin wire in that chain |
| 2.6 | Self-test section 3: bus timing | reading all 8 positions takes about 4 ms per servo (~30 ms per pass), 0 failed. Note the number: a slower bus means more delay for the policy |
| 2.7 | Self-test section 4: IMU, robot upright and still | LSM9DS1 found; <0.5% bad reads; ~200 reads/s; gyro bias under 0.05 rad/s (it's removed anyway); accelerometer 1.00 g; gravity z ≈ −1.00; yaw drift under 2° in 10 s |
| 2.8 | Self-test section 5: camera | over 10 frames/s |
| 2.9 | Run the self-test 3 times | the same results (the timing numbers within ~1 ms) |

## 3. Dashboard, still nothing moves

`python3 -m robot.dashboard`, open `http://isaacpi.local:8000` on the computer and on the phone.

| # | Test | Expect |
|---|---|---|
| 3.1 | Page loads | "8 of 8 servos", every card filled in, a warning that zero isn't set |
| 3.2 | Relax all, then move each joint by hand, one at a time | only that card's angle and counts change, on the correct leg and joint. A different card moving = IDs in the wrong place |
| 3.3 | Bus card on the Pi card | about 4 ms per servo per pass, fail 0% |
| 3.4 | Tilt the robot forward ~20° | IMU "fwd" goes up by ~20. Goes down or "right" changes instead: fix the axis mapping (calibration step 5) |
| 3.5 | Tilt it to its right (its own right) ~20° | "right" goes up by ~20 |
| 3.6 | Turn it to its left (counter-clockwise seen from above) | gyro z goes positive while turning |
| 3.7 | Roll it quickly to its right, then pitch it quickly forward (nose down) | gyro x positive while rolling right; gyro y positive while pitching forward (the simulator's frame: x forward, y left, z up, right-hand rule). Wrong sign: the axis mapping is off |
| 3.8 | Hold it upright, still, then "Level here" | fwd and right read ~0; compare with the phone inclinometer on the torso: within ~1° |
| 3.9 | Show camera | live picture; Pi temperature rises a few degrees at most, not past 70 °C over 5 min |
| 3.10 | Unplug the BusLinker's USB, then plug it back and press Rescan | all 8 servo cards go grey; the page keeps working; after Rescan they come back |
| 3.11 | Unplug the IMU's SDA wire for a few s, then reconnect | IMU card shows an error, then recovers by itself (it re-zeroes the gyro: keep the robot still for 2 s) |
| 3.12 | Ctrl+C the dashboard, restart it | starts again cleanly |

## 4. Calibration (robot hanging)

| # | Test | Expect |
|---|---|---|
| 4.1 | Relax all, hold both legs straight (against a flat board), "Set zero" | all angles read ~0 |
| 4.2 | Scale check: bend one knee to exactly 90° with a set square | the knee reads 90° ± 2. Off by a constant ratio (e.g. 80°): tell me, the counts-per-degree is wrong |
| 4.3 | Same for one hip pitch (leg at 90° forward) and one ankle | ±2° |
| 4.4 | Each joint, torque on, **+1°** first, then **+5°** | moves the way the card says ("+ = knee bends", "+ = leg back", "+ = toes down", left hip roll "+ = leg outward", right hip roll "+ = leg inward"). Wrong way: Flip, and test again |
| 4.5 | After all flips: Relax all, legs straight again, check all read ~0 | yes. If not, Set zero again (flipping doesn't change zero) |
| 4.6 | Each joint relaxed: move by hand to where it touches something, back off ~3°, Lower here / Upper here | the range shown matches what you felt; a yellow note if it's narrower than the sim's |
| 4.7 | Write limits to servos, then Rescan | each card shows the same limits "in servo memory" |
| 4.8 | Try to drag a slider past a limit | it stops at the limit |
| 4.9 | Copy `robot/calibration.json` somewhere safe (or commit it) | |

## 5. First motions (robot hanging, hand near "Relax all")

| # | Test | Expect |
|---|---|---|
| 5.1 | One joint at a time: slider slowly across its whole range, move time 300 ms | smooth, no buzzing, no cable pulled tight anywhere in the range |
| 5.2 | Each joint: hold at home for 30 s | temperature rises by less than ~2 °C (no load yet) |
| 5.3 | Straight, then Home (2.5 s) | both legs move together, ends at 0 / −23 / 46 / −23 on every leg; feet parallel to the torso at home |
| 5.4 | Boot: straight → home | as above, in two smooth parts |
| 5.5 | Stop all during a slow move (move time 3000) | freezes where it is |
| 5.6 | Relax all while holding home | every leg goes limp at once |
| 5.7 | Mirror check: at home, look from the front | both legs mirror each other; hip rolls both at 0, legs parallel |

## 6. Protection (robot hanging)

| # | Test | Expect |
|---|---|---|
| 6.1 | Temporarily set `SERVO_TEMP_LIMIT = 35` in `robot/dashboard.py`, hold home until one servo reaches 35 °C (or warm it with your hand) | that servo relaxes, its card turns red and locks; it unlocks below 30 °C. Set the limit back to 60 |
| 6.2 | Same for the Pi with `PI_TEMP_LIMIT` set just above the current Pi temperature (run something heavy, e.g. the camera) | every servo relaxes; recovers when cooler. Set it back to 75 |
| 6.3 | Servo power switch off while holding home | cards go grey or stale, nothing crashes; on again + Rescan restores |

## 7. Power under load

| # | Test | Expect |
|---|---|---|
| 7.1 | Boot → Straight → Home, 10 times in a row, watching the battery card | it dips during moves and recovers; never below 10.5 V |
| 7.2 | Meanwhile check the Pi's "Power" | stays OK (no undervoltage) with the camera on |
| 7.3 | Multimeter on the logic pack / the 5 V rail during 7.1 | 5.0–5.25 V at the Pi |

## 8. Standing (on the floor, harness slack but attached)

| # | Test | Expect |
|---|---|---|
| 8.1 | Feet on the floor, legs straight, Boot | crouches to home and stands on its own |
| 8.2 | Standing at home: IMU | lean within a few degrees; note the values (it should match the sim's standing tilt) |
| 8.3 | Stand for 5 min, watch temperatures | knees and hip pitches warm the most. Stop at 55 °C and tell me how long it took (sets how hard training should penalize crouching) |
| 8.4 | Light push on the torso | leans and comes back (the servos hold; no policy yet) |
| 8.5 | Straight while standing | stands up taller without falling |

## 9. Walking (policy on the robot)

Before: `python scripts/export_policy.py runs/juan_v7_8_best` on the computer (already done for
v7_8_best: `robot/policies/juan_v7_8_best.npz`), then `git pull` on the Pi.

| # | Test | Expect |
|---|---|---|
| 9.1 | Hanging, feet off the ground: Boot, then Start the policy (command: stand still) | it starts; small leg motions; "step" time under ~5 ms of 20, "late" stays near 0 |
| 9.2 | Hanging: forward 0.1 | the legs step in a regular rhythm, both legs alike (mirror image), no buzzing or jerks; Stop (hold) freezes it |
| 9.3 | Hanging: close the browser tab while it runs | after 1 s the command goes to stand still, after 3 s it stops and holds |
| 9.4 | Hanging: tilt the robot past 40° by hand while it runs | it stops and goes limp |
| 9.5 | On the floor, harness slack: Boot, Start, stand still for 30 s | stands; lean stays within a few degrees |
| 9.6 | On the floor: forward 0.05, then 0.1, 0.15... | walks; stop at the first stumble (Space = stand still, Esc = stop) |
| 9.7 | Turning in place 0.15 → 0.3, then sideways | turns / side-steps in place |
| 9.8 | Copy the logs in `robot/logs/` to the computer | one file per run, for comparing with the simulator |

If 9.2 shows a leg moving the wrong way or out of rhythm, stop: a joint sign (calibration step 2) or
zero is wrong. Servo move time: start with 0 ms (as in training); try 20 ms if the motion is jerky.

## 10. The servo model from walking logs

The simulator's servo model is fitted from the robot's own walking logs, not a bench test:
`python scripts/analysis/replay.py robot/logs/<log>.npz` replays a log through the simulator over a
grid of servo stiffnesses and delays and reports which one matches the real joints best
(the result is in `biped_rl/robots/juan.py`, `MEASURED_KP` and `MEASURED_DELAY_STEPS`).
`python scripts/analysis/replay_video.py robot/logs/<log>.npz` renders the run in 3D with its telemetry.
