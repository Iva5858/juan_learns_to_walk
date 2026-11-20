# The real robot

Everything in [robot/](../robot/) runs on the Pi and needs only `pyserial`, `smbus2` and `numpy` (plus `python3-picamera2`), not the training stack. Setup and first power-on, step by step: [bringup.md](bringup.md).

```bash
# on the Pi, once
sudo apt install python3-serial python3-smbus2 python3-numpy python3-picamera2 i2c-tools alsa-utils espeak-ng
# on the Pi, from the project folder
python3 -m robot.selftest     # reads every sensor, moves nothing
python3 -m robot.dashboard    # then open http://<pi-name>.local:8000
```

The dashboard shows all 8 servos (IDs 1–4 left leg, 5–8 right, in the order hip roll, hip pitch, knee, ankle), the IMU, camera, battery and Pi health, and walks you through calibration: zero with the legs straight, each joint's direction, physical limits set by hand. The walking policy runs from the same page ([walker.py](../robot/walker.py)): start at Home, then steer from the Drive tab: number keys 0–6 set the speed (0.1 m/s each), ↑/↓ change it by 0.05 m/s per press, ←/→ held turn, the preset buttons, the joystick (phone) or a gamepad. The page has four tabs (Drive, Actions, Status, Setup; files in [robot/web/](../robot/web/)); the previous page is still at `/classic`. It stands if the link to the page stalls for 2 s and stops after 6 s, relaxes everything past 40° of tilt, and logs every run to `robot/logs/`. The page offers the policies named in [robot/policies/roles.json](../robot/policies/roles.json), one per role (walk, preselected; speed; getup for "Get up"; set with `python scripts/policies.py <role> <name>`); every other export stays in the folder, unlisted. Without roles.json the page lists every policy in `robot/policies/` and preselects the one named in `robot/policies/default.txt`.

The code: [body.py](../robot/body.py) runs the hardware and the safety rules, [actions.py](../robot/actions.py) declares everything the page can ask for (`POST /api/<action>`; `GET /api/actions` lists them, `GET /api/events` pushes the state), [dashboard.py](../robot/dashboard.py) is the web server. Without the robot, `python3 -m robot.dashboard --fake` runs it all on stand-in hardware ([fake.py](../robot/fake.py)), with its own calibration and logs in `robot/logs/fake/`. On a computer with the training stack, `python -m robot.dashboard --sim` runs the same code on Juan in MuJoCo ([sim.py](../robot/sim.py); `--physics nominal|measured|randomized`, `--seed` picks the randomized robot; Mac only), with its calibration and logs in `robot/logs/sim/`.

[robot/policy.py](../robot/policy.py) is a numpy-only copy of the trained actor and of the simulator's observation pipeline; `export_policy.py` checks both against the simulator, step by step, before saving. Gestures, a chirping voice and a playful mode live in [personality.py](../robot/personality.py), [gestures.py](../robot/gestures.py) and [sound.py](../robot/sound.py).

The camera (Actions tab; OpenCV and three OpenCV-zoo models in `robot/models/`, not in git: the download links are in each file's docstring):
- **Greet faces** ([faces.py](../robot/faces.py), YuNet): a happy bounce when someone appears.
- **React to expressions** ([emotions.py](../robot/emotions.py), MobileFaceNet expression model on the YuNet face, aligned by its 5 landmarks): happy → happy bounce, sad → sad, surprised → excited, once an expression is held over 3 of 5 frames, at most every 8 s; while walking, the sound only. The camera sits 41 cm up and sees 24° above level, so it reads faces at its own height (someone crouching or seated), not a standing adult's.
- **Follow a person** ([follow.py](../robot/follow.py), NanoDet-Plus, people only): steers the running walking policy like the joystick: turns toward the nearest person, walks to ~1.3 m of them (distance from where their feet meet the floor), at most 0.2 m/s and 0.5 rad/s, standing still when they are lost and then turning toward where they were last seen for 6 s. It takes the steering (the joystick, voice and the victory spin wait) and doesn't count as the page being there: with the page gone the walker stands and stops as usual. Use a policy that turns in place, the `follow` role (`python scripts/policies.py follow <name>`); `scripts/analysis/follow_test.py` compares policies on it in the simulator.
Each runs on a computer's webcam too: `python -m robot.faces`, `python -m robot.emotions`, `python -m robot.follow`.

**Test microphone** (Actions tab, Voice card; [mic.py](../robot/mic.py)): records 3 s from the default input (sounddevice, as the voice commands listen, or `arecord`), shows the level and peak in dB and a verdict (ok / silent / very quiet / clipping) with the input's name, then plays the recording back on the speaker, raised up to +30 dB so it can be heard. One press checks both. It waits while voice commands hold the microphone.

Servo and calibration tools, run on the Pi with the dashboard stopped:
- [center_servos.py](../robot/center_servos.py): leg servos to the middle of their range (500 counts), one at a time, each held while its horn goes back on straight (re-zeroing the legs).
- [carry_limits.py](../robot/carry_limits.py): carry a calibration's joint limits and directions over to a new zero, after re-horning the servos.
- [calibrate_camera.py](../robot/calibrate_camera.py): capture checkerboard views and compute the lens calibration for IMU stabilization ([camera-stabilization.md](camera-stabilization.md)).

Joint signs (torso frame: x forward, y left, z up), positive rotation: hip roll = left leg outward / right leg inward; hip pitch = leg swings back; knee = bends; ankle = toes down. Legs straight = 0.
