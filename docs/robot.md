# The real robot

Everything in [robot/](../robot/) runs on the Pi and needs only `pyserial`, `smbus2` and `numpy` (plus `python3-picamera2`), not the training stack. Setup and first power-on, step by step: [bringup.md](bringup.md).

```bash
# on the Pi, once
sudo apt install python3-serial python3-smbus2 python3-numpy python3-picamera2 i2c-tools alsa-utils espeak-ng
# on the Pi, from the project folder
python3 -m robot.selftest     # reads every sensor, moves nothing
python3 -m robot.dashboard    # then open http://<pi-name>.local:8000
```

The dashboard shows all 8 servos (IDs 1–4 left leg, 5–8 right, in the order hip roll, hip pitch, knee, ankle), the IMU, camera, battery and Pi health, and walks you through calibration: zero with the legs straight, each joint's direction, physical limits set by hand. The walking policy runs from the same page ([walker.py](../robot/walker.py)): start at Home, then steer from the Drive tab: number keys 0–6 set the speed (0.1 m/s each), ↑/↓ held ramp it, ←/→ held turn, the preset buttons, the joystick (phone) or a gamepad. The page has four tabs (Drive, Actions, Status, Setup; files in [robot/web/](../robot/web/)); the previous page is still at `/classic`. It stands if the link to the page stalls for 2 s and stops after 6 s, relaxes everything past 40° of tilt, and logs every run to `robot/logs/`. The policy dropdown lists `robot/policies/`; the default is in `robot/policies/default.txt`.

The code: [body.py](../robot/body.py) runs the hardware and the safety rules, [actions.py](../robot/actions.py) declares everything the page can ask for (`POST /api/<action>`; `GET /api/actions` lists them, `GET /api/events` pushes the state), [dashboard.py](../robot/dashboard.py) is the web server. Without the robot, `python3 -m robot.dashboard --fake` runs it all on stand-in hardware ([fake.py](../robot/fake.py)), with its own calibration and logs in `robot/logs/fake/`.

[robot/policy.py](../robot/policy.py) is a numpy-only copy of the trained actor and of the simulator's observation pipeline; `export_policy.py` checks both against the simulator, step by step, before saving. Gestures, a chirping voice and a playful mode live in [personality.py](../robot/personality.py), [gestures.py](../robot/gestures.py) and [sound.py](../robot/sound.py).

Joint signs (torso frame: x forward, y left, z up), positive rotation: hip roll = left leg outward / right leg inward; hip pitch = leg swings back; knee = bends; ankle = toes down. Legs straight = 0.
