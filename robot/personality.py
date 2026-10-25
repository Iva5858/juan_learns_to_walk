"""Personality: gestures played on the servos, the voice, and playful mode, inside the dashboard.

  * Gestures (robot/gestures.py, checked in sim by scripts/check_gestures.py) stream one target pose
    per 20 ms to the servos, clamped to the calibrated limits. They start and end at home; from
    anywhere else the robot first moves to home over a second. Not while the walking policy runs.
  * Playful mode: while the robot stands at home with nothing else going on, every 10-25 s a small
    idle animation (settle, fidget, weight shift), sometimes with a chirp.
  * Reactions: a "whoa" when the robot is bumped (gyro spike), a sleepy whine when the battery is
    low, and sounds when the walking policy starts, stops or the robot falls.
"""
import math
import random
import threading
import time

from robot import gestures as G
from robot.joints import BY_NAME, HOME, NAMES
from robot.sound import Voice

IDLE_EVERY_S = (10.0, 25.0)
AT_HOME_DEG = 6.0           # "standing at home": every joint within this of home
START_TILT_DEG = 20         # gestures only start with the torso this close to upright
FALL_TILT_DEG = 40          # past this during a gesture: stop and relax (falling)
BUMP_RAD_S = 2.0            # gyro roll/pitch rate that counts as a bump
LOW_BATTERY_V = 10.5
WALKER_SOUNDS = {"start": "excited", "stop": "ok", "fall": "scared", "lost": "sad"}


class Personality:
    def __init__(self, robot):
        self.robot = robot
        self.voice = Voice()
        self.gesture, self._cancel = None, threading.Event()
        self.playful = False
        self.last_activity = time.monotonic()
        self._next_idle = random.uniform(*IDLE_EVERY_S)
        self._last_bump = self._last_battery = 0.0
        threading.Thread(target=self._watch, daemon=True).start()

    # ---- gestures ----
    def busy(self):
        return self.gesture is not None

    def idle_playing(self):
        return self.gesture in G.IDLE

    def play(self, name, bpm=None, quiet=False):
        r = self.robot
        if name not in G.ALL:
            return f"no gesture {name}"
        if self.busy():
            return f"already playing {self.gesture}"
        if r.walker.running:
            return "the walking policy is running: stop it first"
        if not r.cal.zero_set:
            return "not calibrated: set zero first"
        if any(BY_NAME[n][1] not in r.bus_of for n in NAMES):
            return "not all 8 servos found"
        if any(r.locked(BY_NAME[n][1]) for n in NAMES):
            return "a servo is locked (too hot)"
        label, _, sound, keys = G.ALL[name]
        if name == "dance" and bpm:
            keys = G.dance(bpm=max(70, min(120, float(bpm))))  # (checked in sim at 70-120 BPM)
        lean = self._tilt()
        if lean is None:
            return "no IMU reading"
        if lean > START_TILT_DEG:
            return f"not upright (tilted {lean:.0f} deg)"
        now = self._current_pose()
        if now is None:
            return "no position reading"
        frames = []
        if max(abs(now[n] - HOME[n]) for n in NAMES) > AT_HOME_DEG:
            frames += G.frames([(1.0, G.HOME, "smooth")], start=now)  # go home first
        frames += G.frames(keys)
        self.gesture = name
        self._cancel.clear()
        if sound and not quiet:
            self.voice.say(sound)
        threading.Thread(target=self._run, args=(frames,), daemon=True).start()
        return None

    def stop(self):
        self._cancel.set()

    def _tilt(self):
        st = self.robot.imu.state if getattr(self.robot, "imu", None) else None  # (the IMU starts after us)
        if st is None or time.time() - st["t"] > 0.2:
            return None
        return math.degrees(math.acos(max(-1.0, min(1.0, -st["gravity"][2]))))  # from upright, any direction

    def _current_pose(self):
        out = {}
        for n in NAMES:
            pos = self.robot.servo.get(BY_NAME[n][1], {}).get("pos")
            if pos is None:
                return None
            out[n] = self.robot.cal.joints[n].degrees(pos)
        return out

    def _run(self, frames):
        r, dt = self.robot, 1.0 / G.CONTROL_HZ
        next_t = time.monotonic()
        try:
            for f in frames:
                if self._cancel.is_set() or r.walker.running:
                    break
                lean = self._tilt()
                if lean is not None and lean > FALL_TILT_DEG:
                    r.message = f"Gesture stopped: tilted {lean:.0f} deg (falling): relaxed"
                    r.relax()
                    self.voice.say("scared")
                    break
                per_bus = {}
                for n in NAMES:
                    sid = BY_NAME[n][1]
                    counts = r.cal.joints[n].clamp_counts(n, f[n])
                    per_bus.setdefault(r.bus_of[sid], {})[sid] = counts
                    r.servo[sid]["target"], r.servo[sid]["torque"] = counts, True
                for bus, targets in per_bus.items():
                    bus.move_many(targets, 20)  # the servo glides over the 20 ms: smoother
                next_t += dt
                time.sleep(max(0.0, next_t - time.monotonic()))
        finally:
            self.gesture = None
            self.last_activity = time.monotonic()

    # ---- reactions and playful mode ----
    def walker_event(self, kind):
        self.last_activity = time.monotonic()
        if kind in WALKER_SOUNDS:
            self.voice.say(WALKER_SOUNDS[kind])

    def _idle(self):
        r = self.robot
        if self.busy() or r.walker.running or not r.cal.zero_set:
            return False
        now = self._current_pose()
        if now is None or any(not r.servo.get(BY_NAME[n][1], {}).get("torque") for n in NAMES):
            return False
        lean = self._tilt()
        if lean is None or lean > START_TILT_DEG:
            return False
        return max(abs(now[n] - HOME[n]) for n in NAMES) < AT_HOME_DEG

    def _watch(self):
        while True:
            time.sleep(0.05)
            now = time.monotonic()
            imu = getattr(self.robot, "imu", None)
            st = imu.state if imu else None
            if st and now - self._last_bump > 3.0 and math.hypot(st["gyro"][0], st["gyro"][1]) > BUMP_RAD_S:
                self._last_bump = now
                self.voice.say("scared")
            vins = [s.get("vin") for s in self.robot.servo.values() if s.get("vin")]
            if vins and max(vins) < LOW_BATTERY_V and now - self._last_battery > 120:
                self._last_battery = now
                self.voice.say("sleepy")
            if self.playful and now - self.last_activity > self._next_idle and self._idle():
                self.play(random.choice(list(G.IDLE)), quiet=True)
                if random.random() < 0.4:
                    self.voice.say("chirp")
                self._next_idle = random.uniform(*IDLE_EVERY_S)

    def state(self):
        return {"gesture": self.gesture, "playful": self.playful, "voice": self.voice.state(),
                "gestures": [{"name": k, "label": v[0], "description": v[1]} for k, v in G.GESTURES.items()]}
