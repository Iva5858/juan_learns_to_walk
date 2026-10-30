"""Personality: gestures played on the servos, the voice, and playful mode, inside the dashboard.

  * Gestures (robot/gestures.py, checked in sim by scripts/check_gestures.py) stream one target pose
    per 20 ms to the servos, clamped to the calibrated limits. They start and end at home; from
    anywhere else the robot first moves to home over a second. Not while the walking policy runs.
  * Playful mode: while the robot stands at home with nothing else going on, every 10-25 s a small
    idle animation (settle, fidget, weight shift), sometimes with a chirp.
  * Reactions: a "whoa" when the robot is bumped (gyro spike), a sleepy whine when the battery is
    low, and sounds when the walking policy starts, stops or the robot falls.
  * Greeting faces (robot/faces.py): a happy bounce and chirp when someone appears, if standing idle.
  * Voice commands (robot/voice_commands.py, needs a microphone): steer the running walking policy,
    play gestures, answer in chord speech. Voice never starts the walking policy.
  * Victory spin: turn in place a full circle with the walking policy, stop, then a happy bounce.
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
SPIN_RATE = 0.6             # rad/s turn command for the victory spin (~11 s per circle in sim)
SPIN_TIMEOUT_S = 25.0


class Personality:
    def __init__(self, robot):
        self.robot = robot
        self.voice = Voice()
        self.gesture, self._cancel = None, threading.Event()
        self.playful = False
        self.last_activity = time.monotonic()
        self._next_idle = random.uniform(*IDLE_EVERY_S)
        self._last_bump = self._last_battery = 0.0
        self.faces = self.listener = None
        self.faces_error = self.listener_error = None
        self.spinning = False
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

    # ---- faces, voice commands, victory spin ----
    def set_faces(self, on):
        """Greet people the camera sees (robot/faces.py)."""
        if not on:
            if self.faces:
                self.faces.stop()
            self.faces = None
            return None
        if self.faces:
            return None
        try:
            from robot.faces import FaceWatcher, jpeg_frames
            self.faces = FaceWatcher(lambda: jpeg_frames(self.robot.camera.stream()), self._greet)
            self.faces_error = None
        except Exception as e:  # no OpenCV, no model, no camera
            self.faces_error = str(e)[:200]
        return self.faces_error

    def _greet(self, face):
        self.last_activity = time.monotonic()
        if self._idle():
            self.play("happy_bounce")  # (its own happy chirp)
        else:
            self.voice.say("happy")

    def set_listening(self, on):
        """Voice commands through a microphone (robot/voice_commands.py)."""
        if not on:
            if self.listener:
                self.listener.close()
            self.listener = None
            return None
        if self.listener:
            return None
        try:
            from robot.voice_commands import VoiceCommands
            self.listener = VoiceCommands(self._voice_command, wake=True)
            self.listener_error = None
        except Exception as e:  # no microphone, no vosk, no model
            self.listener_error = str(e)[:200]
        return self.listener_error

    def _voice_command(self, action, text):
        w, kind = self.robot.walker, action[0]
        self.last_activity = time.monotonic()
        if kind == "speak":
            self.voice.speak_chords(action[1], translate=True)
        elif kind == "gesture":
            if self.play(action[1]):  # an error: walking, already playing...
                self.voice.say("curious")
        elif kind == "spin":
            if self.victory_spin():
                self.voice.say("curious")
        elif not w.running:
            self.voice.say("curious")  # walking commands only steer a running policy
        else:
            vx, vy, wz = w.ctrl.command
            vx, wz = {"walk": (action[-1], 0.0), "turn": (vx, action[-1]), "faster": (vx + 0.05, wz),
                      "slower": (vx - 0.05, wz), "stop": (0.0, 0.0)}[kind]
            w.set_command(vx, 0.0 if kind in ("walk", "stop") else vy, wz)
            self.voice.say("ok")

    def victory_spin(self):
        """Turn in place a full circle (IMU heading), stop the walking policy, then a happy bounce."""
        w = self.robot.walker
        if not w.running:
            return "start the walking policy first"
        if self.spinning:
            return "already spinning"
        self.spinning = True
        threading.Thread(target=self._spin, daemon=True).start()
        return None

    def _spin(self):
        r, w = self.robot, self.robot.walker
        try:
            last, turned, t0 = r.imu.state["yaw"], 0.0, time.monotonic()
            while w.running and abs(turned) < 2 * math.pi and time.monotonic() - t0 < SPIN_TIMEOUT_S:
                w.set_command(0.0, 0.0, SPIN_RATE)
                time.sleep(0.05)
                yaw = r.imu.state["yaw"]
                turned += (yaw - last + math.pi) % (2 * math.pi) - math.pi
                last = yaw
            if not w.running:
                return
            w.set_command(0.0, 0.0, 0.0)
            time.sleep(1.5)  # settle into standing
            w.stop("stopped: victory spin")
            time.sleep(0.5)
            self.play("happy_bounce")
        finally:
            self.spinning = False

    def state(self):
        return {"gesture": self.gesture, "playful": self.playful, "voice": self.voice.state(),
                "greet_faces": self.faces is not None, "faces": self.faces.state() if self.faces else None,
                "faces_error": self.faces_error, "listening": self.listener is not None,
                "listener_error": self.listener_error or (self.listener.error if self.listener else None),
                "spinning": self.spinning,
                "gestures": [{"name": k, "label": v[0], "description": v[1]} for k, v in G.GESTURES.items()]}
