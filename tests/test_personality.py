"""robot/personality.py without hardware: voice commands, greeting and the victory spin, on a stand-in
robot (walker, IMU, servos) and a silent voice."""
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot import personality as P


class SilentVoice:
    def __init__(self, *a, **k):
        self.said, self.chords = [], []

    def say(self, e):
        self.said.append(e)

    def speak_chords(self, text, translate=False):
        self.chords.append(text)

    def state(self):
        return {}


class FakeWalker:
    def __init__(self, running=True):
        self.running = running
        self.ctrl = SimpleNamespace(command=np.zeros(3))
        self.stopped = None

    def set_command(self, vx, vy, wz):
        self.ctrl.command = np.array([vx, vy, wz])

    def stop(self, why="stopped", relax=False):
        self.running, self.stopped = False, why


class FakeIMU:
    """Integrates the walker's turn command into a heading, like the real IMU yaw."""
    def __init__(self, walker):
        self.walker, self.yaw = walker, 0.0
        self.state = {"yaw": 0.0, "gravity": [0, 0, -1], "gyro": [0, 0, 0], "t": time.time()}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            time.sleep(0.01)
            self.yaw += self.walker.ctrl.command[2] * 0.4  # 40x real time
            self.state = {**self.state, "yaw": self.yaw, "t": time.time()}


@pytest.fixture
def pers(monkeypatch):
    monkeypatch.setattr(P, "Voice", SilentVoice)
    walker = FakeWalker()
    robot = SimpleNamespace(walker=walker, servo={}, cal=SimpleNamespace(zero_set=True))
    robot.imu = FakeIMU(walker)
    p = P.Personality(robot)
    played = []
    monkeypatch.setattr(p, "play", lambda name, **k: played.append(name))
    p.played = played
    return p


def test_voice_steers_a_running_policy(pers):
    pers._voice_command(("walk", 0.15), "juan walk")
    assert np.allclose(pers.robot.walker.ctrl.command, [0.15, 0, 0])
    pers._voice_command(("faster",), "juan faster")
    pers._voice_command(("turn", 0.4), "juan turn left")
    assert np.allclose(pers.robot.walker.ctrl.command, [0.2, 0, 0.4])
    pers._voice_command(("stop",), "juan stop")
    assert np.allclose(pers.robot.walker.ctrl.command, 0)


def test_voice_never_starts_walking(pers):
    pers.robot.walker.running = False
    pers._voice_command(("walk", 0.15), "juan walk")
    assert np.allclose(pers.robot.walker.ctrl.command, 0) and pers.voice.said[-1] == "curious"


def test_voice_gestures_and_answers(pers):
    pers._voice_command(("gesture", "dance"), "juan dance")
    pers._voice_command(("speak", "I am Juan."), "juan who are you")
    assert pers.played == ["dance"] and pers.voice.chords == ["I am Juan."]


def test_victory_spin_turns_a_full_circle_then_bounces(pers):
    assert pers.victory_spin() is None
    for _ in range(300):
        if not pers.spinning:
            break
        time.sleep(0.02)
    assert not pers.spinning
    assert pers.robot.imu.yaw >= 2 * math.pi - 0.5        # turned (about) a full circle
    assert pers.robot.walker.stopped == "stopped: victory spin"
    assert pers.played == ["happy_bounce"]


def test_spin_needs_the_policy(pers):
    pers.robot.walker.running = False
    assert pers.victory_spin() == "start the walking policy first"
