"""The dashboard's microphone test (robot/mic.py) with a stand-in recorder: levels, verdicts, playback."""
import time

import numpy as np

from robot import mic
from robot.sound import RATE as VOICE_RATE
from robot.sound import Voice


class Player:
    def __init__(self):
        self.played = []

    def play_samples(self, samples, rate):
        self.played.append((samples, rate))


def tone(amplitude, seconds=1.0):
    t = np.arange(int(mic.RATE * seconds)) / mic.RATE
    return (np.clip(np.sin(2 * np.pi * 440 * t) * amplitude, -1, 1) * 32767).astype(np.int16)


def run(samples):
    player = Player()
    m = mic.MicTest(player, recorder=lambda s, on_level: (on_level(-30.0), (samples, "test mic"))[1])
    assert m.start(1) is None
    for _ in range(100):
        if not m.running:
            break
        time.sleep(0.01)
    return m.state(), player.played


def test_speech_level_is_ok_and_played_back_louder():
    s, played = run(tone(0.1))
    r = s["result"]
    assert s["phase"] == "done" and r["verdict"] == "ok" and r["device"] == "test mic"
    assert abs(r["peak_db"] + 20) < 0.5 and abs(r["rms_db"] + 23) < 0.5 and r["played_back"]
    assert len(played) == 1 and played[0][1] == mic.RATE and np.abs(played[0][0]).max() <= 0.71


def test_silence_quiet_and_clipping():
    s, played = run(np.zeros(mic.RATE, np.int16))
    assert s["result"]["verdict"].startswith("silent") and not played
    assert run(tone(0.002))[0]["result"]["verdict"].startswith("very quiet")
    assert run(tone(1.5))[0]["result"]["verdict"].startswith("clipping")


def test_recorder_errors_are_reported():
    def broken(seconds, on_level):
        raise OSError("device busy")
    m = mic.MicTest(Player(), recorder=broken)
    m.start(1)
    time.sleep(0.1)
    assert m.state()["phase"] == "error" and "device busy" in m.state()["result"]["verdict"]


def test_voice_resamples_the_recording():
    v, queued = Voice(), []
    v._put = queued.append
    v.play_samples(np.zeros(mic.RATE), mic.RATE)
    kind, samples = queued[0]
    assert kind == "samples" and abs(len(samples) - VOICE_RATE) <= 1
