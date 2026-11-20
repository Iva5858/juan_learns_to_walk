"""Microphone test, from the dashboard (Actions tab, Voice card, "Test microphone"): record a few seconds, measure the
level, then play the recording back on the speaker, so one button checks the microphone and the speaker.

Records the way the voice commands listen (sounddevice, 16 kHz mono, the default input), or with `arecord` if
sounddevice is missing. numpy only otherwise.
"""
import shutil
import subprocess
import threading
import time

import numpy as np

RATE = 16000
SILENT_DB, QUIET_DB = -60.0, -50.0   # peak below SILENT_DB: no signal; level below QUIET_DB: too quiet for speech


def _db(v):
    return round(float(20 * np.log10(max(float(v), 1e-6))), 1)


def levels(x):
    """Level of 16-bit samples: rms and peak in dB of full scale (DC removed), and the share of clipped samples."""
    if len(x) == 0:
        return None
    f = x.astype(np.float64) / 32768.0
    f = f - f.mean()
    return {"rms_db": _db(np.sqrt(np.mean(f ** 2))), "peak_db": _db(np.abs(f).max()),
            "clipped": round(float(np.mean(np.abs(x.astype(np.int32)) >= 32767)), 4)}


def verdict(lv):
    if lv is None:
        return "no audio recorded"
    if lv["peak_db"] < SILENT_DB:
        return "silent: no signal (wrong input device, I2S mic not enabled, or it sends on the other channel)"
    if lv["clipped"] > 0.01:
        return "clipping: too loud or too much gain"
    if lv["rms_db"] < QUIET_DB:
        return "very quiet: speak closer, or raise the gain"
    return "ok"


def record(seconds, on_level=lambda db: None):
    """(int16 samples, input device name). Calls on_level(rms dB) about 10 times a second while recording."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        sd = None
    if sd is not None:
        chunks = []

        def callback(data, frames, t, status):
            a = np.frombuffer(bytes(data), dtype="<i2").copy()
            chunks.append(a)
            on_level(levels(a)["rms_db"])

        try:
            name = sd.query_devices(kind="input")["name"]
        except Exception:
            name = "default input"
        with sd.RawInputStream(samplerate=RATE, blocksize=RATE // 10, dtype="int16", channels=1, callback=callback):
            time.sleep(seconds)
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.int16)), name
    if shutil.which("arecord"):
        out = subprocess.run(["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw",
                              "-d", str(max(1, round(seconds)))], capture_output=True, timeout=seconds + 5, check=True).stdout
        return np.frombuffer(out, dtype="<i2"), "arecord default"
    raise OSError("nothing to record with: pip install sounddevice (the voice extra) or sudo apt install alsa-utils")


class MicTest:
    """One test at a time, in a background thread; state() for the page."""

    def __init__(self, voice, recorder=record):
        self.voice, self.recorder = voice, recorder
        self.lock = threading.Lock()
        self.running = False
        self.phase = None          # recording / done / error
        self.live_db = None
        self.result = None

    def start(self, seconds=3.0, play_back=True):
        seconds = max(1.0, min(10.0, float(seconds)))
        with self.lock:
            if self.running:
                return "a microphone test is already running"
            self.running, self.phase, self.live_db, self.result = True, "recording", None, None
        threading.Thread(target=self._run, args=(seconds, play_back), daemon=True).start()
        return None

    def _set_level(self, db):
        self.live_db = db

    def _run(self, seconds, play_back):
        try:
            x, device = self.recorder(seconds, self._set_level)
            lv = levels(x)
            self.result = {"device": device, "seconds": round(len(x) / RATE, 2), **(lv or {}), "verdict": verdict(lv),
                           "played_back": False}
            if play_back and lv and lv["peak_db"] >= SILENT_DB:
                f = x.astype(np.float64) / 32768.0
                f = f - f.mean()
                gain = min(0.7 / max(np.abs(f).max(), 1e-6), 30.0)   # loud enough to hear, at most +30 dB
                self.voice.play_samples(f * gain, RATE)
                self.result.update(played_back=True, playback_gain_db=_db(gain))
            self.phase = "done"
        except Exception as e:      # no device, device busy, no recorder
            self.result = {"verdict": f"error: {str(e)[:200]}"}
            self.phase = "error"
        finally:
            self.running = False

    def state(self):
        return {"running": self.running, "phase": self.phase, "live_db": self.live_db, "result": self.result}
