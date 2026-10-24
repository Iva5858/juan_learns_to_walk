"""Juan's voice: procedurally generated robot chirps per emotion (no recordings needed; a little
different every time), your own .wav files in robot/sounds/, and optional text-to-speech.

Plays through whatever audio output there is: `aplay` (ALSA, Raspberry Pi OS; pick a device with
the JUAN_AUDIO_DEVICE environment variable, e.g. "plughw:1,0" for a USB speaker) or `afplay` (Mac,
for previewing). Without an audio device it quietly does nothing. Sounds queue, one at a time.

    python3 -m robot.sound happy          # play one
    python3 -m robot.sound --all          # every emotion, one after the other
    python3 -m robot.sound --save out/    # write them as .wav files
"""
import io
import math
import os
import queue
import random
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np

RATE = 22050
HERE = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(HERE, "sounds")


# ---------------------------------------------------------------- synthesis
def syllable(f0, f1, seconds, vibrato=0.0, vib_hz=18.0, bright=0.35, curve=1.0):
    """One chirp: pitch glides f0 -> f1 (Hz), with optional vibrato (fraction of pitch), a soft
    attack and release, and a few harmonics for a warm, toy-like timbre."""
    n = max(1, int(seconds * RATE))
    u = np.linspace(0.0, 1.0, n)
    freq = f0 * (f1 / f0) ** (u ** curve)  # exponential glide sounds natural
    freq = freq * (1.0 + vibrato * np.sin(2 * np.pi * vib_hz * u * seconds))
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    wave_ = np.sin(phase) + bright * np.sin(2 * phase) + 0.5 * bright ** 2 * np.sin(3 * phase)
    attack, release = min(0.012, seconds / 4), min(0.04, seconds / 3)
    env = np.minimum(1.0, np.minimum(u * seconds / attack, (1 - u) * seconds / release))
    return wave_ * env


def silence(seconds):
    return np.zeros(int(seconds * RATE))


def phrase(parts):
    return np.concatenate(parts)


def _r(rng, a, b):
    return rng.uniform(a, b)


def make(emotion, seed=None):
    """A short phrase for an emotion, as float samples in [-1, 1]."""
    rng = random.Random(seed)
    base = _r(rng, 700, 900)
    p = []
    if emotion == "happy":       # bright, rising, bouncy
        for i in range(rng.randint(3, 4)):
            f = base * (1 + 0.12 * i) * _r(rng, 0.95, 1.08)
            p += [syllable(f, f * 1.5, _r(rng, 0.07, 0.11), vibrato=0.02), silence(0.035)]
        p += [syllable(base * 1.4, base * 2.2, 0.16, vibrato=0.03)]
    elif emotion == "excited":   # fast, high, trilling
        for _ in range(rng.randint(5, 7)):
            f = base * _r(rng, 1.2, 1.7)
            p += [syllable(f, f * _r(rng, 1.2, 1.6), _r(rng, 0.045, 0.07), vibrato=0.04), silence(0.02)]
        p += [syllable(base * 1.6, base * 2.6, 0.2, vibrato=0.06, vib_hz=24)]
    elif emotion == "sad":       # slow, low, falling
        p += [syllable(base * 0.8, base * 0.55, 0.32, vibrato=0.015, vib_hz=6), silence(0.08),
              syllable(base * 0.6, base * 0.38, 0.5, vibrato=0.02, vib_hz=5, curve=0.6)]
    elif emotion == "curious":   # a question: up at the end
        p += [syllable(base * 0.9, base * 0.85, 0.09), silence(0.05),
              syllable(base * 0.8, base * 1.45, 0.24, vibrato=0.01, curve=1.8)]
    elif emotion == "hello":     # two-note greeting, up then a little flourish
        p += [syllable(base, base * 1.05, 0.12), silence(0.05), syllable(base * 1.33, base * 1.4, 0.16, vibrato=0.02),
              silence(0.03), syllable(base * 1.5, base * 1.9, 0.09)]
    elif emotion == "sleepy":    # long yawn: up, then a slow slide down
        p += [syllable(base * 0.6, base * 0.9, 0.35, bright=0.2, curve=0.7),
              syllable(base * 0.9, base * 0.45, 0.6, bright=0.2, vibrato=0.01, vib_hz=4)]
    elif emotion == "scared":    # quick wobbling "whoa"
        p += [syllable(base * 1.3, base * 2.0, 0.08), syllable(base * 2.0, base * 0.9, 0.3, vibrato=0.06, vib_hz=14)]
    elif emotion == "alert":     # two-tone beep-boop
        for _ in range(2):
            p += [syllable(base * 1.6, base * 1.6, 0.09, bright=0.6), silence(0.03),
                  syllable(base * 1.2, base * 1.2, 0.09, bright=0.6), silence(0.06)]
    elif emotion == "ok":        # short confirming blip
        p += [syllable(base * 1.1, base * 1.5, 0.07), silence(0.03), syllable(base * 1.5, base * 1.5, 0.06)]
    elif emotion == "startup":   # ascending arpeggio
        for i, step in enumerate((1.0, 1.26, 1.5, 2.0)):
            p += [syllable(base * 0.7 * step, base * 0.7 * step * 1.02, 0.09 + 0.02 * (i == 3)), silence(0.02)]
    elif emotion == "goodbye":   # descending arpeggio, soft
        for step in (2.0, 1.5, 1.26, 1.0):
            p += [syllable(base * 0.7 * step, base * 0.7 * step * 0.98, 0.1, bright=0.2), silence(0.03)]
    elif emotion == "chirp":     # tiny idle chirp
        f = base * _r(rng, 1.1, 1.6)
        p += [syllable(f, f * _r(rng, 0.8, 1.4), _r(rng, 0.05, 0.09), vibrato=0.02)]
    else:
        raise KeyError(emotion)
    x = phrase(p)
    return 0.8 * x / max(1e-6, np.max(np.abs(x)))


EMOTIONS = ("happy", "excited", "sad", "curious", "hello", "sleepy", "scared", "alert", "ok", "startup",
            "goodbye", "chirp")


def to_wav(samples, volume=1.0):
    pcm = (np.clip(samples * volume, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------- playback
class Voice:
    """Plays sounds one at a time in a background thread."""

    def __init__(self, volume=0.7):
        self.volume = volume
        self.device = os.environ.get("JUAN_AUDIO_DEVICE")
        self.backend = "aplay" if shutil.which("aplay") else "afplay" if shutil.which("afplay") else None
        self.tts = "espeak-ng" if shutil.which("espeak-ng") else "espeak" if shutil.which("espeak") else None
        self.error = None if self.backend else "no audio player (aplay / afplay) found"
        self.playing = None
        self._q = queue.Queue(maxsize=4)
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, emotion):
        """Queue an emotion's phrase (dropped if a few are already waiting)."""
        self._put(("emotion", emotion))

    def play_file(self, name):
        self._put(("file", os.path.join(SOUND_DIR, os.path.basename(name))))

    def speak(self, text):
        self._put(("tts", text[:200]))

    def files(self):
        return sorted(f for f in os.listdir(SOUND_DIR) if f.endswith(".wav")) if os.path.isdir(SOUND_DIR) else []

    def _put(self, item):
        if self.backend is None:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass

    def _play_wav_bytes(self, data):
        if self.backend == "aplay":
            cmd = ["aplay", "-q"] + (["-D", self.device] if self.device else []) + ["-"]
            subprocess.run(cmd, input=data, timeout=10, check=True, capture_output=True)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
            try:
                subprocess.run(["afplay", "-v", str(self.volume), f.name], timeout=10, check=True)
            finally:
                os.unlink(f.name)

    def _loop(self):
        while True:
            kind, what = self._q.get()
            self.playing = what
            try:
                if kind == "emotion":
                    vol = self.volume if self.backend == "aplay" else 1.0  # afplay sets its own volume
                    self._play_wav_bytes(to_wav(make(what), vol))
                elif kind == "file":
                    with open(what, "rb") as f:
                        self._play_wav_bytes(f.read())
                elif kind == "tts":
                    if self.tts is None:
                        raise OSError("no text-to-speech: sudo apt install espeak-ng")
                    wav = subprocess.run([self.tts, "-s", "150", "-p", "70", "--stdout", what],
                                         capture_output=True, timeout=10, check=True).stdout
                    self._play_wav_bytes(wav)
                self.error = None
            except Exception as e:  # no device, device busy, bad file...
                self.error = str(e)[:200]
            self.playing = None

    def state(self):
        return {"backend": self.backend, "tts": self.tts, "volume": self.volume, "error": self.error,
                "playing": self.playing, "files": self.files(), "emotions": EMOTIONS}


if __name__ == "__main__":
    import sys
    import time
    args = sys.argv[1:]
    if args[:1] == ["--save"]:
        out = args[1] if len(args) > 1 else "."
        os.makedirs(out, exist_ok=True)
        for e in EMOTIONS:
            with open(os.path.join(out, f"{e}.wav"), "wb") as f:
                f.write(to_wav(make(e, seed=1)))
        print(f"wrote {len(EMOTIONS)} files to {out}")
    else:
        v = Voice()
        for e in (EMOTIONS if args[:1] == ["--all"] else args or ["happy"]):
            print(e)
            v.say(e)
            time.sleep(0.3)
            while v.playing or not v._q.empty():
                time.sleep(0.05)
        if v.error:
            print("audio error:", v.error)
