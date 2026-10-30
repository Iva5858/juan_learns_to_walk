"""Voice commands: offline speech recognition (Vosk) limited to a fixed list of phrases, which is far
more reliable than free speech and runs on a Raspberry Pi 4. Works with any microphone: the Mac's for
testing, a USB microphone on the robot.

    python -m robot.voice_commands            # listen and print what it understood (Ctrl+C stops)
    python -m robot.voice_commands --wake     # only react to phrases starting with "Juan"

Setup: pip install "vosk==0.3.44" sounddevice (the project's "voice" extra), and the small English
model (about 70 MB) unzipped into robot/models/:
    https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
"""
import json
import os
import queue
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "vosk-model-small-en-us-0.15")
WAKE_WORD = "juan"
RATE = 16000

# phrase -> action. ("walk", forward m/s) / ("turn", rad/s) / ("faster",) / ("slower",) / ("stop",) /
# ("gesture", name in robot/gestures.py) / ("speak", text) / ("spin",) (victory spin)
PHRASES = {
    "walk": ("walk", 0.15), "go": ("walk", 0.15), "forward": ("walk", 0.15), "walk forward": ("walk", 0.15),
    "come here": ("walk", 0.15),
    "back": ("walk", -0.1), "backward": ("walk", -0.1), "go back": ("walk", -0.1), "walk backward": ("walk", -0.1),
    "faster": ("faster",), "slower": ("slower",),
    "left": ("turn", 0.4), "turn left": ("turn", 0.4), "right": ("turn", -0.4), "turn right": ("turn", -0.4),
    "stop": ("stop",), "stand": ("stop",), "halt": ("stop",), "stay": ("stop",),
    "dance": ("gesture", "dance"), "bounce": ("gesture", "happy_bounce"), "happy": ("gesture", "happy_bounce"),
    "bow": ("gesture", "bow"), "hello": ("gesture", "wake_up"), "sway": ("gesture", "sway"),
    "sad": ("gesture", "sad"), "tap": ("gesture", "toe_tap"),
    "who are you": ("speak", "I am Juan. I am learning to walk."),
    "spin": ("spin",), "victory": ("spin",),
}


def parse(text, wake=False):
    """The action for a recognised phrase, or None. With wake, the phrase must start with "Juan"."""
    words = text.lower().split()
    if wake:
        if not words or words[0] != WAKE_WORD:
            return None
    words = [w for w in words if w != WAKE_WORD]
    phrase = " ".join(words)
    if phrase in PHRASES:
        return PHRASES[phrase]
    for n in range(len(words), 0, -1):  # the longest known phrase inside what was heard
        for i in range(len(words) - n + 1):
            part = " ".join(words[i:i + n])
            if part in PHRASES:
                return PHRASES[part]
    return None


class VoiceCommands:
    """Listens in a background thread; calls on_command(action, text) for every phrase it understands
    and on_heard(text) for everything it hears (also what it doesn't understand)."""

    def __init__(self, on_command, on_heard=None, wake=False, model_dir=MODEL_DIR, device=None):
        import sounddevice
        import vosk
        vosk.SetLogLevel(-1)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"speech model not found: {model_dir} (see robot/voice_commands.py)")
        grammar = sorted({p for p in PHRASES} | {f"{WAKE_WORD} {p}" for p in PHRASES}) + [WAKE_WORD, "[unk]"]
        self.recognizer = vosk.KaldiRecognizer(vosk.Model(model_dir), RATE, json.dumps(grammar))
        self.on_command, self.on_heard, self.wake = on_command, on_heard, wake
        self.error = None
        self._audio = queue.Queue(maxsize=50)
        self._stream = sounddevice.RawInputStream(samplerate=RATE, blocksize=4000, dtype="int16", channels=1,
                                                  device=device, callback=self._on_audio)
        self._stream.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _on_audio(self, data, frames, time, status):
        try:
            self._audio.put_nowait(bytes(data))
        except queue.Full:
            pass  # recognition fell behind: drop audio rather than lag

    def _loop(self):
        while True:
            data = self._audio.get()
            try:
                if self.recognizer.AcceptWaveform(data):
                    text = json.loads(self.recognizer.Result()).get("text", "").replace("[unk]", "").strip()
                    if not text:
                        continue
                    if self.on_heard:
                        self.on_heard(text)
                    action = parse(text, self.wake)
                    if action:
                        self.on_command(action, text)
            except Exception as e:
                self.error = str(e)[:200]

    def close(self):
        self._stream.stop()
        self._stream.close()


if __name__ == "__main__":
    import sys
    import time

    from robot.sound import Voice
    voice = Voice()
    wake = "--wake" in sys.argv

    def command(action, text):
        print(f"  -> {action}", flush=True)
        voice.say("ok")

    listener = VoiceCommands(command, on_heard=lambda t: print(f"heard: {t!r}", flush=True), wake=wake)
    print("Listening" + (' (start with "Juan")' if wake else "") + ". Try: walk, stop, turn left, back, faster, "
          "dance, bounce, bow, who are you. Ctrl+C stops.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        listener.close()
