"""Face detection: notices when someone appears in front of the camera, so Juan can greet them.

OpenCV's YuNet detector (a ~230 kB network, a few tens of ms per frame on a Pi 4 at 320x240) reads
frames from any source: the robot's camera stream on the Pi, a webcam on a computer for testing.
A greeting fires when a face *appears*: seen in 2 of the last 3 frames after at least ABSENT_S
without one, and at most once every COOLDOWN_S, so it doesn't bounce non-stop at someone standing
there.

    python -m robot.faces              # the computer's webcam: boxes on screen, a chirp on each greeting

Setup: OpenCV (Raspberry Pi OS: sudo apt install python3-opencv; elsewhere the "vision" extra) and the
model, saved as robot/models/face_detection_yunet_2023mar.onnx:
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
"""
import os
import threading
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
DETECT_SIZE = (320, 240)  # frames are shrunk to this before detection
MIN_SCORE = 0.75
ABSENT_S = 3.0            # a face must have been gone this long for the next one to count as new
COOLDOWN_S = 20.0         # at most one greeting this often


class FaceDetector:
    def __init__(self, model=MODEL):
        import cv2
        if not os.path.exists(model):
            raise FileNotFoundError(f"face model not found: {model} (see robot/faces.py)")
        self.cv2 = cv2
        self.net = cv2.FaceDetectorYN.create(model, "", DETECT_SIZE, MIN_SCORE)

    def detect(self, image):
        """Faces in a BGR image as (x, y, w, h, score) in that image's pixels, largest first."""
        h, w = image.shape[:2]
        small = self.cv2.resize(image, DETECT_SIZE)
        _, found = self.net.detect(small)
        if found is None:
            return []
        sx, sy = w / DETECT_SIZE[0], h / DETECT_SIZE[1]
        faces = [(f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy, float(f[14])) for f in found]
        return sorted(faces, key=lambda f: -f[2] * f[3])


class FaceWatcher:
    """Reads frames from `frames()` (an iterator of BGR images) in a background thread, at most `fps`
    a second, and calls on_greet(face) when a face appears. With frames=None, feed it with step()."""

    def __init__(self, frames, on_greet, fps=4.0, detector=None):
        self.detector = detector or FaceDetector()
        self.frames, self.on_greet, self.period = frames, on_greet, 1.0 / fps
        self.faces, self.last_seen, self.last_greet = [], 0.0, -COOLDOWN_S
        self.error = None
        self.running = True
        self._recent = deque(maxlen=3)
        if frames is not None:
            threading.Thread(target=self._loop, daemon=True).start()

    def step(self, image, now=None):
        """One frame: detect, update the state, greet if a face just appeared. Returns the faces."""
        now = time.monotonic() if now is None else now
        faces = self.detector.detect(image)
        self.faces = faces
        self._recent.append(bool(faces))
        present = bool(faces) and sum(self._recent) >= 2  # in this frame, and 2 of the last 3
        if present:
            if now - self.last_seen > ABSENT_S and now - self.last_greet > COOLDOWN_S:
                self.last_greet = now
                self.on_greet(faces[0])
            self.last_seen = now
        return faces

    def _loop(self):
        last = 0.0
        while self.running:
            try:
                for image in self.frames():
                    if not self.running:
                        return
                    if time.monotonic() - last < self.period:
                        continue
                    last = time.monotonic()
                    self.step(image)
                    self.error = None
            except Exception as e:  # camera gone, decoding error...
                self.error = str(e)[:200]
            time.sleep(1.0)  # the source ended or failed: try again

    def stop(self):
        self.running = False

    def state(self):
        return {"faces": len(self.faces), "seen_s_ago": time.monotonic() - self.last_seen if self.last_seen else None,
                "error": self.error}


def jpeg_frames(jpeg_stream):
    """BGR images from an iterator of JPEG bytes (the robot camera's MJPEG stream)."""
    import cv2
    import numpy as np
    for jpeg in jpeg_stream:
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            yield image


if __name__ == "__main__":
    import cv2

    from robot.sound import Voice
    voice = Voice()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("no webcam (on a Mac: allow camera access for the terminal in System Settings)")
    greeted = []

    def greet(face):
        greeted.append(time.monotonic())
        voice.say("happy")
        print(f"hello! face {int(face[2])}x{int(face[3])} px, score {face[4]:.2f}", flush=True)

    watcher = FaceWatcher(None, greet)
    print("Looking for faces: q quits.", flush=True)
    last = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if time.monotonic() - last >= 0.25:
            last = time.monotonic()
            watcher.step(frame)
        for x, y, w, h, s in watcher.faces:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (80, 200, 80), 2)
            cv2.putText(frame, f"{s:.2f}", (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
        if greeted and time.monotonic() - greeted[-1] < 1.5:
            cv2.putText(frame, "Hello!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 140, 255), 3)
        cv2.imshow("Juan sees", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
