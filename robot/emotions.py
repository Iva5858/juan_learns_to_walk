"""Facial expressions: what the nearest face shows, so Juan can react (robot/personality.py).

The OpenCV model zoo's expression network (MobileFaceNet, 4.8 MB, 7 expressions, trained on RAF-DB) reads
the face found by robot/faces.py, aligned to 112x112 by its 5 landmarks. Juan reacts to three:

    happy      -> happy bounce        sad -> sad (slow crouch, a sad sound)        surprised -> excited

A single frame decides nothing: an expression counts when it is the top one in HOLD of the last WINDOW
readings with at least MIN_PROB, on a face at least MIN_FACE_PX wide. Then a reaction, at most one every
COOLDOWN_S, and not the same one again until the face has shown something else.

The camera sits 41 cm up and sees 24 deg above level: a standing adult's face is only in the picture
from ~2.7 m, too small to read there. It works on a face at the robot's height (someone crouching or
seated) or with the robot leaning back (the "Look up" gesture).

    python -m robot.emotions          # the computer's webcam: faces, expressions and what Juan would do

Setup: OpenCV, robot/faces.py's model, and this one, saved as
robot/models/facial_expression_recognition_mobilefacenet_2022july.onnx:
    https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx
"""
import os
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "facial_expression_recognition_mobilefacenet_2022july.onnx")
LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
REACTIONS = {"happy": "happy_bounce", "sad": "sad", "surprised": "excited"}   # expression -> gesture
SOUNDS = {"happy": "happy", "sad": "sad", "surprised": "excited"}             # while walking: the sound only
# Where the 5 landmarks sit in the network's 112x112 input (eyes, nose tip, mouth corners)
ALIGNED = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
MIN_FACE_PX = 48            # narrower faces are not read
WINDOW, HOLD = 5, 3         # an expression must top HOLD of the last WINDOW readings ...
MIN_PROB = 0.5              # ... each with at least this probability
COOLDOWN_S = 8.0


def similarity(src, dst):
    """The rotation + uniform scale + shift (2x3) taking points `src` closest to `dst` (least squares)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    a = np.zeros((2 * len(src), 4))
    a[0::2] = np.column_stack([src[:, 0], -src[:, 1], np.ones(len(src)), np.zeros(len(src))])
    a[1::2] = np.column_stack([src[:, 1], src[:, 0], np.zeros(len(src)), np.ones(len(src))])
    (c, s, tx, ty), *_ = np.linalg.lstsq(a, dst.ravel(), rcond=None)
    return np.array([[c, -s, tx], [s, c, ty]])


class EmotionReader:
    """read(image, face) -> {expression: probability} for a robot.faces.Face in a BGR image, or None when
    the face is too small to read."""

    def __init__(self, model=MODEL):
        import cv2
        if not os.path.exists(model):
            raise FileNotFoundError(f"expression model not found: {model} (see robot/emotions.py)")
        self.cv2 = cv2
        self.net = cv2.dnn.readNet(model)

    def read(self, image, face):
        if face.w < MIN_FACE_PX:
            return None
        cv2 = self.cv2
        crop = cv2.warpAffine(image, similarity(face.landmarks, ALIGNED), (112, 112))
        x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5) / 0.5
        self.net.setInput(cv2.dnn.blobFromImage(x))
        z = self.net.forward().ravel().astype(float)
        p = np.exp(z - z.max())
        return dict(zip(LABELS, p / p.sum()))


class EmotionTracker:
    """Readings in (one per frame, None when no face is read), reactions out: update() returns the
    expression to react to (a key of REACTIONS) or None. numpy only."""

    def __init__(self):
        self.recent = deque(maxlen=WINDOW)
        self.shown = None           # the expression held now (any of LABELS), for the page
        self.last_reacted, self.last_t = None, -COOLDOWN_S

    def update(self, probs, now=None):
        now = time.monotonic() if now is None else now
        top = max(probs, key=probs.get) if probs else None
        self.recent.append(top if top and probs[top] >= MIN_PROB else None)
        counts = {e: self.recent.count(e) for e in set(self.recent) if e}
        held = max(counts, key=counts.get) if counts else None
        self.shown = held if held and counts[held] >= HOLD else None
        if self.shown not in REACTIONS:
            if self.shown or not any(self.recent):   # something else held, or no face read lately: may react again
                self.last_reacted = None
            return None
        if self.shown != self.last_reacted and now - self.last_t >= COOLDOWN_S:
            self.last_reacted, self.last_t = self.shown, now
            return self.shown
        return None


if __name__ == "__main__":
    import cv2

    from robot.faces import FaceDetector
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("no webcam (on a Mac: allow camera access for the terminal in System Settings)")
    faces, reader, tracker = FaceDetector(), EmotionReader(), EmotionTracker()
    print("Reading expressions: q quits.", flush=True)
    last, said = 0.0, (None, 0.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if time.monotonic() - last >= 0.25:
            last = time.monotonic()
            found = faces.detect(frame)
            probs = reader.read(frame, found[0]) if found else None
            react = tracker.update(probs)
            if react:
                said = (f"-> {REACTIONS[react]}", time.monotonic())
                print(f"{react}: Juan plays {REACTIONS[react]}", flush=True)
            for i, f in enumerate(found):
                cv2.rectangle(frame, (int(f.x), int(f.y)), (int(f.x + f.w), int(f.y + f.h)), (80, 200, 80), 2)
                if i == 0 and probs:
                    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
                    for k, (e, p) in enumerate(top):
                        cv2.putText(frame, f"{e} {p:.2f}", (int(f.x), int(f.y + f.h) + 20 + 20 * k),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 80), 2)
        cv2.putText(frame, f"held: {tracker.shown or '-'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 140, 255), 2)
        if said[0] and time.monotonic() - said[1] < 2.0:
            cv2.putText(frame, said[0], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 140, 255), 2)
        cv2.imshow("Juan reads faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
