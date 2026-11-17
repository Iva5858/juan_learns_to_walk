"""Following a person: the camera finds people, and the nearest one's direction and distance steer the running
walking policy the way the joystick does (forward speed and turn rate; never sideways or backward).

  * PersonDetector: NanoDet-Plus (OpenCV model zoo, COCO, 3.8 MB), people only. The whole body, not the face:
    the camera sits 41 cm up and sees 24 deg above level, so a standing adult's face is only in the picture
    beyond ~2.7 m, while legs and torso are in view from ~0.9 m.
  * locate(): direction from the box's centre; distance from where the feet meet the floor (camera height
    and the angle below level), so it needs the feet in the picture: closer than ~0.9 m counts as "near".
  * FollowController (numpy only: scripts/analysis/follow_test.py runs it in the simulator): turn toward the
    person, walk to FOLLOW_M away, turn first when they are well off to the side, ease the speed (stopping dead
    from speed is when the robot falls), stand still when they are lost; after LOST_S turn slowly toward
    where they were last seen, for SEARCH_S at most.
  * Follower: a thread reading the camera, steering the walker as "follow" (no other source steers meanwhile).
    It does not keep the page's link alive: with the page gone, the walker stands and stops as usual.

    python -m robot.follow            # the computer's webcam: boxes, direction and distance (no robot)

Setup: OpenCV (see robot/faces.py) and the model, saved as robot/models/object_detection_nanodet_2022nov.onnx:
    https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx
"""
import math
import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "object_detection_nanodet_2022nov.onnx")
INPUT = 416                 # the network's input, square (the frame is letterboxed into it)
MIN_SCORE = 0.40          # test images: people 0.46-0.60; the robot's footage without people: at most 0.33
PERSON = 0                  # COCO class

CAMERA_HEIGHT_M = 0.41      # the Pi camera above the floor, standing at home (robot/sim.py HEAD_CAMERA)
HFOV_DEG = 62.2             # Raspberry Pi Camera v2, full sensor (640x480 is binned from it)
FEET_CUT = 0.97             # a box reaching this far down the picture: feet not in view (near)

FOLLOW_M = 1.3              # walk to this far from the person
NEAR_M = 0.9                # closer: stop walking, only turn to face them
VX_MAX = 0.2                # m/s (on hardware the walkers do ~0.2 m/s)
WZ_MAX = 0.5                # rad/s (the moves test turns at 0.5)
K_FWD = 0.4                 # m/s per m too far
K_TURN = 1.5                # rad/s per rad off centre
DEADBAND_RAD = math.radians(4)
TURN_FIRST_RAD = math.radians(25)   # off to the side by this much: no forward speed, turn first
ACCEL, DECEL = 0.15, 0.25   # m/s per s: speed changes no faster than this
SMOOTH = 0.5                # low-pass on the direction and distance (share of the new reading)
LOST_S = 1.0                # no sighting this long: stand still
SEARCH_S = 6.0              # then turn toward the last sighting for at most this long
SEARCH_WZ = 0.3


def locate(box, size, hfov_deg=HFOV_DEG, camera_height=CAMERA_HEIGHT_M):
    """(direction rad, + = left; distance m or None when the feet are out of the picture) of a box
    (x1, y1, x2, y2) in an image of `size` (w, h), for a level camera."""
    w, h = size
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)      # focal length, pixels
    bearing = math.atan2((w / 2) - (box[0] + box[2]) / 2, f)
    if box[3] >= FEET_CUT * h:
        return bearing, None
    below = math.atan2(box[3] - h / 2, f)                    # the feet's angle below level
    return bearing, (camera_height / math.tan(below) if below > 0.01 else None)


class FollowController:
    """Sightings in, (forward m/s, turn rad/s) out. Call step() at any rate with the latest sighting
    (bearing, distance) or None when nobody is seen; distance None means near (feet out of the picture)."""

    def __init__(self, follow_m=FOLLOW_M, vx_max=VX_MAX, wz_max=WZ_MAX):
        self.follow_m, self.vx_max, self.wz_max = follow_m, vx_max, wz_max
        self.bearing = self.distance = None
        self.vx = self.wz = 0.0
        self.unseen_s = math.inf
        self.state = "looking"

    def step(self, sighting, dt):
        if sighting is not None:
            b, d = sighting
            d = NEAR_M if d is None else d
            fresh = self.unseen_s > LOST_S or self.bearing is None
            self.bearing = b if fresh else self.bearing + SMOOTH * (b - self.bearing)
            self.distance = d if fresh else self.distance + SMOOTH * (d - self.distance)
            self.unseen_s = 0.0
        else:
            self.unseen_s += dt
        vx, wz = 0.0, 0.0
        if self.unseen_s <= LOST_S and self.bearing is not None:
            b = self.bearing
            if abs(b) > DEADBAND_RAD:
                wz = float(np.clip(K_TURN * (b - math.copysign(DEADBAND_RAD, b)), -self.wz_max, self.wz_max))
            side = max(0.0, 1.0 - abs(b) / TURN_FIRST_RAD)          # 1 straight ahead, 0 well off to the side
            if self.distance > NEAR_M:
                vx = float(np.clip(K_FWD * (self.distance - self.follow_m), 0.0, self.vx_max)) * side
            self.state = "following" if vx > 0 else "facing"
        elif self.bearing is not None and self.unseen_s <= LOST_S + SEARCH_S:
            wz = math.copysign(SEARCH_WZ, self.bearing)
            self.state = "searching"
        else:
            self.state = "looking"
        step = (ACCEL if vx > self.vx else DECEL) * dt
        self.vx += float(np.clip(vx - self.vx, -step, step))
        self.wz = wz
        return self.vx, self.wz


class PersonDetector:
    """People in a BGR image as (x1, y1, x2, y2, score) in that image's pixels, largest first."""

    REG_MAX = 7                 # box sides as distributions over 0..7 strides

    def __init__(self, model=MODEL, min_score=MIN_SCORE):
        import cv2
        if not os.path.exists(model):
            raise FileNotFoundError(f"person model not found: {model} (see robot/follow.py)")
        self.cv2, self.min_score = cv2, min_score
        self.net = cv2.dnn.readNet(model)
        self.mean = np.array([103.53, 116.28, 123.675], np.float32)
        self.std = np.array([57.375, 57.12, 58.395], np.float32)
        self.centres = {}                                          # by stride: each cell's centre, pixels
        for s in (8, 16, 32, 64):
            yv, xv = np.mgrid[0:INPUT // s, 0:INPUT // s] * s
            self.centres[s] = np.column_stack([xv.ravel(), yv.ravel()]) + 0.5 * (s - 1)

    def detect(self, image):
        cv2 = self.cv2
        h, w = image.shape[:2]
        scale = INPUT / max(h, w)                                 # letterbox: keep the shape, pad below/right
        small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((INPUT, INPUT, 3), np.float32)
        canvas[:small.shape[0], :small.shape[1]] = (small - self.mean) / self.std
        self.net.setInput(cv2.dnn.blobFromImage(canvas))
        outs = [o.reshape(o.shape[-2], o.shape[-1]) for o in self.net.forward(self.net.getUnconnectedOutLayersNames())]
        cells = lambda o: o.shape[0]                               # (class scores and boxes, one level per stride:
        classes = sorted((o for o in outs if o.shape[1] != 4 * (self.REG_MAX + 1)), key=cells)   # paired by size,
        regs = sorted((o for o in outs if o.shape[1] == 4 * (self.REG_MAX + 1)), key=cells)      # whatever the order)
        boxes, scores = [], []
        for cls, reg in zip(classes, regs):
            stride = INPUT // round(math.sqrt(len(cls)))
            centres, reg = self.centres[stride], reg.reshape(len(cls), 4, self.REG_MAX + 1)
            keep = cls[:, PERSON] >= self.min_score
            if not keep.any():
                continue
            p = np.exp(reg[keep] - reg[keep].max(-1, keepdims=True))
            dist = (p / p.sum(-1, keepdims=True)) @ np.arange(self.REG_MAX + 1) * stride   # l, t, r, b
            c = centres[keep]
            boxes.append(np.column_stack([c - dist[:, :2], c + dist[:, 2:]]) / scale)
            scores.append(cls[keep, PERSON])
        if not boxes:
            return []
        boxes, scores = np.concatenate(boxes), np.concatenate(scores)
        boxes = np.clip(boxes, 0, [w, h, w, h])
        xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.min_score, 0.6)).ravel()
        people = [(*map(float, boxes[i]), float(scores[i])) for i in idx]
        return sorted(people, key=lambda p: -(p[2] - p[0]) * (p[3] - p[1]))


class Follower:
    """Steers the running walking policy toward the nearest person (the largest box) in `frames()` (an
    iterator of BGR images). Reads at most `fps` frames a second; sends a command every frame."""

    def __init__(self, walker, frames, fps=4.0, detector=None):
        self.detector = detector or PersonDetector()
        self.walker, self.frames, self.period = walker, frames, 1.0 / fps
        self.ctrl = FollowController()
        self.people, self.sighting, self.error = [], None, None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def step(self, image, dt):
        people = self.detector.detect(image)
        self.people = people
        self.sighting = locate(people[0][:4], (image.shape[1], image.shape[0])) if people else None
        vx, wz = self.ctrl.step(self.sighting, dt)
        w = self.walker
        if not w.running or w.steer_owner != "follow":   # the policy stopped, or the steering was taken back
            self.running = False
            return
        w.set_command(vx, 0.0, wz, source="follow", alive=False)

    def _loop(self):
        last = time.monotonic()
        while self.running:
            try:
                for image in self.frames():
                    if not self.running:
                        return
                    now = time.monotonic()
                    if now - last < self.period:
                        continue
                    self.step(image, now - last)
                    last = now
                    self.error = None
            except Exception as e:  # camera gone, decoding error...
                self.error = str(e)[:200]
            if self.running:   # the source ended or failed: stand still and try again
                self.ctrl.step(None, 1.0)
                time.sleep(1.0)

    def stop(self):
        self.running = False

    def state(self):
        c = self.ctrl
        return {"state": c.state, "people": len(self.people), "error": self.error,
                "bearing_deg": None if c.bearing is None else round(math.degrees(c.bearing), 1),
                "distance_m": None if c.distance is None else round(c.distance, 2),
                "command": [round(c.vx, 3), round(c.wz, 3)]}


if __name__ == "__main__":
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("no webcam (on a Mac: allow camera access for the terminal in System Settings)")
    det, ctrl, last = PersonDetector(), FollowController(), time.monotonic()
    print("Looking for people: q quits. (Distances assume a level camera 41 cm up: wrong for a webcam.)", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.monotonic()
        people = det.detect(frame)
        ms = 1000 * (time.monotonic() - t0)
        sighting = locate(people[0][:4], (frame.shape[1], frame.shape[0])) if people else None
        vx, wz = ctrl.step(sighting, t0 - last)
        last = t0
        for i, (x1, y1, x2, y2, s) in enumerate(people):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (80, 200, 80) if i == 0 else (120, 120, 120), 2)
            cv2.putText(frame, f"{s:.2f}", (int(x1), int(y1) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
        cv2.putText(frame, f"{ctrl.state}  fwd {vx:.2f} m/s  turn {wz:+.2f} rad/s  ({ms:.0f} ms)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 140, 255), 2)
        cv2.imshow("Juan follows", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
