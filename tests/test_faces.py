"""robot/faces.py: the greeting rules (with a stand-in detector) and the real detector on a blank frame."""
import numpy as np
import pytest

from robot import faces


class Script:
    """A detector that 'sees' a face whenever the test says so."""
    def __init__(self):
        self.face = False

    def detect(self, image):
        return [(10, 10, 50, 60, 0.9)] if self.face else []


def run(watcher, det, seen, start, step=0.25):
    for i, s in enumerate(seen):
        det.face = s
        watcher.step(None, now=start + i * step)


def test_greets_once_when_a_face_appears():
    det, greets = Script(), []
    w = faces.FaceWatcher(None, greets.append, detector=det)
    run(w, det, [False, True, True, True, True], start=100.0)
    assert len(greets) == 1                       # 2 of 3 frames, then no repeat while it stays


def test_a_single_flicker_is_not_a_face():
    det, greets = Script(), []
    w = faces.FaceWatcher(None, greets.append, detector=det)
    run(w, det, [False, True, False, False, True, False], start=100.0)
    assert greets == []


def test_cooldown_and_absence():
    det, greets = Script(), []
    w = faces.FaceWatcher(None, greets.append, detector=det)
    run(w, det, [True, True], start=100.0)                                   # greet
    run(w, det, [False] * 20 + [True, True], start=101.0)                    # back after 5 s: cooldown
    assert len(greets) == 1
    run(w, det, [False] * 20 + [True, True], start=100.0 + faces.COOLDOWN_S + 5)  # after the cooldown
    assert len(greets) == 2


def test_real_detector_finds_nothing_in_a_blank_frame():
    pytest.importorskip("cv2")
    if not __import__("os").path.exists(faces.MODEL):
        pytest.skip("face model not downloaded")
    assert faces.FaceDetector().detect(np.zeros((480, 640, 3), np.uint8)) == []
