"""The camera behaviors' logic, without a camera: robot/follow.py (where a person is, how to steer toward them)
and robot/emotions.py (which expressions count). The networks themselves are checked if their models exist."""
import math
import os

import numpy as np
import pytest

from robot import emotions, follow
from robot.emotions import EmotionTracker
from robot.follow import FollowController, locate


def test_locate_direction_and_distance():
    w, h = 640, 480
    f = (w / 2) / math.tan(math.radians(follow.HFOV_DEG) / 2)
    b, d = locate((300, 100, 340, 300), (w, h))                 # centred, feet 60 px below the middle
    assert b == pytest.approx(0.0)
    assert d == pytest.approx(follow.CAMERA_HEIGHT_M / (60 / f))
    b, _ = locate((0, 100, 40, 300), (w, h))                     # at the left edge: + = left, half the field
    assert b == pytest.approx(math.atan2(300, f))
    assert locate((300, 100, 340, 475), (w, h))[1] is None       # feet out of the picture: near


def run(ctrl, sighting, seconds, dt=0.25):
    for _ in range(int(seconds / dt)):
        out = ctrl.step(sighting, dt)
    return out


def test_follow_turns_toward_and_walks_to_the_person():
    c = FollowController()
    vx, wz = run(c, (math.radians(-30), 3.0), 2.0)               # far, well off to the right: turn first
    assert vx == 0.0 and wz < 0
    vx, wz = run(c, (0.0, 3.0), 4.0)                             # far, ahead: walk, no faster than VX_MAX
    assert vx == pytest.approx(follow.VX_MAX, abs=1e-3) and wz == 0.0
    vx, _ = run(c, (0.0, follow.FOLLOW_M), 4.0)                  # there: eases down to standing
    assert vx == pytest.approx(0.0, abs=1e-3)
    vx, _ = run(c, (0.0, None), 4.0)                             # near (feet out of view): stands, still faces
    assert vx == 0.0 and c.state == "facing"


def test_follow_eases_speed_and_searches_when_lost():
    c = FollowController()
    run(c, (0.0, 3.0), 4.0)
    vx0 = c.vx
    vx, _ = c.step(None, 0.25)                                   # one missed frame: keeps going
    assert vx == pytest.approx(vx0)
    vx, _ = run(c, None, 1.0)                                    # lost: slows at DECEL, not at once
    assert 0.0 < vx < vx0
    c = FollowController()
    run(c, (math.radians(10), 2.0), 1.0)
    _, wz = run(c, None, 2.0)
    assert c.state == "searching" and wz == pytest.approx(follow.SEARCH_WZ)   # toward the last sighting (left)
    vx, wz = run(c, None, follow.SEARCH_S + 1)
    assert c.state == "looking" and vx == pytest.approx(0.0) and wz == 0.0


def probs(top, p=0.8):
    rest = (1 - p) / (len(emotions.LABELS) - 1)
    return {e: p if e == top else rest for e in emotions.LABELS}


def test_an_expression_counts_when_held_and_reacts_once():
    t = EmotionTracker()
    got = [t.update(probs("happy"), now=i) for i in range(6)]
    assert got == [None, None, "happy", None, None, None]        # 3 of the last 5, then not again
    t = EmotionTracker()
    assert [t.update(probs("happy", 0.4), now=i) for i in range(6)] == [None] * 6   # too unsure
    assert [t.update(probs("neutral"), now=i) for i in range(6)] == [None] * 6      # not one Juan reacts to


def test_reactions_repeat_only_after_something_else_and_the_cooldown():
    t = EmotionTracker()
    seq = ["happy"] * 4 + ["neutral"] * 5 + ["happy"] * 4 + ["sad"] * 12   # sad held past the cooldown
    got = [(i, t.update(probs(e), now=i)) for i, e in enumerate(seq)]
    fired = [(i, e) for i, e in got if e]
    assert [e for _, e in fired] == ["happy", "happy", "sad"]
    assert all(b - a >= emotions.COOLDOWN_S for (a, _), (b, _) in zip(fired, fired[1:]))


def test_alignment_matches_the_landmarks():
    src = np.array([[10.0, 20.0], [40, 18], [25, 35], [14, 50], [38, 49]])
    m = emotions.similarity(src, emotions.ALIGNED * 2 + 5)
    back = np.c_[src, np.ones(5)] @ m.T
    m2 = emotions.similarity(back, emotions.ALIGNED)
    assert np.allclose(np.c_[emotions.ALIGNED * 2 + 5, np.ones(5)] @ m2.T, emotions.ALIGNED, atol=1e-6)


@pytest.mark.skipif(not (os.path.exists(follow.MODEL) and os.path.exists(emotions.MODEL)), reason="models not downloaded")
def test_the_networks_load_and_find_nothing_in_a_blank_picture():
    pytest.importorskip("cv2")
    from robot.faces import Face
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert follow.PersonDetector().detect(blank) == []
    face = Face(200, 100, 120, 150, 0.9, [(240, 160), (290, 160), (265, 190), (245, 220), (285, 220)])
    p = emotions.EmotionReader().read(blank, face)
    assert set(p) == set(emotions.LABELS) and sum(p.values()) == pytest.approx(1.0)
