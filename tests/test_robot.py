"""The robot side, without the hardware: its joint table matches the model, and every gesture is safe
on the simulated robot."""
import numpy as np
import pytest

from biped_rl.tasks.locomotion import EnvConfig
from robot import gestures
from tests.conftest import load_script


def test_joint_table_matches_the_model():
    assert load_script("check_robot_joints.py").mismatches() == []


@pytest.mark.slow
@pytest.mark.parametrize("name", list(gestures.ALL))
def test_gesture_is_safe(name):
    check = load_script("check_gestures.py")
    cfg = EnvConfig().use_measured_servos()
    cfg.randomization.enabled = False
    fell, tilt, torque, _ = check.play(cfg, gestures.ALL[name][3], seed=0)
    assert not fell
    assert tilt < gestures.MAX_TILT.get(name, check.MAX_TILT)
    assert torque < check.MAX_TORQUE
    assert np.isfinite(torque)
