import json

import pytest

from biped_rl.robots import juan
from biped_rl.tasks.locomotion import EnvConfig
from biped_rl.training.stages import STAGES, apply_stage
from tests.conftest import CHECKPOINT


def test_round_trip():
    cfg = EnvConfig()
    assert EnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


def test_removed_settings_are_dropped():
    d = EnvConfig().to_dict()
    d["mirror_symmetry"] = True
    d["rewards"]["step_timing"] = 0.0
    assert EnvConfig.from_dict(d) == EnvConfig()


def test_removed_features_refuse_runs_that_used_them():
    d = EnvConfig().to_dict()
    d["rewards"]["forward_speed"], d["start_poses"] = 0.0, ["home"]  # off: dropped
    assert EnvConfig.from_dict(d) == EnvConfig()
    d["rewards"]["forward_speed"] = 2.0  # on: a run that needs the old code
    with pytest.raises(ValueError, match="forward_speed"):
        EnvConfig.from_dict(d)


def test_shipped_checkpoint_loads():
    cfg = EnvConfig.load(CHECKPOINT)
    assert cfg.robot.model_path == "juan/rl_scene.xml"
    assert cfg.history == 3


def test_measured_servos():
    cfg = EnvConfig().use_measured_servos()
    assert cfg.robot.kp == juan.MEASURED_KP
    assert cfg.randomization.action_delay_steps == juan.MEASURED_DELAY_STEPS


def test_stages():
    for stage in STAGES:
        cfg = EnvConfig()
        apply_stage(cfg, stage)
        assert cfg.randomization.enabled == (stage in ("physics", "pushes"))
        assert (cfg.randomization.push_velocity > 0) == (stage == "pushes")
