import json

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
