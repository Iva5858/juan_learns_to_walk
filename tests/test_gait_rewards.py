"""The step-length and air-time reward terms (off by default, on for juan_v7_15)."""
from types import SimpleNamespace

import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.env_cfg import RewardWeights
from biped_rl.tasks.locomotion.mdp import rewards


def fake_env(forward, standing=False):
    cfg = EnvConfig()
    cmd = SimpleNamespace(forward=forward, standing=standing, vec=np.array([forward, 0.0, 0.0]))
    return SimpleNamespace(command=cmd, cfg=cfg)


def touchdown(step=(0.0, 0.0), air=(0.0, 0.0), which=(True, False)):
    return SimpleNamespace(new_td=np.array(which), step_at_touchdown=np.array(step), air_at_touchdown=np.array(air))


def test_off_by_default():
    w = RewardWeights()
    s = touchdown(step=(0.1, 0.0), air=(0.3, 0.0))
    assert rewards.step_length(fake_env(0.3), s, w) == 0.0
    assert rewards.feet_air_time(fake_env(0.3), s, w) == 0.0


def test_step_length_follows_the_command_direction():
    w = RewardWeights(step_length=1.0, step_length_target=0.12)
    assert np.isclose(rewards.step_length(fake_env(0.3), touchdown(step=(0.06, 0.0)), w), 0.5)
    assert np.isclose(rewards.step_length(fake_env(0.3), touchdown(step=(0.30, 0.0)), w), 1.0)   # capped
    assert rewards.step_length(fake_env(0.3), touchdown(step=(-0.06, 0.0)), w) == 0.0         # landed behind
    assert np.isclose(rewards.step_length(fake_env(-0.1), touchdown(step=(-0.06, 0.0)), w), 0.5)  # backward
    assert rewards.step_length(fake_env(0.0), touchdown(step=(0.06, 0.0)), w) == 0.0          # not walking
    assert rewards.step_length(fake_env(0.3), touchdown(step=(0.06, 0.0), which=(False, False)), w) == 0.0


def test_air_time_rewards_using_the_planned_swing():
    w = RewardWeights(feet_air_time=1.0)
    env = fake_env(0.3)
    from biped_rl.tasks.locomotion.mdp import commands
    planned = env.cfg.command.swing_ratio * commands.gait_period(env)
    assert np.isclose(rewards.feet_air_time(env, touchdown(air=(planned, 0.0)), w), 1.0)
    assert np.isclose(rewards.feet_air_time(env, touchdown(air=(planned / 4, 0.0)), w), 0.25)  # early tap
    assert rewards.feet_air_time(fake_env(0.0, standing=True), touchdown(air=(planned, 0.0)), w) == 0.0


def test_terms_stay_bounded_in_the_simulator():
    cfg = EnvConfig()
    cfg.rewards.step_length, cfg.rewards.feet_air_time = 2.0, 2.0
    env = BipedEnv(cfg.to_dict())
    rng = np.random.default_rng(0)
    env.reset(seed=0, options={"command": (0.3, 0.0, 0.0)})
    seen = 0.0
    for _ in range(300):
        _, r, term, trunc, info = env.step(np.clip(0.3 * rng.standard_normal(env.nu), -1, 1))
        for k in ("rew/step_length", "rew/feet_air_time"):
            assert 0.0 <= info[k] <= 2 * 2.0 + 1e-9
            seen += info[k]
        assert np.isfinite(r) and r >= -10.0 - 1e-9
        if term or trunc:
            env.reset(options={"command": (0.3, 0.0, 0.0)})
    assert seen > 0  # touchdowns happened and were scored
