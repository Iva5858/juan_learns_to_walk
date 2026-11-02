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


def swing_state(clearance, contact, swing, sole_vel=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))):
    ep = SimpleNamespace(clearance=np.array(clearance), contact=np.array(contact, dtype=float))
    s = SimpleNamespace(swing=np.array(swing), sole_vel=np.array(sole_vel))
    return ep, s


def test_swing_lift_pays_for_every_millimetre():
    w = RewardWeights(swing_lift=1.0)
    env = fake_env(0.3)
    h = env.cfg.command.swing_height
    values = []
    for lift in (0.0, 0.003, h / 2, h, 2 * h):
        env.episode, s = swing_state((lift, 0.0), (0, 1), (True, False))
        values.append(rewards.swing_lift(env, s, w))
    assert values[0] == 0.0 < values[1] < values[2] < values[3] == values[4]  # rising, capped at swing_height


def test_swing_reach_needs_a_lifted_foot_moving_the_commanded_way():
    w = RewardWeights(swing_reach=1.0)
    env = fake_env(0.3)
    env.command.heading_target = 0.0
    env.episode, s = swing_state((0.02, 0.0), (0, 1), (True, False), ((0.6, 0, 0), (0, 0, 0)))
    assert np.isclose(rewards.swing_reach(env, s, w), 1.0)          # 2x the command (cap), one of two feet
    env.episode, s = swing_state((0.0, 0.0), (0, 1), (True, False), ((0.6, 0, 0), (0, 0, 0)))
    assert rewards.swing_reach(env, s, w) == 0.0                     # sliding, not lifted
    env.episode, s = swing_state((0.02, 0.0), (0, 1), (True, False), ((-0.6, 0, 0), (0, 0, 0)))
    assert rewards.swing_reach(env, s, w) == 0.0                     # moving the wrong way


def test_double_support_only_when_a_foot_should_swing():
    w = RewardWeights(double_support=1.0)
    env = fake_env(0.3)
    env.episode, s = swing_state((0, 0), (1, 1), (True, False))
    assert rewards.double_support(env, s, w) == 1.0
    env.episode, s = swing_state((0, 0), (1, 1), (False, False))
    assert rewards.double_support(env, s, w) == 0.0                  # the clock's double-support phase
    env = fake_env(0.0, standing=True)
    env.episode, s = swing_state((0, 0), (1, 1), (True, False))
    assert rewards.double_support(env, s, w) == 0.0                  # told to stand


def test_stand_feet_charges_a_split_stance_only_when_standing():
    w = RewardWeights(stand_feet=1.0)
    s = SimpleNamespace(feet_x=0.05)
    assert np.isclose(rewards.stand_feet(fake_env(0.0, standing=True), s, w), 1.0)
    assert np.isclose(rewards.stand_feet(fake_env(0.0, standing=True), SimpleNamespace(feet_x=-0.5), w), 2.0)  # capped
    assert rewards.stand_feet(fake_env(0.2), s, w) == 0.0
    assert rewards.stand_feet(fake_env(0.0, standing=True), s, RewardWeights()) == 0.0   # off by default


def test_stopping_starts_the_settle_window():
    cfg = EnvConfig()
    cfg.rewards.stand_settle_s = 1.0
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=0, options={"command": (0.2, 0.0, 0.0)})
    for _ in range(50):
        env.step(np.zeros(env.nu))
    env.set_command(0.0)
    assert env.command.standing and np.isclose(env.command.stand_since, env.episode.t)
    env.reset(seed=1, options={"stand": True})
    assert env.command.stand_since == 0.0
