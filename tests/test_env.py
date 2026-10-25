import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import observations


def rollout(cfg, steps=200, seed=3):
    env = BipedEnv(cfg.to_dict())
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    out = []
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(np.clip(0.3 * rng.standard_normal(env.nu), -1, 1))
        out.append(np.concatenate([obs, [r]]))
        if term or trunc:
            obs, _ = env.reset()
    return env, np.array(out)


def test_spaces():
    env = BipedEnv(EnvConfig().to_dict())
    obs, _ = env.reset(seed=0)
    assert env.nu == 8
    assert obs.shape == env.observation_space.shape
    assert env.actor_obs_dim == observations.actor_dim(env.cfg, env.nu)
    assert env.dt == 0.02


def test_deterministic_with_a_seed():
    cfg = EnvConfig()  # randomized physics, pushes, noise: everything that draws random numbers
    assert np.array_equal(rollout(cfg)[1], rollout(cfg)[1])


def test_stands_at_home_without_actions():
    cfg = EnvConfig()
    cfg.randomization.enabled = False
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=0, options={"stand": True})
    for _ in range(100):
        _, _, term, _, _ = env.step(np.zeros(env.nu))
        assert not term
    assert np.degrees(np.abs(env.data.qpos[env.robot.qpos_idx] - env.robot.home_joint).max()) < 5


def test_hip_roll_outward_signs():
    env = BipedEnv(EnvConfig().to_dict())
    # positive hip roll: left leg outward, right leg inward (biped_rl/robots/juan.py)
    assert env.robot.roll_out_sign.tolist() == [1.0, -1.0]
