"""The shipped policy: it is mirror-symmetric, it walks, and its numpy export (what runs on the robot)
matches the trained network and the simulator's observation pipeline."""
import numpy as np
import pytest

from biped_rl.robots import PROJECT
from biped_rl.runs import load_policy
from biped_rl.symmetry import check_robot_mirror, mirror_maps
from biped_rl.tasks.locomotion import BipedEnv
from tests.conftest import CHECKPOINT, load_script


def test_mirror_map_matches_the_model(cfg):
    env = BipedEnv(cfg.to_dict())
    assert check_robot_mirror(env) < 0.005  # m: left and right part positions agree
    obs_perm, obs_sign, act_perm, act_sign = mirror_maps(env)
    x = np.random.default_rng(0).standard_normal(len(obs_perm))
    twice = np.array(obs_sign) * (np.array(obs_sign) * x[obs_perm])[obs_perm]
    assert np.allclose(twice, x)  # mirroring twice gives back the original


def test_policy_is_mirror_symmetric(cfg):
    model, norm = load_policy(CHECKPOINT, cfg)
    env = BipedEnv(cfg.to_dict())
    obs_perm, obs_sign, act_perm, act_sign = map(np.array, mirror_maps(env))
    obs, _ = env.reset(seed=0, options={"command": (0.2, 0.05, 0.3)})
    for _ in range(50):
        obs, *_ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
    a = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
    a_mirror = model.predict(norm.normalize_obs(obs_sign * obs[obs_perm]), deterministic=True)[0]
    assert np.allclose(act_sign * a[act_perm], a_mirror, atol=1e-5)


@pytest.mark.slow
def test_policy_walks(cfg):
    model, norm = load_policy(CHECKPOINT, cfg)
    env = BipedEnv(cfg.to_dict())
    obs, _ = env.reset(seed=0, options={"speed": 0.3})
    for _ in range(250):  # 5 s
        obs, _, term, _, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        assert not term
    assert np.hypot(*env.data.qpos[:2]) > 1.0


@pytest.mark.slow
def test_export_matches_the_shipped_robot_policy(tmp_path):
    out = tmp_path / f"{CHECKPOINT.name}.npz"
    load_script("export_policy.py").export(CHECKPOINT, out)  # raises if numpy and PyTorch disagree
    exported, shipped = np.load(out), np.load(PROJECT / "robot" / "policies" / out.name)
    assert set(exported.files) == set(shipped.files)
    for k in exported.files:
        assert np.array_equal(exported[k], shipped[k]), k
