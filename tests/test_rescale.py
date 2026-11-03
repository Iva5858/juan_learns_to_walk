"""scripts/rescale_actions.py: a policy given a larger movement range must output the same motion
(in radians) as before, as long as it stays inside the old range."""
import sys

import numpy as np
import pytest
import torch

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from tests.conftest import CHECKPOINT, load_script


def mean_action(model, norm, obs):
    with torch.no_grad():
        t = model.policy.obs_to_tensor(norm.normalize_obs(obs))[0]
        return model.policy._action_distribution(t).distribution.mean.numpy()[0]


@pytest.mark.parametrize("joints", [[], ["--joints", "hip_pitch", "knee", "ankle"]])
def test_rescaled_policy_moves_the_same(tmp_path, monkeypatch, joints):
    out = tmp_path / "scaled"
    monkeypatch.setattr(sys, "argv", ["rescale", str(CHECKPOINT), str(out), "--scale", "0.8", *joints])
    load_script("rescale_actions.py").main()
    old_cfg, new_cfg = EnvConfig.load(CHECKPOINT), EnvConfig.load(out)
    old, new = load_policy(CHECKPOINT, old_cfg), load_policy(out, new_cfg)
    env, new_env = BipedEnv(old_cfg.to_dict()), BipedEnv(new_cfg.to_dict())
    r = env.action_scale / new_env.action_scale
    assert np.allclose(r, 0.625 if not joints else [1, 0.625, 0.625, 0.625] * 2)
    # the same radians cost the same smoothness penalty: actions r x as large, counted 1/r x
    assert np.allclose(new_env.smoothness_unit * r, env.smoothness_unit)
    assert new_cfg.rewards.action_rate == old_cfg.rewards.action_rate
    obs, _ = env.reset(seed=0)
    frame = 6 + 3 * env.nu
    for _ in range(100):
        a_old = mean_action(*old, obs)
        scaled = obs.copy()
        for k in range(old_cfg.history):  # the previous actions it observes, in the new units
            scaled[k * frame + 6 + 2 * env.nu:(k + 1) * frame] *= r
        assert np.allclose(mean_action(*new, scaled), r * a_old, atol=1e-5)
        obs, _, term, trunc, _ = env.step(np.clip(a_old, -1, 1))
        if term or trunc:
            obs, _ = env.reset()
