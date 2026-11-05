"""Guarded training must survive SB3's automatic normalization synchronization."""
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize, sync_envs_normalization

from biped_rl.tasks.locomotion import BipedEnv
from tests.conftest import load_script


@pytest.mark.parametrize("guarded", [False, True])
def test_training_and_evaluation_normalization_stacks_match(cfg, guarded):
    train = DummyVecEnv([lambda: BipedEnv(cfg)])
    if guarded:
        train = VecCheckNan(train, raise_exception=True, check_inf=True)
    train = VecNormalize(train)
    evaluation = VecNormalize(load_script("train.py").evaluation_env(cfg, guarded), training=False, norm_reward=False)
    try:
        train.reset()
        for _ in range(5):
            train.step(np.zeros((1, train.action_space.shape[0])))
        sync_envs_normalization(train, evaluation)  # exact function that crashed the overnight run
        np.testing.assert_array_equal(evaluation.obs_rms.mean, train.obs_rms.mean)
        np.testing.assert_array_equal(evaluation.obs_rms.var, train.obs_rms.var)
        assert evaluation.obs_rms is not train.obs_rms
        assert evaluation.ret_rms.count == train.ret_rms.count
    finally:
        train.close()
        evaluation.close()
