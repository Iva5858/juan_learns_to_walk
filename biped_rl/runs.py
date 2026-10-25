"""Loading a trained run: runs/<name>/ holds config.json, final_model.zip and vecnormalize.pkl
(best_model.zip and best_vecnormalize.pkl: the best evaluation during training)."""
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv


def load_policy(run, cfg, best=False):
    """(model, normalizer) of a run. Use as model.predict(norm.normalize_obs(obs), deterministic=True)
    on a BipedEnv made from `cfg`."""
    run = Path(run)
    model = PPO.load(run / ("best_model" if best else "final_model"), device="cpu")
    norm = VecNormalize.load(str(run / ("best_vecnormalize.pkl" if best else "vecnormalize.pkl")),
                             DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
    norm.training, norm.norm_reward = False, False
    return model, norm
