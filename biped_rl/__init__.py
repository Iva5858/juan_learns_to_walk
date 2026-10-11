import gymnasium as gym

from biped_rl.config import EnvConfig
from biped_rl.env import BipedEnv

# Episode length is set by EnvConfig.episode_seconds (the env truncates itself).
gym.register(id="Biped-v0", entry_point="biped_rl.env:BipedEnv")

__all__ = ["BipedEnv", "EnvConfig"]
