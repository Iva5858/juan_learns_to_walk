"""Reinforcement learning for Juan, a small servo-driven biped, in MuJoCo.

    biped_rl.robots          robot descriptions (biped_rl/robots/juan.py) and model paths
    biped_rl.tasks           tasks: locomotion (environment, configuration, mdp terms)
    biped_rl.training        PPO setup, training stages, TensorBoard dashboard
    biped_rl.policy          asymmetric actor-critic, mirror-symmetric actor
    biped_rl.symmetry        left-right mirror maps
    biped_rl.motions         scripted motions (boot sequence)
"""
import gymnasium as gym

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig

# Episode length is set by EnvConfig.episode_seconds (the env truncates itself).
gym.register(id="Biped-v0", entry_point="biped_rl.tasks.locomotion.env:BipedEnv")

__all__ = ["BipedEnv", "EnvConfig"]
