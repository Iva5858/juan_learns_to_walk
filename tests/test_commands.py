"""Command sampling: the optional share of fast forward commands."""
import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import commands


def draw(cfg, n=300, seed=0):
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=seed)
    out = []
    for _ in range(n):
        commands.sample(env)
        out.append(env.command.vec.copy())
    return env, np.array(out)


def test_fast_commands_are_in_the_top_quarter():
    cfg = EnvConfig()
    c = cfg.command
    c.stand_prob, c.turn_in_place_prob, c.backward_prob, c.sideways_prob, c.fast_prob = 0.0, 0.0, 0.0, 0.0, 1.0
    env, v = draw(cfg)
    top = env.curriculum.speed_max
    assert np.all((v[:, 0] >= 0.75 * top) & (v[:, 0] <= top)) and np.all(v[:, 1:] == 0)


def test_fast_share_off_draws_the_same_commands_as_before():
    a = draw(EnvConfig())[1]
    cfg = EnvConfig()
    cfg.command.fast_prob = 0.0
    assert np.array_equal(a, draw(cfg)[1])
