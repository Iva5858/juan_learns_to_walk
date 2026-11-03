"""The getup task (biped_rl/tasks/getup.py): the same face-down start every time, no early end, and a
reward that is 0 lying still and grows as the robot rises."""
import numpy as np

from biped_rl.tasks.getup import GetUpEnv, config


def test_starts_face_down_the_same_way_every_time():
    env = GetUpEnv(config(episode_seconds=1.0))
    env.reset(seed=0)
    q0 = env.data.qpos.copy()
    assert env.gravity_body()[0] > 0.95            # the torso's forward axis points down: face down
    assert env.lying_height < 0.4 * env.robot.home_height
    env.reset(seed=5)
    assert np.allclose(env.data.qpos, q0)


def test_lying_still_earns_nothing_and_never_ends_early():
    env = GetUpEnv(config(episode_seconds=1.0))
    env.reset(seed=0)
    total, steps = 0.0, 0
    while True:
        _, r, terminated, truncated, info = env.step(np.zeros(env.nu))
        total, steps = total + r, steps + 1
        assert not terminated
        if truncated:
            break
    assert steps == int(1.0 / env.dt) and abs(total) < 1e-3 and info["standing"] == 0


def test_standing_at_home_earns_the_most():
    env = GetUpEnv(config(episode_seconds=1.0))
    env.reset(seed=0)
    env._reset_to_home()                 # put it standing at home, as if it had got up
    env.settle_at_home()
    _, r, *_ , info = env.step(np.zeros(env.nu))
    assert info["standing"] == 1 and info["upright"] > 0.95 and r > 0.1
