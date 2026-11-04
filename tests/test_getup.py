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


def test_idle_ends_the_episode_with_the_bonus():
    """Standing still at home with idle_finish on: idle after 0.5 s, the episode ends there, and the bonus
    covers what standing to the time limit would have earned. Off: it never ends early."""
    weights = {"strict_standing": 1, "flat_feet": 1, "idle_finish": 10.0, "upright_on_feet": 1, "height_right_side_up": 1}
    for finish in (10.0, 0.0):
        env = GetUpEnv(config(episode_seconds=2.0), {**weights, "idle_finish": finish})
        env.reset(seed=0)
        env._reset_to_home()
        env.settle_at_home()
        rewards, ended = [], False
        for _ in range(env._max_steps):
            _, r, terminated, truncated, info = env.step(np.zeros(env.nu))
            rewards.append(r)
            if terminated or truncated:
                ended = terminated
                break
        if finish:
            assert ended and 0.5 <= info["idle_at"] <= 1.0 and rewards[-1] > finish
        else:
            assert not ended and info["idle_done"] == 1   # idle, but the episode runs to its limit
