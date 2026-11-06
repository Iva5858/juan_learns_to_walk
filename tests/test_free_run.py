"""The free run (training stages speed / speed_physics): straight-leg neutral pose, one go command, no clock,
net-speed and lifted-step rewards. All off by default."""
from types import SimpleNamespace

import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.env_cfg import RewardWeights
from biped_rl.tasks.locomotion.mdp import rewards
from biped_rl.training.stages import apply_stage


def speed_cfg(stage="speed"):
    cfg = EnvConfig()
    apply_stage(cfg, stage)
    return cfg


def test_off_by_default():
    cfg, w = EnvConfig(), RewardWeights()
    assert cfg.neutral_pose == "home" and cfg.command.go_command == () and cfg.command.clock
    assert w.net_speed == 0.0 and w.free_air_time == 0.0


def test_straight_neutral_pose_starts_and_holds_straight_legs():
    env = BipedEnv(speed_cfg())
    env._reset_to_home()
    assert np.allclose(env.robot.home_joint, 0.0) and np.allclose(env.data.qpos[env.robot.qpos_idx], 0.0)
    assert abs(env.soles()[0][:, 2].min() - env._key_sole_z) < 1e-6  # on the floor, where the keyframe stands
    env.reset(seed=0)
    assert np.abs(env.data.qpos[env.robot.qpos_idx]).max() <= 0.05 + 1e-9  # straight + the reset jitter
    for _ in range(100):  # zero action = hold straight legs
        _, _, term, _, _ = env.step(np.zeros(env.nu))
        assert not term
    assert np.abs(env.data.qpos[env.robot.qpos_idx]).max() < np.radians(6)


def test_full_range_reaches_every_end_stop():
    env = BipedEnv(speed_cfg())
    assert np.all(env.action_scale >= np.maximum(-env.robot.joint_lo, env.robot.joint_hi) - 1e-3)


def test_one_go_command_no_clock_whatever_is_asked():
    env = BipedEnv(speed_cfg())
    obs, _ = env.reset(seed=0, options={"speed": 0.3})
    assert np.allclose(env.command.vec, (1.0, 0.0, 0.0)) and not env.command.standing
    env.reset(seed=0, options={"command": (0.4, 0.0, 0.0)})  # the training clips ask this way
    assert np.allclose(env.command.vec, (1.0, 0.0, 0.0))
    obs, _ = env.reset(seed=0, options={"speed": 0.3})
    a = env.actor_obs_dim
    assert np.allclose(obs[a - 4:a - 2], 0.0)  # the clock
    env.set_command(0.1, 0.0, 0.5)
    assert np.allclose(env.command.vec, (1.0, 0.0, 0.0))
    env.set_command(0.0)
    assert env.command.standing
    env.reset(seed=1)
    for _ in range(300):  # never a "stand" or another command in training
        env.step(np.zeros(env.nu))
    assert np.allclose(env.command.vec, (1.0, 0.0, 0.0))


def test_net_speed_pays_progress_only():
    w = RewardWeights(net_speed=10.0)
    s = lambda v, slip=0.0: SimpleNamespace(net_vx=v, slip_v=slip)
    assert np.isclose(rewards.net_speed(None, s(0.25), w), 2.5)
    assert np.isclose(rewards.net_speed(None, s(2.0), w), 20.0)  # no cap
    assert rewards.net_speed(None, s(-0.2), w) == 0.0
    assert np.isclose(rewards.net_speed(None, s(0.25, 0.1), w), 2.5)  # sliding counts only when switched on
    w.net_speed_slip = 1.0
    assert np.isclose(rewards.net_speed(None, s(0.25, 0.1), w), 1.5)
    assert rewards.net_speed(None, s(0.1, 0.3), w) == 0.0             # gliding along: nothing


def test_free_air_time_needs_a_real_step_and_progress():
    w = RewardWeights(free_air_time=1.0, free_air_min=0.1, free_air_cap=0.3, free_air_speed=0.1)
    td = lambda air, v: SimpleNamespace(new_td=np.array([True, False]), air_at_touchdown=np.array([air, 0.0]), net_vx=v)
    assert np.isclose(rewards.free_air_time(None, td(0.2, 0.2), w), 0.1)
    assert np.isclose(rewards.free_air_time(None, td(0.5, 0.2), w), 0.2)    # capped
    assert rewards.free_air_time(None, td(0.05, 0.2), w) == 0.0             # a shuffle's tap
    assert rewards.free_air_time(None, td(0.2, 0.0), w) == 0.0              # marching in place
    assert np.isclose(rewards.free_air_time(None, td(0.2, 0.05), w), 0.05)  # half the progress, half the pay


def test_net_speed_is_measured_on_the_centre_of_mass():
    env = BipedEnv(speed_cfg())
    env.reset(seed=0)
    env.episode.com_hist.clear()
    env.episode.com_hist.append(env.com_xy() - 0.1 * np.array([np.cos(env.command.heading_target),
                                                                np.sin(env.command.heading_target)]))
    _, _, _, _, info = env.step(np.zeros(env.nu))
    assert np.isclose(info["net_speed"], 0.1 / env.dt, rtol=0.05)  # 10 cm over one step


def test_speed_stages():
    cfg = speed_cfg()
    w = cfg.rewards
    assert not cfg.randomization.enabled and cfg.randomization.push_velocity == 0
    assert w.knee_hyperextension == 0 and w.action_rate == 0 and w.tracking_vel == 0 and w.gait_contact == 0
    assert w.net_speed > 0 and w.servo_overload > 0 and w.backdrive > 0 and w.break_impact > 0  # limits stay
    phys = speed_cfg("speed_physics")
    assert phys.randomization.enabled and phys.randomization.physics_curriculum
    assert phys.randomization.push_velocity == 0


def test_robot_controller_follows_the_free_run(tmp_path):
    from robot.policy import Controller
    meta = {"control_hz": 50.0, "history": 3, "home": np.zeros(8), "joint_lo": -np.ones(8), "joint_hi": np.ones(8),
            "action_scale": np.ones(8), "action_filter": 0.5, "gait_period": 0.8, "gait_period_fast": 0.45,
            "fast_speed": 0.8, "clock": False, "go_command": np.array([1.0, 0.0, 0.0])}
    ctrl = Controller(SimpleNamespace(meta=meta))
    ctrl.set_command(0.2, 0.1, 0.0)
    assert np.allclose(ctrl.command, (1.0, 0.0, 0.0))
    ctrl.set_command(0.0)
    assert ctrl.standing
    old = Controller(SimpleNamespace(meta={k: v for k, v in meta.items() if k not in ("clock", "go_command")}))
    old.set_command(0.2)
    assert old.clock and np.allclose(old.command, (0.2, 0.0, 0.0))  # policies exported before: unchanged


def test_a_fall_takes_back_the_last_second_of_speed_reward():
    cfg = speed_cfg()
    cfg.rewards.net_speed_slip = 0.0  # (with it, this dive's slipping feet already cancel its speed)
    env = BipedEnv(cfg)
    env.reset(seed=0)
    paid = 0.0
    for _ in range(300):  # lean everything forward: a dive
        a = np.zeros(env.nu)
        a[[1, 5]] = -0.6
        _, r, term, _, info = env.step(a)
        paid += info["rew/net_speed"] * info["penalty_scale"]
        if term:
            break
    assert term
    refunded = sum(env.episode.speed_paid)
    step_pay = (env.cfg.rewards.alive + sum(v for k, v in info.items() if k.startswith("rew/"))) * info["penalty_scale"]
    assert refunded > 0 and np.isclose(r, step_pay + env.cfg.rewards.termination - refunded)
    assert paid - refunded < 0.5 * paid  # most of what the dive earned is taken back


def test_hip_roll_free_band():
    robot = SimpleNamespace(hip_roll_idx=[0, 4])
    env = SimpleNamespace(robot=robot)
    q = lambda deg: SimpleNamespace(q=np.radians(np.array([deg, 0, 0, 0, -deg, 0, 0, 0], dtype=float)))
    w = RewardWeights(hip_roll=1.0)
    assert np.isclose(rewards.hip_roll(env, q(10), w), 2 * np.radians(10) ** 2)  # default: from 0, as before
    w.hip_roll_free_deg = 12.0
    assert rewards.hip_roll(env, q(10), w) == 0.0                                # within the band: free
    assert np.isclose(rewards.hip_roll(env, q(19), w), 2 * np.radians(7) ** 2)   # only the excess


def test_nominal_delay_range():
    cfg = speed_cfg()
    env = BipedEnv(cfg)
    env.reset(seed=0)
    assert env.servo.action_delay == 0   # default: the fixed nominal delay
    cfg.randomization.nominal_delay_range = (1, 4)
    env = BipedEnv(cfg)
    seen = set()
    for seed in range(40):
        env.reset(seed=seed)
        seen.add(env.servo.action_delay)
    assert seen == {1, 2, 3, 4}
    assert env.episode.target_buf.maxlen >= 5  # room for a 4-step delay (the buffer used to be sized for 2)


def test_swing_knee_pays_a_bent_lifted_leg_that_travels():
    env = SimpleNamespace(robot=SimpleNamespace(knee_idx=[2, 6]), episode=SimpleNamespace(contact=np.array([0.0, 1.0])))
    q = lambda left, right: np.radians(np.array([0, 0, left, 0, 0, 0, right, 0], dtype=float))
    w = RewardWeights()
    assert rewards.swing_knee(env, SimpleNamespace(q=q(40, 40), net_vx=0.3), w) == 0.0     # off by default
    w.swing_knee = 1.0
    assert np.isclose(rewards.swing_knee(env, SimpleNamespace(q=q(40, 40), net_vx=0.3), w), 0.5)   # left in the air
    assert np.isclose(rewards.swing_knee(env, SimpleNamespace(q=q(-80, 0), net_vx=0.3), w), 0.5)   # bird knee, capped
    assert np.isclose(rewards.swing_knee(env, SimpleNamespace(q=q(20, 40), net_vx=0.3), w), 0.25)  # stance knee: nothing
    assert rewards.swing_knee(env, SimpleNamespace(q=q(40, 40), net_vx=0.0), w) == 0.0             # in place: nothing


def test_go_with_a_turn_rate():
    cfg = speed_cfg()
    cfg.command.go_turn_range = (-0.6, 0.6)
    env = BipedEnv(cfg)
    rates = set()
    for seed in range(5):
        env.reset(seed=seed)
        assert env.command.vec[0] == 1.0 and not env.command.standing
        rates.add(round(float(env.command.vec[2]), 3))
    assert len(rates) > 1 and all(abs(r) <= 0.6 for r in rates)   # random turn rates while going
    env.set_command(0.2, 0.0, 0.9)
    assert np.allclose(env.command.vec, (1.0, 0.0, 0.6))            # go, the turn clipped to the trained range
    env.reset(seed=0, options={"command": (0.4, 0.0, -0.3)})
    assert np.allclose(env.command.vec, (1.0, 0.0, -0.3))           # a test can ask for a turn rate
    from robot.policy import Controller
    meta = {"control_hz": 50.0, "history": 3, "home": np.zeros(8), "joint_lo": -np.ones(8), "joint_hi": np.ones(8),
            "action_scale": np.ones(8), "action_filter": 0.5, "gait_period": 0.8, "gait_period_fast": 0.45,
            "fast_speed": 0.8, "clock": False, "go_command": np.array([1.0, 0.0, 0.0]),
            "go_turn_range": np.array([-0.6, 0.6])}
    ctrl = Controller(SimpleNamespace(meta=meta))
    ctrl.set_command(0.0, 0.0, 0.4)
    assert np.allclose(ctrl.command, (1.0, 0.0, 0.4))   # the dashboard's turn passes through; any move is go
