import json

import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import observations


def rollout(cfg, steps=200, seed=3):
    env = BipedEnv(cfg.to_dict())
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    out = []
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(np.clip(0.3 * rng.standard_normal(env.nu), -1, 1))
        out.append(np.concatenate([obs, [r]]))
        if term or trunc:
            obs, _ = env.reset()
    return env, np.array(out)


def test_spaces():
    env = BipedEnv(EnvConfig().to_dict())
    obs, _ = env.reset(seed=0)
    assert env.nu == 8
    assert obs.shape == env.observation_space.shape
    assert env.actor_obs_dim == observations.actor_dim(env.cfg, env.nu)
    assert env.dt == 0.02


def test_deterministic_with_a_seed():
    cfg = EnvConfig()  # randomized physics, pushes, noise: everything that draws random numbers
    assert np.array_equal(rollout(cfg)[1], rollout(cfg)[1])


def test_stands_at_home_without_actions():
    cfg = EnvConfig()
    cfg.randomization.enabled = False
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=0, options={"stand": True})
    for _ in range(100):
        _, _, term, _, _ = env.step(np.zeros(env.nu))
        assert not term
    assert np.degrees(np.abs(env.data.qpos[env.robot.qpos_idx] - env.robot.home_joint).max()) < 5


def test_hip_roll_outward_signs():
    env = BipedEnv(EnvConfig().to_dict())
    # positive hip roll: left leg outward, right leg inward (biped_rl/robots/juan.py)
    assert env.robot.roll_out_sign.tolist() == [1.0, -1.0]


def test_limits_action_scale_spans_each_joints_range():
    from robot.policy import action_reach, action_to_target
    cfg = EnvConfig()
    cfg.action_scale = "limits"
    env = BipedEnv(cfg.to_dict())
    r = env.robot
    pivot, up, down = action_reach("limits", r.home_joint, r.joint_lo, r.joint_hi)
    assert np.allclose(action_to_target(np.ones(env.nu), pivot, up, down, r.joint_lo, r.joint_hi), r.joint_hi)
    assert np.allclose(action_to_target(-np.ones(env.nu), pivot, up, down, r.joint_lo, r.joint_hi), r.joint_lo)
    assert np.allclose(action_to_target(np.zeros(env.nu), pivot, up, down, r.joint_lo, r.joint_hi), r.home_joint)
    # a number keeps the classic mapping
    pivot, up, down = action_reach(0.5, r.home_joint, r.joint_lo, r.joint_hi)
    a = np.full(env.nu, 0.4)
    assert np.allclose(action_to_target(a, pivot, up, down, r.joint_lo, r.joint_hi),
                       np.clip(r.home_joint + 0.2, r.joint_lo, r.joint_hi))


def test_straight_pivot_and_start_poses():
    from robot.policy import action_reach, action_to_target, target_to_action
    cfg = EnvConfig()
    cfg.action_scale, cfg.start_poses = "limits_straight", ("home", "knees_back", "straight")
    cfg.randomization.enabled = False  # no varied starting crouch: check the poses themselves
    env = BipedEnv(cfg.to_dict())
    r = env.robot
    pivot, up, down = action_reach("limits_straight", r.home_joint, r.joint_lo, r.joint_hi)
    assert np.allclose(action_to_target(np.zeros(env.nu), pivot, up, down, r.joint_lo, r.joint_hi), 0.0)
    assert np.allclose(action_to_target(np.ones(env.nu), pivot, up, down, r.joint_lo, r.joint_hi), r.joint_hi)
    a = target_to_action(r.home_joint, pivot, up, down)  # the filter holds home when the policy takes over
    assert np.allclose(action_to_target(a, pivot, up, down, r.joint_lo, r.joint_hi), r.home_joint)
    seen = set()
    for seed in range(12):
        env.reset(seed=seed)
        q = env.data.qpos[r.qpos_idx]
        name = min(r.poses, key=lambda k: np.abs(q - r.poses[k]).max())
        assert np.abs(q - r.poses[name]).max() < 0.06, name  # each start is one of the poses (+ noise)
        assert np.allclose(action_to_target(env.episode.filtered, pivot, up, down, r.joint_lo, r.joint_hi),
                           env.episode.target, atol=1e-9)  # no jump at the first step
        seen.add(name)
    assert seen == {"home", "knees_back", "straight"}
    assert r.poses["knees_back"][r.knee_idx].max() < 0  # knees bent backwards


def test_clips_follow_what_the_run_trains_on():
    from biped_rl.training.callbacks import clip_commands
    from biped_rl.training.stages import apply_stage
    cfg = EnvConfig()
    cfg.command.fixed_forward = 1.0
    assert clip_commands(cfg, 0.4) == [("run", (1.0, 0.0, 0.0))]
    cfg = EnvConfig()
    apply_stage(cfg, "walk")
    assert clip_commands(cfg, 0.1) == [("fwd", (0.1, 0.0, 0.0))]          # the speed it trains at, nothing else
    cfg.command.backward_prob, cfg.command.backward_speed = 1.0, 0.25
    assert [c[0] for c in clip_commands(cfg, 0.4)] == ["back"]             # backward only: no forward clip
    cfg = EnvConfig()
    apply_stage(cfg, "commands")
    assert [c[0] for c in clip_commands(cfg, 0.4)] == ["fwd", "back", "side", "turn"]


def test_physics_difficulty_zero_is_the_nominal_robot():
    """Every randomization grows with the physics difficulty: at 0 the randomized robot is the nominal one
    (but the lowest delays), and its start isn't jostled."""
    cfg = EnvConfig()
    cfg.randomization.enabled, cfg.randomization.physics_curriculum = True, True
    env = BipedEnv(cfg.to_dict())
    env.curriculum.difficulty = 0.0
    env.reset(seed=3)
    n, r = env.robot.nominal, cfg.robot
    assert np.allclose(env.model.body_mass, n["mass"]) and np.allclose(env.model.geom_friction[:, 0], n["friction"][:, 0])
    assert np.allclose(env.servo.kp, r.kp) and np.allclose(env.servo.stall, r.stall_torque)
    assert env.servo.latency_steps == 0 and env.servo.action_delay == cfg.randomization.action_delay_steps[0]
    assert np.allclose(env.model.opt.gravity, n["gravity"])
    env.curriculum.difficulty = 1.0
    env.reset(seed=3)
    assert not np.allclose(env.servo.kp, r.kp)  # the full ranges at 1


def test_physics_stage_starts_easy_after_a_stage_without_the_curriculum(tmp_path):
    from biped_rl.training.stages import load_curriculum
    cfg = EnvConfig()
    cfg.randomization.physics_curriculum = False  # as the commands stage saves it
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "curriculum.json").write_text(json.dumps({"speed_max": 0.4, "difficulty": 1.0, "push_level": [0.05] * 8}))
    cfg.randomization.enabled, cfg.randomization.physics_curriculum = True, True
    env = BipedEnv(cfg.to_dict())

    class OneEnv:  # the VecEnv calls load_curriculum makes
        def env_method(self, name, *args):
            return [getattr(env, name)(*args)]
    load_curriculum(OneEnv(), tmp_path)
    assert env.curriculum.difficulty == cfg.randomization.difficulty_start
