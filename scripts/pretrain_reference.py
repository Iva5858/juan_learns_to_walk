"""Start a policy that already walks the reference gait (biped_rl/tasks/locomotion/mdp/reference.py):
behaviour cloning before any reinforcement learning.

1. Examples: the reference gait played through the simulated servos (with their delay and the action
   filter) under the training commands, changing every few seconds (knee flips included). A virtual
   harness (a torque keeping the torso level, a light side force) holds the robot up so the examples
   cover whole strides, not one stumble before a fall: the policy learns the movement here, balance in RL.
2. The policy's actor is fit to the actions that played it (mean squared error), its observation
   normalization set from the examples.
3. Saved as a run (final_model.zip, vecnormalize.pkl, config.json) that train.py can --resume.

    python scripts/pretrain_reference.py runs/v9/juan_v9_bc
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import observations
from biped_rl.tasks.locomotion.mdp import reference as ref
from biped_rl.training.ppo import new_model, std_units
from biped_rl.training.stages import apply_stage, save_curriculum
from robot.policy import target_to_action

HARNESS_KP, HARNESS_KD, HARNESS_SIDE = 12.0, 0.6, 8.0   # N m/rad, N m s/rad, N/m


def ostrich_config():
    """The ostrich run's settings: commands of every kind on the nominal robot with the measured servos."""
    cfg = EnvConfig()
    apply_stage(cfg, "commands")
    cfg.action_scale = "limits_straight"
    cfg.start_poses = ("reference",)  # standing in the reference pose for the first command
    cfg.command.reference_gait = True
    cfg.command.swing_ratio = ref.DEFAULT.swing_ratio
    cfg.command.swing_height = ref.DEFAULT.lift
    cfg.command.backward_prob, cfg.command.backward_speed = 0.0, 0.0   # backward (knees forward) comes later
    cfg.robot.kp = 30
    cfg.randomization.nominal_delay_steps = 3
    cfg.randomization.action_delay_steps = (3, 4)
    return cfg


def teacher_action(env):
    """The action that makes the filtered, delayed servo target land on the reference pose."""
    r, a = env.robot, env.cfg.action_filter
    want = target_to_action(ref.pose_now(env, env.servo.action_delay + 1), r.action_pivot, r.action_up, r.action_down)
    return np.clip((want - (1 - a) * env.episode.filtered) / a, -1.0, 1.0)


def harness(env):
    """Hold the torso up: a torque toward level and a side force toward the commanded path."""
    d, root = env.data, env.robot.root
    g = env.gravity_body()                 # (0, 0, -1) when level
    w = d.qvel[3:6]
    torque_body = np.array([HARNESS_KP * g[1] - HARNESS_KD * w[0], -HARNESS_KP * g[0] - HARNESS_KD * w[1], 0.0])
    side = np.array([-np.sin(env.command.heading_target), np.cos(env.command.heading_target), 0.0])
    off = side @ (d.qpos[0:3] - np.append(env.command.ref_xy, 0.0))
    d.xfrc_applied[root, 3:6] = env.root_rot() @ torque_body
    d.xfrc_applied[root, 0:3] = -HARNESS_SIDE * off * side


def start_at_reference(env, seed):
    """A new episode standing in the reference pose for its first command (the servos already holding it)."""
    env.reset(seed=seed)
    pose = ref.pose_now(env)
    env.place_pose(pose)
    ep, r = env.episode, env.robot
    ep.filtered = target_to_action(pose, r.action_pivot, r.action_up, r.action_down)
    ep.target_buf.extend([pose] * ep.target_buf.maxlen)
    reading = observations.read_sensors(env)
    ep.sensor_buf.extend([reading] * ep.sensor_buf.maxlen)
    return observations.compute(env)


def collect(args):
    seed, steps = args
    cfg = ostrich_config()
    env = BipedEnv(cfg.to_dict())
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []
    obs = start_at_reference(env, int(seed))
    while len(obs_list) < steps:
        a = teacher_action(env)
        obs_list.append(obs)
        act_list.append(a)
        harness(env)
        # a little noise in what is played, so the examples include small slips from the reference
        obs, _, term, trunc, _ = env.step(np.clip(a + rng.normal(0, 0.02, a.shape), -1, 1))
        if term or trunc:
            obs = start_at_reference(env, int(rng.integers(1 << 30)))
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--steps", type=int, default=600_000, help="example steps (12 s of robot time per 600)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--epochs", type=int, default=40)
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = ostrich_config()
    per = a.steps // a.workers
    with Pool(a.workers) as pool:
        parts = pool.map(collect, [(1000 + i, per) for i in range(a.workers)])
    obs = np.concatenate([o for o, _ in parts])
    act = np.concatenate([x for _, x in parts])
    print(f"examples: {len(obs)} steps", flush=True)

    venv = VecNormalize(DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    model, venv = new_model(cfg, venv, tensorboard_log=str(run_dir))
    venv.obs_rms.update(obs)                     # symmetric statistics (new_model installed them)
    norm_obs = torch.as_tensor(np.clip((obs - venv.obs_rms.mean) / np.sqrt(venv.obs_rms.var + venv.epsilon), -10, 10),
                               dtype=torch.float32)
    target = torch.as_tensor(act)
    policy = model.policy
    params = [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    n, batch = len(norm_obs), 4096
    for epoch in range(a.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mean = policy.get_distribution(norm_obs[idx]).distribution.mean
            loss = torch.mean((mean - target[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f"epoch {epoch}: action error (rms) {np.sqrt(total / n):.4f}", flush=True)
    # Exploration: ~4 deg per joint, not a fresh policy's ~11 (that much noise would shake the learned walk apart).
    with torch.no_grad():
        policy.log_std.copy_(torch.as_tensor(np.log(0.13 * std_units(BipedEnv(cfg.to_dict()))), dtype=torch.float32))
    model.save(run_dir / "final_model")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    save_curriculum(venv, run_dir)
    print(f"saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
