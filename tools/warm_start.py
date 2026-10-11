"""Start a v7-style policy from a walking forward-only policy (juan_v3), instead of from scratch.

Learning to walk from zero on Juan's model got stuck standing still; every Juan policy so far
inherited walking from the placeholder biped. The v7 network sees the same inputs as v3 plus two
(sideways and turn-rate commands), in the same order, so v3's weights are copied over with zero
weight on the two new inputs: for a straight command it walks like v3 from the first step. The
training stages then add commands, realistic physics and pushes on top.

    python tools/warm_start.py runs/juan_v3 runs/juan_v7_0_warm
    python train.py --run-name juan_v7_2_commands --stage commands --resume runs/juan_v7_0_warm

With mirror symmetry on (the default), the new policy averages v3 with its mirror image, which
removes v3's left/right lean but also changes its gait a little: the check at the end shows by how
much (distance walked and falls, v3 vs the copy).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biped_rl.config import EnvConfig  # noqa: E402
from biped_rl.env import BipedEnv  # noqa: E402
from biped_rl.symmetry import mirror  # noqa: E402
from train import apply_stage, new_model  # noqa: E402

# Variance given to the normalizer for the two new command inputs (roughly that of the commands
# drawn in the "commands" stage); their weights start at zero, so this only sets their scale.
NEW_INPUT_VAR = {"sideways": 0.003, "turn_rate": 0.08}


def input_map(src_env, dst_env):
    """Index in the new observation of every old observation entry: the history frames and the
    forward command keep their place, the two new commands are inserted after the forward one."""
    frames = src_env.actor_obs_dim - 5  # old tail: forward command, clock (2), heading error (2)
    if src_env._omni or not dst_env._omni or dst_env.actor_obs_dim != src_env.actor_obs_dim + 2:
        raise SystemExit("expects a forward-only source run and an omni (3-command) target")
    old_total = src_env.observation_space.shape[0]
    return np.array([i if i <= frames else i + 2 for i in range(old_total)]), frames + 1


def walk_test(model, norm, cfg, seconds=10.0, speed=0.4, seed=0):
    """Nominal physics, straight command: (metres walked forward, fell?, sideways drift m)."""
    cfg = EnvConfig.from_dict(cfg.to_dict())
    cfg.randomization.enabled = False
    env = BipedEnv(cfg.to_dict())
    obs, _ = env.reset(seed=seed, options={"speed": speed})
    x0 = env.data.qpos[0]
    for _ in range(int(seconds / env.dt)):
        action = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            return env.data.qpos[0] - x0, True, info["lateral_offset"]
    return env.data.qpos[0] - x0, False, info["lateral_offset"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="forward-only run to start from, e.g. runs/juan_v3")
    p.add_argument("target", help="new run directory, e.g. runs/juan_v7_0_warm")
    p.add_argument("--std", type=float, default=0.1,
                   help="minimum exploration noise (v3's has shrunk to ~0.03, too little to learn new commands)")
    args = p.parse_args()
    src, dst = Path(args.source), Path(args.target)

    src_cfg = EnvConfig.from_dict(json.loads((src / "config.json").read_text()))
    src_env = BipedEnv(src_cfg.to_dict())
    src_model = PPO.load(src / "final_model", device="cpu")
    src_norm = VecNormalize.load(str(src / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(src_cfg.to_dict())]))
    src_norm.training = False

    dst_cfg = EnvConfig()
    dst_cfg.robot.model_path, dst_cfg.robot.foot_bodies = src_cfg.robot.model_path, src_cfg.robot.foot_bodies
    apply_stage(dst_cfg, "walk")
    dst_env = BipedEnv(dst_cfg.to_dict())
    model, norm = new_model(dst_cfg, DummyVecEnv([lambda: BipedEnv(dst_cfg.to_dict())]))
    old_to_new, first_new = input_map(src_env, dst_env)
    new_inputs = [first_new, first_new + 1]

    # Weights: every layer copies as is, except the first layer of the actor and of the critic,
    # which read the observation: their columns move to the new positions, new columns are zero.
    old_sd, new_sd = src_model.policy.state_dict(), model.policy.state_dict()
    for key, value in old_sd.items():
        if key.endswith("mask"):
            continue
        if key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
            w = th.zeros_like(new_sd[key])
            w[:, old_to_new] = value
            new_sd[key] = w
        else:
            new_sd[key] = value.clone()
    new_sd["log_std"] = th.maximum(old_sd["log_std"], th.tensor(np.log(args.std), dtype=th.float32))
    model.policy.load_state_dict(new_sd)

    # Normalization: copy v3's statistics to the new positions, give the new inputs a sensible
    # scale, then make them mirror-symmetric (needed for the symmetric policy to be exact).
    rms = norm.obs_rms
    mean, var = np.zeros(rms.mean.shape), np.ones(rms.var.shape)
    mean[old_to_new], var[old_to_new] = src_norm.obs_rms.mean, src_norm.obs_rms.var
    var[new_inputs] = NEW_INPUT_VAR["sideways"], NEW_INPUT_VAR["turn_rate"]
    if dst_cfg.mirror_symmetry:
        m_mean = mirror(mean, rms.perm, rms.sign)
        var = 0.5 * (var + var[rms.perm]) + 0.25 * (mean - m_mean) ** 2  # variance of the 50/50 mix
        mean = 0.5 * (mean + m_mean)
    rms.mean, rms.var = mean, var
    rms.count = 1e6  # v3's count (~1e9) would freeze the statistics; let them adapt to the new physics
    norm.ret_rms = src_norm.ret_rms
    norm.training = False

    dst.mkdir(parents=True, exist_ok=True)
    model.save(dst / "final_model")
    norm.save(str(dst / "vecnormalize.pkl"))
    (dst / "config.json").write_text(json.dumps(dst_cfg.to_dict(), indent=2))
    print(f"saved {dst}: {src}'s weights, {len(old_to_new)} inputs mapped, 2 new (zero weight), "
          f"exploration noise {np.exp(new_sd['log_std'].numpy()).round(3).min()}-{np.exp(new_sd['log_std'].numpy()).round(3).max()}")

    print("check, nominal physics, 0.4 m/s straight, 10 s (3 starts):")
    for name, mdl, nrm, cfg in (("source", src_model, src_norm, src_cfg), ("copy  ", model, norm, dst_cfg)):
        results = [walk_test(mdl, nrm, cfg, seed=s) for s in range(3)]
        print(f"  {name}: walked {', '.join(f'{d:.2f} m' for d, _, _ in results)}; "
              f"falls {sum(f for _, f, _ in results)}/3; sideways drift {', '.join(f'{100 * o:.0f} cm' for *_, o in results)}")


if __name__ == "__main__":
    main()
