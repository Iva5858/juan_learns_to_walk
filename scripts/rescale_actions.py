"""Give a trained policy a larger (or smaller) movement range without changing what it does.

The policy's action is an offset from the home pose, `action_scale` rad per unit: with 0.5 every joint
reaches only +-28.6 deg around home. Raising the scale outright would make a trained policy move
1.6x too far. Instead, everything expressed in action units is converted, so the new run starts out
behaving exactly like the old one and can then learn to use the extra range:

  * the network's action outputs (last layer) and its exploration noise are scaled by old/new;
  * the previous actions it observes keep their normalized values (their running mean and
    variance are scaled the same way);
  * the smoothness penalties on action changes (action_rate, action_acc) are scaled by (new/old)^2,
    so the same motion in radians costs the same.

    python scripts/rescale_actions.py runs/juan_v7_13_straight runs/juan_v7_13_scale08 --scale 0.8
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--scale", type=float, required=True, help="new action_scale [rad per unit action]")
    a = p.parse_args()
    src, out = Path(a.run), Path(a.out)
    cfg = EnvConfig.load(src)
    old = cfg.action_scale
    r = old / a.scale  # old / new: the same radians in the new units
    out.mkdir(parents=True, exist_ok=True)

    model = PPO.load(src / "final_model", device="cpu")
    pol = model.policy
    with torch.no_grad():
        pol.action_net.weight.mul_(r)
        pol.action_net.bias.mul_(r)
        pol.log_std.add_(math.log(r))
    model.save(out / "final_model")

    norm = VecNormalize.load(str(src / "vecnormalize.pkl"), DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
    nu = norm.venv.envs[0].nu  # (from the model: saved configs may list extra, unused joint names)
    frame = 6 + 3 * nu  # gravity (3), gyro (3), joint angles, joint speeds, previous action (nu each)
    for k in range(cfg.history):
        act = slice(k * frame + 6 + 2 * nu, (k + 1) * frame)  # the previous action in frame k
        norm.obs_rms.mean[act] *= r
        norm.obs_rms.var[act] *= r * r
    norm.save(str(out / "vecnormalize.pkl"))

    cfg.action_scale = a.scale
    cfg.rewards.action_rate /= r * r
    cfg.rewards.action_acc /= r * r
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    if (src / "curriculum.json").exists():
        shutil.copy(src / "curriculum.json", out / "curriculum.json")
    print(f"{src} -> {out}: action_scale {old} -> {a.scale} rad "
          f"(reach +-{math.degrees(a.scale):.1f} deg around home); outputs x{r:.4f}")


if __name__ == "__main__":
    main()
