"""States where a getup policy hands over to the walker, for the walker to practise starting from them
(RandomizationConfig.start_pool): the getup policy runs from varied lying poses until it has stood on its feet
for --stand-for s, and the robot's exact state then (joint angles and speeds, torso pose and motion) is saved.
They are mostly lunges, one foot 20+ cm ahead: getting up from the floor ends there, and only a step squares it.

    python scripts/getup_pool.py runs/getup/juan_getup_20 out/getup_pool.npz --n 1000
then  ./scripts/finetune.sh v7/juan_v7_28_takeover --randomization start_pool=out/getup_pool.npz start_pool_prob=0.25
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.getup import GetUpEnv, config

W = {}   # per worker process: its getup env, policy and settings (macOS starts workers fresh)


def init_worker(args):
    run = Path(args["getup"])
    saved = json.loads((run / "getup.json").read_text())
    start = {"side_deg": args["side_deg"], "joint_rad": args["joint_rad"], "any_heading": True,
             "knee_min_deg": saved.get("start", {}).get("knee_min_deg")}
    env = GetUpEnv(config(8.0, args["physics_from"] or None).to_dict(), saved["weights"], start)
    norm = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: env]))
    norm.training = False
    W.update(env=env, model=PPO.load(run / "final_model", device="cpu"), norm=norm, stand_for=args["stand_for"])


def handover_state(seed):
    """(qpos, qvel) when the getup policy has stood --stand-for s, or None if it never did."""
    env, model, norm = W["env"], W["model"], W["norm"]
    obs, _ = env.reset(seed=seed)
    for _ in range(env._max_steps):
        obs, _, terminated, truncated, _ = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        if env.episode.stood_for >= W["stand_for"]:
            return env.data.qpos.copy(), env.data.qvel.copy()
        if terminated or truncated:
            return None
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("getup", help="getup run (final_model.zip, vecnormalize.pkl, getup.json)")
    p.add_argument("out", help=".npz to write")
    p.add_argument("--n", type=int, default=1000, help="getup attempts (the pool keeps the ones that stood)")
    p.add_argument("--side-deg", type=float, default=45.0)
    p.add_argument("--joint-rad", type=float, default=0.25)
    p.add_argument("--stand-for", type=float, default=0.3)
    p.add_argument("--physics-from", default="", help="physics randomization for the getup runs ('' = nominal robot)")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    with Pool(a.workers, initializer=init_worker, initargs=(vars(a),)) as pool:
        states = [s for s in pool.map(handover_state, range(a.n)) if s is not None]
    if not states:
        raise SystemExit("the getup policy never stood: no states")
    qpos, qvel = (np.array(x) for x in zip(*states))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, qpos=qpos, qvel=qvel, getup=str(a.getup))
    print(f"{len(states)}/{a.n} getups stood {a.stand_for} s: saved to {a.out}")


if __name__ == "__main__":
    main()
