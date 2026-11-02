"""Can a run's policy stay up in the next stage's setup? Runs it there (the next stage's commands, physics and
pushes; rewards don't matter), deterministic, and exits 1 if too few episodes last. scripts/from_scratch.sh
calls it before the commands stage: twice (juan_v8d, juan_v8j) a free run fell at the first command it
had never been shown, and the commands stage spent hours falling instead of learning.

    python scripts/check_handoff.py runs/v8/juan_v8k_stride commands
"""
import argparse
import sys

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.training.stages import apply_stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("stage", help="the stage it is handed to (commands, physics, pushes)")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--min-share", type=float, default=0.5, help="share of episodes that must last (default 0.5)")
    p.add_argument("--keep-commands", action="store_true",
                   help="keep the run's own command settings (a pipeline that overrides the stage's commands)")
    a = p.parse_args()
    cfg = EnvConfig.load(a.run)
    own = EnvConfig.load(a.run).command
    apply_stage(cfg, a.stage)
    if a.keep_commands:
        cfg.command = own
    model, norm = load_policy(a.run, cfg)
    lasted, ends = 0, {"fell": 0, "broke": 0}
    for seed in range(a.episodes):
        env = BipedEnv(cfg.to_dict())
        obs, _ = env.reset(seed=1000 + seed)
        for _ in range(int(a.seconds / env.dt)):
            obs, _, term, trunc, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
            if term:
                ends["broke" if info["broke"] else "fell"] += 1
                break
            if trunc:
                lasted += 1
                break
        else:
            lasted += 1
        env.close()
    ok = lasted >= a.min_share * a.episodes
    print(f"hand-off {a.run} -> {a.stage}: {lasted}/{a.episodes} lasted {a.seconds:.0f} s "
          f"(fell {ends['fell']}, hardware limit {ends['broke']}): {'ok' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
