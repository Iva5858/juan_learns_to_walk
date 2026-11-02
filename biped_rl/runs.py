"""Trained runs on disk (docs/runs.md): runs/<generation>/<name>/ holds config.json, final_model.zip and
vecnormalize.pkl (best_model.zip and best_vecnormalize.pkl: the best evaluation during training), and
run.json: where it came from (the run it resumed, the stage, the command line, when, which commit)."""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.tasks.locomotion import BipedEnv

RUNS = Path("runs")
# A run's name: runs/<generation>/<name>. A generation is one line of training from scratch (v7, v8, ...);
# fine-tunes of it stay in it (juan_v7_13 -> juan_v7_14). Failed runs move to runs/archive/<generation>/.
RUN_NAME = re.compile(r"^(v\d+|hop|early|archive/[\w.-]+|test)/[\w.-]+$")


def load_policy(run, cfg, best=False):
    """(model, normalizer) of a run. Use as model.predict(norm.normalize_obs(obs), deterministic=True)
    on a BipedEnv made from `cfg`."""
    run = Path(run)
    model = PPO.load(run / ("best_model" if best else "final_model"), device="cpu")
    norm = VecNormalize.load(str(run / ("best_vecnormalize.pkl" if best else "vecnormalize.pkl")),
                             DummyVecEnv([lambda: BipedEnv(cfg.to_dict())]))
    norm.training, norm.norm_reward = False, False
    return model, norm


def check_name(name):
    """Refuse a run name outside the layout (a bare name would land in runs/ itself, unsorted)."""
    if not RUN_NAME.match(name):
        raise SystemExit(f"run name {name!r}: use <generation>/<name>, e.g. v7/juan_v7_21_speed "
                         "(generations: v1, v2, ..., hop, early, test; see docs/runs.md)")


def _commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+changes" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def write_record(run_dir, **fields):
    """Add to the run's run.json (created at the start of training, completed at the end)."""
    path = Path(run_dir) / "run.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(fields)
    path.write_text(json.dumps(record, indent=2))


def start_record(run_dir, args):
    write_record(run_dir, name=args.run_name, parent=args.resume, stage=args.stage, timesteps=args.timesteps,
                 command=" ".join(["python"] + sys.argv), commit=_commit(),
                 started=datetime.now().isoformat(timespec="seconds"), finished=None)
