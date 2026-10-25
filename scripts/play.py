"""Drive a trained policy with the keyboard in the MuJoCo viewer, starting with the boot sequence
the real robot will use (straight legs -> home crouch -> policy).

    mjpython scripts/play.py --run checkpoints/juan_v7_12b_pushes_best              # nominal physics
    mjpython scripts/play.py --run checkpoints/juan_v7_12b_pushes_best --robust     # the realistic robot (full difficulty)
    mjpython scripts/play.py --run checkpoints/juan_v7_12b_pushes_best --pushes     # + random shoves
    python scripts/play.py --run checkpoints/juan_v7_12b_pushes_best --test         # no window: check the boot sequence

Keys (in the viewer window):
    Up / Down       forward speed  +- 0.05 m/s   (-0.1 .. top speed)
    Left / Right    turn rate      +- 0.15 rad/s (+-0.6, left = +)
    , / .           sideways       +- 0.05 m/s   (+-0.1, left = +)
    Enter           stop: stand still
    Home            restart from the boot sequence
Mouse: double-click a body, then Ctrl + right-drag to push it.

macOS needs `mjpython` (installed with mujoco) for the viewer window.
"""
import argparse
import threading
import time
from pathlib import Path

import numpy as np

from biped_rl.motions import boot_targets
from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import commands

# GLFW key codes (the viewer passes these to key_callback)
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_COMMA, KEY_PERIOD, KEY_ENTER, KEY_HOME = 44, 46, 257, 268


class Keyboard:
    """Command set by key presses (the viewer only reports presses, so each press is a step)."""

    def __init__(self, top_speed):
        self.top = top_speed
        self.lock = threading.Lock()
        self.forward = self.sideways = self.turn = 0.0
        self.restart = False

    def on_key(self, key):
        with self.lock:
            if key == KEY_UP:
                self.forward = min(self.forward + 0.05, self.top)
            elif key == KEY_DOWN:
                self.forward = max(self.forward - 0.05, -0.1)
            elif key == KEY_LEFT:
                self.turn = min(self.turn + 0.15, 0.6)
            elif key == KEY_RIGHT:
                self.turn = max(self.turn - 0.15, -0.6)
            elif key == KEY_COMMA:
                self.sideways = min(self.sideways + 0.05, 0.1)
            elif key == KEY_PERIOD:
                self.sideways = max(self.sideways - 0.05, -0.1)
            elif key == KEY_ENTER:
                self.forward = self.sideways = self.turn = 0.0
            elif key == KEY_HOME:
                self.restart = True
            else:
                return
            self.forward, self.sideways, self.turn = (round(v, 3) for v in (self.forward, self.sideways, self.turn))
            print(f"command: forward {self.forward:+.2f} m/s, sideways {self.sideways:+.2f} m/s, "
                  f"turn {self.turn:+.2f} rad/s" + ("  (stand)" if not (self.forward or self.sideways or self.turn) else ""),
                  flush=True)

    def command(self):
        with self.lock:
            return self.forward, self.sideways, self.turn


def make_env(run, robust, pushes):
    cfg = EnvConfig.load(run)
    rnd = cfg.randomization
    rnd.enabled = robust or pushes
    rnd.physics_curriculum = False  # full difficulty
    rnd.push_curriculum = False
    rnd.push_velocity = 0.3 if pushes else 0.0
    cfg.episode_seconds = 1e6       # play until it falls or you restart
    return cfg, BipedEnv(cfg.to_dict())


def boot(env, on_step=lambda: None, seed=None):
    """Straight legs -> home crouch, then hand over to the policy. Returns (observation, stayed up)."""
    env.reset(seed=seed, options={"stand": True})  # draws this episode's physics (if randomized)
    env.place_pose(np.zeros(env.nu))        # powered up standing with straight legs
    upright = True
    for target in boot_targets(env.data.qpos[env.robot.qpos_idx].copy(), env.robot.home_joint, env.dt):
        upright &= env.hold_pose(target)
        on_step()
    return env.start_policy({"stand": True}), upright


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="checkpoints/juan_v7_12b_pushes_best")
    p.add_argument("--best", action="store_true", help="best_model instead of final_model")
    p.add_argument("--robust", action="store_true", help="randomized physics at full difficulty")
    p.add_argument("--pushes", action="store_true", help="random 0.3 m/s shoves (implies --robust)")
    p.add_argument("--test", action="store_true", help="no viewer: boot, then walk and turn, and report")
    a = p.parse_args()

    run = Path(a.run)
    cfg, env = make_env(run, a.robust, a.pushes)
    model, norm = load_policy(run, cfg, best=a.best)
    act = lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]

    if a.test:
        obs, upright = boot(env, seed=0)  # fixed seed: the same numbers every time
        print(f"boot sequence: {'stayed upright' if upright else 'FELL'}; joint error vs home "
              f"{np.degrees(np.abs(env.data.qpos[env.robot.qpos_idx] - env.robot.home_joint).max()):.1f} deg (max)")
        for label, command, seconds in (("stand", (0, 0, 0), 2), ("forward 0.3", (0.3, 0, 0), 5),
                                        ("turn left 0.4 while walking 0.2", (0.2, 0, 0.4), 5), ("stop", (0, 0, 0), 2)):
            env.set_command(*command)
            x0, yaw0, fell = env.data.qpos[0:2].copy(), commands.heading_error(env) + env.command.heading_target, False
            for _ in range(int(seconds / env.dt)):
                obs, _, terminated, _, _ = env.step(act(obs))
                if terminated:
                    fell = True
                    break
            yaw = commands.heading_error(env) + env.command.heading_target
            print(f"  {label:32s} moved {np.linalg.norm(env.data.qpos[0:2] - x0):.2f} m, "
                  f"turned {np.degrees(yaw - yaw0):+.0f} deg{'  FELL' if fell else ''}")
        return

    import mujoco.viewer
    keys = Keyboard(top_speed=env.curriculum.speed_max)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys.on_key) as viewer:
        paced = lambda: (viewer.sync(), time.sleep(env.dt))
        while viewer.is_running():
            print("boot sequence: straight legs -> home crouch ...", flush=True)
            obs, _ = boot(env, on_step=paced)
            print("policy running: arrows to walk and turn, Enter to stop, Home to restart", flush=True)
            keys.restart = False
            while viewer.is_running() and not keys.restart:
                start = time.time()
                env.set_command(*keys.command())
                obs, _, terminated, _, _ = env.step(act(obs))
                viewer.sync()
                if terminated:
                    print("fell over: restarting in 2 s", flush=True)
                    time.sleep(2.0)
                    break
                time.sleep(max(0.0, env.dt - (time.time() - start)))


if __name__ == "__main__":
    main()
