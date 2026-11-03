"""Test the hand-built walker in the simulator, the way scripts/analysis/walk_test.py tests trained policies:
standing at home, the policy takes over (start_policy), 1 s standing, then a straight command for 8 s, on the
measured servo model (kp 30, 3-4 steps of bus delay). Single process.

    python scripts/handcrafted/test_walker.py out/handcrafted/juan_handcrafted.npz walk  [--random] [--shoves]
    python scripts/handcrafted/test_walker.py out/handcrafted/juan_handcrafted.npz stand  (standing shoves)
    python scripts/handcrafted/test_walker.py out/handcrafted/juan_handcrafted.npz video --speed 0.4
    python scripts/handcrafted/test_walker.py out/handcrafted/juan_handcrafted.npz check  (robot Controller
        vs the env's observation path, step by step)

The policy is robot/policy.py's Policy (the numpy pipeline the robot runs) fed the env's own observation
(as walk_test does; the export check shows the two are identical).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig  # noqa: E402
from robot.policy import Controller, Policy  # noqa: E402

PHYSICS_FROM = PROJECT / "runs/v7/juan_v7_13_straight"
SPEEDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DIRECTIONS = ("fwd", "fwd-L", "left", "back-L", "back", "back-R", "right", "fwd-R")


def make_env(policy, random=False, shoves=0.0, render=False):
    m = policy.meta
    cfg = EnvConfig.load(PHYSICS_FROM) if random else EnvConfig()
    cfg.use_measured_servos()
    r = cfg.randomization
    r.enabled = random
    r.nominal_delay_steps = 3            # the real bus delay also on the nominal robot
    r.physics_curriculum, r.push_curriculum = False, False
    r.push_velocity = shoves
    cfg.action_scale = tuple(np.broadcast_to(m["action_scale"], (8,)).tolist())
    cfg.action_filter = float(m["action_filter"])
    c = cfg.command
    c.gait_period, c.gait_period_fast, c.fast_speed = (float(m["gait_period"]), float(m["gait_period_fast"]),
                                                      float(m["fast_speed"]))
    return BipedEnv(cfg.to_dict(), render_mode="rgb_array" if render else None)


def tilt_deg(env):
    return float(np.degrees(np.arccos(np.clip(-env.gravity_body()[2], -1, 1))))


def run(env, policy, seed, plan, push=None, frames=None, trace=None):
    """plan: [(seconds, command)]. push: (time, (vx, vy) in the robot frame). Returns a dict."""
    env.reset(seed=seed)
    env.settle_at_home()
    obs = env.start_policy()
    env.episode.next_push = 2.0  # no random shove while still standing
    total = sum(s for s, _ in plan)
    t_walk, p_walk, tilts, impacts = None, None, [], []
    fell = False
    t_end = 0.0
    for seconds, cmd in plan:
        env.set_command(*cmd)
        obs[90:93] = cmd  # the command part of the observation (the env's next one has it anyway)
        if cmd[0] and t_walk is None:
            t_walk, p_walk, y0 = env.episode.t, env.data.qpos[:2].copy(), env.yaw()
        t_end += seconds
        while env.episode.t < t_end - 1e-9:
            if push is not None and env.episode.t >= push[0]:
                c, s = np.cos(env.yaw()), np.sin(env.yaw())
                vx, vy = push[1]
                env.data.qvel[0:2] += (c * vx - s * vy, s * vx + c * vy)
                env.episode.last_push, env.episode.last_push_vec = env.episode.t, np.array([c * vx - s * vy, s * vx + c * vy])
                push = None
            obs, _, term, _, info = env.step(policy(obs))
            tilts.append(tilt_deg(env))
            impacts.append(info["peak_force_bw"])
            if frames is not None and env.episode.steps % 2 == 0:
                frames.append(env.render())
            if trace is not None:
                trace.append((env.episode.t, env.data.qpos[:3].copy(), env.gravity_body().copy(),
                              env.data.qpos[env.robot.qpos_idx].copy(), env.episode.contact.copy()))
            if term:
                fell = True
                why = ("broke (impact %.1f bw)" % info["peak_force_bw"]) if info["broke"] else (
                    "touch" if env.non_foot_ground_contact() else "tilt/height")
                g = env.gravity_body()
                why += " " + ("fwd" if g[0] > 0.3 else "back" if g[0] < -0.3 else "") + (
                    "L" if g[1] > 0.3 else "R" if g[1] < -0.3 else "")
                break
        if fell:
            break
    out = {"fell": fell, "why": why if fell else "", "t": env.episode.t, "max_tilt": max(tilts) if tilts else 0.0,
           "mean_tilt": float(np.mean(tilts)) if tilts else 0.0,
           "impact_p95": float(np.percentile(impacts, 95)) if impacts else 0.0}
    if t_walk is not None:
        fwd, left, turn = env.moved_since(p_walk, y0)
        out.update(speed=fwd / max(env.episode.t - t_walk, 1e-3), left=left, turn=turn,
                   walked=fwd)
    else:  # fell before the walk started
        out.update(speed=0.0, left=0.0, turn=0.0, walked=0.0, why="(standing) " + out["why"])
    return out


def walk(policy, seeds, random, shoves, speeds):
    env = make_env(policy, random=random, shoves=shoves)
    print(f"walk: {'randomized (v7_13 ranges)' if random else 'nominal'} robot, measured servos, "
          f"shoves {shoves} m/s; 1 s stand + 8 s walk")
    rows = []
    for v in speeds:
        res = [run(env, policy, 7000 + s, [(1.0, (0, 0, 0)), (8.0, (v, 0, 0))]) for s in range(seeds)]
        falls = sum(r["fell"] for r in res)
        sp = np.mean([r["speed"] for r in res])
        print(f"  cmd {v:.2f}: speed {sp:.3f} m/s, falls {falls}/{seeds}, side {np.mean([r['left'] for r in res]):+.2f} m, "
              f"turn {np.mean([r['turn'] for r in res]):+.0f} deg, tilt mean {np.mean([r['mean_tilt'] for r in res]):.1f} "
              f"max {np.max([r['max_tilt'] for r in res]):.1f} deg, impact p95 {np.mean([r['impact_p95'] for r in res]):.1f} bw, fell at "
              f"{[(round(r['t'], 1), r['why']) for r in res if r['fell']]}", flush=True)
        rows.append((v, sp, falls))
    return rows


def stand(policy, seeds, random, strengths, walking=0.0):
    """One shove from each of 8 directions (robot frame), 1 s after the start (standing) or 2 s into a walk."""
    env = make_env(policy, random=random)
    mode = f"walking at {walking}" if walking else "standing"
    print(f"shoves while {mode} ({'randomized' if random else 'nominal'}, {seeds} seeds): strongest survived per direction")
    held = []
    for k, d in enumerate(DIRECTIONS):
        ang = k * np.pi / 4
        best = 0.0
        for s in strengths:
            plan = [(1.0, (0, 0, 0)), (5.0, (walking, 0, 0))] if walking else [(4.0, (0, 0, 0))]
            t_push = 3.0 if walking else 1.0
            ok = all(not run(env, policy, 20000 + sd, plan, push=(t_push, (s * np.cos(ang), s * np.sin(ang))))["fell"]
                     for sd in range(seeds))
            if not ok:
                break
            best = s
        held.append(best)
        print(f"  {d:7s}: {best:.1f} m/s", flush=True)
    return held


def video(policy, speed, path, seconds=8.0, push=None, random=False):
    import imageio
    env = make_env(policy, random=random, render=True)
    frames = []
    r = run(env, policy, 7000, [(1.0, (0, 0, 0)), (seconds, (speed, 0, 0)), (2.0, (0, 0, 0))], push=push, frames=frames)
    imageio.mimsave(path, frames, fps=25)
    print(f"video {path}: {r}")


def walk_shoved(policy, seeds, speeds, strength=0.3):
    """One shove of `strength` m/s from a random direction 2 s into the walk (nominal robot)."""
    env = make_env(policy)
    rng = np.random.default_rng(0)
    print(f"walk with one {strength} m/s shove from a random direction 2 s into the walk (nominal, {seeds} seeds)")
    for v in speeds:
        res = []
        for s in range(seeds):
            ang = rng.uniform(0, 2 * np.pi)
            res.append(run(env, policy, 7000 + s, [(1.0, (0, 0, 0)), (8.0, (v, 0, 0))],
                           push=(3.0, (strength * np.cos(ang), strength * np.sin(ang)))))
        print(f"  cmd {v:.2f}: falls {sum(r['fell'] for r in res)}/{seeds}, speed {np.mean([r['speed'] for r in res]):.3f} m/s, "
              f"fell at {[(round(r['t'], 1), r['why']) for r in res if r['fell']]}", flush=True)


def check(policy):
    """Drive the env with robot.policy.Controller fed robot_readings() and compare with the env's observation."""
    env = make_env(policy)
    env.reset(seed=1)
    env.settle_at_home()
    obs = env.start_policy()
    ctrl = Controller(policy)
    ctrl.start(env.robot_readings(), phase=env.command.phase)
    worst = 0.0
    for seconds, v in ((1, 0.0), (4, 0.3)):
        env.set_command(v)
        ctrl.set_command(v)
        for _ in range(int(seconds / env.dt)):
            ctrl.tick(env.robot_readings())
            from biped_rl.tasks.locomotion.mdp import observations
            o = observations.compute(env)
            worst = max(worst, np.abs(ctrl.last_obs - o[:97]).max())
            _, _, term, _, _ = env.step(ctrl.last_action)
            if term:
                break
    print(f"check: max obs difference robot Controller vs env {worst:.2e}, walked {np.hypot(*env.data.qpos[:2]):.2f} m, "
          f"{'FELL' if term else 'up'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("mode", choices=["walk", "stand", "video", "check", "walkshove", "report"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--shoves", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--speeds", type=float, nargs="*", default=SPEEDS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    policy = Policy(a.npz)
    if a.mode == "walk":
        walk(policy, a.seeds, a.random, a.shoves, a.speeds)
    elif a.mode == "stand":
        stand(policy, a.seeds, a.random, (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0))
    elif a.mode == "walkshove":
        stand(policy, a.seeds, a.random, (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0), walking=a.speed)
    elif a.mode == "report":
        check(policy)
        walk(policy, 10, False, 0.0, SPEEDS)
        walk_shoved(policy, 10, (0.2, 0.4), strength=0.2)
        walk_shoved(policy, 10, (0.2, 0.4), strength=0.3)
        stand(policy, 3, False, (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0))
        stand(policy, 3, True, (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0))
        walk(policy, 5, True, 0.0, (0.1, 0.2, 0.3, 0.4))
    elif a.mode == "video":
        video(policy, a.speed, a.out or str(Path(a.npz).with_suffix(f".v{a.speed}.mp4")), random=a.random)
    else:
        check(policy)


if __name__ == "__main__":
    main()
