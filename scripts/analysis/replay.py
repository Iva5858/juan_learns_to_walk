"""Real vs simulated: replay a robot log (robot/logs/*.npz) through the simulator, window by window.

Each window (1 s by default) starts the simulated Juan in the real robot's pose (joint angles and
speeds, torso tilt from the IMU) and feeds it the servo targets the real robot was sent, step by
step, through the simulator's servo model. Then the simulated joints are compared with the real ones.
(A whole run can't be replayed in one go: the policy was reacting to the real robot, so an open-loop
copy drifts away within a second or two.)

Run over a grid of servo models and delays, it also tells which ones match the real robot best:
system identification from walking data instead of a bench test.

    python scripts/analysis/replay.py robot/logs/2026-10-24_125742.348_juan_v7_11_upright.npz
    python scripts/analysis/replay.py <log> --plot      # + out/replay/<log>.png: real vs best vs training model
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig  # noqa: E402

# Servo models to compare: (label, no-load speed rad/s, kp Nm/rad, kd)
SERVOS = [("original 5.8/15", 5.8, 15.0, 0.2), ("bench fit 3.0/6", 3.0, 6.0, 0.2), ("stiff 5.8/25", 5.8, 25.0, 0.3),
          ("stiffer 5.8/40", 5.8, 40.0, 0.4), ("fast stiff 8/25", 8.0, 25.0, 0.3)]
DELAYS = [1, 2, 3, 4, 5, 6]   # whole control steps between the log's target and the servo acting on it
LATENCY_MS = 10         # within the step


def tilt_quat(gravity):
    """Torso orientation (yaw 0) whose down vector in the body frame is `gravity`."""
    g = np.asarray(gravity, float) / np.linalg.norm(gravity)
    roll = math.atan2(-g[1], -g[2])
    pitch = math.atan2(g[0], math.hypot(g[1], g[2]))
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, 0.0]), "xyz")
    return q


def replay_window(env, log, k0, n, delay, servo):
    speed, kp, kd = servo[1:]
    d, robot = env.data, env.robot
    env.reset(seed=0)
    sv = env.servo
    sv.speed[:], sv.kp[:], sv.kd[:] = speed, kp, kd
    d.qpos[3:7] = tilt_quat(log["gravity"][k0])
    d.qpos[robot.qpos_idx] = log["q"][k0]
    d.qvel[:] = 0.0
    d.qvel[robot.dof_idx] = log["qd"][k0]
    mujoco.mj_forward(env.model, d)
    env.ground()
    lat = min(int(LATENCY_MS / 1000 / env.model.opt.timestep), env.frame_skip - 1)
    target = log["target"][max(k0 - delay - 1, 0)]
    sim_q, sim_g = [], []
    for k in range(k0, k0 + n):
        new = log["target"][max(k - delay, 0)]
        for s in range(env.frame_skip):
            if s == lat:
                target = new
            d.ctrl[:] = env.servo_torque(target, None, 1.0 / sv.speed)[2]
            mujoco.mj_step(env.model, d)
        sim_q.append(d.qpos[robot.qpos_idx].copy())
        sim_g.append(env.gravity_body())
    return np.array(sim_q), np.array(sim_g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--window", type=float, default=1.0, help="seconds per replayed window")
    p.add_argument("--config", default="runs/juan_v7_12b_pushes", help="run whose robot model to use")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    log = dict(np.load(a.log))
    names = [str(n) for n in log["joints"]]
    cfg = EnvConfig.from_dict(__import__("json").loads((PROJECT / a.config / "config.json").read_text()))
    cfg.randomization.enabled = False
    env = BipedEnv(cfg.to_dict())
    n = int(a.window / env.dt)
    T = len(log["t"])
    starts = list(range(5, T - n - 1, n))
    print(f"{Path(a.log).name}: {T} steps ({T * env.dt:.0f} s), {len(starts)} windows of {a.window:.1f} s")
    print(f"{'servo model':18s} {'delay':>5s} | joint error vs real, rms deg (all | per joint: {' '.join(x.split('_', 1)[1][:5] for x in names[:4])} L/R)"
          f" | tilt error deg")
    results = []
    for servo, delay in itertools.product(SERVOS, DELAYS):
        errs, terr, traces = [], [], []
        for k0 in starts:
            sq, sg = replay_window(env, log, k0, n, delay, servo)
            real = log["q"][k0 + 1:k0 + 1 + n]
            errs.append(sq - real)
            rg, sg2 = log["gravity"][k0 + 1:k0 + 1 + n], sg
            terr.append(np.degrees(np.arccos(np.clip(np.sum(rg / np.linalg.norm(rg, axis=1, keepdims=True) * sg2, axis=1), -1, 1))))
            traces.append(sq)
        e = np.degrees(np.concatenate(errs))
        rms = np.sqrt(np.mean(e ** 2, axis=0))
        tilt = float(np.mean(np.concatenate(terr)))
        results.append((float(np.sqrt(np.mean(e ** 2))), servo[0], delay, rms, tilt, traces))
        print(f"{servo[0]:18s} {delay * 20 + LATENCY_MS:3d}ms | {results[-1][0]:5.2f} | "
              f"{' '.join(f'{v:4.1f}' for v in rms[:4])} / {' '.join(f'{v:4.1f}' for v in rms[4:])} | {tilt:5.1f}", flush=True)
    best = min(results, key=lambda r: r[0])
    print(f"\nbest match: {best[1]}, {best[2] * 20 + LATENCY_MS} ms ({best[0]:.2f} deg rms joint error)")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        train = next(r for r in results if r[1] == SERVOS[0][0] and r[2] == 1)
        fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        t = np.arange(T) * env.dt
        for j, ax in enumerate(axes.T.ravel()):
            ax.plot(t, np.degrees(log["q"][:, j]), color="black", lw=1.2, label="real")
            ax.plot(t, np.degrees(log["target"][:, j]), color="gray", lw=0.6, ls=":", label="target sent")
            for k0, tr_b, tr_t in zip(starts, best[5], train[5]):
                tt = t[k0 + 1:k0 + 1 + n]
                ax.plot(tt, np.degrees(tr_b[:, j]), color="tab:blue", lw=1, label="sim, best match" if k0 == starts[0] else None)
                ax.plot(tt, np.degrees(tr_t[:, j]), color="tab:orange", lw=1, alpha=0.7,
                        label="sim, training model" if k0 == starts[0] else None)
            ax.set_title(names[j], fontsize=9)
            ax.set_ylabel("deg")
        axes[0, 0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"{Path(a.log).name}: best {best[1]}, {best[2] * 20 + LATENCY_MS} ms (blue) vs "
                     f"training model {SERVOS[0][0]}, {train[2] * 20 + LATENCY_MS} ms (orange)", fontsize=10)
        fig.tight_layout()
        out = PROJECT / "out" / "replay" / (Path(a.log).stem + ".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=90)
        print(f"plot: {out.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
