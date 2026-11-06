"""Free-run policies (training stages speed / speed_physics): how fast, and how. Each episode starts the way the
robot would: standing still in the policy's neutral pose (legs straight), the policy taking over with its "go"
command, 20 s. Four robots: the nominal one (what the speed stage trains on), the measured servos, randomized
robots with the measured servos (the randomization of --physics-from, as walk_test), and the same shoved (random
0.3 m/s shoves every few seconds, as walk_test's push test); and the measured servos told to turn left at TURN while
going (the "turn" column: deg/s actually turned; ~23 deg/s = all of it).

Per robot: net speed = distance along the start heading / 20 s (a fall counts as standing still for the rest
of the 20 s), the feet's sliding speed on the floor (mean of both feet: speed that won't carry over to a real
floor; the gate counts net speed minus it), falls (of them, hardware-limit stops) and when, steps per second, foot lift, landing force,
peak torque, the knees' bend (mean; + = forward like the home crouch, - = backward, bird-legged), the hip-roll
joints' swing (peak to peak) and the torso's side-to-side roll (+-, from its rms).

    python scripts/analysis/free_run_test.py runs/speed/juan_speed_1 [runs/speed/juan_speed_1_best ...]
    python scripts/analysis/free_run_test.py runs/speed/a --gate 0.15     # exit 1 unless nominal >= 0.15 m/s, falls <= 30%
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from biped_rl.tasks.locomotion.mdp import observations

PHYSICS_FROM = "runs/v7/juan_v7_13_straight"
RUN_S = 20.0
ROBOTS = ("nominal", "measured", "randomized", "shoved", "turning")
TURN = 0.4  # rad/s, left: the turning robot's command (free runs trained with CommandConfig.go_turn_range)


def episode(args):
    run, robot, seed, physics_from = args
    cfg = EnvConfig.load(run)
    if robot in ("randomized", "shoved"):
        cfg.randomization = EnvConfig.load(physics_from).randomization
        cfg.randomization.physics_curriculum = cfg.randomization.push_curriculum = False
        cfg.randomization.push_velocity = 0.3 if robot == "shoved" else 0.0
    else:
        cfg.randomization.enabled = False
    if robot != "nominal":
        cfg.use_measured_servos()
        if cfg.randomization.nominal_delay_range:  # the measured delay, not the run's own training range
            cfg.randomization.nominal_delay_range = tuple(cfg.randomization.action_delay_steps)
    model, norm = load_policy(run, cfg)
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=seed)
    env.settle_at_home()
    obs = env.start_policy({"speed": 1.0})   # a free-run policy turns any move into its go command
    if robot == "turning":
        env.set_command(1.0, 0.0, TURN)
        obs = observations.compute(env)
    turned, last_yaw = 0.0, env.yaw()   # summed step by step: 20 s at 23 deg/s is more than a full circle
    heading, p0 = env.command.heading_target, env.data.qpos[:2].copy()
    touchdowns, lifts, impact, torque, knees, ended, slide, hip_roll, roll = 0, [], 0.0, 0.0, [], None, [], [], []
    knee, prev = env.robot.knee_idx, env.episode.contact.copy()
    for _ in range(int(RUN_S / env.dt)):
        peak_lift = env.episode.swing_peak.copy()   # each foot's highest clearance since it last touched
        obs, _, term, _, info = env.step(model.predict(norm.normalize_obs(obs), deterministic=True)[0])
        td = (env.episode.contact == 1) & (prev == 0)
        prev = env.episode.contact.copy()
        touchdowns += int(td.sum())
        lifts += list(peak_lift[td])
        impact, torque = max(impact, info["peak_force_bw"]), max(torque, info["peak_torque_stall"])
        knees.append(np.degrees(env.data.qpos[env.robot.qpos_idx][knee]).mean())
        turned += (env.yaw() - last_yaw + np.pi) % (2 * np.pi) - np.pi
        last_yaw = env.yaw()
        slide.append(info["slip_speed"])
        hip_roll.append(np.degrees(env.data.qpos[env.robot.qpos_idx][env.robot.hip_roll_idx]))
        g = env.gravity_body()
        roll.append(np.degrees(np.arctan2(g[1], -g[2])))
        if term:
            ended = ("broke" if info["broke"] else "fell", env.episode.t)
            break
    dist = float(np.dot(env.data.qpos[:2] - p0, (np.cos(heading), np.sin(heading))))
    if robot == "turning":  # the path curves: its length, not the progress along the start heading
        dist = float(np.linalg.norm(env.data.qpos[:2] - p0))
    return {"speed": dist / RUN_S, "turn_dps": float(np.degrees(turned) / env.episode.t) if robot == "turning" else 0.0,
            "slide": float(np.mean(slide)), "ended": ended, "steps_per_s": touchdowns / env.episode.t,
            "lift_mm": 1000 * float(np.mean(lifts)) if lifts else 0.0, "impact_bw": impact, "torque_stall": torque,
            "knee_deg": float(np.mean(knees)),
            "hip_roll_swing_deg": float(np.ptp(np.array(hip_roll[50:]), axis=0).mean()) if len(hip_roll) > 60 else 0.0,
            "roll_deg": float(np.sqrt(2) * np.std(roll[50:])) if len(roll) > 60 else 0.0}


def summarize(results):
    falls = [r["ended"] for r in results if r["ended"]]
    return {"speed": float(np.mean([r["speed"] for r in results])), "speed_best": float(max(r["speed"] for r in results)),
            "episodes": len(results), "falls": len(falls), "broke": sum(e[0] == "broke" for e in falls),
            "fall_s": float(np.median([e[1] for e in falls])) if falls else None,
            **{k: float(np.mean([r[k] for r in results])) for k in ("slide", "turn_dps", "steps_per_s", "lift_mm", "knee_deg",
                                                              "hip_roll_swing_deg", "roll_deg")},
            **{k: float(max(r[k] for r in results)) for k in ("impact_bw", "torque_stall")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--episodes", type=int, default=6, help="per robot (randomized: one robot each)")
    p.add_argument("--physics-from", default=PHYSICS_FROM)
    p.add_argument("--workers", type=int, default=4, help="few while a training runs")
    p.add_argument("--gate", type=float, help="exit 1 unless the first run reaches this speed less sliding [m/s] "
                   "on the nominal robot with at most 30%% falls (between chained stages)")
    p.add_argument("--pick", choices=ROBOTS, help="print last the run to continue from: the fastest net of sliding on "
                   "this robot among those with <= 30%% falls (else the one with the fewest falls)")
    p.add_argument("--json")
    a = p.parse_args()
    jobs = [(run, robot, seed, a.physics_from) for run in a.runs for robot in ROBOTS for seed in range(a.episodes)]
    with Pool(a.workers) as pool:
        out = pool.map(episode, jobs)
    table = {}
    for (run, robot, _, _), r in zip(jobs, out):
        table.setdefault(run, {}).setdefault(robot, []).append(r)
    report = {run: {robot: summarize(rs) for robot, rs in robots.items()} for run, robots in table.items()}
    print(f"{'policy':34s} {'robot':10s} {'m/s':>6s} {'best':>5s} {'slide':>5s}  falls (hw)  at   steps/s  lift   impact  torque"
          "  knee  hiproll  roll  turn")
    for run, robots in report.items():
        for robot, s in robots.items():
            at = f"{s['fall_s']:4.1f}s" if s["fall_s"] is not None else "   - "
            print(f"{Path(run).name:34s} {robot:10s} {s['speed']:6.3f} {s['speed_best']:5.2f} {s['slide']:5.3f}"
                  f"  {s['falls']:2d}/{s['episodes']} ({s['broke']})"
                  f"  {at}  {s['steps_per_s']:5.1f}  {s['lift_mm']:4.0f}mm  {s['impact_bw']:4.1f}x  {s['torque_stall']:4.2f}x"
                  f"  {s['knee_deg']:+5.0f}°  {s['hip_roll_swing_deg']:5.0f}°  ±{s['roll_deg']:3.0f}°  {s['turn_dps']:+4.0f}°/s")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
    if a.pick:
        score = {run: robots[a.pick] for run, robots in report.items()}
        ok = [r for r, v in score.items() if v["falls"] <= 0.3 * v["episodes"]]
        best = (max(ok, key=lambda r: score[r]["speed"] - score[r]["slide"]) if ok
                else min(score, key=lambda r: (score[r]["falls"], -score[r]["speed"])))
        print(f"pick ({a.pick}): {best}")
    if a.gate is not None:
        s = report[a.runs[0]]["nominal"]
        ok = s["speed"] - s["slide"] >= a.gate and s["falls"] <= 0.3 * s["episodes"]
        print(f"gate {a.gate} m/s net of sliding, falls <= 30% on the nominal robot: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
