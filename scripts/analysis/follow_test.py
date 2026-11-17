"""Following a person in the simulator: the robot's own follow controller (robot/follow.py FollowController)
steering a walking policy toward a virtual person, the way the dashboard's "Follow a person" does on the robot.

The person starts 1.5 m ahead and walks a random path for FOLLOW_S: legs of 2-5 s at 0 (a pause), 0.1, 0.15 or
0.2 m/s, turning up to 120 deg between legs (sometimes past the robot's side, out of view). Every VISION_S the
controller gets what the camera would give: the direction and distance of the person, measured from the torso,
with the camera's limits: seen only within the field of view (+-31 deg), MISS of the frames missed, noise on
the direction and distance, one frame late, and no distance closer than follow.NEAR_M (feet out of the picture).
On randomized robots with the measured servos (walk_test's robots), standing still at home, the policy taking
over, 1 s standing, then following.

Per policy: falls, the share of the time the person is in view, the mean distance to them while in view, the
longest time out of view, and how far the robot walked.

--joystick: rapid joystick changes instead: a new command every 0.5-2 s for FOLLOW_S (stand still, turn in place,
walk with a turn, walk straight; forward up to 0.3 m/s, turn up to 0.6 rad/s). Falls, and how far the forward speed
and turn rate are from the command (mean absolute error, from 0.5 s after each change, 0.5 s averages).

    python scripts/analysis/follow_test.py runs/v7/juan_v7_31_takeover36_best runs/v7/juan_v7_34_final
    python scripts/analysis/follow_test.py runs/v7/a --seeds 24 --json out/follow.json
    python scripts/analysis/follow_test.py runs/v7/a runs/v7/b --joystick
"""
import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from biped_rl.runs import load_policy, physics_run
from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from robot import follow
from robot.follow import FollowController

FOLLOW_S, STAND_S = 40.0, 1.0
VISION_S = 0.25                      # the Pi's person detector: ~4 frames a second
MISS = 0.1                           # share of frames where the detector misses the person
BEARING_NOISE = math.radians(2.0)
DISTANCE_NOISE = 0.1                 # relative
HALF_FOV = math.radians(follow.HFOV_DEG / 2)
PERSON_SPEEDS = (0.0, 0.1, 0.15, 0.2)


def joystick_command(rng):
    """One rapid joystick command: (forward m/s, turn rad/s)."""
    kind = rng.choice(4, p=(0.2, 0.2, 0.45, 0.15))
    if kind == 0:
        return 0.0, 0.0                                                   # stand still
    if kind == 1:
        return 0.0, float(rng.choice((-1, 1)) * rng.uniform(0.2, 0.6))     # turn in place
    if kind == 2:
        return float(rng.uniform(0.05, 0.3)), float(rng.uniform(-0.5, 0.5))   # walk with a turn
    return float(rng.uniform(0.05, 0.3)), 0.0                             # walk straight


def joystick_episode(env, act, obs, rng):
    n, cmd, left, held = int(FOLLOW_S / env.dt), (0.0, 0.0), 0.0, 0.0
    errs_v, errs_w, buf, term = [], [], [], False
    last_xy, last_yaw = env.data.qpos[:2].copy(), env.yaw()
    for _ in range(n):
        left -= env.dt
        held += env.dt
        if left <= 0:
            cmd, left, held, buf = joystick_command(rng), rng.uniform(0.5, 2.0), 0.0, []
            env.set_command(cmd[0], 0.0, cmd[1])
        obs, _, term, _, _ = env.step(act(obs))
        if term:
            break
        d, yaw = env.data.qpos[:2] - last_xy, env.yaw()
        vx = (math.cos(yaw) * d[0] + math.sin(yaw) * d[1]) / env.dt
        wz = ((yaw - last_yaw + math.pi) % (2 * math.pi) - math.pi) / env.dt
        last_xy, last_yaw = env.data.qpos[:2].copy(), yaw
        if held >= 0.5:
            buf.append((vx, wz))
            if len(buf) * env.dt >= 0.5:
                v, w = np.mean(buf, axis=0)
                errs_v.append(abs(v - cmd[0]))
                errs_w.append(abs(w - cmd[1]))
                buf = []
    return {"fell": bool(term), "fwd_err": float(np.mean(errs_v)) if errs_v else None,
            "turn_err": float(np.mean(errs_w)) if errs_w else None}


def episode(args):
    run, seed, physics_from, joystick = args
    cfg = EnvConfig.load(run)
    cfg.randomization = EnvConfig.load(physics_from).randomization
    cfg.use_measured_servos()
    r = cfg.randomization
    r.physics_curriculum, r.push_curriculum, r.push_velocity = False, False, 0.0
    model, norm = load_policy(run, cfg)
    env = BipedEnv(cfg.to_dict())
    env.reset(seed=seed)
    env.settle_at_home()
    obs = env.start_policy()
    act = lambda o: model.predict(norm.normalize_obs(o), deterministic=True)[0]
    for _ in range(int(STAND_S / env.dt)):
        obs, _, term, _, _ = env.step(act(obs))
        if term:
            return None
    rng = np.random.default_rng(seed)
    if joystick:
        out = joystick_episode(env, act, obs, rng)
        env.close()
        return out
    yaw0, xy0 = env.yaw(), env.data.qpos[:2].copy()
    person = xy0 + 1.5 * np.array([math.cos(yaw0), math.sin(yaw0)])
    heading, speed, leg_left = yaw0, 0.0, 0.0
    ctrl, pending = FollowController(), None
    in_view, dists, out_s, out_max, walked, last = 0, [], 0.0, 0.0, 0.0, xy0.copy()
    every = max(1, round(VISION_S / env.dt))
    n = int(FOLLOW_S / env.dt)
    term = False
    for i in range(n):
        leg_left -= env.dt
        if leg_left <= 0:
            leg_left = rng.uniform(2.0, 5.0)
            speed = float(rng.choice(PERSON_SPEEDS))
            heading += rng.uniform(-1, 1) * math.radians(120)
        person += speed * env.dt * np.array([math.cos(heading), math.sin(heading)])
        if i % every == 0:
            d = person - env.data.qpos[:2]
            bearing = (math.atan2(d[1], d[0]) - env.yaw() + math.pi) % (2 * math.pi) - math.pi
            dist = float(np.hypot(*d))
            seen = abs(bearing) <= HALF_FOV
            in_view += seen
            out_s = 0.0 if seen else out_s + VISION_S
            out_max = max(out_max, out_s)
            if seen:
                dists.append(dist)
            sighting = None
            if seen and rng.random() > MISS:
                b = bearing + rng.normal(0, BEARING_NOISE)
                sighting = (b, None if dist < follow.NEAR_M else dist * (1 + rng.normal(0, DISTANCE_NOISE)))
            sighting, pending = pending, sighting       # one frame late
            vx, wz = ctrl.step(sighting, VISION_S)
            env.set_command(vx, 0.0, wz)
        obs, _, term, _, _ = env.step(act(obs))
        walked += float(np.hypot(*(env.data.qpos[:2] - last)))
        last = env.data.qpos[:2].copy()
        if term:
            break
    env.close()
    frames = max(1, (i // every) + 1)
    return {"fell": bool(term), "in_view": in_view / frames, "distance_m": float(np.mean(dists)) if dists else None,
            "out_max_s": out_max, "walked_m": walked}


def measure(run, seeds, pool, physics_from, joystick=False):
    res = pool.map(episode, [(run, 9000 + s, physics_from, joystick) for s in range(seeds)])
    eps = [e for e in res if e is not None]
    if joystick:
        mean = lambda k: float(np.mean([e[k] for e in eps if e[k] is not None]))
        return {"episodes": len(eps), "standing_falls": len(res) - len(eps), "falls": sum(e["fell"] for e in eps),
                "fwd_err": mean("fwd_err"), "turn_err": mean("turn_err")}
    dist = [e["distance_m"] for e in eps if e["distance_m"] is not None]
    return {"episodes": len(eps), "standing_falls": len(res) - len(eps), "falls": sum(e["fell"] for e in eps),
            "in_view": float(np.mean([e["in_view"] for e in eps])), "distance_m": float(np.mean(dist)) if dist else None,
            "out_max_s": float(np.max([e["out_max_s"] for e in eps])), "walked_m": float(np.mean([e["walked_m"] for e in eps]))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=12, help="randomized robots (default 12)")
    p.add_argument("--physics-from", default=physics_run(), help="the run whose randomization makes the test robots")
    p.add_argument("--processes", type=int, default=6)
    p.add_argument("--joystick", action="store_true", help="rapid joystick changes instead of a person")
    p.add_argument("--json", help="save the numbers to this file")
    a = p.parse_args()
    out = {}
    with Pool(a.processes) as pool:
        for run in a.runs:
            m = out[run] = measure(run, a.seeds, pool, a.physics_from, a.joystick)
            fell = f"  (fell standing: {m['standing_falls']})" if m["standing_falls"] else ""
            if a.joystick:
                print(f"{Path(run).name:32s} falls {m['falls']}/{m['episodes']}  forward error {m['fwd_err']:.3f} m/s  "
                      f"turn error {m['turn_err']:.3f} rad/s" + fell, flush=True)
                continue
            d = f"{m['distance_m']:.2f} m" if m["distance_m"] is not None else "-"
            print(f"{Path(run).name:32s} falls {m['falls']}/{m['episodes']}  in view {100 * m['in_view']:3.0f}%  "
                  f"distance {d}  longest out of view {m['out_max_s']:4.1f} s  walked {m['walked_m']:4.1f} m" + fell, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
