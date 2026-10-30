"""Can Juan stand still on one foot? A static check before training one-leg skills (stand on one foot,
wave a leg, kick): search for a pose with the standing foot flat on the floor, the other foot lifted,
the centre of mass over the standing sole, and every servo under 80% of its stall torque holding it.

Without ankle roll, the weight can only move sideways through hip roll and by leaning the torso, so
this is not obvious. Static only: it says whether a balanced pose exists, not whether the robot can
get into it or hold it against delays (that is training).

    python scripts/analysis/one_foot.py                  # the best poses found and their margins
"""
import argparse

import mujoco
import numpy as np

from biped_rl.robot_logs import kinematic_env
from robot.joints import NAMES

LIFT_M = 0.02          # the other sole at least this high
MAX_TORQUE = 0.8       # x stall


def stance_frame_pose(env, q, stance):
    """Pose the robot with joint angles q and the stance foot flat (as at home), sole on the floor.
    Returns (com xy relative to the stance sole hull, hull points, other sole height, torques)."""
    m, d, robot = env.model, env.data, env.robot
    foot = robot.foot_bodies[stance]
    # orientation that makes the stance foot's world rotation equal to its home rotation
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    d.qpos[robot.qpos_idx] = robot.home_joint
    mujoco.mj_kinematics(m, d)
    R_foot_home = d.xmat[foot].reshape(3, 3).copy()
    d.qpos[robot.qpos_idx] = q
    mujoco.mj_kinematics(m, d)
    R_foot = d.xmat[foot].reshape(3, 3)          # (root at identity)
    R_root = R_foot_home @ R_foot.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_root.ravel())
    d.qpos[3:7] = quat
    d.qpos[0:3] = [0, 0, 0.3]
    mujoco.mj_forward(m, d)
    soles = env.soles()[0]
    d.qpos[2] -= soles[stance, 2]
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)
    soles = env.soles()[0]
    # the stance sole's footprint: the lowest collision points of that foot
    pts = []
    for g in robot.foot_geoms[stance]:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        else:
            c = env._box_corners * m.geom_size[g]
            w = d.geom_xpos[g] + c @ d.geom_xmat[g].reshape(3, 3).T
        pts.append(w[w[:, 2] < w[:, 2].min() + 0.002])
    pts = np.concatenate(pts)[:, :2]
    com = d.subtree_com[robot.root][:2]
    other = soles[1 - stance, 2] - soles[stance, 2]
    # holding torques: inverse dynamics at rest, with the stance foot carrying the weight
    mujoco.mj_inverse(m, d)
    tau = d.qfrc_inverse[robot.dof_idx]
    return com, pts, other, tau


def hull_margin(p, pts):
    """Signed distance from p to the edge of the convex hull of pts (+ inside)."""
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1])  # counter-clockwise
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / np.linalg.norm(edges, axis=1, keepdims=True)
    return float(np.min(np.sum((hull - p) * normals, axis=1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=20000)
    a = p.parse_args()
    env = kinematic_env()
    robot = env.robot
    stall = env.cfg.robot.stall_torque
    rng = np.random.default_rng(0)
    lo, hi = robot.joint_lo, robot.joint_hi
    best = []
    for stance in (0, 1):
        for _ in range(a.samples):
            q = robot.home_joint + rng.normal(0, 0.35, env.nu)
            q = np.clip(q, lo, hi)
            com, pts, other, tau = stance_frame_pose(env, q, stance)
            if other < LIFT_M:
                continue
            margin = hull_margin(com, pts)
            torque = float(np.max(np.abs(tau)) / stall)
            best.append((margin, torque, other, stance, q))
    ok = [b for b in best if b[0] > 0 and b[1] < MAX_TORQUE]
    print(f"{len(best)} poses with a foot lifted >= {100 * LIFT_M:.0f} cm; {len(ok)} balanced with every servo "
          f"under {100 * MAX_TORQUE:.0f}% of stall")
    for margin, torque, other, stance, q in sorted(ok or best, key=lambda b: -b[0])[:5]:
        print(f"  stand on {'left' if stance == 0 else 'right'}: centre of mass {1000 * margin:+5.1f} mm inside the sole edge, "
              f"other foot {100 * other:.1f} cm up, peak torque {100 * torque:.0f}% of stall")
        print("     " + "  ".join(f"{n.replace('left_', 'L ').replace('right_', 'R ')} {np.degrees(v):+5.1f}" for n, v in zip(NAMES, q)))


if __name__ == "__main__":
    main()
