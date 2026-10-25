"""Check that the robot code's joint table (robot/joints.py) matches the simulator: joint names,
order, ranges and the home pose. Run after changing the model or the table.

    python scripts/check_robot_joints.py
"""

import mujoco
import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from robot.joints import JOINTS


def mismatches():
    """Differences between robot/joints.py and the model, as readable lines (none: they match)."""
    env = BipedEnv(EnvConfig().to_dict())
    m, robot = env.model, env.robot
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.jnt_qposadr.tolist().index(a)) for a in robot.qpos_idx]
    out = []
    if [j[0] for j in JOINTS] != names:
        out.append(f"joint order differs:\n  sim   {names}\n  robot {[j[0] for j in JOINTS]}")
    for (name, _, lo, hi, home, _), s_lo, s_hi, s_home in zip(JOINTS, robot.joint_lo, robot.joint_hi, robot.home_joint):
        sim = np.degrees([s_lo, s_hi, s_home])
        if not np.allclose([lo, hi, home], sim, atol=0.05):
            out.append(f"{name}: robot {lo}/{hi}/home {home}, sim {sim[0]:.2f}/{sim[1]:.2f}/home {sim[2]:.2f}")
    return out


def main():
    problems = mismatches()
    print("\n".join(problems) or "robot/joints.py matches the simulator")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
