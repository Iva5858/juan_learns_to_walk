"""Check that the robot code's joint table (robot/joints.py) matches the simulator: joint names,
order, ranges and the home pose. Run after changing the model or the table.

    python scripts/check_robot_joints.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the robot/ package isn't installed

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from robot.joints import JOINTS


def main():
    env = BipedEnv(EnvConfig().to_dict())
    m, robot = env.model, env.robot
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.jnt_qposadr.tolist().index(a)) for a in robot.qpos_idx]
    ok = [j[0] for j in JOINTS] == names
    if not ok:
        print(f"joint order differs:\n  sim   {names}\n  robot {[j[0] for j in JOINTS]}")
    for (name, _, lo, hi, home, _), s_lo, s_hi, s_home in zip(JOINTS, robot.joint_lo, robot.joint_hi, robot.home_joint):
        sim = np.degrees([s_lo, s_hi, s_home])
        if not np.allclose([lo, hi, home], sim, atol=0.05):
            ok = False
            print(f"{name}: robot {lo}/{hi}/home {home}, sim {sim[0]:.2f}/{sim[1]:.2f}/home {sim[2]:.2f}")
    print("robot/joints.py matches the simulator" if ok else "MISMATCH: fix robot/joints.py")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
