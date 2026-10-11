"""Sanity-check a robot model before spending hours training on it.

    python scripts/inspect_model.py models/my_robot.xml            # report + 3 s stand test
    python scripts/inspect_model.py models/my_robot.xml --view     # also open the viewer

Check the printed masses/inertias against your CAD's mass properties:
a wrong unit (mm vs m, g vs kg) is the most common CAD-export bug.
"""
import argparse

import mujoco
import mujoco.viewer
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--view", action="store_true")
    args = p.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or f"#{i}"

    print(f"\n{'body':<24}{'mass [kg]':>10}   principal inertia [kg m^2]")
    for b in range(1, m.nbody):
        flag = "   <-- suspicious" if m.body_mass[b] < 1e-3 or m.body_inertia[b].min() < 1e-7 else ""
        print(f"{name(mujoco.mjtObj.mjOBJ_BODY, b):<24}{m.body_mass[b]:>10.3f}   "
              f"{np.array2string(m.body_inertia[b], precision=5)}{flag}")
    print(f"{'TOTAL':<24}{m.body_mass.sum():>10.3f}")

    print(f"\n{'actuator':<24}{'joint range [rad]':>22}{'torque limit [Nm]':>20}{'kp':>8}")
    for a in range(m.nu):
        j = m.actuator_trnid[a, 0]
        frc = m.jnt_actfrcrange[j] if m.jnt_actfrclimited[j] else m.actuator_forcerange[a]
        print(f"{name(mujoco.mjtObj.mjOBJ_ACTUATOR, a):<24}"
              f"{np.array2string(m.jnt_range[j], precision=2):>22}"
              f"{np.array2string(frc, precision=1):>20}{m.actuator_gainprm[a, 0]:>8.0f}")

    if m.njnt == 0 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        print("\nWARNING: first joint is not a <freejoint/>; the robot is bolted to the world.")
    n_mesh_collide = sum(1 for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                         and m.geom_contype[g])
    if n_mesh_collide > 4:
        print(f"\nNOTE: {n_mesh_collide} colliding mesh geoms. Consider --feet in cad_to_mjcf.py "
              "or primitive (box/capsule) collision shapes for speed and stable contacts.")

    # Stand test: hold the home pose with the servos for 3 s.
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key) if key >= 0 else mujoco.mj_resetData(m, d)
    for a in range(m.nu):
        d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
    z0 = d.qpos[2]
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"\nStand test: base height {z0:.3f} -> {d.qpos[2]:.3f} m after 3 s, "
          f"{d.ncon} contacts, warnings: {[w.number for w in d.warning if w.number]}")

    if args.view:
        mujoco.viewer.launch(m)


if __name__ == "__main__":
    main()
