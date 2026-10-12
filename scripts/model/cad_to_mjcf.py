"""Convert a CAD-exported URDF (+ STL/OBJ meshes) into an RL-ready MuJoCo MJCF.

    python scripts/model/cad_to_mjcf.py path/to/robot.urdf --meshdir path/to/meshes \
        --out models/my_robot.xml --feet left_foot right_foot

What it does on top of MuJoCo's own URDF importer:
  * adds a <freejoint/> to the root link (floating-base biped)
  * places the robot so its lowest point sits just above the floor
  * adds floor, light, physics options and joint damping/armature
  * adds a <position> (PD servo) actuator for every revolute/prismatic joint;
    URDF <limit effort> becomes the torque limit
  * optionally (--feet) keeps collision only on the feet: CAD meshes are
    collided as convex hulls, so legs/torso hulls cause bogus self-contacts
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def urdf_to_raw_mjcf(urdf: Path, meshdir: Path, out: Path):
    tree = ET.parse(urdf)
    robot = tree.getroot()
    for old in robot.findall("mujoco"):
        robot.remove(old)
    ext = ET.SubElement(robot, "mujoco")
    ET.SubElement(ext, "compiler", meshdir=str(meshdir.resolve()), discardvisual="false",
                  fusestatic="false", strippath="true", balanceinertia="true")
    tmp = urdf.with_name(urdf.stem + "._mjtmp.urdf")  # same folder so relative paths still work
    tree.write(tmp)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    mujoco.mj_saveLastXML(str(out), model)


def lowest_point(model, data):
    mujoco.mj_forward(model, data)
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            start, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[start:start + n]
            world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
            zmin = min(zmin, world[:, 2].min())
        elif model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
            zmin = min(zmin, data.geom_xpos[g][2] - model.geom_rbound[g])
    return zmin


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urdf", type=Path)
    p.add_argument("--meshdir", type=Path, help="folder with the mesh files (default: <urdf dir>/meshes)")
    p.add_argument("--out", type=Path, default=Path("models/my_robot.xml"))
    p.add_argument("--kp", type=float, default=100.0, help="servo stiffness [Nm/rad]; tune to your motors")
    p.add_argument("--damping", type=float, default=1.0, help="joint damping [Nm s/rad]")
    p.add_argument("--armature", type=float, default=0.01,
                   help="reflected rotor inertia [kg m^2] = rotor inertia * gear_ratio^2")
    p.add_argument("--feet", nargs="*", default=None,
                   help="link names of the feet; if given, only these links keep collision geoms")
    args = p.parse_args()

    meshdir = args.meshdir or args.urdf.parent / "meshes"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    urdf_to_raw_mjcf(args.urdf, meshdir, args.out)

    tree = ET.parse(args.out)
    root = tree.getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", os.path.relpath(meshdir.resolve(), args.out.parent.resolve()))
    compiler.set("autolimits", "true")
    root.insert(1, ET.Element("option", timestep="0.002", integrator="implicitfast"))

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(2, asset)
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512",
                  height="512", rgb1="0.2 0.3 0.4", rgb2="0.3 0.4 0.5")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.1")

    world = root.find("worldbody")
    base = world.find("body")
    if base is None:
        raise SystemExit("No body found in worldbody. Is the URDF root link empty?")
    world.insert(0, ET.Element("light", pos="0 0 4", dir="0 0 -1", directional="true"))
    world.insert(1, ET.Element("geom", name="floor", type="plane", size="0 0 0.05", material="grid"))
    base.insert(0, ET.Element("freejoint", name="root"))
    ET.SubElement(base, "site", name="imu", pos="0 0 0")

    actuator = ET.SubElement(root, "actuator")
    for body in base.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("type", "hinge") not in ("hinge", "slide"):
                continue
            joint.set("damping", str(args.damping))
            joint.set("armature", str(args.armature))
            if joint.get("range") is None:
                print(f"WARNING: joint {joint.get('name')} has no limits; servo ctrlrange will be unbounded")
            ET.SubElement(actuator, "position", name=joint.get("name"), joint=joint.get("name"),
                          kp=str(args.kp), inheritrange="1")

    if args.feet is not None:
        feet = set(args.feet)
        for body in base.iter("body"):
            if body.get("name") in feet:
                continue
            for geom in body.findall("geom"):
                if geom.get("contype") is None:  # collision geom -> visual only
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                    geom.set("group", "1")

    ET.indent(tree)
    tree.write(args.out)

    # Lift the robot so it starts just above the floor.
    model = mujoco.MjModel.from_xml_path(str(args.out))
    data = mujoco.MjData(model)
    lift = 0.005 - lowest_point(model, data)
    pos = np.array([float(v) for v in (base.get("pos") or "0 0 0").split()])
    pos[2] += lift
    base.set("pos", " ".join(f"{v:.4g}" for v in pos))
    tree.write(args.out)

    model = mujoco.MjModel.from_xml_path(str(args.out))
    print(f"Wrote {args.out}: {model.nbody - 1} bodies, {model.nu} actuators, "
          f"total mass {model.body_mass.sum():.2f} kg, base height {pos[2]:.3f} m")
    print("Next: python scripts/model/inspect_model.py", args.out)


if __name__ == "__main__":
    main()
