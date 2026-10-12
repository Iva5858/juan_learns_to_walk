"""Turn the onshape-to-robot export into the RL-ready model.

    python scripts/model/postprocess_onshape.py juan            # writes juan/rl_robot.xml + juan/rl_scene.xml

Re-run after every `onshape-to-robot juan`; the original robot.xml / scene.xml are never modified.

What it does:
  * masses from the pitch-deck BoM (never Onshape's material properties): every part gets its
    BoM mass, MuJoCo computes each part's inertia from its mesh, links are summed from parts
  * joint conventions = the training conventions (torso frame x forward, y left, z up):
    roll joints rotate about +x, pitch joints (hip pitch, knee, ankle) about +y. So hip pitch
    negative = leg forward, knee positive = bends back, ankle negative = toes up
  * hip roll range: 5 deg inward / 25 deg outward on both legs (direction measured on the model)
  * removes any coupling between joints (Onshape gear relations) and gives all 8 joints a servo
  * foot soles = flat boxes + rubber pads (stable contact instead of mesh hulls), capsules on
    thighs and shins (legs cannot pass through each other; knee on the floor = a fall)
  * IMU site at the BNO055, home keyframe (crouched, feet flat, just touching the floor)
"""
import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

# --------------------------------------------------------------------------- BoM (grams)
# (regex on the part name, grams, "each" = per matching part | "total" = split over all matches)
MASS_RULES = [
    # printed parts (PLA)
    (r"^1[ab]_hip_", 113 / 4, "each"),
    (r"^2[ab]_thigh_", 135 / 4, "each"),
    (r"^3[ab]_shin_", 131 / 4, "each"),
    (r"^4_ankle_foot", 64 / 2, "each"),
    (r"^5_pelvis", 191, "each"),
    (r"^6_battery_door", 14, "each"),
    (r"^7_head_tray", 44, "each"),
    (r"^8_head_shelf", 36, "each"),
    (r"^9_head_cover", 94, "each"),
    (r"^debug_board_holder", 7, "each"),
    # electronics: HTD-35H "servo with horns" 64 g = body 58 g + 2 horns x 3 g (split is an estimate)
    (r"^htd_35h_servo_body", 58, "each"),
    (r"^htd_35h_horn", 3, "each"),
    (r"^battery_3s_2200", 176, "each"),
    (r"^battery_3s_650", 59, "each"),
    (r"^raspberry_pi_4", 47, "each"),
    (r"^pi_camera_v2", 4, "each"),
    (r"^lm2596_regulator", 11, "total"),
    (r"^adafruit_bno055_imu", 3, "total"),
    # debug board (BusLinker) and its components
    (r"^(circuit_board_40_x|chip|capacitor_100_uf|screw_terminal|serial_port|usb_socket|vin_jumper"
     r"|servo_socket_[ab])", 8, "total"),
    # servo leads (BoM: 8 leads, 23 g), split over every lead segment in the CAD
    (r"^(servo_bus_lead|lead__|lead_in_from|trunk_lead)", 23, "total"),
    # power and data cables + plugs (BoM: 42 g)
    (r"^(camera_ribbon|imu_wires|motor_battery_leads|pi_battery_leads|usb_c_power_cable|usb_cable__"
     r"|usb_a_plug|usb_c_plug|usb_plug_at|xt30_pair|xt60_pair|servo_plug_[ab])", 42, "total"),
    # fasteners (BoM totals / counts)
    (r"^m2_heat_set_insert", 7 / 55, "each"),
    (r"^m2_x_8_socket_head_screw", 14 / 52, "each"),
    (r"^m2_x_12_socket_head_screw", 1 / 3, "each"),
    (r"^m2_x_5_horn_screw", 10 / 64, "each"),
]
# Thin / hollow parts: mass on the surface ("shell"); dense parts: solid convex hull ("convex")
SHELL_PARTS = r"^(\d[ab]?_|debug_board_holder|circuit_board|lead|servo_bus_lead|trunk_lead|camera_ribbon|imu_wires|usb_c_power|usb_cable|motor_battery_leads|pi_battery_leads|adafruit_bno055|lm2596)"

ROLL = ("left_hip_roll", "right_hip_roll")
PITCH = ("left_hip_pitch", "left_knee", "left_ankle", "right_hip_pitch", "right_knee", "right_ankle")
ACTUATOR_ORDER = ("left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                  "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
FEET = ("left_foot_link", "right_foot_link")
LEG_CAPSULES = {"left_thigh_link": "left_shin_link", "left_shin_link": "left_foot_link",
                "right_thigh_link": "right_shin_link", "right_shin_link": "right_foot_link"}
ROLL_INWARD_DEG, ROLL_OUTWARD_DEG = 5.0, 25.0
HOME = {"hip_roll": 0.0, "hip_pitch": -0.4, "knee": 0.8, "ankle": -0.4}  # rad, training conventions


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def parse(path):
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def part_geoms(root):
    """(part name, geom element, parent body) for every geom, from the '<!-- Part X -->' comments."""
    out = []
    for body in root.iter("body"):
        name = None
        for child in list(body):
            if child.tag is ET.Comment:
                m = re.match(r"\s*Part (\S+)", child.text or "")
                name = m.group(1) if m else name
            elif child.tag == "geom":
                out.append((re.sub(r"_\d+$", "", name or "?"), child, body))
    return out


def convex_hull_2d(points):
    """Monotone-chain convex hull of 2D points, counter-clockwise."""
    pts = sorted(set(map(tuple, np.round(points, 5))))
    if len(pts) < 3:
        return np.array(pts)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def compile_xml(tree, path):
    tree.write(path)
    return mujoco.MjModel.from_xml_path(str(path))


def sole_bottom(model, data, foot):
    """World z of the lowest point of a foot's pad mesh."""
    g = model.geom(f"{foot}_sole").id
    mid = model.geom_dataid[g]
    v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
    return float((data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())


def write_training_model(tree, model, export_dir):
    """Same physics without the ~370 visual meshes (MuJoCo repositions every geom each step):
    each link's BoM-derived mass/inertia is written explicitly, only collision shapes are kept.
    Training uses rl_scene_train.xml; videos and the viewer use rl_scene.xml."""
    import copy
    t = copy.deepcopy(tree)
    root = t.getroot()
    root.find("compiler").set("inertiafromgeom", "false")
    for body in root.iter("body"):
        b = model.body(body.get("name"))
        for g in [g for g in body.findall("geom") if g.get("class") == "visual"]:
            body.remove(g)
        quat = model.body_iquat[b.id]
        inertial = ET.Element("inertial", pos=fmt(model.body_ipos[b.id]), quat=fmt(quat),
                              mass=f"{model.body_mass[b.id]:.6g}", diaginertia=fmt(model.body_inertia[b.id]))
        body.insert(1, inertial)
        for g in body.findall("geom"):  # mass now lives in <inertial>
            g.set("mass", "0")
            g.attrib.pop("density", None)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    for mesh in list(asset.iter("mesh")):
        name = mesh.get("name") or Path(mesh.get("file", "")).stem
        if name not in used:
            asset.remove(mesh)
    for mat in list(asset.iter("material")):
        asset.remove(mat)
    for g in root.iter("geom"):
        g.attrib.pop("material", None)
    t.write(export_dir / "rl_robot_train.xml")
    scene = (export_dir / "scene.xml").read_text().replace('file="robot.xml"', 'file="rl_robot_train.xml"')
    (export_dir / "rl_scene_train.xml").write_text(scene)
    m2 = mujoco.MjModel.from_xml_path(str(export_dir / "rl_scene_train.xml"))
    assert np.allclose(m2.body_mass, model.body_mass, atol=1e-6), "training model masses differ"
    print(f"training model: {m2.ngeom} geoms (full model {model.ngeom}), same masses and inertias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--pad-thickness", type=float, default=0.002, help="rubber sole pad [m]")
    ap.add_argument("--pad-mass", type=float, default=0.0, help="per foot [g]")
    ap.add_argument("--friction", type=float, default=0.8)
    args = ap.parse_args()
    src, out = args.export_dir / "robot.xml", args.export_dir / "rl_robot.xml"
    tree = parse(src)
    root = tree.getroot()
    report = []

    # ---- physics options, joint defaults (match the placeholder the training was tuned on)
    root.find("compiler").set("inertiafromgeom", "true")
    if root.find("option") is None:
        root.insert(1, ET.Element("option", timestep="0.002", integrator="implicitfast"))
    for jd in root.iter("default"):
        j = jd.find("joint")
        if j is not None:
            j.attrib.update(damping="0.02", armature="0.001", frictionloss="0.02")

    # ---- remove couplings (gear relations) and the exported actuators; rebuilt below
    for tag in ("equality", "actuator"):
        for el in root.findall(tag):
            removed = [e.attrib for e in el]
            root.remove(el)
            if tag == "equality" and removed:
                report.append(f"removed {len(removed)} joint coupling(s) (Onshape gear relation): {removed}")

    # ---- masses from the BoM
    geoms = part_geoms(root)
    for body in root.iter("body"):
        for inertial in body.findall("inertial"):
            body.remove(inertial)
    matches = defaultdict(list)
    unmatched = []
    for part, geom, body in geoms:
        if geom.get("class") == "collision":
            geom.set("density", "0")  # collision copies carry no mass (the visual copy does)
            continue
        rule = next((r for r in MASS_RULES if re.search(r[0], part)), None)
        if rule is None:
            unmatched.append(part)
            geom.set("density", "0")
        else:
            matches[rule[0]].append(geom)
    for pattern, grams, kind in MASS_RULES:
        gs = matches.get(pattern, [])
        for g in gs:
            g.set("mass", f"{(grams / len(gs) if kind == 'total' else grams) / 1000:.6g}")
        if not gs:
            report.append(f"WARNING: BoM line {pattern!r} matched no part")
    if unmatched:
        report.append(f"WARNING: no BoM mass for {sorted(set(unmatched))} (given zero mass)")
    for mesh in root.find("asset").iter("mesh"):
        name = Path(mesh.get("file")).stem
        mesh.set("inertia", "shell" if re.search(SHELL_PARTS, name) else "convex")

    # ---- joint conventions: flip axes so roll = +x and pitch = +y in the torso frame
    tmp = args.export_dir / "_rl_tmp.xml"
    model = compile_xml(tree, tmp)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = {j.get("name"): j for j in root.iter("joint")}
    for name, want in [(n, 0) for n in ROLL] + [(n, 1) for n in PITCH]:
        axis_world = data.xaxis[model.joint(name).id]
        if np.argmax(np.abs(axis_world)) != want:
            report.append(f"WARNING: {name} axis {np.round(axis_world, 2)} is not along torso "
                          f"{'xyz'[want]}: check the CAD frames")
        if axis_world[want] < 0:
            j = joints[name]
            j.set("axis", fmt(-np.array([float(x) for x in j.get("axis", "0 0 1").split()])))
            if j.get("range"):
                lo, hi = (float(x) for x in j.get("range").split())
                j.set("range", fmt([-hi, -lo]))
            report.append(f"flipped {name} to match training conventions")

    # ---- hip roll limits: 5 deg inward / 25 deg outward (roll axis is +x now: left outward = +)
    for name in ROLL:
        out_sign = 1.0 if name.startswith("left") else -1.0
        lim = sorted([out_sign * np.radians(ROLL_OUTWARD_DEG), -out_sign * np.radians(ROLL_INWARD_DEG)])
        joints[name].set("range", fmt(lim))

    # ---- feet: flat sole box + rubber pad instead of the mesh hull; capsules on thighs/shins
    model = compile_xml(tree, tmp)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for foot in FEET:
        bid = model.body(foot).id
        R, x = data.xmat[bid].reshape(3, 3), data.xpos[bid]
        verts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH \
                    and re.search(r"ankle_foot", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) or ""):
                mid = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                verts.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
        verts = np.vstack(verts)
        z0 = verts[:, 2].min()
        sole = verts[verts[:, 2] < z0 + 0.003]
        # Pad = the foot's own outline (2D hull of the sole region), extruded by the pad thickness
        # below the CAD sole, plus 1 mm overlap into the sole so the pad is attached.
        outline = convex_hull_2d(sole[:, :2])
        t = args.pad_thickness
        prism_w = np.vstack([np.column_stack([outline, np.full(len(outline), z)]) for z in (z0 - t, z0 + 0.001)])
        prism_local = (prism_w - x) @ R  # world -> foot body frame
        fb = bodies[foot]
        for g in [g for g in fb.findall("geom") if g.get("class") == "collision"]:
            fb.remove(g)
        ET.SubElement(root.find("asset"), "mesh", name=f"{foot}_pad", vertex=fmt(prism_local.flatten()),
                      inertia="shell")
        ET.SubElement(fb, "geom", name=f"{foot}_sole", type="mesh", mesh=f"{foot}_pad",
                      friction=f"{args.friction} 0.005 0.0001", mass=f"{max(args.pad_mass, 1e-3) / 1000:.6g}",
                      rgba="0.12 0.12 0.12 1")
        ext = outline.max(0) - outline.min(0)
        report.append(f"{foot}: rubber pad following the foot outline ({len(outline)} points, "
                      f"{1000 * ext[0]:.0f} x {1000 * ext[1]:.0f} mm), {1000 * t:.1f} mm thick, friction {args.friction}")
    radii = {}
    for link, child in LEG_CAPSULES.items():
        bid, cid = model.body(link).id, model.body(child).id
        R, x = data.xmat[bid].reshape(3, 3), data.xpos[bid]
        a, b = data.xpos[bid], data.xpos[cid]
        shells = []
        for g in range(model.ngeom):
            mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g]) or ""
            if model.geom_bodyid[g] == bid and re.match(r"^\d[ab]_", mname):
                mid = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                shells.append(data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)
        pts = np.vstack(shells)
        ab = b - a
        tpar = np.clip((pts - a) @ ab / (ab @ ab), 0, 1)
        radii[link] = 0.9 * np.median(np.linalg.norm(pts - (a + np.outer(tpar, ab)), axis=1))
        # shorten so the capsule ends stop short of the joints (adjacent links do not collide anyway)
        p0, p1 = a + 0.15 * ab, b - 0.15 * ab
        ET.SubElement(bodies[link], "geom", name=f"{link}_capsule", type="capsule", size="0",
                      fromto=fmt(np.concatenate([R.T @ (p0 - x), R.T @ (p1 - x)])), density="0",
                      rgba="0.8 0.3 0.3 0.25", group="3")
    for seg in ("thigh", "shin"):  # same radius on both legs: mirrored meshes differ slightly
        r = np.mean([radii[f"{side}_{seg}_link"] for side in ("left", "right")])
        for side in ("left", "right"):
            next(g for g in bodies[f"{side}_{seg}_link"].findall("geom")
                 if g.get("name") == f"{side}_{seg}_link_capsule").set("size", f"{r:.4g}")
        report.append(f"{seg}s: collision capsule radius {1000 * r:.0f} mm (both legs)")

    # ---- IMU site at the BNO055 (axes = torso axes; confirm the chip mounting on the robot)
    imu = [geom for part, geom, body in part_geoms(root) if part.startswith("adafruit_bno055")
           and body.get("name") == "body"]
    if imu:
        pos = np.mean([[float(v) for v in g.get("pos").split()] for g in imu], axis=0)
        ET.SubElement(bodies["body"], "site", name="imu", pos=fmt(pos), size="0.005")

    # ---- actuators: one servo per joint, in the training's order
    act = ET.SubElement(root, "actuator")
    for name in ACTUATOR_ORDER:
        ET.SubElement(act, "position", name=name, joint=name, kp="15", inheritrange="1",
                      forcerange="-3.43 3.43")

    # ---- home keyframe: crouch, feet flat, soles 1 mm above the floor
    model = compile_xml(tree, tmp)
    data = mujoco.MjData(model)
    q = model.qpos0.copy()
    for name in ACTUATOR_ORDER:
        key = next(k for k in HOME if name.endswith(k))
        q[model.joint(name).qposadr[0]] = HOME[key]
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    lowest = min(sole_bottom(model, data, f) for f in FEET)
    q[2] += 0.001 - lowest
    ctrl = [q[model.joint(n).qposadr[0]] for n in ACTUATOR_ORDER]
    for k in root.findall("keyframe"):
        root.remove(k)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=fmt(q), ctrl=fmt(ctrl))

    ET.indent(tree)
    model = compile_xml(tree, out)
    tmp.unlink(missing_ok=True)
    scene = (args.export_dir / "scene.xml").read_text().replace('file="robot.xml"', 'file="rl_robot.xml"')
    (args.export_dir / "rl_scene.xml").write_text(scene)
    write_training_model(tree, model, args.export_dir)

    # ---- report
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    print(f"\nwrote {out} and {args.export_dir / 'rl_scene.xml'}\n")
    for line in report:
        print(" -", line)
    print(f"\n{'link':<18}{'mass g':>8}")
    for b in range(1, model.nbody):
        print(f"{model.body(b).name:<18}{1000 * model.body_mass[b]:>8.1f}")
    total = model.body_subtreemass[1]
    print(f"{'TOTAL':<18}{1000 * total:>8.1f}   (deck BoM: ~1747 g + pads)")
    straight = mujoco.MjData(model)
    straight.qpos[:] = model.qpos0
    mujoco.mj_forward(model, straight)
    floor = min(sole_bottom(model, straight, f) for f in FEET)
    print(f"CoM height, legs straight: {1000 * (straight.subtree_com[1][2] - floor):.0f} mm  (deck: 244 mm)")
    print(f"home pose (crouched): CoM {1000 * (data.subtree_com[1][2]):.0f} mm above floor\n")
    print(f"{'joint':<18}{'range deg':>18}")
    for name in ACTUATOR_ORDER:
        lo, hi = np.degrees(model.jnt_range[model.joint(name).id])
        print(f"{name:<18}{lo:>8.1f} .. {hi:>6.1f}")


if __name__ == "__main__":
    main()
