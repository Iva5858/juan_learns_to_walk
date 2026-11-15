"""Juan's model for the dashboard's 3D view (robot/web/view3d.js): the body tree, the joints, and the CAD meshes cut
down to a size a phone draws quickly. Run again whenever juan/ changes (scripts/model/postprocess_onshape.py).

    python scripts/model/web_model.py

Writes robot/web/model/:
    juan.json      bodies (parent, position, orientation), joints (axis, anchor), per-body boxes, and per
                   detail level where each body's mesh sits in its .bin file
    juan_low.bin   ~6 mm detail; juan_high.bin ~2 mm detail. Per body: float32 xyz positions (m, body frame),
                   uint8 rgb colours (padded to 4 bytes), uint32 triangle indices.
Detail is cut by vertex clustering: vertices merged on a grid of the level's cell size, collapsed triangles
dropped. The page draws it with flat shading, so the merged vertices need no normals.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

from biped_rl.robots import PROJECT, resolve_model_path
from biped_rl.tasks.locomotion import EnvConfig

OUT = PROJECT / "robot" / "web" / "model"
LEVELS = {"low": 0.006, "high": 0.002}   # grid cell (m)


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def body_meshes(m):
    """Every visual mesh in its body's frame: body -> (vertices, triangles, rgb per vertex)."""
    out = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[a:a + n] @ quat_mat(m.geom_quat[g]).T + m.geom_pos[g]
        f = m.mesh_face[fa:fa + fn].astype(np.int64)
        mat = m.geom_matid[g]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        rgb = np.tile((np.clip(rgba[:3], 0, 1) * 255).astype(np.uint8), (n, 1))
        out.setdefault(int(m.geom_bodyid[g]), []).append((v, f, rgb))
    return {b: (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] + off for p, off in zip(parts, np.cumsum([0] + [len(p[0]) for p in parts[:-1]]))]),
                np.concatenate([p[2] for p in parts]))
            for b, parts in out.items()}


def cluster(v, f, rgb, cell):
    """Vertex clustering: one vertex per occupied grid cell (the mean of its vertices), triangles that still
    span three cells kept once."""
    keys = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    k = inv.max() + 1
    counts = np.bincount(inv, minlength=k)[:, None]
    nv = np.stack([np.bincount(inv, weights=v[:, i], minlength=k) for i in range(3)], 1) / counts
    ncol = np.zeros((k, 3), np.uint8)
    ncol[inv] = rgb
    tri = inv[f]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    keep = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)[1]   # each triangle once (winding kept)
    return nv, ncol, tri[np.sort(keep)]


def main():
    path = resolve_model_path(EnvConfig().robot.model_path)
    m = mujoco.MjModel.from_xml_path(str(path))
    meshes = body_meshes(m)
    bodies = []
    for b in range(1, m.nbody):
        joint = next((j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE), None)
        v = meshes[b][0] if b in meshes else np.zeros((1, 3))
        bodies.append({"name": m.body(b).name, "parent": m.body(int(m.body_parentid[b])).name if m.body_parentid[b] else None,
                       "pos": m.body_pos[b].round(6).tolist(), "quat": m.body_quat[b].round(6).tolist(),
                       "joint": None if joint is None else {"name": m.joint(joint).name, "axis": m.jnt_axis[joint].tolist(),
                                                            "anchor": m.jnt_pos[joint].round(6).tolist()},
                       "box": [v.min(0).round(4).tolist(), v.max(0).round(4).tolist()]})
    OUT.mkdir(parents=True, exist_ok=True)
    levels = {}
    for level, cell in LEVELS.items():
        blob, parts, tris = bytearray(), {}, 0
        for b in range(1, m.nbody):
            if b not in meshes:
                continue
            v, f, rgb = meshes[b]
            nv, ncol, tri = cluster(v, f, rgb, cell)
            entry = {"pos": len(blob), "verts": len(nv)}
            blob += nv.astype("<f4").tobytes()
            entry["col"] = len(blob)
            blob += ncol.tobytes() + b"\0" * (-len(ncol.tobytes()) % 4)
            entry["idx"], entry["tris"] = len(blob), len(tri)
            blob += tri.astype("<u4").tobytes()
            parts[m.body(b).name] = entry
            tris += len(tri)
        (OUT / f"juan_{level}.bin").write_bytes(bytes(blob))
        levels[level] = {"file": f"juan_{level}.bin", "bytes": len(blob), "tris": tris, "bodies": parts}
        print(f"{level}: {tris} triangles, {len(blob) / 1e6:.2f} MB")
    (OUT / "juan.json").write_text(json.dumps({"source": Path(path).name, "bodies": bodies, "levels": levels}, indent=1))
    print(f"wrote {OUT}/juan.json")


if __name__ == "__main__":
    main()
