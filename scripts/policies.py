"""Which exported policy plays which role on the robot: robot/policies/roles.json. The dashboard offers only these
(walk, preselected, and speed as walking policies; getup for Get up), and deploy_pi.sh copies only these to the Pi.
Every other export stays in robot/policies/ on this computer, the library, ready to be given a role again.

    python scripts/policies.py                                   # the roles
    python scripts/policies.py speed juan_v11n_servos_best       # give a role (a runs/... path works too: its name)
    python scripts/policies.py speed none                        # leave a role empty
    python scripts/policies.py --files                           # the files deploy copies (relative to robot/policies)
"""
import argparse
import json
import sys
from pathlib import Path

POLICIES = Path(__file__).resolve().parents[1] / "robot" / "policies"
ROLES = POLICIES / "roles.json"
NAMES = ("walk", "speed", "getup")


def path(role, name):
    return POLICIES / ("getup" if role == "getup" else "") / f"{name}.npz"


def describe(role, name):
    import numpy as np  # here, not at the top: deploy_pi.sh runs --files with the system python3
    if not name:
        return "-"
    p = path(role, name)
    if not p.exists():
        return f"{name}  (MISSING: {p.relative_to(POLICIES.parents[1])})"
    m = np.load(p, allow_pickle=False)
    meta = {k[5:]: m[k] for k in m.files if k.startswith("meta_")}
    start = "straight legs" if not np.any(meta["home"]) else "home"
    kind = ("free run" + (", steers" if meta.get("go_turn_range", np.zeros(0)).size else "")
            if meta.get("go_command", np.zeros(0)).size else "commands")
    return name if role == "getup" else f"{name}  (start: {start}; {kind})"


def check(role, name):
    import numpy as np
    p = path(role, name)
    if not p.exists():
        sys.exit(f"{p} doesn't exist: export it first (scripts/export_policy.py or export_getup.py)")
    task = str(np.load(p, allow_pickle=False).get("meta_task", "walk"))
    if (task == "getup") != (role == "getup"):
        sys.exit(f"{name} is a {task} policy: not for the {role} role")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("role", nargs="?", choices=NAMES)
    p.add_argument("name", nargs="?", help="exported policy name (or runs/... path), or none")
    p.add_argument("--files", action="store_true")
    a = p.parse_args()
    roles = json.loads(ROLES.read_text()) if ROLES.exists() else {}
    if a.files:
        for role in NAMES:
            if roles.get(role) and path(role, roles[role]).exists():
                print(path(role, roles[role]).relative_to(POLICIES))
        return
    if a.role:
        if a.name is None:
            p.error("give a policy name (or none)")
        name = None if a.name == "none" else Path(a.name).name
        if name:
            check(a.role, name)
        roles[a.role] = name
        ROLES.write_text(json.dumps({k: roles.get(k) for k in NAMES}, indent=1) + "\n")
    for role in NAMES:
        print(f"{role:6s} {describe(role, roles.get(role))}")
    rest = len(list(POLICIES.glob("*.npz"))) + len(list(POLICIES.glob("getup/*.npz"))) - sum(1 for r in NAMES if roles.get(r))
    print(f"(library: {rest} more exported policies here, not on the dashboard and not deployed)")


if __name__ == "__main__":
    main()
