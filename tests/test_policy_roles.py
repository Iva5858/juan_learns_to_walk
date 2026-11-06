"""robot/policies/roles.json: the dashboard offers one policy per role (walk, speed, getup), and scripts/policies.py
sets them."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from robot import walker as W


def _library(tmp_path, monkeypatch, roles=None):
    (tmp_path / "getup").mkdir()
    for name in ("old_a", "old_b", "best", "fast"):
        np.savez(tmp_path / f"{name}.npz", meta_task=np.asarray("walk"))
    for name in ("getup_1", "getup_2"):
        np.savez(tmp_path / "getup" / f"{name}.npz", meta_task=np.asarray("getup"))
    if roles is not None:
        (tmp_path / "roles.json").write_text(json.dumps(roles))
    monkeypatch.setattr(W, "POLICY_DIR", str(tmp_path))


def test_roles_limit_what_the_page_offers(tmp_path, monkeypatch):
    _library(tmp_path, monkeypatch, {"walk": "best", "speed": "fast", "getup": "getup_2"})
    assert W.policies() == ["best", "fast"]                 # walk first, then speed; the library is hidden
    assert W.getup_policies() == ["getup_2"]
    assert W.default_policy() == "best"
    assert W.policy_labels() == {"best": "walk", "fast": "speed", "getup_2": "getup"}


def test_missing_or_empty_roles(tmp_path, monkeypatch):
    _library(tmp_path, monkeypatch, {"walk": "best", "speed": None, "getup": "gone"})
    assert W.policies() == ["best"] and W.getup_policies() == []   # an empty or missing one is just left out


def test_without_roles_everything_is_listed(tmp_path, monkeypatch):
    _library(tmp_path, monkeypatch)
    assert W.policies() == ["best", "fast", "old_a", "old_b"] and len(W.getup_policies()) == 2


def test_the_shipped_roles_point_at_real_policies():
    roles = json.loads((Path(W.__file__).parent / "policies" / "roles.json").read_text())
    assert set(roles) == {"walk", "speed", "getup"} and roles["walk"]


def test_policies_script_refuses_the_wrong_kind():
    script = Path(__file__).resolve().parents[1] / "scripts" / "policies.py"
    out = subprocess.run([sys.executable, str(script), "walk", "juan_getup_x_does_not_exist"], capture_output=True, text=True)
    assert out.returncode != 0 and "doesn't exist" in out.stderr
