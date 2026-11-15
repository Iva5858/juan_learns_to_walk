"""The dashboard's simulator backend (robot/sim.py) and the 3D view's model (robot/web/model/)."""
import json

import pytest

from robot import sim
from robot.joints import BY_NAME, HOME, NAMES, Calibration


@pytest.fixture
def world():
    cal = Calibration(zero_set=True)
    for n in NAMES:
        cal.joints[n].zero, cal.joints[n].sign = 500, 1
    w = sim.World(cal, run=False)
    sim.Bus.world = w
    return w


def counts(world, name, deg):
    return int(round(world.cal.joints[name].counts(deg)))


def test_stands_at_home_holding(world):
    world.step(1.0)
    s = world.scene()
    assert not s["fallen"]
    g = world.readings()["gravity"]
    assert g[2] < -0.99   # upright
    bus = sim.Bus("sim0")
    for n in NAMES:
        assert abs(world.cal.joints[n].degrees(bus.position(BY_NAME[n][1])) - HOME[n]) < 3.0


def test_servo_command_arrives_after_the_bus_delay(world):
    bus, sid = sim.Bus("sim0"), BY_NAME["left_ankle"][1]
    before = bus.position(sid)
    assert world.delay > 0.05   # the measured servos: 3-4 control steps
    bus.move_many({sid: counts(world, "left_ankle", HOME["left_ankle"] + 15)})
    world.step(0.02)
    assert abs(bus.position(sid) - before) <= 2          # not yet
    world.step(0.5)
    assert bus.position(sid) - before > 0.5 * 15 * 1000 / 240


def test_relaxed_servos_collapse_and_reset_stands_again(world):
    bus = sim.Bus("sim0")
    bus.set_torque(254, False)
    world.step(2.0)
    assert world.scene()["fallen"]
    world.reset("home")
    world.step(0.5)
    assert not world.scene()["fallen"]


def test_face_down_start(world):
    world.reset("face_down")
    world.step(0.5)
    s = world.scene()
    assert s["fallen"] and world.readings()["gravity"][0] > 0.8   # lying on its front: down is the torso's forward


def test_shove_moves_the_torso(world):
    world.step(0.2)
    world.shove(0.0, 0.5)
    world.step(0.1)
    assert world.d.qpos[1] > 0.01


def test_web_model_matches_the_robot():
    from biped_rl.robots import PROJECT
    model = json.loads((PROJECT / "robot" / "web" / "model" / "juan.json").read_text())
    joints = {b["joint"]["name"] for b in model["bodies"] if b["joint"]}
    assert joints == set(NAMES)
    for level in ("low", "high"):
        info = model["levels"][level]
        size = (PROJECT / "robot" / "web" / "model" / info["file"]).stat().st_size
        assert size == info["bytes"]
        for part in info["bodies"].values():   # every part lies inside the file, aligned for the typed arrays
            assert part["pos"] % 4 == 0 and part["idx"] % 4 == 0
            assert part["idx"] + 4 * 3 * part["tris"] <= size
    assert model["levels"]["low"]["tris"] < 40000   # quick on a phone


def test_sim_actions_are_registered():
    from robot import actions
    assert {"sim/reset", "sim/physics", "sim/shove", "sim/view", "sim/pause"} <= set(actions.ACTIONS)
