"""The dashboard's API on the fake robot (robot/fake.py): actions, the walking policy, command sources,
calibration in its own file, the old GET routes, and the event stream."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from robot import dashboard, fake
from robot.body import Robot
from robot.joints import NAMES

POLICY = "juan_v7_13_straight"   # committed (CI has it)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    folder = tmp_path_factory.mktemp("fake_robot")
    cal = str(folder / "calibration.json")
    robot = Robot(fake.hardware(cal), cal)
    robot.walker.log_dir = str(folder)
    dashboard.robot = robot
    srv = dashboard.serve(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://localhost:{srv.server_address[1]}", robot
    robot.stop()
    srv.shutdown()


def call(base, action, **params):
    req = urllib.request.Request(f"{base}/api/{action}", data=json.dumps(params).encode(), method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.load(r)


def test_every_action_is_described(server):
    base, _ = server
    names = {a["name"]: a for a in get(base, "/api/actions")}
    assert {"stop", "boot", "policy/start", "policy/cmd", "cal/zero", "gesture", "heartbeat"} <= set(names)
    assert names["cal/zero"]["confirm"] and names["stop"]["danger"]
    assert names["policy/cmd"]["params"]["vx"] == {"type": "float"}


def test_bad_parameters_are_refused(server):
    base, _ = server
    assert call(base, "move", joint="no_such_joint", deg=1)[0] == 400
    assert call(base, "policy/cmd")[0] == 400
    assert call(base, "no/such/action")[0] == 404


def test_walking_with_command_sources(server):
    base, robot = server
    assert call(base, "policy/cmd", vx=0.1)[1]["error"] == "policy not running"
    assert call(base, "policy/start", name=POLICY) == (200, {"error": None})
    assert get(base, "/api/state")["mode"] == "walking"
    assert call(base, "move", joint="left_knee", deg=30)[0] == 409   # the policy has the servos
    assert call(base, "policy/cmd", vx=0.3, wz=0.2)[0] == 200
    time.sleep(0.3)
    w = get(base, "/api/state")["walker"]
    assert w["running"] and w["command"] == [0.3, 0.0, 0.2] and w["command_source"] == "page" and w["ticks"] > 5
    robot.walker.steer_owner = "spin"   # a behavior steering
    assert call(base, "policy/cmd", vx=0.1)[0] == 409
    robot.walker.steer_owner = None
    with urllib.request.urlopen(base + "/policy/stop") as r:   # the old GET route
        assert json.load(r) == {"error": None}
    time.sleep(0.2)
    assert get(base, "/api/state")["mode"] == "idle"


def test_calibration_stays_in_the_fake_robots_file(server):
    base, robot = server
    assert call(base, "cal/flip", joint="left_knee")[0] == 200
    saved = json.loads(open(robot.cal_path).read())
    assert saved["joints"]["left_knee"]["sign"] == -1
    call(base, "cal/flip", joint="left_knee")


def test_event_stream_pushes_fast_and_full_state(server):
    base, _ = server
    seen = {}
    with urllib.request.urlopen(base + "/api/events", timeout=5) as r:
        event = None
        while sum(seen.values()) < 4:
            line = r.readline().decode()
            if line.startswith("event:"):
                event = line.split()[1]
            elif line.startswith("data:"):
                seen[event] = seen.get(event, 0) + 1
                assert "mode" in json.loads(line[5:])
    assert seen.get("full") and seen.get("fast")


def test_pages_are_served_and_nothing_else(server):
    base, _ = server
    for path in ("/", "/web/app.js", "/web/drive.js", "/web/style.css", "/classic"):
        with urllib.request.urlopen(base + path) as r:
            assert r.status == 200 and r.read()
    for path in ("/web/../dashboard.py", "/web/missing.js"):
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + path)


def test_shutdown_saves_the_walk_log_and_stops_the_server(tmp_path, monkeypatch):
    from robot import body
    monkeypatch.setattr(body, "SHUTDOWN_MS", (50, 100, 20))
    cal = str(tmp_path / "calibration.json")
    robot = Robot(fake.hardware(cal), cal)
    robot.walker.log_dir = str(tmp_path)
    dashboard.robot = robot
    srv = dashboard.serve(0)
    dashboard.attach_exit(robot, srv)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://localhost:{srv.server_address[1]}"
    time.sleep(0.5)   # the first sensor readings
    assert call(base, "policy/start", name=POLICY)[0] == 200
    time.sleep(0.5)
    status, reply = call(base, "shutdown")
    assert status == 200 and reply["log"] and reply["log"].endswith(".npz")
    assert list(tmp_path.glob("*.npz")) and not robot.walker.running
    thread.join(timeout=3)
    assert not thread.is_alive()   # serve_forever returned: the dashboard exits


def test_getup_then_walker_hands_over_when_upright(tmp_path):
    """The fake robot is upright, so the getup phase hands over to the walking policy after UP_HOLD_S; a getup
    policy is refused as a walker, and commands wait until it is up."""
    import shutil

    from robot import walker as W
    cal = str(tmp_path / "calibration.json")
    robot = Robot(fake.hardware(cal), cal)
    robot.walker.log_dir = str(tmp_path)
    getups = W.getup_policies()
    if not getups:
        pytest.skip("no getup policy exported (scripts/export_getup.py)")
    time.sleep(0.5)
    assert "getup policy" in robot.walker.start(f"getup/{getups[0]}")   # never started as a walker
    assert robot.walker.start_getup(getups[0], POLICY) is None
    assert robot.walker.phase == "getup" and robot.walker.set_command(0.1, 0, 0)
    time.sleep(W.UP_HOLD_S + 0.6)
    assert robot.walker.phase == "walk" and robot.walker.running
    assert robot.walker.set_command(0.1, 0, 0) is None
    robot.walker.stop()
    shutil.rmtree(tmp_path, ignore_errors=True)


def _fake_robot(tmp_path, monkeypatch):
    from robot import body
    monkeypatch.setattr(body, "SHUTDOWN_MS", (50, 100, 20))
    cal = str(tmp_path / "calibration.json")
    robot = Robot(fake.hardware(cal), cal)
    time.sleep(0.5)   # the first sensor readings
    return robot


def test_shutdown_squats_then_relaxes(tmp_path, monkeypatch):
    from robot.joints import SQUAT
    robot = _fake_robot(tmp_path, monkeypatch)
    moves = []
    move = robot.move
    monkeypatch.setattr(robot, "move", lambda name, deg, ms=300, _boot=False: moves.append((name, deg)) or move(name, deg, ms))
    robot.shutdown()
    assert [d for n, d in moves[-8:]] == [SQUAT[n] for n in NAMES]
    assert not any(s["torque"] for s in robot.hw.world.servos.values())
    assert "squatted" in robot.message


def test_shutdown_relaxes_at_once_when_fallen_or_failing(tmp_path, monkeypatch):
    robot = _fake_robot(tmp_path, monkeypatch)
    robot.cal.imu_level = (45.0, 0.0)   # the IMU reads 45 deg off its level: lying on its side
    robot.shutdown()
    assert "no squat: tilted" in robot.message and not any(s["torque"] for s in robot.hw.world.servos.values())

    robot = _fake_robot(tmp_path, monkeypatch)
    robot.move(NAMES[0], 0.0)
    monkeypatch.setattr(robot, "pose", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        robot.shutdown()
    assert not any(s["torque"] for s in robot.hw.world.servos.values())   # relaxed all the same
