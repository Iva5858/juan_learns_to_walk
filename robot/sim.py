"""Simulated hardware: the dashboard driving Juan in MuJoCo, on a computer (not the Pi: it needs mujoco and
biped_rl, which the robot never imports).

    python -m robot.dashboard --sim                        # the measured servos (the default)
    python -m robot.dashboard --sim --physics randomized   # nominal / measured / randomized (one drawn robot)

Everything above the hardware is the robot's own code: robot/body.py reads the servo buses, robot/walker.py
runs the exported policies (robot/policies/), gestures, safety rules and the page are unchanged. Underneath:
  * servos: the simulator's servo model (the one training uses: PD, torque-speed line, braking with the
    measured servos), each command applied after the bus delay the robot measured; a servo's own timed move
    (move_many(ms)) ramps its target over that time; a relaxed servo gives no torque;
  * IMU: gravity and gyro of the torso (gyro low-passed at 4 Hz, as robot/imu.py does), heading;
  * camera: rendered, from Juan's head ("head") or following him ("chase");
  * the world: reset standing at home, legs straight or lying face down; shoves. Actions in group "sim".
The calibration is the sim's own (robot/logs/sim/calibration.json): zero 500, sign +1 on every joint.
Physics runs in real time in its own thread (2 ms steps, 20 ms at a time).
"""
import math
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from biped_rl.tasks.locomotion import BipedEnv, EnvConfig
from robot import actions
from robot.fake import PiHealth
from robot.joints import BY_ID, NAMES, Calibration
from robot.stabilization import OrientationHistory, sensor_time

PHYSICS = ("nominal", "measured", "randomized")
POSES = ("home", "straight", "face_down")
VIEWS = ("chase", "head")
PORTS = ("sim0",)                         # one bus for all 8, as on the robot
HEAD_CAMERA = np.array([0.045, 0.0, 0.41])   # the Pi camera on the torso (juan/rl_robot.xml), body frame
GYRO_LOWPASS_HZ = 4.0
CAMERA_FPS = 15


class World:
    """One simulated Juan, stepped in real time; every fake part reads and writes it under `lock`."""

    def __init__(self, cal, physics="measured", seed=0, run=True):
        self.cal = cal
        self.lock = threading.RLock()
        self.view = "chase"
        self.turn_rate = lambda: 0.0     # (robot/body.py sets this on any world; unused here)
        self.paused = False
        self.realtime = 1.0
        self.vis = None                  # the render model, loaded on the first camera frame
        self.render_lock = threading.Lock()
        self.setup(physics, seed)
        if run:   # (tests step it themselves: World.step)
            threading.Thread(target=self._loop, daemon=True).start()

    def step(self, seconds):
        """Advance the simulation (whole 20 ms control steps), as fast as it computes."""
        for _ in range(round(seconds / self.env.dt)):
            with self.lock:
                self._control_step()

    # ---- setup ----
    def setup(self, physics="measured", seed=0, pose="home"):
        if physics not in PHYSICS:
            raise ValueError(f"physics: one of {', '.join(PHYSICS)}")
        cfg = EnvConfig()
        if physics != "nominal":
            cfg.use_measured_servos()
        r = cfg.randomization
        r.enabled = physics == "randomized"
        r.push_velocity, r.physics_curriculum = 0.0, False
        with self.lock:
            self.env = env = BipedEnv(cfg.to_dict())
            env.reset(seed=seed)
            self.physics, self.seed = physics, seed
            self.m, self.d = env.model, env.data
            sv = env.servo
            self.slack = sv.backlash if sv.backlash is not None and sv.backlash.any() else None
            # The bus delay: whole control steps plus the latency within one. Randomized: this robot's draw. Measured:
            # the fastest the robot measured (3 steps + the mean latency; the dashboard's own bus reads add the rest).
            # Nominal: none, as the perfect-physics training stages.
            if physics == "randomized":
                self.delay = sv.action_delay * env.dt + sv.latency_steps * self.m.opt.timestep
            elif physics == "measured":
                self.delay = min(r.action_delay_steps) * env.dt + float(np.mean(r.latency))
            else:
                self.delay = 0.0
            self.t = 0.0
            self.pending = []            # (apply at sim time, joint index, target rad, ms)
            self.limits = {}             # servo ID -> (lo, hi) counts written to the servo
            self.gyro = np.zeros(3)
            self.reset(pose)

    def reset(self, pose="home"):
        """Put Juan back: standing at home or with straight legs (servos holding), or lying face down
        (servos holding home: the getup policies' start)."""
        if pose not in POSES:
            raise ValueError(f"pose: one of {', '.join(POSES)}")
        env, n = self.env, len(NAMES)
        with self.lock:
            joints = np.zeros(n) if pose == "straight" else env.robot.home_joint.copy()
            yaw = env.yaw()   # upright where it is, facing the same way
            self.d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            self.d.qvel[:] = 0.0
            env.place_pose(joints)
            if pose == "face_down":
                down = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]   # torso front down, then the heading
                mujoco.mju_mulQuat(self.d.qpos[3:7], np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]), np.array(down))
                mujoco.mj_forward(self.m, self.d)
                low = min(env._geom_lowest_point(g)[2] for g in range(self.m.ngeom) if not env.robot.geom_is_world[g])
                self.d.qpos[2] += 0.005 - low
                mujoco.mj_forward(self.m, self.d)
            self.target = joints.copy()                  # what each servo is driving to now
            self.ramp = [None] * n                       # (start rad, goal rad, t0, t1) of a timed move
            self.torque = np.ones(n, dtype=bool)
            self.pending.clear()
            self.gyro[:] = 0.0

    def shove(self, forward, left):
        """A push: the torso's velocity changes by (forward, left) m/s, in its own heading."""
        with self.lock:
            yaw = self.env.yaw()
            c, s = math.cos(yaw), math.sin(yaw)
            self.d.qvel[0] += c * forward - s * left
            self.d.qvel[1] += s * forward + c * left

    # ---- servos ----
    def command(self, idx, rad, ms):
        with self.lock:
            self.pending.append((self.t + self.delay, idx, rad, ms))
            self.torque[idx] = True

    def relax(self, idx):
        with self.lock:
            self.torque[idx] = False
            self.ramp[idx] = None
            self.pending = [p for p in self.pending if p[1] != idx]

    def hold_here(self, idx):
        with self.lock:
            self.target[idx] = self.joint_rad(idx)
            self.ramp[idx] = None
            self.pending = [p for p in self.pending if p[1] != idx]

    def joint_rad(self, idx):
        return float(self.d.qpos[self.env.robot.qpos_idx[idx]])

    def _apply_pending(self):
        due = [p for p in self.pending if p[0] <= self.t]
        if not due:
            return
        self.pending = [p for p in self.pending if p[0] > self.t]
        for _, idx, rad, ms in due:
            if ms > 0:
                self.ramp[idx] = (self.target[idx], rad, self.t, self.t + ms / 1000)
            else:
                self.ramp[idx] = None
                self.target[idx] = rad

    def _ramps(self):
        for idx, r in enumerate(self.ramp):
            if r is not None:
                a, b, t0, t1 = r
                u = min(1.0, (self.t - t0) / max(t1 - t0, 1e-6))
                self.target[idx] = a + (b - a) * u
                if u >= 1.0:
                    self.ramp[idx] = None

    # ---- physics ----
    def _control_step(self):
        """20 ms of physics."""
        env, sv = self.env, self.env.servo
        self._apply_pending()
        target = np.clip(self.target, env.robot.joint_lo, env.robot.joint_hi)
        for _ in range(env.frame_skip):
            self._ramps()
            target = np.clip(self.target, env.robot.joint_lo, env.robot.joint_hi)
            qd, _, tau_c = env.servo_torque(target, self.slack, 1.0 / sv.speed)
            env.drive(np.where(self.torque, tau_c, 0.0), qd)
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep
        a = env.dt / (1.0 / (2 * math.pi * GYRO_LOWPASS_HZ) + env.dt)
        self.gyro += a * (self.d.qvel[3:6] - self.gyro)
        env.episode.target = target

    def _loop(self):
        wall0, sim0 = time.monotonic(), 0.0
        while True:
            if self.paused:
                time.sleep(0.05)
                wall0, sim0 = time.monotonic(), self.t
                continue
            with self.lock:
                self._control_step()
                ahead = (self.t - sim0) / self.realtime - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.2:  # can't keep up: don't try to catch up
                wall0, sim0 = time.monotonic(), self.t

    # ---- what the page and the camera show ----
    def readings(self):
        with self.lock:
            env = self.env
            return {"gravity": env.gravity_body().tolist(), "gyro": self.gyro.tolist(), "yaw": env.yaw(),
                    "accel_g": 1.0}

    def scene(self):
        """The world as the page's 3D view draws it: torso position (m) and orientation (w, x, y, z)."""
        with self.lock:
            return {"physics": self.physics, "seed": self.seed, "t": round(self.t, 3), "view": self.view,
                    "paused": self.paused, "pos": self.d.qpos[0:3].tolist(), "quat": self.d.qpos[3:7].tolist(),
                    "fallen": bool(self.env.non_foot_ground_contact() or self.env.gravity_body()[2] > -0.6)}

    def frame(self, width=640, height_px=480):
        """One camera image (RGB array), from the head camera or following the robot. Always from the same
        thread (Camera's render thread): the OpenGL context belongs to the thread that made it."""
        with self.lock:   # a copy of the state: rendering (tens of ms) must not hold up the physics
            qpos, R = self.d.qpos.copy(), self.env.root_rot().copy()
            root, height, view = self.env.robot.root, self.env.robot.home_height, self.view
        with self.render_lock:
            if self.vis is None:   # the full model with the CAD meshes, drawn posed like the simulation
                self.vis = SimpleNamespace(model=mujoco.MjModel.from_xml_path(str(self.env._model_path)))
                self.vis.data = mujoco.MjData(self.vis.model)
                self.vis.renderer = mujoco.Renderer(self.vis.model, height_px, width)
                self.vis.cam = mujoco.MjvCamera()
            v = self.vis
            v.data.qpos[:] = qpos
            mujoco.mj_forward(v.model, v.data)
            cam = v.cam
            if view == "head":
                eye = qpos[0:3] + R @ HEAD_CAMERA
                fwd = R[:, 0]
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.distance = 0.5
                cam.lookat[:] = eye + fwd * cam.distance
                cam.azimuth = math.degrees(math.atan2(fwd[1], fwd[0]))
                cam.elevation = math.degrees(math.asin(np.clip(fwd[2], -1, 1)))
            else:
                cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                cam.trackbodyid = root
                cam.distance = 4 * height
                cam.elevation = -15
                cam.azimuth = 120
            v.renderer.update_scene(v.data, camera=cam)
            return v.renderer.render()


def _counts(cal, idx, rad):
    name = NAMES[idx]
    return cal.joints[name].counts(math.degrees(rad))


class Bus:
    """hiwonder.Bus's interface on the simulated servos."""
    world = None

    def __init__(self, port):
        self.port = port
        self.lock = threading.RLock()

    def close(self):
        pass

    def ping(self, sid):
        return sid in BY_ID

    def _idx(self, sid):
        return NAMES.index(BY_ID[sid][0])

    def position(self, sid):
        time.sleep(0.0012)   # one query on the real 115200 baud bus
        w = self.world
        with w.lock:
            return int(round(_counts(w.cal, self._idx(sid), w.joint_rad(self._idx(sid)))))

    def temperature(self, sid):
        return 38 + sid % 3

    def voltage(self, sid):
        return 11.6

    def torque_on(self, sid):
        return bool(self.world.torque[self._idx(sid)])

    def target(self, sid):
        w = self.world
        return int(round(_counts(w.cal, self._idx(sid), w.target[self._idx(sid)])))

    def angle_limits(self, sid):
        return self.world.limits.get(sid, (0, 1000))

    def info(self, sid):
        return {"offset": 0, "angle_limits": self.angle_limits(sid), "vin_limits": (4.5, 14.0), "temp_max": 85}

    def move(self, sid, counts, ms=0):
        self.move_many({sid: counts}, ms)

    def move_many(self, targets, ms=0):
        w = self.world
        for sid, counts in targets.items():
            lo, hi = w.limits.get(sid, (0, 1000))
            counts = max(lo, min(hi, counts))
            idx = self._idx(sid)
            w.command(idx, math.radians(w.cal.joints[NAMES[idx]].degrees(counts)), ms)

    def set_torque(self, sid, on):
        for s in (BY_ID if sid == 254 else (sid,)):
            if on:
                self.world.hold_here(self._idx(s))
                self.world.torque[self._idx(s)] = True
            else:
                self.world.relax(self._idx(s))

    def stop(self, sid=254):
        for s in (BY_ID if sid == 254 else (sid,)):
            self.world.hold_here(self._idx(s))

    def write_angle_limits(self, sid, lo, hi):
        self.world.limits[sid] = (min(lo, hi), max(lo, hi))


def ports():
    return list(PORTS)


class IMU:
    """robot.imu.IMU's interface on the simulated torso (200 readings a second)."""
    world = None

    def __init__(self, bus=1, axes=("x", "y", "z")):
        self.bus, self.axes = bus, tuple(axes)
        self.error, self.device, self.state = None, "simulated IMU", None
        self.reads = self.failures = 0
        self.read_ms = 0.0
        self.orientation = OrientationHistory()
        self.yaw0 = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def set_axes(self, axes):
        self.axes = tuple(axes)

    def set_bus(self, bus):
        self.bus = int(bus)

    def zero_gyro(self):
        pass

    def reset(self):
        self.yaw0 = self.world.readings()["yaw"]

    def _loop(self):
        while True:
            time.sleep(0.005)
            r = self.world.readings()
            yaw = (r["yaw"] - self.yaw0 + math.pi) % (2 * math.pi) - math.pi
            self.reads += 1
            self.orientation.append(sensor_time(), r["gyro"])
            self.state = {"gravity": r["gravity"], "gyro": r["gyro"], "yaw": yaw, "accel_g": r["accel_g"],
                          "gyro_raw_rms": 0.0, "gyro_lowpass_hz": GYRO_LOWPASS_HZ, "biasing": False,
                          "gyro_bias": [0.0, 0.0, 0.0], "t": time.time(), "device": self.device}


class Camera:
    """robot.camera.Camera's interface: frames rendered from the simulation."""
    world = None
    available = True

    def __init__(self):
        self.error, self.stabilizer, self.stabilization_error, self.corrected = None, None, None, False
        self.last_view = 0.0
        self.jpeg, self.seq, self.thread = None, 0, None
        self.new_frame = threading.Condition()

    def configure_stabilization(self, imu, path=None):
        pass

    def _render_loop(self):
        """The only thread that renders (an OpenGL context belongs to the thread that made it), while
        anyone has watched in the last 2 s; every stream sends its latest JPEG."""
        encode = _jpeg_encoder()
        while True:
            t0 = time.monotonic()
            if t0 - self.last_view > 2.0:
                time.sleep(0.1)
                continue
            try:
                jpeg = encode(self.world.frame())
            except Exception as e:   # no OpenGL (a headless machine): say so
                self.error = f"can't render the simulation: {e}"
                with self.new_frame:
                    self.new_frame.notify_all()
                return
            with self.new_frame:
                self.jpeg, self.seq = jpeg, self.seq + 1
                self.new_frame.notify_all()
            time.sleep(max(0.0, 1 / CAMERA_FPS - (time.monotonic() - t0)))

    def stream(self, stabilized=False):
        self.last_view = time.monotonic()
        with self.new_frame:
            if self.thread is None:
                self.thread = threading.Thread(target=self._render_loop, daemon=True)
                self.thread.start()
        seen = 0
        while self.error is None:
            with self.new_frame:
                self.new_frame.wait_for(lambda last=seen: self.seq != last or self.error is not None, timeout=5.0)
                if self.seq == seen:
                    continue
                seen, jpeg = self.seq, self.jpeg
            self.last_view = time.monotonic()
            yield jpeg


def _jpeg_encoder():
    try:
        import cv2
        return lambda img: cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
    except ImportError:
        import io

        from PIL import Image

        def encode(img):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "JPEG", quality=75)
            return buf.getvalue()
        return encode


def hardware(cal_path, physics="measured", seed=0):
    """The simulated parts for robot.body.Robot and their calibration (the sim's own file)."""
    cal = Calibration.load(cal_path)
    if not cal.zero_set:
        cal.zero_set = True
        for name in NAMES:
            cal.joints[name].zero, cal.joints[name].sign = 500, 1
    cal.save(cal_path)
    world = World(cal, physics, seed)
    Bus.world = IMU.world = Camera.world = world
    return SimpleNamespace(ports=ports, Bus=Bus, IMU=IMU, Camera=Camera, pi=PiHealth, world=world, sim=world,
                           broadcast=254)


# ---------------------------------------------------------------- the page's controls for the simulation
def _sim(robot):
    world = getattr(robot.hw, "sim", None)
    if world is None:
        raise RuntimeError("not running in the simulator (python -m robot.dashboard --sim)")
    return world


@actions.action("sim/reset", "Reset", "sim", pose=(str, "home"))
def sim_reset(robot, pose):
    """Put the simulated Juan back: home, straight or face_down. Stops the policy and any gesture."""
    robot.stop()
    _sim(robot).reset(pose)
    robot.imu.reset()


@actions.action("sim/physics", "Physics", "sim", physics=(str, "measured"), seed=(int, 0))
def sim_physics(robot, physics, seed):
    """Change the simulated robot: nominal, measured (the servo model fitted on the real robot) or randomized
    (one robot drawn from training's randomization; the seed picks which). Starts again at home."""
    robot.stop()
    _sim(robot).setup(physics, seed)
    robot.imu.reset()


@actions.action("sim/shove", "Shove", "sim", forward=(float, 0.0), left=(float, 0.3))
def sim_shove(robot, forward, left):
    """Push the torso: a sudden change of velocity (m/s), forward and to the left (negative: back, right)."""
    _sim(robot).shove(forward, left)


@actions.action("sim/view", "Camera view", "sim", view=(str, "chase"))
def sim_view(robot, view):
    """The camera stream: chase (following Juan) or head (Juan's own camera)."""
    if view not in VIEWS:
        return f"view: one of {', '.join(VIEWS)}"
    _sim(robot).view = view


@actions.action("sim/pause", "Pause", "sim", on=(bool, True), speed=(float, 1.0))
def sim_pause(robot, on, speed):
    """Pause or resume the simulation; speed: slow motion below 1 (0.1-2)."""
    w = _sim(robot)
    w.paused = on
    w.realtime = min(2.0, max(0.1, speed))
