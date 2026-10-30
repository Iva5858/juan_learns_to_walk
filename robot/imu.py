"""The IMU as the policy sees it in the simulator (biped_rl/tasks/locomotion/mdp/observations.py):
    gravity   unit vector pointing DOWN, in the torso frame (x forward, y left, z up):
              (0, 0, -1) standing upright
    gyro      rad/s, torso frame, right-hand rule: + x = rolling right (right side down),
              + y = pitching forward (nose down), + z = turning left (counter-clockwise from above)
    yaw       rad, integrated from the gyro (for the heading error; drifts slowly, fine for a walk)

The LSM9DS1 (accelerometer + gyro, magnetometer unused) has no on-chip fusion, so gravity is
estimated here: the gyro rotates the previous estimate, and the accelerometer pulls it back with a
0.5 s time constant (a complementary filter), ignoring the accelerometer during impacts.
"""
import math
import threading
import time

from robot.stabilization import OrientationHistory, sensor_time

RATE_HZ = 200
TAU_S = 0.5           # accelerometer correction time constant
ACCEL_TRUST = 0.15    # use the accelerometer only within +-15% of 1 g (not during impacts)
BIAS_SECONDS = 2.0    # gyro bias averaged over this long at start (robot still)
# Low-pass on the gyro the POLICY sees. The servos make the torso buzz at ~15-20 Hz (the gyro reads
# 0.5-2.7 rad/s rms while standing still), and policies trained without the filter amplified it into a
# 16-18 Hz chatter. In sim, a 4 Hz filter cut that chatter from 41% to 8% of the action energy without
# changing clean walking; the simulator applies the same filter (RandomizationConfig.gyro_lowpass_hz).
# The gravity estimate and the yaw use the raw gyro (filtering would only slow them down). None = off.
GYRO_LOWPASS_HZ = 4.0

# LSM9DS1 accelerometer + gyro (magnetometer is a separate device at 0x1C / 0x1E, not used)
LSM_ADDRESSES = (0x6B, 0x6A)
WHO_AM_I, WHO_AM_I_VALUE = 0x0F, 0x68
CTRL_REG1_G, CTRL_REG6_XL, CTRL_REG8 = 0x10, 0x20, 0x22
OUT_X_G, OUT_X_XL = 0x18, 0x28
GYRO_CFG = 0b100_11_000   # 238 Hz, +-2000 deg/s
ACCEL_CFG = 0b100_10_000  # 238 Hz, +-4 g
GYRO_SCALE = 0.070 * math.pi / 180   # 70 mdps per count at +-2000 deg/s -> rad/s
ACCEL_SCALE = 0.122e-3 * 9.80665     # 0.122 mg per count at +-4 g -> m/s^2


def _i16(b, i):
    v = b[i] | (b[i + 1] << 8)
    return v - 65536 if v > 32767 else v


def remap(v, axes):
    """Sensor vector -> torso frame. axes[k] names the sensor axis along torso axis k, e.g. "-y"."""
    return [(-1.0 if a.startswith("-") else 1.0) * v["xyz".index(a[-1])] for a in axes]


def tilt_deg(gravity):
    """(lean right, lean forward) in degrees, from the downward gravity vector."""
    gx, gy, gz = gravity
    return math.degrees(math.atan2(-gy, -gz)), math.degrees(math.atan2(gx, -gz))


def tilted(gravity, right_deg, forward_deg):
    """The gravity vector as an IMU tilted by (right_deg, forward_deg) more would read it: tilt_deg of the
    result is tilt_deg(gravity) plus the offsets."""
    gx, gy, gz = gravity
    p, r = math.radians(forward_deg), math.radians(right_deg)
    gx, gz = gx * math.cos(p) - gz * math.sin(p), gz * math.cos(p) + gx * math.sin(p)
    gy, gz = gy * math.cos(r) + gz * math.sin(r), gz * math.cos(r) - gy * math.sin(r)
    return [gx, gy, gz]


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


class GravityFilter:
    """Complementary filter: gravity (down, body frame) and yaw from accelerometer + gyro."""

    def __init__(self, tau=TAU_S, trust=ACCEL_TRUST):
        self.tau, self.trust = tau, trust
        self.g, self.yaw, self.a_norm = None, 0.0, 9.80665

    def update(self, accel, gyro, dt):
        """accel: specific force (m/s^2, points UP at rest), gyro: rad/s, both in the torso frame."""
        a_norm = math.sqrt(sum(x * x for x in accel))
        self.a_norm = a_norm
        g_acc = [-x / a_norm for x in accel] if a_norm > 1e-3 else None
        if self.g is None:
            if g_acc is None:
                return None
            self.g = g_acc
        # A world-fixed direction seen from a body turning at w changes as dg/dt = -w x g.
        wxg = _cross(gyro, self.g)
        g = [self.g[k] - wxg[k] * dt for k in range(3)]
        if g_acc is not None and abs(a_norm / 9.80665 - 1.0) < self.trust:
            alpha = dt / self.tau
            g = [(1 - alpha) * g[k] + alpha * g_acc[k] for k in range(3)]
        self.g = _normalize(g)
        # Yaw rate = the turn rate about the world's vertical, i.e. the gyro along "up" (-gravity).
        self.yaw -= sum(gyro[k] * self.g[k] for k in range(3)) * dt
        return self.g


class LSM9DS1:
    name = "LSM9DS1"

    def __init__(self, bus):
        from smbus2 import SMBus
        self.i2c = SMBus(bus)
        for address in LSM_ADDRESSES:
            try:
                if self.i2c.read_byte_data(address, WHO_AM_I) == WHO_AM_I_VALUE:
                    self.address = address
                    break
            except OSError:
                continue
        else:
            self.i2c.close()
            raise OSError("no LSM9DS1 at 0x6B / 0x6A")
        self.i2c.write_byte_data(self.address, CTRL_REG8, 0x44)  # block data update, auto-increment
        self.i2c.write_byte_data(self.address, CTRL_REG1_G, GYRO_CFG)
        self.i2c.write_byte_data(self.address, CTRL_REG6_XL, ACCEL_CFG)
        time.sleep(0.05)

    def read(self):
        """(accel m/s^2, gyro rad/s) in the sensor's axes. Accel points UP at rest (+1 g on z flat)."""
        g = self.i2c.read_i2c_block_data(self.address, OUT_X_G, 6)
        a = self.i2c.read_i2c_block_data(self.address, OUT_X_XL, 6)
        return [_i16(a, 2 * k) * ACCEL_SCALE for k in range(3)], [_i16(g, 2 * k) * GYRO_SCALE for k in range(3)]

    def close(self):
        self.i2c.close()


class IMU:
    """Reads the IMU in its own thread at RATE_HZ and keeps the latest estimate."""

    def __init__(self, bus=1, axes=("x", "y", "z")):
        self.bus, self.axes = bus, tuple(axes)
        self.state, self.error, self.device = None, None, None
        self.reads = self.failures = 0
        self.read_ms = 0.0
        self.gyro_bias = [0.0, 0.0, 0.0]
        self._bias_samples, self._bias_until = [], 0.0
        self._lock = threading.Lock()
        self.orientation = OrientationHistory()
        self.filter = GravityFilter()
        self.gyro_lowpass_hz = GYRO_LOWPASS_HZ
        self._gyro_f = None
        self.gyro_raw_rms = 0.0  # how much the raw gyro shakes (a vibration gauge for the page)
        threading.Thread(target=self._loop, daemon=True).start()

    def set_axes(self, axes):
        self.axes = tuple(axes)
        self.reset()

    def set_bus(self, bus):
        self.bus = int(bus)
        self._close()

    def zero_gyro(self):
        """Average the gyro over the next BIAS_SECONDS as its bias. Keep the robot still."""
        with self._lock:
            self._bias_samples, self._bias_until = [], time.monotonic() + BIAS_SECONDS

    def reset(self):
        with self._lock:
            self.filter = GravityFilter()  # restart the estimate from the accelerometer
            self.orientation.clear()

    def _close(self):
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
        self.device = None

    def _open(self):
        try:
            self.device = LSM9DS1(self.bus)
        except Exception as e:
            raise OSError(f"I2C bus {self.bus}: {e}") from e
        self.error = None
        self.reset()
        self.zero_gyro()

    def _loop(self):
        dt = 1.0 / RATE_HZ
        next_t = time.monotonic()
        while True:
            if self.device is None:
                try:
                    self._open()
                except Exception as e:
                    self.error, self.state = str(e), None
                    time.sleep(3)
                    continue
                next_t = time.monotonic()
            t0 = time.perf_counter()
            self.reads += 1
            try:
                sample_start = sensor_time()
                accel, gyro = self.device.read()
                sample_time = 0.5 * (sample_start + sensor_time())
                self._update(remap(accel, self.axes), remap(gyro, self.axes), dt, sample_time)
            except OSError as e:
                self.failures += 1
                self.error = str(e)
                if self.failures > 50 and self.failures > 0.5 * self.reads:
                    self._close()
                    continue
            self.read_ms = 0.9 * self.read_ms + 0.1 * (time.perf_counter() - t0) * 1000
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - next_t > 0.1:  # fell far behind (Pi busy): don't try to catch up
                next_t = time.monotonic()

    def _update(self, accel, gyro, dt, timestamp=None):
        with self._lock:
            now = time.monotonic()
            if now < self._bias_until:
                self._bias_samples.append(gyro)
            elif self._bias_samples:
                n = len(self._bias_samples)
                self.gyro_bias = [sum(s[k] for s in self._bias_samples) / n for k in range(3)]
                self._bias_samples = []
            gyro = [gyro[k] - self.gyro_bias[k] for k in range(3)]
            self.orientation.append(sensor_time() if timestamp is None else timestamp, gyro)
            g = self.filter.update(accel, gyro, dt)
        if g is not None:
            self._publish(g, gyro, self.filter.a_norm)

    def _smooth_gyro(self, gyro, dt=1.0 / RATE_HZ):
        self.gyro_raw_rms = math.sqrt(0.995 * self.gyro_raw_rms ** 2 + 0.005 * (gyro[0] ** 2 + gyro[1] ** 2))
        if not self.gyro_lowpass_hz:
            return list(gyro)
        if self._gyro_f is None:
            self._gyro_f = list(gyro)
        a = dt / (1.0 / (2 * math.pi * self.gyro_lowpass_hz) + dt)
        self._gyro_f = [f + a * (g - f) for f, g in zip(self._gyro_f, gyro)]
        return list(self._gyro_f)

    def _publish(self, gravity, gyro, a_norm):
        gyro = self._smooth_gyro(gyro)
        self.state = {"gravity": list(gravity), "gyro": gyro, "yaw": self.filter.yaw, "accel_g": a_norm / 9.80665,
                      "gyro_raw_rms": self.gyro_raw_rms, "gyro_lowpass_hz": self.gyro_lowpass_hz, "biasing": time.monotonic() < self._bias_until,
                      "gyro_bias": list(self.gyro_bias), "t": time.time(), "device": self.device.name}
