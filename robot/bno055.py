"""BNO055 IMU (the robot's first IMU; robot/imu.py uses it if no LSM9DS1 answers), in IMU mode (accelerometer + gyro fusion, no magnetometer, 100 Hz), read the way the
policy sees it in the simulator (biped_rl/tasks/locomotion/mdp/observations.py):
    gravity   unit vector pointing DOWN, in the torso frame (x forward, y left, z up):
              (0, 0, -1) standing upright
    gyro      rad/s, torso frame

The Pi's hardware I2C mishandles the BNO055's clock stretching (~6% corrupted reads at 100 kHz).
Use a bit-banged bus instead, e.g. in /boot/firmware/config.txt:
    dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=23,i2c_gpio_scl=24
and set the IMU bus to 3 on the dashboard. Bad reads are counted and dropped (plausibility gate).
"""
import math
import time

from smbus2 import SMBus, i2c_msg

ADDRESS = 0x28
CHIP_ID, OPR_MODE, PWR_MODE, PAGE_ID, UNIT_SEL = 0x00, 0x3D, 0x3E, 0x07, 0x3B
MODE_CONFIG, MODE_IMU = 0x00, 0x08
BURST_START, BURST_LEN = 0x14, 0x36 - 0x14   # gyro 0x14 .. calibration status 0x35
GRAVITY_RANGE = (9.0, 10.6)  # m/s^2: anything else is a corrupted read or a hard impact


def _i16(b, i):
    v = b[i] | (b[i + 1] << 8)
    return v - 65536 if v > 32767 else v


def remap(v, axes):
    """Sensor vector -> torso frame. axes[k] names the sensor axis along torso axis k, e.g. "-y"."""
    out = []
    for a in axes:
        sign = -1.0 if a.startswith("-") else 1.0
        out.append(sign * v["xyz".index(a[-1])])
    return out


def tilt_deg(gravity):
    """(lean right, lean forward) in degrees, from the downward gravity vector."""
    gx, gy, gz = gravity
    return math.degrees(math.atan2(-gy, -gz)), math.degrees(math.atan2(gx, -gz))


class BNO055:
    name = "BNO055"

    def __init__(self, bus=1, address=ADDRESS):
        self.address = address
        self.i2c = SMBus(bus)
        self.reads = self.failures = 0
        self._start()

    def _write(self, reg, value):
        self.i2c.write_byte_data(self.address, reg, value)

    def _start(self):
        if self.i2c.read_byte_data(self.address, CHIP_ID) != 0xA0:
            raise OSError("no BNO055 at this address (chip ID is not 0xA0)")
        self._write(OPR_MODE, MODE_CONFIG)
        time.sleep(0.025)
        self._write(PWR_MODE, 0x00)   # normal power
        self._write(PAGE_ID, 0x00)
        self._write(UNIT_SEL, 0x00)   # m/s^2, deg/s, degrees, Celsius
        self._write(OPR_MODE, MODE_IMU)
        time.sleep(0.02)

    def close(self):
        self.i2c.close()

    def read(self, axes=("x", "y", "z")):
        """One burst read. Returns None if the read failed or isn't plausible, else a dict with
        gravity (unit, down, torso frame), gyro (rad/s, torso frame) and calibration status (0-3)."""
        self.reads += 1
        try:
            msg_w, msg_r = i2c_msg.write(self.address, [BURST_START]), i2c_msg.read(self.address, BURST_LEN)
            self.i2c.i2c_rdwr(msg_w, msg_r)
            b = bytes(list(msg_r))  # (iterating is the documented way to get the data)
        except OSError:
            self.failures += 1
            return None
        gyro = [_i16(b, 0 + 2 * k) / 16.0 * math.pi / 180 for k in range(3)]   # 0x14: 16 LSB per deg/s
        grv = [_i16(b, 0x2E - BURST_START + 2 * k) / 100.0 for k in range(3)]  # 0x2E: 100 LSB per m/s^2
        norm = math.sqrt(sum(g * g for g in grv))
        if not GRAVITY_RANGE[0] <= norm <= GRAVITY_RANGE[1]:
            self.failures += 1
            return None
        grv, gyro = remap(grv, axes), remap(gyro, axes)
        calib = b[0x35 - BURST_START]
        return {
            "gravity": [-g / norm for g in grv],   # the sensor reports "up"; the policy uses "down"
            "gyro": gyro,
            "calib": {"sys": calib >> 6 & 3, "gyro": calib >> 4 & 3, "accel": calib >> 2 & 3},
        }
