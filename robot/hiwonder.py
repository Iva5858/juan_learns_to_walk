"""Hiwonder bus-servo protocol (HTD-35H, LX series) through a BusLinker USB board.

Frames: 0x55 0x55 ID Len Cmd Params Checksum, 115200 baud, half duplex: the board echoes every
byte we send, so replies are searched for (by ID, command and checksum) rather than assumed to
come first. One Bus per BusLinker; a lock keeps one conversation on the wire at a time.
"""
import glob
import threading
import time

import serial

BAUD = 115200
BROADCAST = 254

MOVE_TIME_WRITE, MOVE_TIME_READ, MOVE_STOP, ID_READ = 1, 2, 12, 14
ANGLE_OFFSET_READ, ANGLE_LIMIT_WRITE, ANGLE_LIMIT_READ = 19, 20, 21
VIN_LIMIT_READ, TEMP_MAX_LIMIT_READ, TEMP_READ, VIN_READ, POS_READ = 23, 25, 26, 27, 28
LOAD_OR_UNLOAD_WRITE, LOAD_OR_UNLOAD_READ = 31, 32


def packet(sid, cmd, params=()):
    body = [sid, len(params) + 3, cmd] + list(params)
    return bytes([0x55, 0x55] + body + [~sum(body) & 0xFF])


def find_reply(buf, sid, cmd, nparams):
    """A valid reply packet in buf, skipping the echo of our own request."""
    i = buf.find(b"\x55\x55")
    while i >= 0:
        if len(buf) >= i + 4:
            length = buf[i + 3]
            chk = i + 2 + length
            if length == nparams + 3 and len(buf) > chk:
                body = buf[i + 2:chk]
                if body[0] == sid and body[2] == cmd and (~sum(body) & 0xFF) == buf[chk]:
                    return body[3:]
        i = buf.find(b"\x55\x55", i + 1)
    return None


def u16(b, i=0):
    return b[i] | (b[i + 1] << 8)


def s16(b, i=0):
    v = u16(b, i)
    return v - 65536 if v > 32767 else v


def s8(v):
    return v - 256 if v > 127 else v


def ports():
    return sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))


class Bus:
    def __init__(self, port):
        self.port = port
        self.ser = serial.Serial(port, BAUD, timeout=0.005, exclusive=True)
        self.lock = threading.RLock()

    def close(self):
        self.ser.close()

    def send(self, sid, cmd, params=()):
        with self.lock:
            self.ser.write(packet(sid, cmd, params))

    def query(self, sid, cmd, nparams, timeout=0.03):
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write(packet(sid, cmd))
            buf = b""
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                buf += self.ser.read(self.ser.in_waiting or 1)
                reply = find_reply(buf, sid, cmd, nparams)
                if reply is not None:
                    return reply
        return None

    # ---- reads (None when the servo doesn't answer) ----
    def ping(self, sid):
        return any(self.query(sid, ID_READ, 1) is not None for _ in range(2))

    def position(self, sid):
        r = self.query(sid, POS_READ, 2)
        return None if r is None else s16(r)

    def temperature(self, sid):
        r = self.query(sid, TEMP_READ, 1)
        return None if r is None else r[0]

    def voltage(self, sid):
        r = self.query(sid, VIN_READ, 2)
        return None if r is None else u16(r) / 1000

    def torque_on(self, sid):
        r = self.query(sid, LOAD_OR_UNLOAD_READ, 1)
        return None if r is None else bool(r[0])

    def target(self, sid):
        r = self.query(sid, MOVE_TIME_READ, 4)
        return None if r is None else u16(r)

    def angle_limits(self, sid):
        r = self.query(sid, ANGLE_LIMIT_READ, 4)
        return None if r is None else (u16(r), u16(r, 2))

    def info(self, sid):
        """Settings that rarely change: offset, angle limits, voltage limits, max temperature."""
        out = {}
        r = self.query(sid, ANGLE_OFFSET_READ, 1)
        out["offset"] = s8(r[0]) if r else None
        out["angle_limits"] = self.angle_limits(sid)
        r = self.query(sid, VIN_LIMIT_READ, 4)
        out["vin_limits"] = (u16(r) / 1000, u16(r, 2) / 1000) if r else None
        r = self.query(sid, TEMP_MAX_LIMIT_READ, 1)
        out["temp_max"] = r[0] if r else None
        return out

    # ---- writes ----
    def move(self, sid, counts, ms=0):
        counts = max(0, min(1000, int(round(counts))))
        ms = max(0, min(30000, int(ms)))
        self.send(sid, MOVE_TIME_WRITE, [counts & 0xFF, counts >> 8, ms & 0xFF, ms >> 8])

    def move_many(self, targets, ms=0):
        """Several servos in one write ({id: counts}): the control loop's per-tick command."""
        ms = max(0, min(30000, int(ms)))
        out = b"".join(packet(sid, MOVE_TIME_WRITE, [c & 0xFF, c >> 8, ms & 0xFF, ms >> 8])
                       for sid, c in ((sid, max(0, min(1000, int(round(c))))) for sid, c in targets.items()))
        with self.lock:
            self.ser.write(out)

    def set_torque(self, sid, on):
        self.send(sid, LOAD_OR_UNLOAD_WRITE, [1 if on else 0])

    def stop(self, sid=BROADCAST):
        self.send(sid, MOVE_STOP)

    def write_angle_limits(self, sid, lo, hi):
        """Stored in the servo's memory: it refuses targets outside [lo, hi] (counts) from then on."""
        lo, hi = max(0, min(1000, int(lo))), max(0, min(1000, int(hi)))
        lo, hi = min(lo, hi), max(lo, hi)
        self.send(sid, ANGLE_LIMIT_WRITE, [lo & 0xFF, lo >> 8, hi & 0xFF, hi >> 8])
