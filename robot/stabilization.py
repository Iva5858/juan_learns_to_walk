"""Calibrated rotational image stabilization; numpy core, optional OpenCV image warp."""
import json
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


def sensor_time():
    """Linux camera SensorTimestamp clock (CLOCK_BOOTTIME), seconds."""
    return time.clock_gettime(getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC))


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolate(a, b, t):
    if np.dot(a, b) < 0:
        b = -b
    q = (1-t)*a + t*b
    return q / np.linalg.norm(q)


class OrientationHistory:
    """Bounded raw-gyro integration in torso axes, independent of policy filtering.

    Absolute heading is arbitrary. Samples across a gap >50 ms invalidate the old
    history; images without a bracketed orientation are left uncorrected.
    """

    def __init__(self):
        self.samples = deque(maxlen=600)
        self.lock = threading.Lock()
        self.generation = 0

    def clear(self):
        with self.lock:
            self.samples.clear()
            self.generation += 1

    def append(self, timestamp, gyro):
        with self.lock:
            q = np.array([1., 0., 0., 0.])
            if self.samples:
                previous, q = self.samples[-1]
                dt = timestamp - previous
                if dt <= 0:
                    return
                if dt > 0.05:
                    self.samples.clear()
                    self.generation += 1
                    q = np.array([1., 0., 0., 0.])
                else:
                    vector = np.asarray(gyro) * dt
                    angle = np.linalg.norm(vector)
                    dq = np.r_[np.cos(angle/2), vector * (np.sin(angle/2)/angle if angle else 0.5)]
                    q = multiply(q, dq)
                    q /= np.linalg.norm(q)
            self.samples.append((timestamp, q.copy()))

    def at(self, timestamp):
        return self.sample(timestamp)[0]

    def sample(self, timestamp):
        with self.lock:
            samples = list(self.samples)
            generation = self.generation
        if len(samples) < 2 or not samples[0][0] <= timestamp <= samples[-1][0]:
            return None, generation
        for (ta, qa), (tb, qb) in zip(samples, samples[1:]):
            if ta <= timestamp <= tb:
                return interpolate(qa, qb, (timestamp-ta)/(tb-ta)), generation
        return None, generation


@dataclass
class CameraCalibration:
    size: tuple
    matrix: object
    distortion: object
    camera_to_body: object
    time_offset_s: float = 0.0
    smoothing_s: float = 0.25
    crop_scale: float = 1.15

    def __post_init__(self):
        self.size = tuple(self.size)
        self.matrix = np.asarray(self.matrix, dtype=float)
        self.distortion = np.asarray(self.distortion, dtype=float)
        self.camera_to_body = np.asarray(self.camera_to_body, dtype=float)
        if (not np.isfinite([self.time_offset_s, self.smoothing_s, self.crop_scale]).all()
                or not np.isfinite(self.size).all()):
            raise ValueError("calibration settings must be finite")
        if self.matrix.shape != (3, 3) or self.camera_to_body.shape != (3, 3):
            raise ValueError("camera matrices must be 3 by 3")
        if not all(np.isfinite(v).all() for v in (self.matrix, self.distortion, self.camera_to_body)):
            raise ValueError("calibration must be finite")
        if self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0 or abs(np.linalg.det(self.matrix)) < 1e-9:
            raise ValueError("invalid camera intrinsics")
        if not np.allclose(self.camera_to_body.T @ self.camera_to_body, np.eye(3), atol=1e-5) or not np.isclose(
                np.linalg.det(self.camera_to_body), 1, atol=1e-5):
            raise ValueError("camera_to_body must be a proper rotation")
        if len(self.size) != 2 or min(self.size) <= 0 or self.smoothing_s <= 0 or self.crop_scale < 1:
            raise ValueError("invalid size, smoothing or crop")
        if self.distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError("unsupported distortion coefficient count")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


class Stabilizer:
    def __init__(self, calibration, history):
        self.cal, self.history = calibration, history
        self.previous = self.smoothed = None
        self.generation = None

    def homography(self, timestamp):
        q, generation = self.history.sample(timestamp + self.cal.time_offset_s)
        if q is None:
            self.previous = self.smoothed = None
            return None
        if (self.previous is None or generation != self.generation
                or timestamp <= self.previous or timestamp-self.previous > 0.5):
            self.smoothed = q
        else:
            alpha = 1 - np.exp(-(timestamp-self.previous)/self.cal.smoothing_s)
            self.smoothed = interpolate(self.smoothed, q, alpha)
        self.previous = timestamp
        self.generation = generation
        mount, k = self.cal.camera_to_body, self.cal.matrix
        relative = mount.T @ rotation(self.smoothed).T @ rotation(q) @ mount
        return k @ relative @ np.linalg.inv(k)

    def process(self, image, timestamp):
        import cv2
        if (image.shape[1], image.shape[0]) != self.cal.size:
            raise ValueError("camera image size differs from calibration")
        h = self.homography(timestamp)
        if h is None:
            return image, False
        k = self.cal.matrix
        clean = cv2.undistort(image, k, self.cal.distortion)
        width, height = self.cal.size
        scale = self.cal.crop_scale
        crop = np.array([[scale, 0, (1-scale)*width/2], [0, scale, (1-scale)*height/2], [0, 0, 1]])
        return cv2.warpPerspective(clean, crop @ h, self.cal.size), True
