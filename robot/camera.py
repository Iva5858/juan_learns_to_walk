"""Pi camera as an MJPEG stream, encoded by the Pi's hardware encoder so it doesn't compete with the
control loop for CPU. Started on first use and stopped when nobody has watched for a while.

Needs picamera2 (Raspberry Pi OS: sudo apt install python3-picamera2).
"""
import io
import os
import queue
import threading
import time

from robot.stabilization import CameraCalibration, Stabilizer

try:
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput
except ImportError:  # not on a Pi, or picamera2 missing: the dashboard runs without a camera
    Picamera2 = None

SIZE = (640, 480)
FPS = 15
IDLE_STOP_S = 10


class _Frames(io.BufferedIOBase):
    def __init__(self):
        self.frame, self.cond = None, threading.Condition()

    def write(self, buf):
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


class Camera:
    available = Picamera2 is not None

    def __init__(self):
        self.cam = None
        self.frames = _Frames()
        self.lock = threading.Lock()
        self.last_view = 0.0
        self.error = None if self.available else "picamera2 not installed"
        self.stabilized = _Frames()
        self.stabilizer = None
        self.stabilization_error = None
        self.corrected = False
        self._pending = queue.Queue(maxsize=1)
        threading.Thread(target=self._stabilize_loop, daemon=True).start()
        threading.Thread(target=self._idle_stop, daemon=True).start()

    def configure_stabilization(self, imu, path=None):
        """Opt in with JUAN_CAMERA_CALIBRATION=/path/to/camera.json, before streaming."""
        path = path or os.environ.get("JUAN_CAMERA_CALIBRATION")
        if not path:
            return
        try:
            import cv2  # noqa: F401
            calibration = CameraCalibration.load(path)
            if calibration.size != SIZE:
                raise ValueError(f"calibration size must match {SIZE}")
            self.stabilizer = Stabilizer(calibration, imu.orientation)
        except (ImportError, OSError, ValueError, TypeError) as exc:
            self.stabilization_error = str(exc)

    def _capture(self, request):
        # Camera callback only copies a frame into a bounded latest-frame queue.
        # Warping/JPEG encoding run outside the camera and walking threads.
        if self.stabilizer is None:
            return
        metadata = request.get_metadata()
        if "SensorTimestamp" not in metadata:
            self.stabilization_error = "camera supplied no SensorTimestamp"
            return
        timestamp = metadata["SensorTimestamp"] * 1e-9 + metadata.get("ExposureTime", 0) * 0.5e-6
        item = (request.make_array("main").copy(), timestamp)
        try:
            self._pending.get_nowait()
        except queue.Empty:
            pass
        try:
            self._pending.put_nowait(item)
        except queue.Full:
            pass

    def _stabilize_loop(self):
        while True:
            image, timestamp = self._pending.get()
            try:
                import cv2
                image, self.corrected = self.stabilizer.process(image, timestamp)
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                self.stabilized.write(jpeg.tobytes())
                self.stabilization_error = None
            except Exception as exc:
                self.corrected = False
                self.stabilization_error = str(exc)

    def _start(self):
        with self.lock:
            if self.cam is not None:
                return True
            cam = None
            try:
                cam = Picamera2()
                main = {"size": SIZE}
                if self.stabilizer is not None:
                    main["format"] = "RGB888"
                cam.configure(cam.create_video_configuration(main=main,
                                                             controls={"FrameRate": FPS}))
                if self.stabilizer is not None:
                    cam.post_callback = self._capture
                cam.start_recording(MJPEGEncoder(), FileOutput(self.frames))
                self.cam, self.error = cam, None
                return True
            except Exception as e:  # no camera connected, ribbon loose, camera busy...
                if cam is not None:
                    try:
                        cam.close()
                    except Exception:
                        pass
                self.error = str(e)
                return False

    def _idle_stop(self):
        while True:
            time.sleep(2)
            with self.lock:
                if self.cam is not None and time.monotonic() - self.last_view > IDLE_STOP_S:
                    self.cam.stop_recording()
                    self.cam.close()
                    self.cam = None

    def stream(self, stabilized=False):
        """Yields JPEG frames while the viewer keeps reading."""
        if not self.available or not self._start():
            return
        frames = self.stabilized if stabilized else self.frames
        while True:
            with frames.cond:
                frames.cond.wait(timeout=1.0)
                frame = frames.frame
            self.last_view = time.monotonic()
            if frame is not None:
                yield frame
