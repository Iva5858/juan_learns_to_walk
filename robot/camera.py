"""Pi camera as an MJPEG stream, encoded by the Pi's hardware encoder so it doesn't compete with the
control loop for CPU. Started on first use and stopped when nobody has watched for a while.

Needs picamera2 (Raspberry Pi OS: sudo apt install python3-picamera2).
"""
import io
import threading
import time

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
        threading.Thread(target=self._idle_stop, daemon=True).start()

    def _start(self):
        with self.lock:
            if self.cam is not None:
                return True
            try:
                cam = Picamera2()
                cam.configure(cam.create_video_configuration(main={"size": SIZE},
                                                             controls={"FrameRate": FPS}))
                cam.start_recording(MJPEGEncoder(), FileOutput(self.frames))
                self.cam, self.error = cam, None
                return True
            except Exception as e:  # no camera connected, ribbon loose, camera busy...
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

    def stream(self):
        """Yields JPEG frames while the viewer keeps reading."""
        if not self.available or not self._start():
            return
        while True:
            with self.frames.cond:
                self.frames.cond.wait(timeout=1.0)
                frame = self.frames.frame
            self.last_view = time.monotonic()
            if frame is not None:
                yield frame
