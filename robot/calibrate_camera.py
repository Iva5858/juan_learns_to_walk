"""Calibrate the camera for IMU stabilization (docs/camera-stabilization.md): capture checkerboard views at the
live stream's size, then compute the lens calibration. On the Pi, from the project folder, dashboard stopped:

    python3 -m robot.calibrate_camera ~/juan-camera-views --capture 30
    python3 -m robot.calibrate_camera ~/juan-camera-views --camera-to-body 0 0 1  -1 0 0  0 -1 0

While capturing, http://<pi-name>.local:8000 shows what the camera sees, with the checkerboard's corners drawn
and "board found" in green when this view will count; Enter saves a view only then.
"""
import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

PORT = 8000
PAGE = b"""<!doctype html><meta name=viewport content="width=device-width"><title>Juan camera</title>
<body style="margin:0;background:#111;display:flex;justify-content:center"><img src="/stream.mjpg" style="max-width:100%">"""


class DualStackServer(ThreadingHTTPServer):
    """IPv4 and IPv6, so http://name.local:8000 works either way (as the dashboard's)."""
    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def find_board(cv2, image, size, fast=False):
    """The checkerboard's inner corners in a BGR image (refined to sub-pixel), or None."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | (cv2.CALIB_CB_FAST_CHECK if fast else 0)
    found, corners = cv2.findChessboardCorners(grey, size, flags)
    if not found:
        return None
    # a 11 x 11 px search window: wider ones span whole squares of a board a little way off and pull the
    # corners off (synthetic views: 2.7 px rms and a 14% wrong focal length with 23 x 23)
    return cv2.cornerSubPix(grey, corners, (5, 5), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def capture(cv2, folder, count, size):
    """Show the camera live over HTTP with the board's detection drawn; save `count` views with the board."""
    from picamera2 import Picamera2

    from robot.camera import FPS, SIZE

    folder.mkdir(parents=True, exist_ok=True)
    state = {"frame": None, "jpeg": None, "found": False}
    lock, stop = threading.Lock(), threading.Event()

    def grab(camera):
        while not stop.is_set():
            frame = camera.capture_array()  # RGB888 is BGR byte order, as OpenCV expects
            corners = find_board(cv2, frame, size, fast=True)
            shown = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(shown, size, corners, True)
            label, colour = ("board found: Enter saves", (0, 200, 0)) if corners is not None else ("board not found", (0, 0, 230))
            cv2.putText(shown, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            with lock:
                state.update(frame=frame, jpeg=jpeg, found=corners is not None)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not stop.is_set():
                    with lock:
                        jpeg = state["jpeg"]
                    if jpeg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                    time.sleep(1 / FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    try:
        server = DualStackServer(("::", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is busy: stop the dashboard first (pkill -f robot.dashboard); "
                         "it also holds the camera") from None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with Picamera2() as camera:
        camera.configure(camera.create_video_configuration(main={"size": SIZE, "format": "RGB888"},
                                                           controls={"FrameRate": FPS}))
        camera.start()
        threading.Thread(target=grab, args=(camera,), daemon=True).start()
        print(f"Live view: http://{socket.gethostname().removesuffix('.local')}.local:{PORT}")
        saved = 0
        try:
            while saved < count:
                input(f"View {saved + 1}/{count}: move the board (or the robot), hold still, Enter: ")
                with lock:
                    frame, found = state["frame"], state["found"]
                if frame is None or not found or find_board(cv2, frame, size) is None:
                    print("   board not found in this view: not saved, try again")
                    continue
                cv2.imwrite(str(folder / f"view_{saved:03d}.png"), frame)
                saved += 1
        finally:
            stop.set()
            server.shutdown()
    print(f"Saved {saved} views in {folder}")


def calibrate(cv2, folder, size, camera_to_body, output):
    points = np.zeros((size[0] * size[1], 3), np.float32)
    points[:, :2] = np.mgrid[:size[0], :size[1]].T.reshape(-1, 2)
    objects, pixels, image_size = [], [], None
    for file in sorted(folder.glob("*.png")):
        image = cv2.imread(str(file))
        if image is None:
            continue
        current = (image.shape[1], image.shape[0])
        if image_size is not None and image_size != current:
            raise SystemExit("all calibration images must have the same size")
        image_size = current
        corners = find_board(cv2, image, size)
        if corners is not None:
            objects.append(points.copy())
            pixels.append(corners)
    if len(objects) < 10:
        raise SystemExit(f"need at least 10 usable checkerboard views; found {len(objects)}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(objects, pixels, image_size, None, None)
    spec = dict(size=image_size, matrix=matrix.tolist(), distortion=distortion.ravel().tolist(),
                camera_to_body=np.array(camera_to_body).reshape(3, 3).tolist(),
                time_offset_s=0.0, smoothing_s=0.25, crop_scale=1.15)
    from robot.stabilization import CameraCalibration
    CameraCalibration(**spec)  # validates it
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Saved {output}; reprojection RMS {rms:.3f} pixels ({len(objects)} views)")


def main():
    import cv2

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path)
    p.add_argument("--capture", type=int, default=0, help="capture this many views (with a live view on port 8000)")
    p.add_argument("--columns", type=int, default=9, help="checkerboard INNER corners")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--camera-to-body", type=float, nargs=9, help="row-major proper rotation: camera (right, down, forward) to torso")
    p.add_argument("--output", type=Path, default=Path("robot/camera_calibration.json"))
    args = p.parse_args()
    if args.columns < 2 or args.rows < 2 or args.capture < 0:
        p.error("invalid checkerboard size or capture count")
    size = (args.columns, args.rows)
    if args.capture:
        capture(cv2, args.images, args.capture, size)
    elif args.camera_to_body is None:
        p.error("supply the measured --camera-to-body rotation (not inferred from checkerboard intrinsics)")
    else:
        calibrate(cv2, args.images, size, args.camera_to_body, args.output)


if __name__ == "__main__":
    main()
