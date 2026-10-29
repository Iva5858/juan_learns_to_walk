"""Calibrate intrinsics from checkerboard PNGs captured at the live stream size/mode."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import cv2

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", type=Path)
    p.add_argument("--capture", type=int, default=0, help="capture this many views on the Pi; stop dashboard camera first")
    p.add_argument("--columns", type=int, default=9, help="checkerboard INNER corners")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--camera-to-body", type=float, nargs=9, help="row-major proper rotation: camera (right, down, forward) to torso")
    p.add_argument("--output", type=Path, default=Path("robot/camera_calibration.json"))
    args = p.parse_args()
    if args.columns < 2 or args.rows < 2 or args.capture < 0:
        p.error("invalid checkerboard size or capture count")
    if args.capture:
        from picamera2 import Picamera2

        from robot.camera import FPS, SIZE

        args.images.mkdir(parents=True, exist_ok=True)
        with Picamera2() as camera:
            camera.configure(camera.create_video_configuration(main={"size": SIZE, "format": "RGB888"},
                                                               controls={"FrameRate": FPS}))
            camera.start()
            for i in range(args.capture):
                input(f"Move checkerboard to view {i+1}/{args.capture}, then press Enter: ")
                cv2.imwrite(str(args.images / f"view_{i:03d}.png"), camera.capture_array())
        return
    if args.camera_to_body is None:
        p.error("supply measured --camera-to-body rotation (not inferred from checkerboard intrinsics)")
    points = np.zeros((args.columns * args.rows, 3), np.float32)
    points[:, :2] = np.mgrid[:args.columns, :args.rows].T.reshape(-1, 2)
    objects, pixels = [], []
    size = None
    for file in sorted(args.images.glob("*.png")):
        image = cv2.imread(str(file))
        if image is None:
            continue
        current = (image.shape[1], image.shape[0])
        if size is not None and size != current:
            p.error("all calibration images must have the same size")
        size = current
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(grey, (args.columns, args.rows))
        if found:
            refined = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objects.append(points.copy())
            pixels.append(refined)
    if len(objects) < 10:
        p.error(f"need at least 10 usable checkerboard views; found {len(objects)}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(objects, pixels, size, None, None)
    spec = dict(size=size, matrix=matrix.tolist(), distortion=distortion.ravel().tolist(),
                camera_to_body=np.array(args.camera_to_body).reshape(3, 3).tolist(),
                time_offset_s=0.0, smoothing_s=0.25, crop_scale=1.15)
    from robot.stabilization import CameraCalibration
    CameraCalibration(**spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"Saved {args.output}; reprojection RMS {rms:.3f} pixels ({len(objects)} views)")


if __name__ == "__main__":
    main()
