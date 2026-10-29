# IMU aided camera stabilization

Juan has a rigid camera and torso IMU. The optional live `/camera-stabilized.mjpg`
dashboard endpoint removes rapid **rotational** camera movement. The original
`/camera.mjpg` remains available and stabilization is off unless configured.
The implementation is in `robot/stabilization.py`; it uses a separate bounded
worker for warping and JPEG encoding so slow image processing drops frames instead
of accumulating delay or blocking the walking control loop.

Install Raspberry Pi OS packages `python3-opencv`, `python3-numpy`, and
`python3-picamera2`. No training dependencies are needed on the robot.

## Calibration and first run

1. Stop dashboard camera streaming. Capture at least 15 varied checkerboard views:
   `python scripts/calibrate_camera.py /tmp/juan-camera-views --capture 20`.
   Move the board through different distances, tilts, and image corners. The
   default board has 9 by 6 **inner** corners; change `--columns` and `--rows` to match.
2. Measure the camera-to-torso rotation from the rigid mount/CAD. Camera coordinates
   are right, down, forward; torso coordinates are forward, left, up, after the
   IMU's configured axis remapping. The three columns of `camera_to_body` are
   the camera's right, down, and forward unit vectors expressed in torso axes.
   This must be a proper rotation, not a reflection. No mounting orientation is assumed.
3. Run `python scripts/calibrate_camera.py /tmp/juan-camera-views --camera-to-body`
   followed by the nine measured matrix entries in row-major order. This writes
   `robot/camera_calibration.json` and prints reprojection error. Inspect the error
   and held-out checkerboards before trusting the calibration. Do not change lens,
   focus, sensor crop, transform, or image size afterwards without recalibrating.
4. Set `JUAN_CAMERA_CALIBRATION=/absolute/path/to/camera_calibration.json` in the
   dashboard's environment and restart it. Open `/camera-stabilized.mjpg` on the
   dashboard host, or choose **Stabilized** in the dashboard's Camera view selector.
   Raw remains the default. `/state` reports calibration errors and whether recent frames
   had matching IMU data. Without matching data frames pass through uncorrected.

The calibration JSON includes `time_offset_s` (added to the camera timestamp to
query the IMU), `smoothing_s` (default 0.25), and `crop_scale` (default 1.15).
Zero time offset is an initial estimate, not a measured synchronization result.
For calibration, turn the powered robot by hand while stationary, view a distant
textured scene, and compare residual image movement at different offsets. Rotation
should be reduced in both directions without a lagging or reversed correction.
Validate pitch, roll, and yaw independently; wrong signs indicate a mount transform
error. Do this with the motors relaxed before walking tests.

## Timing and limits

The IMU retains about three seconds of bias-corrected raw-gyro orientation samples;
it does not use the policy's 4 Hz filtered gyro. Integration uses measured sample
intervals. Camera `SensorTimestamp` and IMU samples use Linux `CLOCK_BOOTTIME`.
IMU read midpoints approximate sample time; sensor/bus latency still requires offset
calibration. Half the metadata `ExposureTime` is added to `SensorTimestamp` to
approximate the first row's exposure midpoint; a single pose cannot describe all
rows of a rolling-shutter exposure. See the [Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
and [Raspberry Pi's timestamp clarification](https://github.com/raspberrypi/picamera2/issues/821).
The opt-in RGB888 stream is BGR byte order, matching OpenCV, and is supported by
Picamera2's [V4L2 encoder format table](https://github.com/raspberrypi/picamera2/blob/main/picamera2/encoders/v4l2_encoder.py),
which the hardware MJPEG encoder uses. This keeps the original raw MJPEG endpoint
available alongside software-encoded stabilized images on supported Pi hardware.

The image is undistorted and transformed by a calibrated rotation homography toward
a smoothed camera orientation. Intentional turns are followed over time. Cropping
hides some empty borders but large movements can still reveal black pixels.
This does not remove translation/parallax, motion blur, or rolling-shutter wobble;
it is not a replacement for visual-inertial odometry or a global-shutter camera.
Gyro drift and I2C timing limit accuracy. Hardware alignment, visual quality, CPU
load and walking-loop timing must still be measured on Juan; desktop tests verify
geometry and timestamp handling, not achieved onboard stabilization performance.
