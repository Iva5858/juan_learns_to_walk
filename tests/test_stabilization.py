import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot.stabilization import CameraCalibration, OrientationHistory, Stabilizer, rotation


def calibration(**kwargs):
    return CameraCalibration(size=(640, 480), matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                             distortion=[0]*5, camera_to_body=np.eye(3), **kwargs)


def test_gyro_integration_and_timestamp_interpolation():
    history = OrientationHistory()
    for i in range(101):
        history.append(i * .01, [0, 0, 1])
    q = history.at(.505)
    expected = np.array([np.cos(.505), np.sin(.505), 0])
    np.testing.assert_allclose(rotation(q) @ [1, 0, 0], expected, atol=1e-6)
    assert history.at(-.01) is None
    assert history.at(1.01) is None
    history.append(2, [0, 0, 1])
    assert history.at(.5) is None


def test_warp_restores_world_ray_after_rotation():
    history = OrientationHistory()
    for i in range(21):
        history.append(i * .01, [0, 0, 2])
    cal = calibration(smoothing_s=1000)
    stabilizer = Stabilizer(cal, history)
    np.testing.assert_allclose(stabilizer.homography(0), np.eye(3), atol=1e-12)
    h = stabilizer.homography(.2)
    ray = np.array([.1, .05, 1.])
    current_pixel = cal.matrix @ rotation(history.at(.2)).T @ ray
    restored = h @ current_pixel
    expected = cal.matrix @ rotation(stabilizer.smoothed).T @ ray
    np.testing.assert_allclose(restored/restored[2], expected/expected[2], atol=1e-8)
    assert stabilizer.homography(10) is None


def test_calibration_rejects_reflected_mount():
    with pytest.raises(ValueError, match="proper rotation"):
        CameraCalibration((640, 480), np.eye(3), [0]*5, np.diag([1, 1, -1]))


@pytest.mark.parametrize("settings", [dict(crop_scale=float("nan")), dict(smoothing_s=0),
                                      dict(time_offset_s=float("inf"))])
def test_invalid_calibration_settings(settings):
    with pytest.raises(ValueError):
        calibration(**settings)


def test_gap_resets_smoothing_even_when_frames_arrive_within_half_second():
    history = OrientationHistory()
    for i in range(11):
        history.append(i * .01, [0, 0, 3])
    stabilizer = Stabilizer(calibration(), history)
    stabilizer.homography(0)
    assert not np.allclose(stabilizer.homography(.1), np.eye(3))
    history.append(.2, [0, 0, 0])  # 100 ms missed samples resets gyro origin
    history.append(.21, [0, 0, 0])
    np.testing.assert_allclose(stabilizer.homography(.205), np.eye(3), atol=1e-12)


def test_stream_restart_discards_old_smoothed_reference():
    history = OrientationHistory()
    for i in range(201):
        history.append(i * .01, [0, 0, 1])
    stabilizer = Stabilizer(calibration(), history)
    stabilizer.homography(0)
    stabilizer.homography(.1)
    # IMU continued while the image worker/stream stopped; do not correct toward
    # the old viewing direction when frames resume.
    np.testing.assert_allclose(stabilizer.homography(1.5), np.eye(3), atol=1e-12)


def test_camera_opt_in_wiring_and_exposure_timestamp(monkeypatch, tmp_path):
    from robot import camera

    monkeypatch.setattr(camera.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(camera.Camera, "available", True)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.delenv("JUAN_CAMERA_CALIBRATION", raising=False)
    instance = camera.Camera()
    imu = SimpleNamespace(orientation=OrientationHistory())
    instance.configure_stabilization(imu)
    assert instance.stabilizer is None
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(dict(size=[640, 480], matrix=[[500, 0, 320], [0, 500, 240], [0, 0, 1]],
                                    distortion=[0]*5, camera_to_body=np.eye(3).tolist())))
    instance.configure_stabilization(imu, path)
    assert instance.stabilizer is not None
    fake = SimpleNamespace(create_video_configuration=lambda **kwargs: kwargs,
                           configure=lambda cfg: setattr(instance, "test_config", cfg),
                           start_recording=lambda *args: None)
    monkeypatch.setattr(camera, "Picamera2", lambda: fake)
    monkeypatch.setattr(camera, "MJPEGEncoder", lambda: object(), raising=False)
    monkeypatch.setattr(camera, "FileOutput", lambda frames: frames, raising=False)
    assert instance._start()
    assert instance.test_config["main"]["format"] == "RGB888"
    assert fake.post_callback == instance._capture
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    request = SimpleNamespace(get_metadata=lambda: dict(SensorTimestamp=1_000_000_000, ExposureTime=20_000),
                              make_array=lambda name: frame)
    instance._capture(request)
    copied, timestamp = instance._pending.get_nowait()
    assert timestamp == pytest.approx(1.01)
    assert copied is not frame


def test_failed_camera_start_closes_device_and_can_retry(monkeypatch):
    from robot import camera

    monkeypatch.setattr(camera.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(camera.Camera, "available", True)
    instance = camera.Camera()
    closed = []

    def configure(_):
        raise RuntimeError("test configuration failure")

    fake = SimpleNamespace(create_video_configuration=lambda **kwargs: kwargs,
                           configure=configure, close=lambda: closed.append(True))
    monkeypatch.setattr(camera, "Picamera2", lambda: fake)
    assert not instance._start()
    assert closed == [True]
    assert instance.cam is None
    assert "configuration failure" in instance.error
