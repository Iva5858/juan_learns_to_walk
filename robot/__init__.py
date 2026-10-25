"""Code that runs on Juan's Raspberry Pi (setup and first power-on: docs/bringup.md).

    robot.dashboard     web dashboard (python3 -m robot.dashboard, port 8000): servos, IMU, camera,
                        battery, calibration, joint limits, walking and gestures
    robot.walker        runs the walking policy at 50 Hz inside the dashboard, with the safety checks
    robot.policy        numpy-only copy of the trained policy and the simulator's observation pipeline
    robot.joints        joint map (servo IDs, signs, zero, limits) and the saved calibration
    robot.hiwonder      Hiwonder bus-servo protocol, one Bus per BusLinker
    robot.imu           LSM9DS1 IMU: gravity, gyro and yaw in the simulator's conventions
    robot.camera        Pi camera MJPEG stream (hardware encoder)
    robot.gestures      keyframe gestures; robot.personality plays them, robot.sound is the voice
    robot.selftest      hardware check that moves nothing (python3 -m robot.selftest)

Only needs numpy, pyserial and smbus2 (and picamera2 for the camera), not the training stack.
"""
