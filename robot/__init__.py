"""Code that runs on Juan's Raspberry Pi: servo bus, IMU, camera, and the robot dashboard.

    robot.joints      joint map (servo IDs, signs, zero, limits) and the saved calibration
    robot.hiwonder    Hiwonder bus-servo protocol, one Bus per BusLinker
    robot.bno055      BNO055 IMU (IMU mode, burst reads), in the simulator's conventions
    robot.camera      Pi camera MJPEG stream (hardware encoder)
    robot.dashboard   web dashboard: monitoring, calibration, joint limits, boot pose

Only needs pyserial and smbus2 (and picamera2 for the camera), not the training stack.
"""
