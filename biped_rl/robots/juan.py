"""Juan: 43 cm, 1.75 kg (CAD; ~2 kg expected built) biped with 4 joints per leg, driven by 8
Hiwonder HTD-35H serial bus servos.

The model, juan/rl_scene.xml, is made from the Onshape assembly by
scripts/model/postprocess_onshape.py: part masses from the BoM, the joint sign conventions below,
box feet with rubber pads, leg capsules, the IMU site and the home keyframe.

Joint sign conventions (torso frame: x forward, y left, z up), positive rotation:
    hip roll    left leg outward, right leg inward
    hip pitch   leg swings back
    knee        shin swings back (the knee bends)
    ankle       toes down
Legs straight = all angles 0. Home (crouch) = hip pitch -22.9 deg, knee +45.8 deg, ankle -22.9 deg:
the three sum to 0, so the feet stay parallel to the torso.
"""

MODEL_PATH = "juan/rl_scene.xml"
FOOT_BODIES = ("left_foot_link", "right_foot_link")  # (left, right)

KNEE_JOINTS = ("left_knee", "right_knee")
HIP_ROLL_JOINTS = ("left_hip_roll", "right_hip_roll")
# (left joint, right joint, sign): -1 where mirroring the motion flips the angle. Checked on the
# model before every training run (biped_rl.symmetry.check_robot_mirror).
MIRROR_PAIRS = (("left_hip_roll", "right_hip_roll", -1), ("left_hip_pitch", "right_hip_pitch", 1),
                ("left_knee", "right_knee", 1), ("left_ankle", "right_ankle", 1))

# Hiwonder HTD-35H at 11.1 V. Only the stall torque and the speed are published; the internal
# gains are estimates until measured on the bench.
STALL_TORQUE = 3.43   # Nm (35 kg*cm x 0.0981)
NO_LOAD_SPEED = 5.8   # rad/s (0.18 s per 60 deg); torque falls linearly to 0 at this speed
KP = 15.0             # Nm/rad (estimate: full torque at ~13 deg of error)
KD = 0.2              # Nm*s/rad (estimate)
