"""Scripted joint motions that run outside the policy: the same functions drive the simulation
(play.py) and, later, the real robot.

Boot sequence: the robot powers up standing with straight legs (all joint angles 0), lowers itself
into the home crouch, holds it, and only then hands over to the policy. Moving all joints
linearly together keeps hip pitch + knee + ankle = 0 at every instant (it is 0 in both poses), so
the feet stay flat and the torso upright the whole way.
"""
import numpy as np


def smooth_step(x):
    """0 -> 1 with zero speed at both ends (cosine ease-in/ease-out)."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def boot_targets(start, home, dt, settle=0.5, move=2.5, hold=1.0):
    """Joint targets, one per control step: hold `start`, move to `home`, hold `home`.
    `start` should be the measured angles (not an assumed zero), so the first target is no jump."""
    start, home = np.asarray(start, float), np.asarray(home, float)
    for _ in range(int(settle / dt)):
        yield start
    n = int(move / dt)
    for i in range(1, n + 1):
        yield start + smooth_step(i / n) * (home - start)
    for _ in range(int(hold / dt)):
        yield home
