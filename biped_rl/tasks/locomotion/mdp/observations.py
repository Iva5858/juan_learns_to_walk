"""Observations. The policy (actor) sees only what the real robot can measure; the critic also
sees privileged simulator state.

Actor observation, oldest history frame first:
    `history` frames of [gravity in torso frame (3), gyro (3), joint angles - home (nu),
                         0.1 x joint speeds (nu), previous action (nu)]
    command: (forward, sideways, turn rate) in omni mode, forward only in forward mode
    gait clock (sin, cos), (0, 0) when standing
    heading error (sin, cos)
Privileged (critic only; zero-filled on the robot):
    path offset, body velocity (3), height below home, foot contacts (2), sole clearances (2),
    true joint angles - home (nu), true 0.1 x joint speeds (nu), floor friction scale
"""
import numpy as np

from biped_rl.tasks.locomotion.mdp import commands


def read_sensors(env):
    """What the real robot measures this control step: IMU gravity direction and gyro (with a
    per-episode bias and noise), and servo angles (with a calibration offset, the servo's 0.24 deg
    resolution and noise) and speeds."""
    d, c, rng, robot = env.data, env.cfg.randomization, env.np_random, env.robot
    noisy = c.enabled
    noise = lambda scale, n: rng.normal(0, scale, n) if noisy else 0.0
    parts = [env.domain.imu_tilt @ env.gravity_body() + noise(c.noise_gravity, 3),
             d.qvel[3:6] + env.domain.gyro_bias + noise(c.noise_gyro, 3)]  # free-joint ang. vel. is in body frame
    if env.cfg.robot.joint_feedback:
        q = d.qpos[robot.qpos_idx] - env.servo.joint_offset  # what a miscalibrated servo reports
        if noisy and c.joint_resolution > 0:
            q = np.round(q / c.joint_resolution) * c.joint_resolution
        parts += [q - robot.home_joint + noise(c.noise_joint_pos, env.nu),
                  0.1 * (d.qvel[robot.dof_idx] + noise(c.noise_joint_vel, env.nu))]
    return np.concatenate(parts)


def history_frames(env):
    """Last `history` frames, oldest first: each is [sensor reading, action of that step].
    Readings are obs_delay steps old; the oldest available one stands in at episode start."""
    ep, frames = env.episode, []
    for j in reversed(range(env.cfg.history)):
        back = min(1 + j + env.servo.obs_delay, len(ep.sensor_buf))
        frames.append(np.concatenate([ep.sensor_buf[-back], ep.action_buf[-1 - j]]))
    return frames


def compute(env):
    """The full observation: [actor part | privileged part]."""
    d, cmd, ep, robot, rc = env.data, env.command, env.episode, env.robot, env.cfg.randomization
    ph = 2 * np.pi * commands.phase(env)
    yaw_err = commands.heading_error(env) + (env.np_random.normal(0, rc.noise_yaw) if rc.enabled else 0.0)
    command = cmd.vec if env.omni else [cmd.forward]
    clock = [0.0, 0.0] if cmd.standing else [np.sin(ph), np.cos(ph)]  # stopped clock = stand
    actor = np.concatenate([*history_frames(env), command, clock, [np.sin(yaw_err), np.cos(yaw_err)]])
    privileged = np.concatenate([
        [commands.lateral_offset(env)],
        env.root_rot().T @ d.qvel[0:3],
        [env.com_height() - robot.home_height],
        ep.contact,
        ep.clearance,
        d.qpos[robot.qpos_idx] - robot.home_joint,
        0.1 * d.qvel[robot.dof_idx],
        [env.domain.friction_scale],
    ])
    return np.concatenate([actor, privileged])


def actor_dim(cfg, nu, omni):
    frame = 6 + (2 * nu if cfg.robot.joint_feedback else 0) + nu
    return cfg.history * frame + (3 if omni else 1) + 2 + 2


def privileged_dim(nu):
    return 1 + 3 + 1 + 2 + 2 + 2 * nu + 1
