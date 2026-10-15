"""Events: domain randomization when an episode starts, varied starting states, and shoves during
an episode. Settings: RandomizationConfig.

Random draws that were added after a run was trained only happen when switched on, so older runs
see exactly the random numbers they were trained with.
"""
import mujoco
import numpy as np

from biped_rl.tasks.locomotion.mdp import curriculums


def set_nominal_physics(env):
    """The model's own masses, frictions and servo numbers, no delays, no sensor biases."""
    m, n, r = env.model, env.robot.nominal, env.cfg.robot
    m.body_mass[:], m.body_inertia[:], m.body_ipos[:] = n["mass"], n["inertia"], n["ipos"]
    m.geom_friction[:] = n["friction"]
    m.dof_armature[:], m.dof_frictionloss[:] = n["armature"], n["frictionloss"]
    m.opt.gravity[:] = n["gravity"]
    sv, nu = env.servo, env.nu
    sv.stall = np.full(nu, r.stall_torque)
    sv.speed = np.full(nu, r.no_load_speed)
    sv.kp = np.full(nu, r.kp)
    sv.kd = np.full(nu, r.kd)
    sv.latency_steps = 0
    sv.action_delay = sv.obs_delay = 0
    sv.backlash = np.zeros(nu)
    sv.joint_offset = np.zeros(nu)
    env.domain.friction_scale = 1.0
    env.domain.imu_tilt = np.eye(3)
    env.domain.gyro_bias = np.zeros(3)


def randomize_physics(env):
    """Masses, friction, floor slope, servo strength, speed, gains and delays, sensor biases."""
    m, n, c, rng = env.model, env.robot.nominal, env.cfg.randomization, env.np_random
    nu, di, root, sv = env.nu, env.robot.dof_idx, env.robot.root, env.servo
    scale = rng.uniform(*c.mass_scale, size=m.nbody)
    m.body_mass[:] = n["mass"] * scale
    m.body_inertia[:] = n["inertia"] * scale[:, None]
    m.body_ipos[root] = n["ipos"][root] + rng.uniform(-c.com_offset, c.com_offset, 3)
    env.domain.friction_scale = rng.uniform(*c.friction) / n["friction"][:, 0].max()
    m.geom_friction[:, 0] = n["friction"][:, 0] * env.domain.friction_scale
    m.dof_armature[di] = n["armature"][di] * rng.uniform(*c.armature_scale, size=nu)
    m.dof_frictionloss[di] = n["frictionloss"][di] * rng.uniform(*c.frictionloss_scale, size=nu)
    tilt, heading = np.deg2rad(rng.uniform(0, c.floor_tilt_deg)), rng.uniform(0, 2 * np.pi)
    g = np.linalg.norm(n["gravity"])
    m.opt.gravity[:] = g * np.array([np.sin(tilt) * np.cos(heading),
                                     np.sin(tilt) * np.sin(heading), -np.cos(tilt)])
    r = env.cfg.robot
    if c.voltage_coupled:  # one battery voltage: torque and speed rise and fall together
        volt = rng.uniform(*c.servo_strength)
        sv.stall = r.stall_torque * volt * rng.uniform(0.95, 1.05, nu)
        sv.speed = r.no_load_speed * volt * rng.uniform(0.95, 1.05, nu)
    else:
        sv.stall = r.stall_torque * rng.uniform(*c.servo_strength) * rng.uniform(0.95, 1.05, nu)
        sv.speed = r.no_load_speed * rng.uniform(*c.servo_speed) * rng.uniform(0.95, 1.05, nu)
    sv.kp = r.kp * rng.uniform(*c.kp_scale, size=nu)
    sv.kd = r.kd * rng.uniform(*c.kd_scale, size=nu)
    lat = int(rng.uniform(*c.latency) / m.opt.timestep)
    sv.latency_steps = min(lat, env.frame_skip - 1)
    # The v7 realism below is scaled by the physics-curriculum difficulty (1 = the full ranges).
    k = env.curriculum.difficulty
    upto = lambda span: (span[0], span[0] + k * (span[1] - span[0]))
    if tuple(c.total_mass_scale) != (1.0, 1.0) or tuple(c.torso_mass_scale) != (1.0, 1.0):
        extra = np.full(m.nbody, rng.uniform(*upto(c.total_mass_scale)))
        extra[root] *= rng.uniform(*upto(c.torso_mass_scale))
        m.body_mass[:] *= extra
        m.body_inertia[:] *= extra[:, None]
    if c.action_delay_steps[1] > 0:
        lo, hi = upto(c.action_delay_steps)
        sv.action_delay = int(rng.integers(int(lo), int(round(hi)) + 1))
    if c.obs_delay_steps[1] > 0:
        lo, hi = upto(c.obs_delay_steps)
        sv.obs_delay = int(rng.integers(int(lo), int(round(hi)) + 1))
    if c.backlash[1] > 0:
        sv.backlash = rng.uniform(*upto(c.backlash), size=nu)
    if c.joint_offset > 0:
        sv.joint_offset = rng.uniform(-k * c.joint_offset, k * c.joint_offset, nu)
    if c.imu_tilt_bias_deg > 0:
        bias = k * c.imu_tilt_bias_deg
        roll, pitch = np.radians(rng.uniform(-bias, bias, 2))
        cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
        env.domain.imu_tilt = (np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
                               @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]))
    if c.gyro_bias > 0:
        env.domain.gyro_bias = rng.uniform(-k * c.gyro_bias, k * c.gyro_bias, 3)


def vary_start_state(env):
    """Start from a perturbed crouch, already moving (as in Berkeley Humanoid Lite): joint angles =
    home x a per-joint scale, plus a torso velocity. Then the robot is put back on the floor."""
    c, d, rng, robot = env.cfg.randomization, env.data, env.np_random, env.robot
    if tuple(c.reset_joint_scale) == (1.0, 1.0) and c.reset_lin_vel == 0 and c.reset_ang_vel == 0:
        return  # off (runs up to juan_v7): no extra random draws
    qi = robot.qpos_idx
    noise = d.qpos[qi] - robot.home_joint
    d.qpos[qi] = np.clip(robot.home_joint * rng.uniform(*c.reset_joint_scale, env.nu) + noise,
                         robot.joint_lo, robot.joint_hi)
    d.qvel[0:2] += rng.uniform(-c.reset_lin_vel, c.reset_lin_vel, 2)
    d.qvel[3:6] += rng.uniform(-c.reset_ang_vel, c.reset_ang_vel, 3)
    mujoco.mj_forward(env.model, d)
    env.ground()


def push_robot(env, s, terminated, truncated):
    """Random shoves every push_interval seconds: a velocity kick to the torso. With the push
    curriculum, from one of 8 directions relative to the robot at that direction's strength, and
    its outcome (fell within 2 s or not) feeds the curriculum; otherwise uniform per axis."""
    rc, ep, cur, rng = env.cfg.randomization, env.episode, env.curriculum, env.np_random
    if cur.pending_push is not None and (terminated or truncated or ep.t - cur.pending_push[1] >= 2.0):
        curriculums.score_push(env, cur.pending_push[0], stayed_up=not terminated)
    if rc.enabled and rc.push_velocity > 0 and ep.t >= ep.next_push:
        if rc.push_curriculum:
            k = int(rng.integers(8))
            yaw = np.arctan2(s.R[1, 0], s.R[0, 0])
            angle = yaw + (k + rng.uniform()) * np.pi / 4  # direction k: [k*45, (k+1)*45) deg
            strength = min(cur.push_level[k] * rng.uniform(0.75, 1.1), rc.push_velocity)
            push = strength * np.array([np.cos(angle), np.sin(angle)])
            cur.pending_push = (k, ep.t)
        else:
            push = rng.uniform(-rc.push_velocity, rc.push_velocity, 2)
        env.data.qvel[0:2] += push
        ep.last_push = ep.t
        ep.last_push_vec = push  # for drawing the shove in videos
        ep.next_push += rng.uniform(*rc.push_interval)
