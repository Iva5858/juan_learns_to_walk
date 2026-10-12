"""When an episode ends early. Both end it with RewardWeights.termination."""


def fell(env, s):
    """Anything but a foot touches the ground, the torso tilts more than 53 deg, or the centre of
    mass sinks below half its home height."""
    return bool(s.bad_contact or s.grav[2] > -0.6 or env.com_height() < 0.5 * env.robot.home_height)


def broke(env, s):
    """A hardware limit was crossed (the real robot could break): a landing above break_impact x
    body weight, or a joint forced past break_joint_speed x servo speed while the servo pushes back
    with more than break_requires_load x stall torque (0: any such over-speed, runs up to v6)."""
    w = env.cfg.rewards
    impact_bw = s.peak_force.max() / env.robot.weight
    overspeed = s.loaded_overspeed if w.break_requires_load > 0 else s.peak_speed_ratio > w.break_joint_speed
    return bool(impact_bw > w.break_impact or overspeed)
