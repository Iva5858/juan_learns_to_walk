"""When an episode ends early. Both end it with RewardWeights.termination."""
import numpy as np


def fell(env, s):
    """Anything but a foot touches the ground, the torso tilts more than 53 deg, or the centre of
    mass sinks below half its home height."""
    return bool(s.bad_contact or s.grav[2] > -0.6 or env.com_height() < 0.5 * env.robot.home_height
                or strayed(env, s))


def strayed(env, s):
    """Imitation's early termination (DeepMimic): the joints more than imitation_terminate_deg (rms) from the
    reference gait ends the episode like a fall, so leaving the style costs as much as falling. Without it,
    juan_v9 drifted off the reference and flipped its knees forward within 30M steps."""
    limit = env.cfg.rewards.imitation_terminate_deg
    if not limit or not env.cfg.command.reference_gait:
        return False
    from biped_rl.tasks.locomotion.mdp import reference
    return bool(np.degrees(np.sqrt(np.mean((s.q - reference.pose_now(env)) ** 2))) > limit)


def broke(env, s):
    """A hardware limit was crossed (the real robot could break): a landing above break_impact x
    body weight, or a joint forced past break_joint_speed x servo speed while the servo pushes back
    with more than break_requires_load x stall torque."""
    w = env.cfg.rewards
    return bool(s.peak_force.max() / env.robot.weight > w.break_impact or s.loaded_overspeed)
