"""Left-right mirror symmetry for the policy.

Juan is basically left-right symmetric, so a good policy is too: in the mirror image of a situation it
should do the mirror image of what it does now. Instead of rewarding symmetry (which bent the
gait: v4 skipped), the actor is symmetric by construction (see policy.py):

    action(s) = 1/2 * ( f(s) + M_a f(M_s s) )

where M_s mirrors an observation and M_a mirrors an action. Both are "swap left and right, flip
the sign of sideways things": each is a permutation with signs, and applying it twice gives back
the original. This module builds those two maps from the environment and checks them on the
robot model.

Observation normalization must be symmetric too (normalize(M_s s) = M_s normalize(s)), which
holds when the running statistics are symmetric: SymmetricRunningMeanStd keeps them so.
"""
import mujoco
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd


def joint_mirror(env):
    """Per actuated joint: index of its mirror partner and the sign (-1 where mirroring the
    motion flips the angle, e.g. hip roll)."""
    perm = np.arange(env.nu)
    sign = np.ones(env.nu)
    for l, r, s in zip(env.robot.mirror_l, env.robot.mirror_r, env.robot.mirror_sign):
        perm[l], perm[r] = r, l
        sign[l] = sign[r] = s
    if len(env.robot.mirror_l) * 2 != env.nu:
        raise ValueError("every actuated joint needs a mirror partner in RobotConfig.mirror_pairs")
    return perm, sign


def mirror_maps(env):
    """(obs_perm, obs_sign, act_perm, act_sign): mirrored[i] = sign[i] * x[perm[i]].

    Actor observation layout (see BipedEnv._sensor_frame / _get_obs), repeated per history frame:
        gravity in torso frame (3)   -> (x, -y, z)        a vector: its sideways part flips
        gyro (3)                     -> (-x, y, -z)       an angular rate: roll and yaw flip
        joint angles, joint speeds, previous actions (nu each, if joint feedback) -> partner, sign
    then the command (forward, sideways, turn rate) -> (vx, -vy, -yaw rate) [omni] or vx [forward],
    the gait clock (sin, cos) -> (-sin, -cos): mirroring swaps the legs, i.e. half a cycle,
    and the heading error (sin, cos) -> (-sin, cos). The privileged part (critic only) is
    left as it is: the actor never sees it.
    """
    jp, js = joint_mirror(env)
    nu, cfg = env.nu, env.cfg
    perm, sign = [], []

    def block(p, s):
        base = len(perm)
        perm.extend(base + np.asarray(p))
        sign.extend(s)

    for _ in range(cfg.history):
        block([0, 1, 2], [1, -1, 1])
        block([0, 1, 2], [-1, 1, -1])
        for _ in range(3 if cfg.robot.joint_feedback else 1):
            block(jp, js)
    block([0, 1, 2], [1, -1, -1]) if env.omni else block([0], [1])
    block([0, 1], [-1, -1])
    block([0, 1], [-1, 1])
    assert len(perm) == env.actor_obs_dim, (len(perm), env.actor_obs_dim)
    total = env.observation_space.shape[0]
    perm.extend(range(env.actor_obs_dim, total))
    sign.extend([1.0] * (total - env.actor_obs_dim))
    return [int(i) for i in perm], [float(s) for s in sign], [int(i) for i in jp], [float(s) for s in js]


def mirror(x, perm, sign):
    return np.asarray(sign) * np.asarray(x)[..., perm]


class SymmetricRunningMeanStd(RunningMeanStd):
    """Observation statistics that every sample updates twice: as seen and mirrored. Then the
    mean and variance are exactly mirror-symmetric, so mirroring commutes with normalizing."""

    def __init__(self, perm, sign, epsilon=1e-4, shape=()):
        super().__init__(epsilon=epsilon, shape=shape)
        self.perm, self.sign = np.asarray(perm), np.asarray(sign)

    def update(self, arr):
        super().update(np.concatenate([arr, mirror(arr, self.perm, self.sign)]))


def check_robot_mirror(env, trials=20, seed=0):
    """Check the joint mirror map on the model itself: bending the joints by d from the home pose
    and by M d must give mirror-image robots, i.e. every left part's centre of mass where the
    mirror image of its right twin's is, in the torso frame. A wrong sign in mirror_pairs fails
    here (by 5-15 cm). Centres of mass, not body frames: CAD exports place the frames of
    mirrored parts differently (on Juan, thigh and shin frames differ by ~5 cm), which changes
    nothing physical. Returns the largest mismatch in metres (Juan: ~1 mm, from the CAD)."""
    m = env.model
    d = mujoco.MjData(m)
    jp, js = joint_mirror(env)
    names = [m.body(i).name for i in range(m.nbody)]
    twin = {}
    for i, n in enumerate(names):
        for a, b in (("left", "right"), ("l_", "r_")):
            if n.startswith(a):
                j = names.index(b + n[len(a):]) if b + n[len(a):] in names else -1
                if j >= 0:
                    twin[i], twin[j] = j, i
    if not twin:
        raise ValueError("no left/right body pairs found by name")

    def part_coms(delta):
        d.qpos[:] = env.data.qpos
        d.qpos[0:7] = [0, 0, 0, 1, 0, 0, 0]  # torso at the origin, upright: positions are in its frame
        d.qpos[env.robot.qpos_idx] = env.robot.home_joint + delta
        mujoco.mj_kinematics(m, d)
        return d.xipos.copy()

    rng = np.random.default_rng(seed)
    worst = 0.0
    flip = np.array([1.0, -1.0, 1.0])
    for _ in range(trials + 1):
        delta = rng.uniform(-0.3, 0.3, env.nu) if _ else np.zeros(env.nu)  # first trial: home itself
        a, b = part_coms(delta), part_coms(mirror(delta, jp, js))
        for i, j in twin.items():
            worst = max(worst, float(np.abs(b[i] - flip * a[j]).max()))
    return worst
