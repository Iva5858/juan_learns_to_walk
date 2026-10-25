"""PPO setup (Stable-Baselines3): hyperparameters and a fresh model with its normalization."""
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from biped_rl.policy import AsymmetricActorCritic
from biped_rl.symmetry import SymmetricRunningMeanStd, check_robot_mirror, mirror_maps
from biped_rl.tasks.locomotion import BipedEnv

# Samples per PPO update (n_steps x n_envs), kept constant whatever the worker count (2048 x 18). More
# workers than cores trains faster on the Mac (36: +43% steps/s), and with the same update size the
# learning per update is unchanged.
ROLLOUT_STEPS = 36864


def n_steps_for(n_envs):
    return max(1, ROLLOUT_STEPS // n_envs)


PPO_CFG = dict(
    n_steps=2048,          # x n_envs samples per update (replaced by n_steps_for(n_envs))
    batch_size=4096,
    n_epochs=5,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
)
POLICY_CFG = dict(
    net_arch={"pi": [256, 128, 64], "vf": [512, 256, 128]},
    log_std_init=-1.0,
    activation_fn=torch.nn.ELU,
)


def new_model(cfg, env, tensorboard_log=None):
    """A fresh PPO model and the observation/reward normalization around `env` (a VecEnv).
    Mirror-symmetric (policy and normalization). Returns (model, env)."""
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=PPO_CFG["gamma"])
    probe = BipedEnv(cfg.to_dict())
    mismatch = check_robot_mirror(probe)
    print(f"mirror check: left/right part positions agree to {1000 * mismatch:.1f} mm", flush=True)
    if mismatch > 0.005:
        raise SystemExit("RobotConfig.mirror_pairs does not mirror this robot (a joint sign is wrong, "
                         "or the robot is not symmetric)")
    mirror = mirror_maps(probe)
    env.obs_rms = SymmetricRunningMeanStd(mirror[0], mirror[1], shape=env.observation_space.shape)
    model = PPO(AsymmetricActorCritic, env, **{**PPO_CFG, "n_steps": n_steps_for(env.num_envs)},
                policy_kwargs={"actor_obs_dim": probe.actor_obs_dim, "mirror": mirror, **POLICY_CFG},
                tensorboard_log=tensorboard_log, device="cpu", verbose=1)
    return model, env
