"""Asymmetric actor-critic for Stable-Baselines3.

The critic sees the full observation (including privileged simulator state such as
true velocity, foot contacts and friction); the actor sees only the first
`actor_obs_dim` entries, i.e. what the real robot can sense. The privileged entries
are zeroed for the actor, so on hardware you can feed zeros there.

With `mirror` (obs_perm, obs_sign, act_perm, act_sign from biped_rl.symmetry.mirror_maps),
the actor is left-right symmetric by construction:
    mean_action(s) = 1/2 * ( f(s) + M_a f(M_s s) )
so in the mirror image of a situation it does exactly the mirror image of what it does now:
no favoured leg, no built-in veer. The observations must be normalized with symmetric
statistics (biped_rl.symmetry.SymmetricRunningMeanStd) for this to hold on raw sensor values.
On the robot, run the same forward pass (it is part of the network: export it as is).
"""
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActorMask(BaseFeaturesExtractor):
    def __init__(self, observation_space, actor_obs_dim):
        super().__init__(observation_space, observation_space.shape[0])
        mask = th.zeros(observation_space.shape[0])
        mask[:actor_obs_dim] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, obs):
        return obs * self.mask


class AsymmetricActorCritic(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, actor_obs_dim, mirror=None, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # The mask has no parameters, so swapping it in after the optimizer is built is safe.
        self.pi_features_extractor = ActorMask(observation_space, actor_obs_dim)
        self.features_extractor = self.pi_features_extractor
        self.mirror = mirror is not None
        if self.mirror:
            obs_perm, obs_sign, act_perm, act_sign = mirror
            self.register_buffer("obs_perm", th.as_tensor(obs_perm, dtype=th.long))
            self.register_buffer("obs_sign", th.as_tensor(obs_sign, dtype=th.float32))
            self.register_buffer("act_perm", th.as_tensor(act_perm, dtype=th.long))
            self.register_buffer("act_sign", th.as_tensor(act_sign, dtype=th.float32))

    def _actor_net(self, obs):
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        return self.action_net(self.mlp_extractor.forward_actor(features))

    def _action_distribution(self, obs):
        mean = self._actor_net(obs)
        log_std = self.log_std
        if self.mirror:
            mirrored_mean = self._actor_net(obs[:, self.obs_perm] * self.obs_sign)
            mean = 0.5 * (mean + mirrored_mean[:, self.act_perm] * self.act_sign)
            log_std = 0.5 * (log_std + log_std[self.act_perm])  # same exploration noise on both legs
        return self.action_dist.proba_distribution(mean, log_std)

    def forward(self, obs, deterministic=False):
        distribution = self._action_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), self.predict_values(obs), log_prob

    def evaluate_actions(self, obs, actions):
        distribution = self._action_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        return self._action_distribution(obs)
