"""Asymmetric actor-critic for Stable-Baselines3.

The critic sees the full observation (including privileged simulator state such as
true velocity, foot contacts and friction); the actor sees only the first
`actor_obs_dim` entries, i.e. what the real robot can sense. The privileged entries
are zeroed for the actor, so on hardware you can feed zeros there.
"""
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
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
    def __init__(self, observation_space, action_space, lr_schedule, actor_obs_dim, **kwargs):
        kwargs["share_features_extractor"] = False
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # The mask has no parameters, so swapping it in after the optimizer is built is safe.
        self.pi_features_extractor = ActorMask(observation_space, actor_obs_dim)
        self.features_extractor = self.pi_features_extractor
