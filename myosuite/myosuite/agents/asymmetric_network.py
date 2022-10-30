from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

class FlattenPrefixExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, length: int = 1):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()
        self.length = length
        self._features_dim = length

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)[..., :self.length]

class CustomNetwork(SACPolicy) :
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self.actor_kwargs.copy()
        actor_feature_extractor = self.features_extractor_class(self.observation_space, length = self.features_extractor_kwargs["actor_obs_dim"])
        actor_kwargs.update(dict(features_extractor=actor_feature_extractor, features_dim=actor_feature_extractor.features_dim))
        return Actor(**actor_kwargs).to(self.device)
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self.critic_kwargs.copy()
        critic_feature_extractor = self.features_extractor_class(self.observation_space, length = self.features_extractor_kwargs["critic_obs_dim"])
        critic_kwargs.update(dict(features_extractor=critic_feature_extractor, features_dim=critic_feature_extractor.features_dim))
        return ContinuousCritic(**critic_kwargs).to(self.device)
