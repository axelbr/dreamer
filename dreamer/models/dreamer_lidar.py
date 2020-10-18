from dataclasses import dataclass

import gym
import tensorflow as tf
from dreamer.models import RSSM, ConvLidarEncoder, ConvLidarDecoder, DenseDecoder
from dreamer.models.actor_critic import ActorCritic, ActionDecoder
from rlephant import Dataset

class Dreamer(tf.Module):
    @dataclass
    class Config:
        dataset_path: str
        B: int
        L: int

    def __init__(self, config: Config, env: gym.Env):
        self._config = config
        self._dataset = Dataset(filename=config.dataset_path)
        self._dynamics = RSSM(stoch=30, deter=200, hidden=200)
        self._encoder = ConvLidarEncoder(embedding_size=16)
        self._decoder = ConvLidarDecoder(output_size=1080)
        self._action = ActionDecoder(size=env.action_space.shape[0],
                                     layers=4,
                                     units=400,
                                     dist='tanh_normal',
                                     init_std=5.0,
                                     act='elu')
        self._value = DenseDecoder(shape=3, layers=3, units=400, act='elu')


    def __call__(self, obs, state=None):
        pass

    def learn_dynamics(self, dataset: Dataset):
        data_sequences = dataset.sample_sequences(count=self._config.B, sequence_length=self._config.L)

    def learn_behaviour(self):
        pass

    def interact_with_env(self, env: gym.Env):
        pass
