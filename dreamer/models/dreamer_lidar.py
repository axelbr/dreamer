from dataclasses import dataclass
from typing import Dict

import gym
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow_probability import distributions as tfd
from dreamer.models import RSSM, ConvLidarEncoder, ConvLidarDecoder, DenseDecoder
from dreamer.models.actor_critic import ActorCritic, ActionDecoder
from rlephant import Dataset

class Dreamer(tf.Module):
    @dataclass
    class Config:
        B: int = 50
        L: int = 15
        beta: float = 1.0

    def __init__(self, config: Config, env: gym.Env):
        self._config = config
        self._dynamics = RSSM(stoch=30, deter=200, hidden=200)
        self._encoder = ConvLidarEncoder(embedding_size=16)
        self._decoder = ConvLidarDecoder(output_size=1080)
        self._reward = DenseDecoder(shape=(), layers=2, units=400, act='elu', dist='normal')

        self._action = ActionDecoder(size=env.action_space.shape[0],
                                     layers=4,
                                     units=400,
                                     dist='tanh_normal',
                                     init_std=5.0,
                                     act='elu')
        self._value = DenseDecoder(shape=3, layers=3, units=400, act='elu')

        self._dynamics_optimizer = Adam()

        modules = [self._encoder, self._dynamics, self._decoder, self._reward]
        self._dynamics_variables = tf.nest.flatten([module.trainable_variables for module in modules])




    def __call__(self, obs, state=None):
        pass

    def learn_dynamics(self, dataset: Dataset):

        # Steps plus one, because we need the previous action of the first observation!
        sequence_length = self._config.L + 1
        data_sequences = dataset.sample_sequence_batch(count=self._config.B, sequence_length=sequence_length)

        observation_batch = tf.cast(data_sequences.observations['lidar'], dtype=tf.float32)
        action_batch = tf.cast(data_sequences.actions['action'], dtype=tf.float32)
        reward_batch = tf.cast(data_sequences.rewards, dtype=tf.float32)

        with tf.GradientTape() as model_tape:

            # Encode observations
            obs_embedding_batch = self._encode_observations(observation_batch)

            # Compute model states for L steps
            posterior_states, prior_states = self._dynamics.observe(obs_embedding_batch, action_batch)

            # Drop first timesteps, only needed for state computation
            observation_batch = observation_batch[:, 1:, :]
            reward_batch = reward_batch[:, 1:]

            loss = self._compute_dynamics_loss(
                posteriors=posterior_states,
                priors=prior_states,
                observation=observation_batch,
                rewards=reward_batch
            )


        gradients = model_tape.gradient(loss, self._dynamics_variables)
        self._dynamics_optimizer.apply_gradients(zip(gradients, self._dynamics_variables))




            #
            # observation_loss = tf.reduce_mean(image_pred.log_prob(observation_batch))
            # reward_loss = tf.reduce_mean(reward_pred.log_prob(reward_batch))
            #
            # prior_dist = self._dynamics.get_dist(prior)
            # post_dist = self._dynamics.get_dist(post)
            # div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
            # div = tf.maximum(div, self._c.free_nats)
            # model_loss = self._c.kl_scale * div - sum(likes.values())
            # model_loss /= float(self._strategy.num_replicas_in_sync)

    def _encode_observations(self, observations: tf.Tensor) -> tf.Tensor:
        batched_obs = tf.reshape(observations, shape=(-1, *observations.shape[2:]))
        embed = self._encoder(batched_obs)
        batch_size = observations.shape[0]
        sequence_length = observations.shape[1]
        embed = tf.reshape(embed, shape=(batch_size, sequence_length, *embed.shape[1:]))
        return embed

    def _compute_dynamics_loss(self, posteriors: Dict, priors: Dict, observation: tf.Tensor, rewards: tf.Tensor) -> tf.Tensor:

        # Construct state and state conditional from posteriors and priors.
        state = self._dynamics.get_state(posteriors)
        posterior_distribution = self._dynamics.get_dist(posteriors)
        prior_distribution = self._dynamics.get_dist(priors)

        # Predict reconstruction and reward conditionals, based on states
        reconstruction_distribution = self._compute_reconstruction_distribution(states=state)
        reward_distribution = self._reward(state)

        # Compute observation and reward losses
        observation_loss = reconstruction_distribution.log_prob(observation)
        reward_loss = reward_distribution.log_prob(rewards)
        regularization = -self._config.beta * tfd.kl_divergence(posterior_distribution, prior_distribution)

        batch_loss = tf.reduce_sum(observation_loss + reward_loss + regularization, axis=1)
        loss = -tf.reduce_mean(batch_loss)
        return loss

    def _compute_reconstruction_distribution(self, states: tf.Tensor) -> tfd.Distribution:
        batched_states = tf.reshape(states, shape=(-1, *states.shape[2:]))
        distribution = self._decoder(batched_states)
        batch_size = states.shape[0]
        sequence_length = states.shape[1]
        distribution = tfd.BatchReshape(distribution, batch_shape=(batch_size, sequence_length))
        return distribution

    def learn_behaviour(self):
        pass

    def interact_with_env(self, env: gym.Env):
        pass
