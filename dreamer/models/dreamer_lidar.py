from dataclasses import dataclass
from typing import Dict, Iterable

import gym
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow_probability import distributions as tfd
from dreamer.models import RSSM, ConvLidarEncoder, ConvLidarDecoder, DenseDecoder
from dreamer.models.actor_critic import ActorCritic, ActionDecoder
from rlephant import Dataset, Transition, Episode

from dreamer import tools

class Dreamer(tf.Module):
    @dataclass
    class Config:
        T: int = 300
        C: int = 3
        B: int = 30
        L: int = 20
        beta: float = 1.0
        H: int = 10
        discount_gamma: float = 0.99
        discount_lambda: float = 0.95

    def __init__(self, config: Config, env: gym.Env):
        self._config = config
        self._dynamics = RSSM(stoch=30, deter=200, hidden=200)
        self._encoder = ConvLidarEncoder(embedding_size=16)
        self._decoder = ConvLidarDecoder(output_size=1080)
        self._reward = DenseDecoder(shape=(), layers=2, units=400, act='elu', dist='normal')

        self._actor = ActionDecoder(size=env.action_space.shape[0],
                                     layers=4,
                                     units=400,
                                     dist='tanh_normal',
                                     init_std=5.0,
                                     act='elu')
        self._value = DenseDecoder(shape=(), layers=3, units=400, act='elu')

        self._dynamics_optimizer = Adam()
        self._actor_optimizer = Adam()
        self._value_optimizer = Adam()

        self._act_dim = env.action_space.shape
        self._info = dict(dynamics={}, critic={}, actor={}, logs={})

        modules = [self._encoder, self._dynamics, self._decoder, self._reward]
        self._dynamics_variables = tf.nest.flatten([module.trainable_variables for module in modules])
        self._pcont = False

    def __call__(self, obs, state=None, training=False):
        obs = obs['lidar']
        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, axis=0)
            batch_size = 1
        else:
            batch_size = obs.shape[0]

        if state is None:
            latent = self._dynamics.initial(batch_size=batch_size)
            action = tf.zeros((batch_size, *self._act_dim), tf.float32)
        else:
            latent, action = state

        embed = self._encoder(obs)
        posterior, _ = self._dynamics.obs_step(prev_state=latent, prev_action=action, embed=embed)
        state = self._dynamics.get_state(posterior)
        action_distribution = self._actor(state)

        if training:
            action = action_distribution.sample()
        else:
            action = action_distribution.mode()

        #action = self._exploration(action, training)
        state = (latent, action)
        return action, state

    def train(self, steps: int, env: gym.Env, dataset: Dataset) -> Iterable[Dict]:
        for step in range(steps):
            self._info['logs']['step'] = step
            for c in range(self._config.C):
                posteriors = self.learn_dynamics(dataset=dataset)
                self.learn_behaviour(starting_state_posteriors=posteriors)

            self.interact_with_env(env=env, dataset=dataset)
            yield self._info

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

        return posterior_states


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

    def learn_behaviour(self, starting_state_posteriors: Dict):

        with tf.GradientTape() as actor_tape:
            imagined_states = self._imagine_horizon(posteriors=starting_state_posteriors, horizon=self._config.H)

            reward = self._reward(imagined_states).mode()
            if self._pcont:
                pcont = self._pcont(imagined_states).mean()
            else:
                pcont = self._config.discount_gamma * tf.ones_like(reward)

            values = self._value(imagined_states).mode()
            returns = tools.lambda_return(
                reward=reward[:-1],
                value=values[:-1],
                pcont=pcont[:-1],
                bootstrap=values[-1],
                lambda_=self._config.discount_lambda, axis=0
            )
            discount = tf.stop_gradient(tf.math.cumprod(tf.concat([tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
            actor_loss = -tf.reduce_mean(discount * returns)

        with tf.GradientTape() as value_tape:
            value_pred = self._value(imagined_states)[:-1]
            target = tf.stop_gradient(returns)
            value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

        gradients = actor_tape.gradient(actor_loss, self._actor.variables)
        self._actor_optimizer.apply_gradients(zip(gradients, self._actor.variables))

        gradients = value_tape.gradient(value_loss, self._value.variables)
        self._value_optimizer.apply_gradients(zip(gradients, self._value.variables))


    def _imagine_horizon(self, posteriors: Dict, horizon: int) -> tf.Tensor:
        if self._pcont:  # Last step could be terminal.
            posteriors = {k: v[:, :-1] for k, v in posteriors.items()}
        flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
        policy = lambda state: self._actor(tf.stop_gradient(self._dynamics.get_state(state))).sample()
        start = {k: flatten(v) for k, v in posteriors.items()}
        trajectories = {}
        for k, v in start.items():
            trajectories[k] = tf.expand_dims(v, axis=1)
        previous = start
        for step in range(horizon):
            actions = policy(previous)
            prior = self._dynamics.img_step(prev_state=previous, prev_action=actions)

            for k in trajectories:
                data = tf.expand_dims(prior[k], axis=1)
                trajectories[k] = tf.concat((trajectories[k], data), axis=1)

            previous = prior

        states = self._dynamics.get_state(trajectories)
        return states

    def interact_with_env(self, env: gym.Env, dataset: Dataset):
        t = 0
        while t < self._config.T:
            done = False
            obs = env.reset()
            episode = Episode()
            state = None
            while not done:
                action, state = self.__call__(obs=obs, state=state, training=True)
                obs, reward, done, info = env.step(action[0])
                transition = Transition(
                    observation=obs,
                    action={'action': action[0]},
                    reward=reward,
                    done=done
                )
                episode.append(transition)
                t += 1

                done = done or t >= self._config.T
            dataset.save(episode)
