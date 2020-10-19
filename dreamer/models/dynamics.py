import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras import layers as tfkl
from tensorflow.keras.mixed_precision import experimental as precision

from dreamer import tools


class RSSM(tools.Module):

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = precision.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])

    # This swaps time and batch dimension
    embed = tf.transpose(embed, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])

    step = action.shape[0]
    prev_state = state
    prev_action = action[0]
    posteriors, priors = {}, {}
    for i in range(1, step):
      posterior, prior = self.obs_step(prev_state=prev_state, prev_action=prev_action, embed=embed[i])
      prev_state = posterior
      prev_action = action[i]

      for k in state:
        if k in posteriors:
          expanded_posterior = tf.expand_dims(posterior[k], axis=1)
          expanded_prior = tf.expand_dims(prior[k], axis=1)
          posteriors[k] = tf.concat((posteriors[k], expanded_posterior), axis=1)
          priors[k] = tf.concat((priors[k], expanded_prior), axis=1)
        else:
          posteriors[k] = tf.expand_dims(posterior[k], axis=1)
          priors[k] = tf.expand_dims(prior[k], axis=1)

    #post, prior = tools.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs),
    #  (action, embed),
    #  (state, state)
    #)
    # post = {k: tf.transpose(v, [1, 0, 2]) for k, v in posteriors.items()}
    #prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in priors.items()}


    return posteriors, priors

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_state(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)    # get distribution+ of the current state
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}    # get distr+ of next state
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior