from dataclasses import dataclass

import tensorflow as tf
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense, Conv1DTranspose, Reshape, Conv2D, Conv2DTranspose
from tensorflow_probability.python.layers import IndependentNormal

from dreamer import tools
from tensorflow_probability import distributions as tfd

tf.config.run_functions_eagerly(run_eagerly=True)


class ConvEncoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = self.get('h1', Conv2D, 1 * self._depth, 4, **kwargs)(x)
    x = self.get('h2', Conv2D, 2 * self._depth, 4, **kwargs)(x)
    x = self.get('h3', Conv2D, 4 * self._depth, 4, **kwargs)(x)
    x = self.get('h4', Conv2D, 8 * self._depth, 4, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
    return tf.reshape(x, shape)


class ConvDecoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
    self._act = act
    self._depth = depth
    self._shape = shape

  def __call__(self, features):
    kwargs = dict(strides=2, activation=self._act)
    x = self.get('h1', Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    x = self.get('h2', Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
    x = self.get('h3', Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
    x = self.get('h4', Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
    x = self.get('h5', Conv2DTranspose, self._shape[-1], 6, strides=2)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))



class ConvLidarEncoder(tf.keras.layers.Layer):

  @dataclass
  class Hyperparams:
    beta: float = 1.0

  def __init__(self, embedding_size: int, act=tf.nn.relu, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.conv_layers = [
      Conv1D(filters=32, kernel_size=3, strides=2, activation=act, padding='same'),
      Conv1D(filters=64, kernel_size=3, strides=2, activation=act, padding='same'),
    ]
    self.flatten_layer = Flatten()
    self.dense_layer = Dense(units=embedding_size)

  def call(self, obs, **kwargs) -> tf.Tensor:
    if len(obs.shape) == 2:
      obs = tf.expand_dims(obs, axis=1)
    x = obs
    for layer in self.conv_layers:
      x = layer(x)
    x = self.flatten_layer(x)
    x = self.dense_layer(x)
    return x

class ConvLidarDecoder(tf.keras.layers.Layer):

  def __init__(self, output_size: int, act=tf.nn.relu, trainable=True, name=None, dtype=None, dynamic=False,
               **kwargs):
    super().__init__(trainable, name, dtype, dynamic, **kwargs)
    self.deconv_layers = [
      Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation=act, padding='same'),
      Conv1DTranspose(filters=32, kernel_size=3, strides=2, activation=act, padding='same'),
      Conv1D(filters=1, kernel_size=3, strides=1, padding='same')
    ]
    self.flatten_layer = Flatten()
    self.dense_layer = Dense(units=int(output_size / 4) * 64)
    self.reshape = Reshape(target_shape=(int(output_size / 4), 64))
    self.dist_params_layer = Dense(units=IndependentNormal.params_size(event_shape=(output_size,)))
    self.dist = IndependentNormal(event_shape=(output_size,), convert_to_tensor_fn=tfd.Distribution.sample)

  def call(self, obs, **kwargs) -> tf.Tensor:
    x = self.dense_layer(obs)
    x = self.reshape(x)
    for layer in self.deconv_layers:
      x = layer(x)
    x = tf.squeeze(x, axis=-1)
    x = self.dist_params_layer(x)
    dist = self.dist(x)
    return dist

