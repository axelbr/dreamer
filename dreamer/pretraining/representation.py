import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv1DTranspose, Conv1D, Flatten, Reshape
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.layers import IndependentNormal, KLDivergenceRegularizer
from dreamer.models import ConvLidarEncoder

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

  def call(self, obs, **kwargs) -> tf.Tensor:
    x = self.dense_layer(obs)
    x = self.reshape(x)
    for layer in self.deconv_layers:
      x = layer(x)
    x = tf.expand_dims(x, axis=1)
    return x


class CVAE(tf.keras.models.Model):

  def __init__(self, input_size: int, embedding_size: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.encoder = ConvLidarEncoder(embedding_size=embedding_size)
    self.decoder = ConvLidarDecoder(output_size=input_size)

    self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(embedding_size), scale=1.), reinterpreted_batch_ndims=1)
    self.dense_layer = Dense(IndependentNormal.params_size(event_shape=(embedding_size,)))
    self.latent_dist_layer = IndependentNormal(
      embedding_size,
      convert_to_tensor_fn=tfd.Distribution.sample,
      activity_regularizer=KLDivergenceRegularizer(self.prior, weight=1.0),
    )

    self.flatten_layer = Flatten()
    self.dense_recon_layer = Dense(2*input_size)
    self.recon_dist_layer = IndependentNormal(event_shape=(input_size,))



  def call(self, obs, training=None, mask=None):
    encoding = self.encoder.call(obs)
    x = self.dense_layer(encoding)
    dist = self.latent_dist_layer.call(x)
    reconstructed_obs = self.decoder.call(dist)

    x = self.flatten_layer(reconstructed_obs)
    x = self.dense_recon_layer(x)
    output_dist = self.recon_dist_layer(x)

    return output_dist

