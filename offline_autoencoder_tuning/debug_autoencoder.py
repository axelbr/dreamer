import models
import numpy as np
import h5py as h5
import os
import tensorflow as tf
import tools
import tensorflow_probability as tfpl
from tensorflow_probability import distributions as tfd

tf.executing_eagerly = True

epochs = 1
batch_size = 32
learning_rate = 0.01
shuffle_batch = 1000
encoded_obs_dim = 128
lidar_shape = 1000

class MyLidarEncoder(tools.Module):
    def __init__(self,  output_dim, act=tf.nn.relu):
        self._act = act
        self._output_dim = output_dim

    def __call__(self, obs):
        kwargs = dict(strides=1, activation=self._act, padding='same')
        lidar = obs['lidar']
        if len(lidar.shape) > 2:
            x = tf.reshape(lidar, shape=(-1, *lidar.shape[2:], 1))
        else:
            x = tf.expand_dims(lidar, axis=-1)
        x = self.get('conv1', tf.keras.layers.Conv1D, filters=4, kernel_size=5, **kwargs)(x)
        x = self.get('conv2', tf.keras.layers.Conv1D, filters=8, kernel_size=3, **kwargs)(x)
        x = self.get('flat', tf.keras.layers.Flatten)(x)
        x = self.get('dense', tf.keras.layers.Dense, units=self._output_dim)(x)

        shape = (*lidar.shape[:-1], *x.shape[1:])
        return tf.reshape(x, shape=shape)

class MyLidarDecoder(tools.Module):
  def __init__(self, output_dim, act=tf.nn.relu):
    self._act = act
    self._output_dim = output_dim

  def __call__(self, features):
    params = tfpl.IndependentNormal.params_size(self._output_dim)
    x = tf.reshape(features, shape=(-1, *features.shape[2:]))
    x = self.get('params', tf.keras.layers.Dense, params, activation=self._act)(x)
    x = self.get('dist', tfpl.IndependentNormal, event_shape=self._output_dim)(x)
    dist = tfd.BatchReshape(x, batch_shape=features.shape[:2])
    return dist

class Autoencoder(tf.keras.Model):
    def __init__(self, latent_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = MyLidarEncoder(output_dim=latent_dim)
        self.decoder = MyLidarDecoder(output_dim=original_dim)

    def call(self, input):
        latent = self.encoder({'lidar': input})
        reconstructed = self.decoder(latent)
        return reconstructed

output_dir = os.path.join("offline_autoencoder_tuning", "out")
dataset_filename = "dataset_random_starts_austria_2000episodes_1000maxobs.h5"
data = h5.File(os.path.join(output_dir, dataset_filename), "r")

all_obs = np.vstack([np.array(data[episode]['obs']['lidar']) for episode in list(data.keys())])
training_data = tf.data.Dataset.from_tensor_slices(all_obs).shuffle(shuffle_batch).batch(batch_size)
"""
for batch in dataset:
    print(batch.shape)
    batch_obs = {'lidar': batch}
    latent = encode(batch_obs)
"""
autoencoder = Autoencoder(encoded_obs_dim, lidar_shape)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

def loss(model, original):
    image_pred = model(original)
    return - tf.reduce_mean(image_pred.log_prob(original))      # max loglikelihood = min entropy, entropy=-log p
"""
autoencoder.compile(opt, loss)
autoencoder.fit(training_data, training_data, batch_size, epochs=1, verbose=2)
"""
def train(loss, model, optimizer, original):
    with tf.GradientTape() as tape:
        loss = loss(model, original)
        gradients = tape.gradient(loss, model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        optimizer.apply_gradients(gradient_variables)

for epoch in range(epochs):
    for step, batch_features in enumerate(training_data):
        train(loss, autoencoder, opt, batch_features)
        loss_values = loss(autoencoder, batch_features)
        reconstructed = autoencoder(batch_features)