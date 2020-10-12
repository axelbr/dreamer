import time
from datetime import datetime
import numpy as np
import h5py as h5
import os
import tensorflow as tf
import tools
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
from offline_autoencoder_tuning.rendering.rendering_tools import _image_summaries

tf.executing_eagerly = True
tf.keras.backend.set_floatx('float64')

epochs = 20
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
        lidar = obs
        if type(obs)==dict:
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
            x = tf.reshape(features, shape=(-1, *features.shape[1:]))
            x = self.get('params', tf.keras.layers.Dense, params, activation=self._act)(x)
            x = self.get('dist', tfpl.IndependentNormal, event_shape=self._output_dim)(x)
            dist = tfd.BatchReshape(x, batch_shape=features.shape[:1])
            return dist

class Autoencoder(tools.Module):
    def __init__(self, latent_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = MyLidarEncoder(output_dim=latent_dim)
        self.decoder = MyLidarDecoder(output_dim=original_dim)

    def __call__(self, input):
        latent = self.encoder({'lidar': input})
        reconstructed = self.decoder(latent)
        return reconstructed

def loss(model, original):
    image_pred = model(original)
    return - tf.reduce_mean(image_pred.log_prob(original))      # max loglikelihood = min entropy, entropy=-log p

def train(loss, model, optimizer, original):
    with tf.GradientTape() as tape:
        loss = loss(model, original)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    optimizer.apply_gradients(gradient_variables)


# load data
output_dir = os.path.join("offline_autoencoder_tuning", "out")
dataset_filename = "dataset_random_starts_austria_2000episodes_1000maxobs.h5"
data = h5.File(os.path.join(output_dir, dataset_filename), "r")
# prepare dataset
all_obs = np.vstack([np.array(data[episode]['obs']['lidar']) for episode in list(data.keys())[:200]])
data = tf.data.Dataset.from_tensor_slices(all_obs).shuffle(shuffle_batch)
test_size = 100    # just to create gif
val_data = data.take(test_size)
train_data = data.skip(test_size)
# model def
autoencoder = Autoencoder(encoded_obs_dim, lidar_shape)
opt = tf.optimizers.Adam(learning_rate=learning_rate)
# TODO: not working with keras 'fit' for some issue wt type (it found a tuple during training)
#autoencoder.compile(opt, loss)
#autoencoder.fit(all_obs, all_obs, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)

train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)
init = time.time()
for epoch in range(epochs):
    print("{}/{} => init time: {:.3f}s".format(epoch + 1, epochs, time.time()-init))
    init = time.time()
    for step, batch_features in enumerate(train_data):
        train(loss, autoencoder, opt, batch_features)
        loss_values = loss(autoencoder, batch_features)
        reconstructed = autoencoder(batch_features)
        if (step % 10 == 0):
            print("\t{}/{} => loss: {}".format(step + 1, len(train_data), loss_values))

for step, batch_features in enumerate(val_data):
    if (step % 10 == 0):
        print("\tTest: {}/{}".format(step + 1, len(val_data)))
    latent = autoencoder.encoder(batch_features)
    reconstructed = autoencoder(batch_features)
    _image_summaries(batch_features, latent, reconstructed, name="{}_{}".format(epoch+1, step+1))

"""
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
autoencoder.save("offline_autoencoder_tuning/models/autoencoder_{}".format(timestamp))
"""