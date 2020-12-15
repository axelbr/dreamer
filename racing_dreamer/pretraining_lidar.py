import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from racing_dreamer.dataset import load_lidar
import matplotlib.pyplot as plt
import time
import tools
import models
from racing_dreamer.pretraining_multihead_vae_lidar import plot_reconstruction_sample

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class MLPLidarEncoder(tools.Module):
  def __init__(self,  output_dim, act=tf.nn.relu):
    self._act = act
    self._output_dim = output_dim
    self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self._output_dim), scale=1),
                                 reinterpreted_batch_ndims=1)
  def __call__(self, obs):
    if type(obs) == dict:
        lidar = obs['lidar']
    else:
        lidar = obs
    if len(lidar.shape) > 2:
      x = tf.reshape(lidar, shape=(-1, *lidar.shape[2:], 1))
    else:
      x = lidar
    #x = tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5)(x)
    x = self.get('dense1', tfkl.Dense, units=128, activation=self._act)(x)
    x = self.get('dense2', tfkl.Dense, units=64, activation=self._act)(x)
    x = self.get('dense3', tfkl.Dense, units=tfpl.MultivariateNormalTriL.params_size(self._output_dim))(x)
    x = tfpl.MultivariateNormalTriL(self._output_dim, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))(x)
    return x

class MLPLidarDecoder(tools.Module):
  def __init__(self, output_dim, shape, act=tf.nn.relu):
    self._act = act
    self._output_dim = output_dim
    self._shape = shape


  def __call__(self, features):
    #params = tfpl.MultivariateNormalTriL.params_size(self._output_dim)
    #x = tf.reshape(features, shape=(-1, *features.shape[2:]))
    x = features
    x = self.get('params', tfkl.Dense, self._output_dim, activation=self._act)(x)
    x = self.get('dense1', tfkl.Dense, units=64, activation=self._act)(x)
    x = self.get('dense2', tfkl.Dense, units=128, activation=self._act)(x)
    x = self.get('dense3', tfkl.Dense, units=self._shape[0], activation=tf.nn.leaky_relu)(x)
    mean = x
    return mean

class MLP_CVAE_Dist(tools.Module):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoded_size = 16

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                     reinterpreted_batch_ndims=1)

        self.encoder = models.MLPLidarEncoder(encoded_size)
        self.decoder = MLPLidarDecoder(encoded_size, input_shape)

    def __call__(self, features):
        z = self.encoder(features)
        m = self.decoder(z)
        return tfd.Independent(tfd.Normal(m, 1))

class CVAE(tfk.Model):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoded_size = 16
        base_depth = 32

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                reinterpreted_batch_ndims=1)

        self.encoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=input_shape),
            tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),                                        # (1000, 1)
            tfkl.Conv1D(1 * base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),     # (1000, 32)
            tfkl.Conv1D(2 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),     # (500, 64)
            tfkl.Conv1D(4 * base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),    # (494, 128)
            tfkl.Conv1D(4 * base_depth, 7, strides=2, padding='same', activation=tf.nn.leaky_relu),     # (247, 128)
            tfkl.Flatten(),                                                                             # (31616)
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),                          # (152)
            tfpl.MultivariateNormalTriL(encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior)),
        ], name="encoder")

        dense_size = (int((np.ceil((input_shape[0] / 2) - 7 + 1))/2 * 4 * base_depth))
        self.decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=(encoded_size,)),
            tfkl.Dense(units=dense_size),
            tfkl.Reshape(target_shape=[int(dense_size / (4*base_depth)), 4 * base_depth]),
            tfkl.Conv1DTranspose(4 * base_depth, 7, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv1DTranspose(2 * base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Conv1DTranspose(1 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv1DTranspose(filters=1, kernel_size=5, strides=1, padding='same', activation=None),
            tfkl.Flatten(),
        ], name="decoder")

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        m = self.decoder(z.sample())
        return tfd.Independent(tfd.Normal(m, 1))

class ConvLidarEncoder(tools.Module):
  def __init__(self,  output_dim, input_field=None, act=tf.nn.relu, pref='h'):
    self._act = act
    self._output_dim = output_dim
    self._input_field = input_field
    self._pref = pref
    self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self._output_dim), scale=1),
                                 reinterpreted_batch_ndims=1)
  def __call__(self, obs):
    if type(obs) == dict:
        lidar = obs['lidar']
    else:
        lidar = obs
    #if len(lidar.shape) > 2:
    #  x = tf.reshape(lidar, shape=(-1, *lidar.shape[2:], 1))
    #else:
    #  x = lidar
    x = lidar if self._input_field==None else tf.expand_dims(lidar[:, self._input_field, :], 1)
    x = tf.transpose(x, (0, 2, 1))
    x = self.get(f'{self._pref}_conv1', tfkl.Conv1D, filters=32,  kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get(f'{self._pref}_conv2', tfkl.Conv1D, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get(f'{self._pref}_conv3', tfkl.Conv1D, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get(f'{self._pref}_conv4', tfkl.Conv1D, filters=28, kernel_size=5, strides=2, padding='same', activation=self._act)(x)
    x = self.get(f'{self._pref}_flat', tfkl.Flatten)(x)
    x = tfpl.MultivariateNormalTriL(self._output_dim, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))(x)
    return x

class ConvLidarDecoder(tools.Module):
  def __init__(self, output_dim, shape, act=tf.nn.relu, out_distr="normal", pref='h'):
    self._act = act
    self._output_dim = output_dim
    self._pref = pref
    self._shape = shape
    self._out_distr = out_distr


  def __call__(self, features):
    x = features
    x = self.get(f'{self._pref}_params', tfkl.Dense, 560, activation=self._act)(x)
    x = tf.reshape(x, shape=(-1, 20, 28))
    x = self.get(f'{self._pref}_deconv1', tfkl.Conv1DTranspose, filters=32, kernel_size=5, strides=2, padding='same', activation=self._act)(x)
    x = self.get(f'{self._pref}_deconv2', tfkl.Conv1DTranspose, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get(f'{self._pref}_deconv3', tfkl.Conv1DTranspose, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    # last layer different for 2 distributions
    if self._out_distr=="normal":
      # distances = Independent Normal Distr => nr params 2 * 1080
      x = self.get(f'{self._pref}_deconv4', tfkl.Conv1DTranspose, filters=2, kernel_size=5, strides=3, padding='same',
                   activation=self._act)(x)
      x = self.get(f'{self._pref}_flat', tfkl.Flatten)(x)
      distr = self.get(f'{self._pref}_dist', tfpl.IndependentNormal, event_shape=(self._shape[0]))(x)
    elif self._out_distr=="bernoulli":
      # distances = Independent Normal Distr => nr params 1 * 1080
      x = self.get(f'{self._pref}_deconv4', tfkl.Conv1DTranspose, filters=1, kernel_size=5, strides=3, padding='same',
                   activation=self._act)(x)
      x = self.get(f'{self._pref}_flat', tfkl.Flatten)(x)
      distr = self.get(f'{self._pref}_dist', tfpl.IndependentBernoulli, event_shape=(self._shape[0]))(x)
    else:
      raise NotImplementedError(f"not implemented for {self._out_distr}")
    return distr

class MLPAutoencoder(tools.Module):
    def __init__(self, input_shape, encoded_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_shape = input_shape
        self._encoded_size = encoded_size

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self._encoded_size), scale=1),
                                     reinterpreted_batch_ndims=1)

        self.distances_encoder = MLPLidarEncoder(self._encoded_size)
        self.distances_decoder = MLPLidarDecoder(self._encoded_size, self._input_shape)
        #self.distances_encoder = MLPLidarEncoder(encoded_size, input_field=0, pref='d')
        #self.obstacles_encoder = MLPLidarEncoder(encoded_size, input_field=None, pref='o')
        #self.distances_decoder = MLPLidarDecoder(encoded_size, input_shape, out_distr="normal", pref='d')
        #self.obstacles_decoder = MLPLidarDecoder(encoded_size, input_shape, out_distr="bernoulli", pref='o')

    def __call__(self, features):
      z = self.distances_encoder(features)
      m = self.distances_decoder(z)
      return tfd.Independent(tfd.Normal(m, 1))
      #latent_distance_sample = self.distances_encoder(features).sample()
      #latent_obstacle_sample = self.obstacles_encoder(features).sample()
      #return self.distances_decoder(latent_distance_sample), self.obstacles_decoder(latent_obstacle_sample)
      #return self.distances_decoder(latent_distance_sample)

def preprocess(x, max=5.0):
    x = tf.cast(x, tf.float32)
    #binary_wall = tf.where(x >= 5.0, 0.0, 1.0)    # if wall then 1 else 0
    sample = x / max - 0.5                        # normalize input
    #return tf.stack([sample, binary_wall])
    return sample


def plot_lidar(distances, obstacles, figure=None, title=""):
  import math, matplotlib.pyplot as plt
  angles = tf.linspace(math.pi/2-math.radians(270.0 / 2), math.pi/2 + math.radians(270.0 / 2), distances.shape[-1])[::-1]
  x = distances * np.cos(angles)
  y = distances * np.sin(angles)
  #c = np.where(obstacles > 0, 'k', 'gray')
  plt.subplot(2, 1, figure)
  plt.title(title)
  #plt.scatter(x, y, c=c)
  plt.scatter(x, y)
  #plt.xlim(-1, 1)
  #plt.ylim(-1, 1)

def main():
    lidar_file = "/home/luigi/PycharmProjects/dreamer/offline_autoencoder_tuning/" \
                 "out/dataset_single_agent_austria_lidar30_random_starts_austria_1000episodes_100steps.h5"
    n_epochs = 20
    batch_size = 128
    lr = 0.01
    lidar_rays = 1080
    encoded_size = 16

    training_data, test_data = load_lidar(lidar_file, train=0.8, shuffle=True)
    training_data = training_data\
        .map(lambda x : preprocess(x, max=30.0))\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = test_data\
        .map(lambda x : preprocess(x, max=30.0)) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    model_name = "MLP_Autoencoder"
    vae = MLPAutoencoder(input_shape=(lidar_rays, 2), encoded_size=32)

    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    #vae.model.compile(optimizer=optimizer, loss=negloglik)
    #vae.encoder.summary()
    #vae.decoder.summary()

    init = time.time()
    from datetime import datetime
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    writer = tf.summary.create_file_writer('log/{}_{}'.format(model_name, timestamp))

    training_step = 0
    for epoch in range(n_epochs):
      print(f'Epoch {epoch+1}/{n_epochs}')
      epoch_loss = 0
      for i_batch, batch in enumerate(iter(training_data)):
        training_step += batch_size
        with tf.GradientTape() as tape:
          #batch_distances, batch_obstacles = batch[:, 0, :], batch[:, 1, :]
          batch_distances = batch
          #distance_distr, obstacle_distr = vae(batch)
          distance_distr = vae(batch)
          #loss = tf.reduce_mean(negloglik(batch_distances, distance_distr) + negloglik(batch_obstacles, obstacle_distr))
          loss = tf.reduce_mean(negloglik(batch_distances, distance_distr))
          if i_batch == 0:
            true_distances = batch[0, :] + .5
            #true_obstacles = batch[0, 1, :]
            true_obstacles = np.ones((1, 1080))
            distances = distance_distr.sample()[0, :] + .5
            #obstacle = obstacle_distr.sample()[0, :]
            obstacle = true_obstacles
            truth_image = plot_lidar(distances=true_distances, obstacles=true_obstacles, figure=1, title=f"True observation - Epoch {epoch}/{n_epochs}")
            recon_image = plot_lidar(distances=distances, obstacles=obstacle, figure=2, title=f"Reconstruction - Epoch {epoch}/{n_epochs}")
            plt.show()
          # loss = tf.reduce_mean(tf.losses.mse(tf.expand_dims(batch, -1), recon_dist))
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        epoch_loss += loss.numpy()
        with writer.as_default():
          tf.summary.scalar('train_loss', epoch_loss / (i_batch+1), step=training_step)
        if i_batch>0 and i_batch % 50 == 0:
          print("epoch {}, batch {} => avg loss {:.10f}".format(epoch, i_batch, epoch_loss / i_batch))

    print("[Info] Training completed in {:.3}s".format(time.time()-init))
    vae.encoder.save(f"pretrained_models/pretrained_encoder_{timestamp}")
    vae.decoder.save(f"pretrained_models/pretrained_decoder_{timestamp}")
    print("[Info] Pretrained models stored")

    init = time.time()
    test_loss = 0
    b = 0
    for step, batch in enumerate(iter(test_data)):
        b += 1
        recon_dist = vae(batch)
        tools.create_reconstruction_gif(batch, recon_dist, name="{}_{}epochs_{}".format(model_name, n_epochs, b))
        loss = tf.reduce_mean(negloglik(batch, recon_dist))
        tf.summary.scalar('test_loss', epoch_loss / (step + 1), step=training_steps * batch_size)
        #loss = tf.reduce_mean(tf.losses.mse(tf.expand_dims(batch, -1), recon_dist))
        test_loss += loss.numpy()
        print("test, batch {} => avg loss {:.10f}".format(b, test_loss/b))
        if b >= 5:
            break
    print("[Info] Testing completed in {:.3}s".format(time.time()-init))

if __name__=="__main__":
    main()
