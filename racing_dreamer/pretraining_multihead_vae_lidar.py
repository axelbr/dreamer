import math, matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from racing_dreamer.dataset import load_lidar
import time
import tools
import models

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class ConvLidarEncoder(tools.Module):
  def __init__(self,  encoded_size, input_field=None, act=tf.nn.relu, pref='h'):
    self._act = act
    self._encoded_size = encoded_size
    self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self._encoded_size), scale=1),
                                 reinterpreted_batch_ndims=1)
  def __call__(self, obs):
    if type(obs) == dict:
        x = obs['lidar']
    else:
        x = obs
    x = tf.transpose(x, (0, 2, 1))
    x = self.get('conv1', tfkl.Conv1D, filters=32,  kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get('conv2', tfkl.Conv1D, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get('conv3', tfkl.Conv1D, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get('conv4', tfkl.Conv1D, filters=28, kernel_size=5, strides=2, padding='same', activation=self._act)(x)
    x = self.get('flat', tfkl.Flatten)(x)
    assert tfpl.MultivariateNormalTriL.params_size(self._encoded_size) == x.shape[1]
    x = tfpl.MultivariateNormalTriL(self._encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))(x)
    return x

class MultiHeadConvLidarDecoder(tools.Module):
  def __init__(self, encoded_dim, shape, act=tf.nn.relu):
    self._act = act
    self._encoded_size = encoded_dim
    self._shape = shape

  def __call__(self, features):
    x = features
    x = self.get('params', tfkl.Dense, tfpl.MultivariateNormalTriL.params_size(self._encoded_size), activation=self._act)(x)
    x = tf.reshape(x, shape=(-1, 20, 28))    # 20*28=560
    x = self.get('deconv1', tfkl.Conv1DTranspose, filters=32, kernel_size=5, strides=2, padding='same', activation=self._act)(x)
    x = self.get('deconv2', tfkl.Conv1DTranspose, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x = self.get('deconv3', tfkl.Conv1DTranspose, filters=32, kernel_size=5, strides=3, padding='same', activation=self._act)(x)

    # distance branch
    x_n = self.get('deconv4', tfkl.Conv1DTranspose, filters=2, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x_n = self.get('flat4', tfkl.Flatten)(x_n)
    distance_dist = self.get('dist1', tfpl.IndependentNormal, event_shape=(self._shape[0]))(x_n)

    # obstacle detection branch
    x_b = self.get('deconv5', tfkl.Conv1DTranspose, filters=1, kernel_size=5, strides=3, padding='same', activation=self._act)(x)
    x_b = self.get('flat5', tfkl.Flatten)(x_b)
    obstacle_dist = self.get('dist2', tfpl.IndependentBernoulli, event_shape=(self._shape[0]))(x_b)

    # return 2 distributions
    return distance_dist, obstacle_dist

class Autoencoder(tools.Module):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoded_size = 16

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                     reinterpreted_batch_ndims=1)

        self.encoder = ConvLidarEncoder(32)
        self.decoder = MultiHeadConvLidarDecoder(32, input_shape)

    def __call__(self, features):
        z = self.encoder(features)
        dist_distr, obs_distr = self.decoder(z)
        return dist_distr, obs_distr

def plot_reconstruction_sample(distance_sample, obstacle_sample, recon_distance, recon_obstacle, title):
  angles = tf.linspace(math.pi / 2 - math.radians(270.0 / 2), math.pi / 2 + math.radians(270.0 / 2),
                       distance_sample.shape[-1])[::-1]
  for i, (distance_scan, obstacle_scan) in enumerate(
          [(distance_sample, obstacle_sample), (recon_distance, recon_obstacle)]):
    x = distance_scan * np.cos(angles)
    y = distance_scan * np.sin(angles)
    c = np.where(obstacle_scan >= 1, 'k', 'grey')
    plt.subplot(2, 1, i + 1)
    plt.scatter(x, y, c=c)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.title(title)
  plt.show()

def preprocess(x, max=5.0):
    x = tf.cast(x, tf.float32)
    sample = x / max
    binary_wall = tf.where(x >= 5.0, 0.0, 1.0)  # if wall then 1 else 0
    return tf.stack([sample, binary_wall])

def main():
    lidar_file = "data/pretraining_austria_single_wt_4_action_repeat.h5"
    n_epochs = 20
    batch_size = 128
    lr = 0.001
    lidar_rays = 1080

    training_data, test_data = load_lidar(lidar_file, train=0.8, shuffle=True)
    training_data = training_data\
        .map(preprocess)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = test_data\
        .map(preprocess) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    model_name = "multihead_autoencoder"
    vae = Autoencoder(input_shape=(lidar_rays, 1))

    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    init = time.time()
    from datetime import datetime
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    writer = tf.summary.create_file_writer('log/{}_{}'.format(model_name, timestamp))
    training_steps = 0
    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(n_epochs):
                print(f'Epoch {epoch}/{n_epochs}')
                training_steps += 1
                epoch_loss = 0
                for step, batch in enumerate(iter(training_data)):
                    # optimization
                    with tf.GradientTape() as tape:
                        y_distance, y_obst_detection = batch[:, 0, :], batch[:, 1, :]
                        distance_distr, obstacle_distr = vae(batch)
                        loss = tf.reduce_mean(negloglik(y_distance, distance_distr) +
                                              negloglik(y_obst_detection, obstacle_distr))
                    gradients = tape.gradient(loss, vae.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
                    # log
                    epoch_loss += loss.numpy()
                    tf.summary.scalar('loss', epoch_loss, step=training_steps * batch_size)
                    # rendering for debug
                    if step==0:
                      rnd_id = np.random.randint(0, batch.shape[0])
                      distance_sample, obstacle_sample = batch[rnd_id, 0, :], batch[rnd_id, 1, :]
                      recon_distance = distance_distr.mode()[rnd_id, :]
                      recon_obstacle = obstacle_distr.mode()[rnd_id, :]
                      text = f'Sample Rendering - Epoch {epoch}'
                      plot_reconstruction_sample(distance_sample, obstacle_sample, recon_distance, recon_obstacle, text)
                    if step + 1 % 50 == 0:
                        print("epoch {}, batch {} => avg loss {:.10f}".format(epoch, step+1, epoch_loss / step+1))
    print("[Info] Training completed in {:.3}s".format(time.time()-init))

    vae.encoder.save(f"pretrained_models/pretrained_{model_name}_encoder")
    vae.decoder.save(f"pretrained_models/pretrained_{model_name}_decoder")

    init = time.time()
    test_loss = 0
    b = 0
    for batch in iter(test_data):
        b += 1

        y_distance, y_obst_detection = batch[:, 0, :], batch[:, 1, :]
        distance_distr, obstacle_distr = vae(batch)
        loss = tf.reduce_mean(negloglik(y_distance, distance_distr) +
                              negloglik(obstacle_distr, obstacle_distr))
        test_loss += loss.numpy()
        tools.create_reconstruction_gif(y_distance, y_obst_detection, distance_distr, obstacle_distr,
                                        name="{}_{}epochs_{}".format(model_name, n_epochs, b))

        print("test, batch {} => avg loss {:.10f}".format(b, test_loss/b))
        if b >= 10:
            break
    print("[Info] Testing completed in {:.3}s".format(time.time()-init))

if __name__=="__main__":
    main()