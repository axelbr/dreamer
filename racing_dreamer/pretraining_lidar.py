import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from racing_dreamer.dataset import load_lidar
import time
import tools
import models
from racing_dreamer.pretraining_multihead_vae_lidar import plot_reconstruction_sample

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

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
    x = tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5)(x)
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

class MLP_VAE(tools.Module):
    def __init__(self, input_shape, encoded_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                     reinterpreted_batch_ndims=1)

        self.encoder = models.MLPLidarEncoder(encoded_size)
        self.decoder = MLPLidarDecoder(encoded_size, input_shape)

    def __call__(self, features):
        z = self.encoder(features)
        m = self.decoder(z)
        return tfd.Independent(tfd.Normal(m, 1))

def preprocess(x, max=5.0):
    sample = tf.cast(x, tf.float32) / max
    return sample

def main():
    lidar_file = "/home/luigi/PycharmProjects/dreamer/offline_autoencoder_tuning/" \
                 "out/dataset_single_agent_austria_lidar30_random_starts_austria_1000episodes_100steps.h5"
    n_epochs = 20
    batch_size = 128
    lr = 0.001
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

    model_name = "MLP_Lidar30"
    vae = MLP_CVAE_Dist(input_shape=(lidar_rays, 1))

    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    #vae.model.compile(optimizer=optimizer, loss=negloglik)
    #vae.encoder.summary()
    #vae.decoder.summary()

    init = time.time()
    from datetime import datetime
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    writer = tf.summary.create_file_writer('log/{}_{}'.format(model_name, timestamp))
    training_steps = 0
    with writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(n_epochs):
                print(f'Epoch {epoch}/{n_epochs}')
                epoch_loss = 0
                b = 0
                for step, batch in enumerate(iter(training_data)):
                    b += 1
                    training_steps += 1
                    with tf.GradientTape() as tape:
                        recon_dist = vae(batch)
                        loss = tf.reduce_mean(negloglik(batch, recon_dist))
                        #loss = tf.reduce_mean(tf.losses.mse(tf.expand_dims(batch, -1), recon_dist))
                    gradients = tape.gradient(loss, vae.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
                    # log
                    epoch_loss += loss.numpy()
                    tf.summary.scalar('train_loss', epoch_loss / (step + 1), step=training_steps * batch_size)
                    if step + 1 % 50 == 0:
                      print("epoch {}, batch {} => avg loss {:.10f}".format(epoch, step + 1, epoch_loss / (step + 1)))
                    # rendering for debug
                    if step == 0:
                      rnd_id = np.random.randint(0, batch.shape[0])
                      distance_sample = batch[rnd_id, :]
                      recon_distance = recon_dist.mode()[rnd_id, :]
                      text = f'Sample Rendering - Epoch {epoch}'
                      plot_reconstruction_sample(distance_sample, recon_distance, text)
    print("[Info] Training completed in {:.3}s".format(time.time()-init))

    vae.encoder.save(f"pretrained_models/pretrained_{model_name}_encoder")
    vae.decoder.save(f"pretrained_models/pretrained_{model_name}_decoder")


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