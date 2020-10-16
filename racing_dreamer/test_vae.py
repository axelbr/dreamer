import tensorflow as tf
import tensorflow_probability as tfp
from racing_dreamer.dataset import load_lidar
import tools
import time

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

class MLP_CVAE_Dist(tools.Module):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        encoded_size = 8

        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                     reinterpreted_batch_ndims=1)

        self.encoder = MLPLidarEncoder(encoded_size)
        self.decoder = MLPLidarDecoder(encoded_size, input_shape)

    def __call__(self, features):
        z = self.encoder(features)
        m = self.decoder(z.sample())
        return tfd.Independent(tfd.Normal(m, 1))

def preprocess(x, max=5.0):
    sample = tf.cast(x, tf.float32) / max
    return sample

lidar_file = "data/pretraining_austria_single_wt_4_action_repeat.h5"
n_epochs = 1
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

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

model_name = "MLP_CVAE_NormDist_stddev80"
vae = MLP_CVAE_Dist(input_shape=(lidar_rays, 1))
vae.encoder.load("models/pretrained_encoder")
vae.decoder.load("models/pretrained_decoder")

init = time.time()
test_loss = 0
b = 0
for batch in iter(test_data):
    b += 1
    recon_dist = vae(batch)
    tools.create_reconstruction_gif(batch, None, recon_dist,
                                    distribution=True, name="mlp_vae_4actionrepeat_lidar_{}epochs_{}_reload_models".format(n_epochs, b))
    loss = tf.reduce_mean(negloglik(batch, recon_dist))
    #loss = tf.reduce_mean(tf.losses.mse(tf.expand_dims(batch, -1), recon_dist))
    test_loss += loss.numpy()
    if b % 10 == 0:
        print("test, batch {} => avg loss {:.10f}".format(b, test_loss/b))
    if b >= 10:
        break
print("[Info] Testing completed in {:.3}s".format(time.time()-init))