import tensorflow as tf
import tensorflow_probability as tfp
from racing_dreamer.dataset import load_lidar
import tools
import time
from racing_dreamer.pretraining_lidar import MLP_CVAE_Dist

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

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

init = time.time()
batch = next(iter(training_data))
recon_dist = vae(batch)
vae.encoder.load("pretrained_models/pretrained_encoder")
vae.decoder.load("pretrained_models/pretrained_decoder")
print("[Info] Loaded pretrained_models")



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