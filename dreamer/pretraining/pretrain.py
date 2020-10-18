from datetime import time

import h5py
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

from dreamer import tools
from dreamer.pretraining.representation import CVAE

tf.config.run_functions_eagerly(run_eagerly=True)


def _image_summaries(lidar, embed, image_pred, name=""):
  recon = image_pred.mean() * 5.0
  if len(lidar.shape) < 3:
    lidar = tf.expand_dims(lidar, axis=0)
  if len(recon.shape) < 3:
    recon = tf.expand_dims(recon, axis=0)
  lidar_img = tools.lidar_to_image(tf.cast(lidar, dtype='float32'))
  recon_img = tools.lidar_to_image(tf.cast(recon, dtype='float32'))
  video = tf.concat([lidar_img, recon_img], 1)
  gif_summary(video, name=name)


def gif_summary(video, name="", fps=30):
  frames = []
  for i in range(video.shape[0]):
    frames.append(video[i].numpy().astype(np.uint8))
  imageio.mimsave(f"{name}.gif", frames)


def negative_logprob(x, dist):
  return -dist.log_prob(x)

def preprocess(sample, obs_field='lidar', obs_max_val=5.0):
  image = tf.cast(sample, tf.float32) / obs_max_val  # Scale to unit interval.
  return image


if __name__ == '__main__':

  dataset_filename = 'data/pretraining_austria_single.h5'
  data = h5py.File(dataset_filename, 'r')
  # prepare dataset

  epochs = 20
  batch_size = 64
  learning_rate = 0.01
  shuffle_batch = 10000
  encoded_obs_dim = 128
  lidar_shape = 1000

  all_obs = np.vstack([data[episode]['obs']['lidar'] for episode in list(data.keys())[:10]])
  data = tf.data.Dataset.from_tensor_slices(all_obs) \
  .map(preprocess)
   # .map(lambda t: tf.expand_dims(t, axis=0)) \
   # .map(lambda t: (t, t))
  test_size = 100  # just to create gif
  val_data = data.take(test_size) \
    .batch(batch_size)
  train_data = data.skip(test_size) \
    .shuffle(shuffle_batch) \
    .batch(batch_size)

  model = CVAE(input_size=1080, embedding_size=16)
  model.compile(optimizer='adam', loss=negative_logprob)
 # model.fit(train_data, epochs=10)

  optimizer = Adam()
  for epoch in range(epochs):
    total_loss = 0
    if True or epoch % 10 == 0:
      for step, batch_features in enumerate(val_data):
        if (step % 10 == 0):
          print("\tTest: {}/{}".format(step + 1, len(val_data)))
        reconstructed = model.call(tf.expand_dims(batch_features, axis=1), training=False)
        _image_summaries(batch_features * 5.0, reconstructed.sample() * 5.0, reconstructed,
                         name="{}_{}".format(epoch + 1, step + 1))

    for step, batch_features in enumerate(train_data):
      with tf.GradientTape() as tape:
        reconstructed = model.call(tf.expand_dims(batch_features, axis=1), training=True)
        negloglik = negative_logprob(batch_features, reconstructed)
        loss = tf.reduce_mean(negloglik)
      total_loss += loss

      gradients = tape.gradient(loss, model.trainable_variables)
      gradient_variables = zip(gradients, model.trainable_variables)
      optimizer.apply_gradients(gradient_variables)
      if (step % 10 == 0):
        print("\t{}/{} => loss: {}".format(step + 1, len(train_data), total_loss/step+1))

  for step, batch_features in enumerate(val_data):
    if (step % 10 == 0):
      print("\tTest: {}/{}".format(step + 1, len(val_data)))
    reconstructed = model.call(batch_features)
    _image_summaries(batch_features, reconstructed.sample(), reconstructed, name="{}_{}".format(epoch + 1, step + 1))

  np.savez_compressed('decoder_weights.npz', model.encoder.weights)
