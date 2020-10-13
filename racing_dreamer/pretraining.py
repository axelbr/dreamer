import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
from rlephant import ReplayStorage

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

def generator(file: str):
    storage = ReplayStorage(filename=file)
    def _gen():
        for episode in storage:
            for transition in episode:
                yield transition.observation['image']
    return _gen

def preprocess(x, max=255.0):
    sample = tf.cast(x, tf.float32) / max
    sample = sample < tf.random.uniform(tf.shape(sample))
    return sample, sample

file = 'pretraining_austria_single.h5'
lidar_dataset = tf.data.Dataset.from_generator(generator=generator(file=file), output_types=tf.float64)

size = 48_000
train_size = int(0.8 * size)
test_size = int(0.1 * size)
eval_size = size - train_size - test_size

train = lidar_dataset.take(train_size)\
    .map(preprocess)\
    .batch(256)\
    .prefetch(tf.data.experimental.AUTOTUNE)\
    .shuffle(int(10e3))
test = lidar_dataset.skip(train_size).take(test_size) \
    .map(preprocess) \
    .batch(256) \
    .prefetch(tf.data.experimental.AUTOTUNE) \
    .shuffle(int(10e3))

eval = lidar_dataset.skip(train_size+test_size).take(eval_size) \
    .map(preprocess) \
    .batch(256) \
    .prefetch(tf.data.experimental.AUTOTUNE) \
    .shuffle(int(10e3))


input_shape = (64,64,3)
encoded_size = 16
base_depth = 32

prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1), reinterpreted_batch_ndims=1)

encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
    tfkl.Conv2D(base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(2 * base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(2 * base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(4 * encoded_size, 7, strides=1,
                padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Flatten(),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size), activation=None),
    tfpl.MultivariateNormalTriL(encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
])

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Reshape([1, 1, encoded_size]),
    tfkl.Conv2DTranspose(2 * base_depth, 7, strides=1,
                         padding='valid', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2 * base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(2 * base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
    tfkl.Conv2D(filters=1, kernel_size=5, strides=1,
                padding='same', activation=None),
    tfkl.Flatten(),
    tfkl.Dense(tfpl.IndependentBernoulli.params_size(tf.reduce_prod(input_shape)), activation=None),
    tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
])

class CVAE(tf.Module):

    def __init__(self, encoder, decoder, name=None):
        super().__init__(name)
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

vae = CVAE(encoder, decoder)
vae = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss=negloglik)

_ = vae.fit(train, epochs=1, validation_data=test)

x = next(iter(eval))[0][:10]
xhat = vae(x)
assert isinstance(xhat, tfd.Distribution)

import matplotlib.pyplot as plt

def display_imgs(x, y=None):
  if not isinstance(x, (np.ndarray, np.generic)):
    x = np.array(x)
  plt.ioff()
  n = x.shape[0]
  fig, axs = plt.subplots(1, n, figsize=(n, 1))
  if y is not None:
    fig.suptitle(np.argmax(y, axis=1))
  for i in range(n):
    axs.flat[i].imshow(x[i].squeeze(), interpolation='none', cmap='gray')
    axs.flat[i].axis('off')
  plt.show()
  plt.close()
  plt.ion()

print('Originals:')
display_imgs(x)

print('Decoded Random Samples:')
display_imgs(xhat.sample())

print('Decoded Modes:')
display_imgs(xhat.mode())

print('Decoded Means:')
display_imgs(xhat.mean())


#vae.save('vae')
