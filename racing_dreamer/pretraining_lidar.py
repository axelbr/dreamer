import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from racing_dreamer.dataset import load_lidar
import time
import tools

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class CVAE:
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
            tfkl.Flatten(),                                                                             # (63232)
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),                          # (152)
            tfpl.MultivariateNormalTriL(encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior)),
        ])

        self.decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=(encoded_size,)),                                                       # (16)
            tfkl.Dense(units=int(np.ceil((input_shape[0] / 2) - 7 + 1) * 4 * base_depth)),                      # (63232)
            tfkl.Reshape(target_shape=[int(np.ceil((input_shape[0] / 2) - 7 + 1)), 4 * base_depth]),            # (494, 128)
            tfkl.Conv1DTranspose(2 * base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),   # (500, 64)
            tfkl.Conv1DTranspose(1 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),    # (1000, 32)
            tfkl.Conv1D(filters=1, kernel_size=5, strides=1, padding='same', activation=None),                  # (1000, 1)
            tfkl.Flatten(),
            tfpl.IndependentBernoulli(input_shape),
        ])

        self.model = tfk.Model(inputs=self.encoder.inputs, outputs=self.encoder.outputs[0])


    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=None, **kwargs):
        self.model.compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, **kwargs)

    def build(self, input_shape):
        return self.model.build(input_shape)

    def summary(self, line_length=None, positions=None, print_fn=None):
        return self.model.summary(line_length, positions, print_fn)

    def call(self, inputs, training=None, mask=None):
        x_hat = self.model(inputs, training, mask)
        return x_hat


def preprocess(x, max=5.0):
    sample = tf.cast(x, tf.float32) / max
    return sample

lidar_file = "../offline_autoencoder_tuning/out/dataset_random_starts_austria_2000episodes_1000maxobs.h5"
n_epochs = 20
training_batch = 256

training_data, test_data = load_lidar(lidar_file, train=0.8, shuffle=True)
training_data = training_data\
    .map(preprocess)\
    .batch(training_batch)\
    .prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data\
    .map(preprocess) \
    .batch(32) \
    .prefetch(tf.data.experimental.AUTOTUNE)

vae = CVAE(input_shape=(1000, 1))

negloglik = lambda x, rv_x: -rv_x.log_prob(x)
optimizer = tf.optimizers.Adam(learning_rate=1e-3)
vae.model.compile(optimizer=optimizer, loss=negloglik, run_eagerly=True)
vae.encoder.summary()
vae.decoder.summary()

init = time.time()
for epoch in range(n_epochs):
    print(f'Epoch {epoch}/{n_epochs}')
    epoch_loss = 0
    b = 0
    for batch in iter(training_data):
        b += 1
        with tf.GradientTape() as tape:
            latent = vae.encoder(batch)
            sample = latent.sample()
            recon_dist = vae.decoder(latent)
            loss = tf.reduce_mean(negloglik(tf.expand_dims(batch, -1), recon_dist))
        gradients = tape.gradient(loss, vae.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.model.trainable_variables))
        epoch_loss += loss.numpy()
        if b % 10 == 0:
            print("epoch {}, batch {} => avg loss {:.3f}".format(epoch, b, epoch_loss / b))
print("[Info] Training completed in {:.3}s".format(time.time()-init))

init = time.time()
test_loss = 0
for batch in iter(test_data):
    b += 1
    latent = vae.encoder(batch)
    sample = latent.sample()
    recon_dist = vae.decoder(latent)
    tools.
    loss = tf.reduce_mean(negloglik(tf.expand_dims(batch, -1), recon_dist))
    test_loss += loss.numpy()
    if b % 10 == 0:
        print("test, batch {} => avg loss {:.3f}".format(b, test_loss/b))
print("[Info] Testing completed in {:.3}s".format(time.time()-init))

#vae.encoder.save("racing_dreamer/models/encoder_{}epochs_{}batch".format(n_epochs, training_batch))
#vae.decoder.save("racing_dreamer/models/decoder_{}epochs_{}batch".format(n_epochs, training_batch))