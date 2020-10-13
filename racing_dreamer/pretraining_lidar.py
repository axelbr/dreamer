import tensorflow as tf
import tensorflow_probability as tfp

from racing_dreamer.dataset import load_lidar

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
            tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            tfkl.Conv1D(base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv1D(2 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv1D(4 * encoded_size, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Flatten(),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),
            tfpl.MultivariateNormalTriL(encoded_size, activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior)),
        ])

        self.decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=self.encoder.output_shape),
            tfkl.Reshape([encoded_size,1]),
            tfkl.Conv1DTranspose(2 * base_depth, 7, strides=1, padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Conv1DTranspose(2 * base_depth, 5, strides=1, padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv1DTranspose(2 * base_depth, 5, strides=2, padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv1D(filters=1, kernel_size=5, strides=1, padding='same', activation=None),
            tfkl.Flatten(),
            tfkl.Dense(tfpl.IndependentNormal.params_size(tf.reduce_prod(input_shape)), activation=None),
            tfpl.IndependentNormal(input_shape),
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


training_data, test_data = load_lidar('pretraining_data/austria_single_20000.h5', train=0.8, shuffle=True)
training_data = training_data\
    .map(preprocess)\
    .batch(256)\
    .prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data\
    .map(preprocess) \
    .batch(32) \
    .prefetch(tf.data.experimental.AUTOTUNE)

vae = CVAE(input_shape=(1080, 1))

negloglik = lambda x, rv_x: -rv_x.log_prob(x)
optimizer = tf.optimizers.Adam(learning_rate=1e-3)
vae.model.compile(optimizer=optimizer,loss=negloglik)
vae.encoder.summary()
vae.decoder.summary()

for epoch in range(15):
    print(f'Epoch {epoch}/15')
    epoch_loss = 0
    b = 0
    for batch in iter(training_data):
        b+=1
        with tf.GradientTape() as tape:
            latent = vae.encoder(batch)
            sample = latent.sample()
            recon_dist = vae.decoder(latent)
            loss = tf.reduce_mean(negloglik(tf.expand_dims(batch, -1), recon_dist))
        gradients = tape.gradient(loss, vae.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.model.trainable_variables))
        epoch_loss += loss.numpy()
        print(epoch_loss/b)