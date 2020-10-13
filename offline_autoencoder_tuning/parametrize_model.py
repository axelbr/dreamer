import tensorflow as tf
import tools
from tensorflow.keras import layers
from tensorflow.keras import models


class Encoder(layers.Layer):
    def __init__(self, parameters, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.model = models.Sequential(name=name)
        for l in range(parameters.n_layers):
            layer = tf.keras.layers.Conv1D(filters=parameters.n_filter,
                                           kernel_size=parameters.kernel_sz,
                                           activation=parameters.h_activation,
                                           strides=parameters.stride
                                           )
            self.model.add(layer)
        self.model.add(layers.Flatten())
        output_layer = layers.Dense(units=parameters.latent_sz,
                                    activation=parameters.o_activation,
                                    )
        self.model.add(output_layer)

    def call(self, inputs, training=None, mask=None):
        return tf.expand_dims(self.model(inputs), -1)

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.model.summary()


class Decoder(layers.Layer):
    def __init__(self, parameters, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.model = models.Sequential(name=name)
        for l in range(parameters.n_layers):
            layer = tf.keras.layers.Conv1DTranspose(filters=parameters.n_filter,
                                                    kernel_size=parameters.kernel_sz,
                                                    activation=parameters.h_activation,
                                                    strides=parameters.stride
                                                    )
            self.model.add(layer)
        self.model.add(layers.Flatten())
        output_layer = tf.keras.layers.Dense(units=parameters.input_sz,
                                             activation=parameters.o_activation,
                                             )
        self.model.add(output_layer)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.model.summary()

class Autoencoder(models.Model):
    def __init__(self, parameters, name="autoencoder"):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(parameters)
        self.decoder = Decoder(parameters)

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        return self.decoder(z)

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary()
        self.decoder.summary()


# parameters
param = tools.AttrDict()
param.n_layers = 3
param.n_filter = 4
param.stride = 2
param.kernel_sz = 5
param.input_sz = 1000
param.latent_sz = 128
param.h_activation = "relu"
param.o_activation = "relu"
param.batch_sz = 32

model = Autoencoder(param)
model.build(input_shape=(None, param.input_sz, 1))
model.summary()
