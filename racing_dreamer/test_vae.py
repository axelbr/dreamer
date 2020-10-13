import tensorflow as tf

from racing_dreamer.pretraining import encoder, decoder

vae = tf.keras.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0])