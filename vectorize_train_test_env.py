from dreamer import define_config, make_env
import tensorflow as tf
import tools
import wrappers

# Create environments.
config = define_config()
datadir = config.logdir / 'episodes'
writer = tf.summary.create_file_writer(
    str(config.logdir), max_queue=1000, flush_millis=20000)
writer.set_as_default()

env = wrappers.SingleForkedRaceCarWrapper(id='A', name=config.task + '-v0')
