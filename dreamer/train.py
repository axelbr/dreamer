import argparse
import functools
import os
import pathlib
import sys
from typing import List

import gym
import tensorflow as tf
from agents import gap_follower
from tensorflow.keras.mixed_precision import experimental as precision

from dreamer.environments import make_async_env

sys.path.append(str(pathlib.Path(__file__).parent))

import dreamer.tools as tools
from dreamer.models import Dreamer

def configure_agent(config, train_envs: List[gym.Env], writer: tf.summary.SummaryWriter) -> Dreamer:
  datadir = config.logdir / 'episodes'
  agent = Dreamer(config, datadir, train_envs[0].action_space, train_envs[0].observation_space, writer)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  return agent


def train_dreamer(agent, config, test_envs, train_envs, writer):
  datadir = config.logdir / 'episodes'
  state = None
  state = tools.simulate(agent, train_envs, 10000, state=state, training=False)

  # Train and regularly evaluate the agent.
  step = tools.count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')

  while step < config.steps:
    print('Start evaluation.')
    tools.simulate(
      functools.partial(agent, training=False), test_envs, episodes=1)
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate(agent, train_envs, steps, state=state)
    step = tools.count_steps(datadir, config)
    agent.save(config.logdir / 'variables.pkl')
  for env in train_envs + test_envs:
    env.close()


def initialize_dataset(config, train_envs, writer):
  datadir = config.logdir / 'episodes'

  step = tools.count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset with {prefill} steps.')
  gapfollower = gap_follower.GapFollower()
  random_agent = lambda o, d, _: ([gapfollower.action({'lidar':o['lidar'][0]}) for _ in d], None)
  tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()


def configure_envs(config, writer: tf.summary.SummaryWriter):
  datadir = config.logdir / 'episodes'
  train_envs = [make_async_env(config, writer, 'train', datadir, store=True, gui=False)
                for _ in range(config.envs)]

  test_envs = [make_async_env(config, writer, 'test', datadir, store=False, gui=False)
               for _ in range(config.envs)]
  return test_envs, train_envs


def setup_logging(config):
  config.logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', config.logdir)
  writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  return writer


def setup_tf(config):
  tf.config.run_functions_eagerly(run_eagerly=True)
  tf.get_logger().setLevel('ERROR')
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    precision.set_policy(precision.Policy('mixed_float16'))


def main(config):
  setup_tf(config)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  os.environ['MUJOCO_GL'] = 'egl'
  writer = setup_logging(config)
  test_envs, train_envs = configure_envs(config, writer=writer)
  initialize_dataset(config, train_envs, writer)
  agent = configure_agent(config, train_envs, writer)
  train_dreamer(agent, config, test_envs, train_envs, writer)


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  from dreamer.config import default
  parser = argparse.ArgumentParser()
  for key, value in default().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())
