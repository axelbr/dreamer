import argparse
import functools
import os
import pathlib
import sys
from typing import List

import gym
import tensorflow as tf
from dreamer.baselines import GapFollower
from rlephant import Dataset
from tensorflow.keras.mixed_precision import experimental as precision

from .environments import make_async_env, make_env

sys.path.append(str(pathlib.Path(__file__).parent))

import dreamer.tools as tools
from dreamer import RacingDreamer

def configure_agent(config, train_envs: List[gym.Env], writer: tf.summary.SummaryWriter) -> RacingDreamer:
  datadir = config.logdir / 'episodes'
  agent = RacingDreamer(RacingDreamer.Config(), train_envs[0])
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  return agent


def train_dreamer(agent, config, test_envs, train_envs, dataset, writer):
  datadir = config.logdir / 'episodes'
  state = None
  #state = tools.simulate(agent, train_envs, 10000, state=state, training=False)

  # Train and regularly evaluate the agent.
  step = tools.count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')

  for logs in agent.train(steps=config.steps, dataset=dataset, env=train_envs[0]):
    print(logs)
    logs['value_losses'] = tf.reduce_mean(logs['value_losses'])
    logs['actor_losses'] = tf.reduce_mean(logs['actor_losses'])
    logs['dynamics_losses'] = tf.reduce_mean(logs['dynamics_losses'])
    for k, v in logs.items():
      with writer.as_default():
        tf.summary.scalar(k, v, step=logs['step'])

def initialize_dataset(config, train_envs, writer):
  datadir = config.logdir / 'episodes.h5'
  step = tools.count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset with {prefill} steps.')
  gapfollower = GapFollower()
  dataset = Dataset(filename=datadir)
  if len(dataset) == 0:
    tools.collect_data(agent=lambda o: gapfollower.action(o), env=train_envs[0], steps=prefill, dataset=dataset)
  writer.flush()
  return dataset

def configure_envs(config, writer: tf.summary.SummaryWriter):
  datadir = config.logdir / 'episodes'
  train_envs = [make_env(config, writer, 'train', datadir, store=True, gui=False)
                for _ in range(config.envs)]

  test_envs = [make_env(config, writer, 'test', datadir, store=False, gui=False)
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
  dataset = initialize_dataset(config, train_envs, writer)
  agent = configure_agent(config, train_envs, writer)
  train_dreamer(
    agent=agent,
    config=config,
    test_envs=test_envs,
    train_envs=train_envs,
    dataset=dataset,
    writer=writer
  )


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
