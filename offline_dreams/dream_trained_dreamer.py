from dreamer import make_env, define_config, Dreamer, count_steps
import tensorflow as tf
import wrappers
import functools
import tools
import argparse
import racecar_gym
from racecar_gym.envs import MultiAgentRaceEnv, MultiAgentScenario
from time import sleep
import os

# disable run on gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#if tf.test.gpu_device_name():
#    print('GPU found')
#else:
#    print("No GPU found")

def create_one_episode_for_init(config, writer, datadir):
  prefill_envs = [make_env(config, writer, 'prefill', datadir, store=True, gui=False)]
  actspace = prefill_envs[0].action_space
  random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
  tools.simulate(random_agent, prefill_envs, episodes=1, training=False)

def main(config):
  # Create environments.
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()

  init_steps = count_steps(datadir, config)
  if init_steps == 0:
    create_one_episode_for_init(config, writer, datadir)

  # hard-code parameters
  config.batch_length = 1
  config.batch_size = 1
  config.log_images = False

  # create test environment
  test_env = make_env(config, writer, 'train', datadir, store=False, gui=True)
  actspace = test_env.action_space
  obspace = test_env.observation_space

  # load dreamer
  dreamer = Dreamer(config, datadir, actspace, obspace, writer)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    dreamer.load(config.logdir / 'variables.pkl')
  else:
    raise FileNotFoundError(config.logdir / 'variables.pkl')

  simulate(functools.partial(dreamer, training=False), test_env, steps=10)
  #todo: collect first steps (e.g. 10) and dream for a certain horizon
  test_env.close()

import numpy as np
def simulate(agent, env, steps=False, episodes=False):
  step, episode = 0, 0
  done = True
  obs = None
  agent_state = None
  while (steps and step < steps) or (episodes and episode < episodes):
    # Reset envs if necessary.
    if done:
      obs = env.reset()
    # Step agents.
    obs = {k: np.array([obs[k]]) for k in obs.keys()}
    action, agent_state = agent(obs, np.array([done]), agent_state)
    action = action[0]
    # Step envs.
    obs, reward, done, info = env.step(action)
    episode += int(done)
    step += 1


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())