import functools
import pathlib
import shutil
import time
import numpy as np

import imageio
from racecar_gym.envs import ChangingTrackMultiAgentRaceEnv
from racecar_gym import MultiAgentScenario
from agents.gap_follower import GapFollower
import argparse

import tools
import wrappers
from dreamer import define_config, Dreamer

import tensorflow as tf

#tf.config.run_functions_eagerly(run_eagerly=True)

def init_agent(agent_name: str, env):
  if agent_name == "dreamer":
    agent = init_dreamer(env)
  else:
    raise NotImplementedError(f'not implemented {agent_name}')
  return agent

def init_dreamer(env):
  config = define_config()
  config.log_scalars = True     # note: important log_scalars True for build_models
  config.log_images = False
  config.training = False
  config.batch_length = 5
  config.batch_size = 5
  config.horizon = 5
  # prefill environment for model initialization
  datadir = pathlib.Path('.tmp')
  writer = tf.summary.create_file_writer(str(datadir), max_queue=1000, flush_millis=20000)
  callbacks = []
  callbacks.append(lambda episodes: tools.save_episodes(datadir, episodes))
  prefill_env = wrappers.Collect(env, callbacks, config.precision)
  # random agent to prefill
  random_agent = lambda o, d, s: ([prefill_env.action_space['A'].sample()], None)  # note: it must work as single_agent agent
  tools.simulate([random_agent for _ in range(prefill_env.n_agents)], prefill_env, config, datadir, writer,
                 prefix='prefill', episodes=10)
  # initialize model
  actspace = env.action_space
  obspace = env.observation_space
  dreamer = Dreamer(config, datadir, actspace, obspace, writer=None)
  # remove tmp directory
  shutil.rmtree(datadir)
  return dreamer


def load_checkpoint(agent_name, base_agent, checkpoint, env):
  if agent_name == "dreamer":
    base_agent.load(checkpoint)
    agent = functools.partial(base_agent, training=False)
  else:
    raise NotImplementedError(f'not implemented {agent_name}')
  return agent


def count_videos(directory):
  filenames = directory.glob('**/*.mp4')
  return sum(1 for _ in filenames)

def save_videos(videos, video_dir, action_repeat, track, checkpoint_id):
  video_dir.mkdir(parents=True, exist_ok=True)
  episodes = count_videos(video_dir)
  for filename, video in videos.items():
    writer = imageio.get_writer(f'{video_dir}/{filename}_{episodes + 1}_{track}_checkpoint{checkpoint_id}.mp4', fps=100 // action_repeat)
    for image in video:
      writer.append_data(image)
    writer.close()

def summarize_episode(episodes, outdir, writer, prefix, action_repeat):
  # note: in multi-agent, each agent produce 1 episode
  episode = episodes[0]  # we summarize w.r.t. the episode of the first agent
  episodes = count_videos(outdir / 'videos')
  episode_len = len(episode['reward']) - 1
  length = episode_len * action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} - Episode of length {episode_len} ({length} sim steps) with return {ret:.1f}.')
  metrics = [
    (f'{prefix}/return', float(episode['reward'].sum())),
    (f'{prefix}/length', len(episode['reward']) - 1),
    (f'{prefix}/progress', float(max(episode['progress']))),
    (f'{prefix}/time', float(max(episode['time'])))]
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(episodes)
    [tf.summary.scalar(k, v) for k, v in metrics]

def make_multi_track_env(tracks, action_repeat, rendering=True):
  scenarios = [MultiAgentScenario.from_spec(f'scenarios/eval/{track}.yml', rendering=rendering) for track in tracks]
  env = ChangingTrackMultiAgentRaceEnv(scenarios=scenarios, order='manual')
  env = wrappers.RaceCarWrapper(env)
  env = wrappers.FixedResetMode(env, mode='grid')
  env = wrappers.ActionRepeat(env, action_repeat)
  env = wrappers.ReduceActionSpace(env, low=[0.005, -1.0], high=[1.0, 1.0])
  return env

def wrap_wrt_track(env, action_repeat, outdir, writer, track, checkpoint_id):
  render_callbacks = []
  render_callbacks.append(lambda videos: save_videos(videos, outdir / 'videos', args.action_repeat, track, checkpoint_id))
  env = wrappers.Render(env, render_callbacks, follow_view=False)
  callbacks = []
  callbacks.append(lambda episodes: summarize_episode(episodes, outdir, writer, f'{track}', action_repeat))
  env = wrappers.Collect(env, callbacks)
  return env

def make_log_dir(args):
  out_dir = args.outdir / f'eval_{args.agent}_{args.trained_on}_{time.time()}'
  out_dir.mkdir(parents=True, exist_ok=True)
  writer = tf.summary.create_file_writer(str(out_dir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  return out_dir, writer

def copy_checkpoint(checkpoint_file, outdir, checkpoint_id):
  cp_dir = outdir / f'checkpoints/{checkpoint_id}'
  cp_dir.mkdir(parents=True, exist_ok=True)
  shutil.copy(checkpoint_file, cp_dir)

def main(args):
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  basedir, writer = make_log_dir(args)
  base_env = make_multi_track_env(args.tracks, action_repeat=args.action_repeat, rendering=False)
  base_agent = init_agent(args.agent, base_env)
  print(f"[Info] Agent Variables: {len(base_agent.variables)}")
  for i, checkpoint in enumerate(args.checkpoints):
    copy_checkpoint(checkpoint, basedir, checkpoint_id=i+1)
    # load agent
    agent = load_checkpoint(args.agent, base_agent, checkpoint, base_env)
    for track in args.tracks:
      print(f"[Info] Checkpoint {i + 1}: {checkpoint}, Track: {track}")
      env = wrap_wrt_track(base_env, args.action_repeat, basedir, writer, track, checkpoint_id=i+1)
      for episode in range(args.eval_episodes):
        obs = env.reset()
        done = False
        agent_state = None
        while not done:
          obs = {id: {k: np.stack([v]) for k, v in o.items()} for id, o in obs.items()}  # dreamer needs size (1, 1080)
          action, agent_state = agent(obs['A'], np.array([done]), agent_state)
          action = {'A': np.array(action[0])}  # dreamer returns action of shape (1,2)
          obs, rewards, dones, infos = env.step(action)
          done = dones['A']
      env.set_next_env()
  env.close()


def parse():
  tracks = ['austria', 'barcelona', 'gbr', 'treitlstrasse', 'treitlstrasse_v2']
  parser = argparse.ArgumentParser()
  parser.add_argument('--agent', type=str, choices=["dreamer"], required=True)
  parser.add_argument('--trained_on', type=str, required=True, choices=tracks)
  parser.add_argument('--checkpoints', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--tracks', nargs='+', type=str, default=tracks)
  parser.add_argument('--eval_episodes', nargs='?', type=int, default=10)
  parser.add_argument('--action_repeat', nargs='?', type=int, default=8)
  return parser.parse_args()

if __name__=="__main__":
  args = parse()
  main(args)