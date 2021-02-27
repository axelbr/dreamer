import tensorflow as tf
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

from dreamer import preprocess
import tools

def make_log_dir(args):
  out_dir = args.outdir / f'eval_{args.agent}_{args.trained_on.replace("_", "")}_{args.obs_type.replace("_", "")}_{time.time()}'
  out_dir.mkdir(parents=True, exist_ok=True)
  writer = tf.summary.create_file_writer(str(out_dir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  return out_dir, writer

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

def save_trajectory(episodes, outdir, action_repeat, track, checkpoint_id):
  trajectory_dir = outdir / 'trajectories'
  trajectory_dir.mkdir(parents=True, exist_ok=True)
  # note: in multi-agent, each agent produce 1 episode
  episode = episodes[0]  # we save w.r.t. the episode of the first agent
  episodes = count_videos(outdir / 'videos')
  positions = episode['pose'][:, :2]
  velocities = episode['velocity'][:, 0]
  filename = trajectory_dir / f"trajectory_{episodes}_{track}_checkpoint{checkpoint_id}"
  np.savez(filename, position=positions, velocity=velocities)


