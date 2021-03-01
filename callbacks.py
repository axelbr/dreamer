import datetime
import io
import pathlib
import time
import uuid

import imageio

import tools
import numpy as np

import tensorflow as tf

def save_videos(videos, config, datadir):
    if not config.log_videos:
        return
    step = tools.count_steps(datadir, config)
    video_dir = config.logdir / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    for filename, video in videos.items():
        writer = imageio.get_writer(f'{video_dir}/{filename}_{step}_{time.time()}.mp4', fps=100 // config.action_repeat)
        for image in video:
            writer.append_data(image)
        writer.close()


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(episode['reward'])
        filename = directory / f'{timestamp}-{identifier}-{length}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())


def summarize_episode(episode_list, config, datadir, writer, prefix):
    """
    Write episode summary in tensorflow logs. Even if multi-agent returns multiple episodes (1 for each agent),
    the summary is written w.r.t. the first agent.

    :param episode_list:    list of episodes (in multi-agent setting, each agent produce 1 episode)
    :param config:          config dictionary
    :param datadir:         dir where the episodes are stored
    :param writer:          tf writer for logging
    :param prefix:          either `train` or `test`
    :return:
    """
    episode = episode_list[0]
    episodes, steps = tools.count_episodes(datadir)
    episode_len = len(episode['reward']) - 1
    length = episode_len * config.action_repeat
    ret = episode['reward'].sum()
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        (f'{prefix}/progress', float(max(episode['progress']))),
        (f'episodes', episodes)]
    step = tools.count_steps(datadir, config)
    with writer.as_default():  # Env might run in a different thread.
        tf.summary.experimental.set_step(step)
        [tf.summary.scalar(k, v) for k, v in metrics]
    print(f'\t[Summary] {prefix.title()} episode of length {episode_len} ({length} sim steps) with return {ret:.1f}.')

