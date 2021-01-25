import argparse
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import warnings
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime

PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')

Run = collections.namedtuple('Run', 'logdir track method seed x y')

def process_logdir_name(logdir):
  splitted = logdir.split('_')
  if len(splitted) == 6:    # assume logdir: track_algo_max_progress_seed_timestamp
    track, algo, _, _, seed, _ = splitted
    seed = int(seed)
  elif len(splitted) == 9:  # assume logdir: track_dreamer_max_progress_ArK_BlL_HH_seed_timestamp
    track, algo, _, _, action_repeat, batch_len, horizon, seed, _ = splitted
    action_repeat = int(''.join(filter(str.isdigit, action_repeat)))  # keep param value
    batch_len = int(''.join(filter(str.isdigit, batch_len)))
    horizon = int(''.join(filter(str.isdigit, horizon)))
    seed = int(seed)
    algo = f'{algo} (AR{action_repeat}, BL{batch_len}, H{horizon})'
  else:
    raise NotImplementedError(f'cannot parse {logdir}')
  return track, algo, seed

def process_filepath(path_list):
  if len(path_list) == 1:     # e.g. austria_dreamer_max_progress_seed_timestamp
    logdir = path_list[0]
    track, method, seed = process_logdir_name(logdir)
  elif len(path_list) == 2:   # e.g. AR8/austria_dreamer_max_progress_seed_timestamp
    param, logdir = path_list
    track, method, seed = process_logdir_name(logdir)
    method = method + ' ' + param
  else:
    raise NotImplementedError(f'processing not defined for {path_list}')
  return track, method, seed

def load_runs(args):
  runs = []
  for dir in args.indir:
    #files = list(dir.glob('**/*jsonl'))
    files = list(dir.glob('**/events*'))
    for file in files:
      try:
        filepath = file.relative_to(dir).parts[:-1]  # path to jsonl file, e.g. columbia/h20/seed
        track, method, seed = process_filepath(filepath)
        if not track in args.tracks:
          continue
        event_acc = EventAccumulator(str(file), size_guidance={'tensors': 1000})  # max number of items to keep
        event_acc.Reload()
        if args.yaxis in event_acc.Tags()['tensors']:
          tag = args.yaxis
        elif 'sim/' + args.yaxis in event_acc.Tags()['tensors']:
          tag = 'sim/' + args.yaxis
        else:
          continue
        y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
        x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
        runs.append(Run(filepath, track, method, seed, x, y))
        print(f'Track: {track}, method: {method}, seed: {seed}.')
      except ValueError as err:
        print(f'Error {file}: {err}')
        continue
      except NotImplementedError as err:
        print(f'Error {file}: {err}')
        continue
  return runs


def aggregate_mean_std(runs, binning):
  all_x = np.concatenate([r.x for r in runs])
  all_y = np.concatenate([r.y for r in runs])
  order = np.argsort(all_x)
  all_x, all_y = all_x[order], all_y[order]
  reducer = lambda y: (np.nanmean(np.array(y)), np.nanstd(np.array(y)))
  binned_x = np.arange(all_x.min(), all_x.max() + binning, binning)
  binned_mean = []
  binned_std = []
  for start, stop in zip([-np.inf] + list(binned_x), list(binned_x)):
    left = (all_x <= start).sum()
    right = (all_x <= stop).sum()
    with warnings.catch_warnings():
      warnings.filterwarnings('error')
      try:
        mean, std = reducer(all_y[left:right])
      except RuntimeWarning as wrn:
        print(f'[WARNING] {wrn}. Consider to increase the binning')
        exit(1)
    binned_mean.append(mean)
    binned_std.append(std)
  return np.array(binned_x), np.array(binned_mean), np.array(binned_std)

def aggregate_mean_min_max(runs, binning):
  all_x = np.concatenate([r.x for r in runs])
  all_y = np.concatenate([r.y for r in runs])
  order = np.argsort(all_x)
  all_x, all_y = all_x[order], all_y[order]
  reducer = lambda y: (np.nanmean(np.array(y)), np.nanmin(np.array(y)), np.nanmax(np.array(y)))
  binned_x = np.arange(all_x.min(), all_x.max() + binning, binning)
  binned_mean = []
  binned_min = []
  binned_max = []
  for start, stop in zip([-np.inf] + list(binned_x), list(binned_x)):
    left = (all_x <= start).sum()
    right = (all_x <= stop).sum()
    with warnings.catch_warnings():
      warnings.filterwarnings('error')
      try:
        mean, min, max = reducer(all_y[left:right])
      except RuntimeWarning as wrn:
        print(f'[WARNING] {wrn}. Consider to increase the binning')
        exit(1)
    binned_mean.append(mean)
    binned_min.append(min)
    binned_max.append(max)
  return np.array(binned_x), np.array(binned_mean), np.array(binned_min), np.array(binned_max)


def plot_mean_std(args, runs, axes):
  tracks = sorted(set([r.track for r in runs]))
  methods = sorted(set([r.method for r in runs]))
  if not type(axes)==np.ndarray:
    axes = [axes]
  for i, (track, ax) in enumerate(zip(tracks, axes)):
    ax.set_title(track.title())
    ax.set_xlabel(args.xlabel if args.xlabel else args.xaxis)
    if i <= 0:  # show y label only on first row
      ax.set_ylabel(args.ylabel if args.ylabel else args.yaxis)
    for j, method in enumerate(methods):
      color = PALETTE[j]
      filter_runs = [r for r in runs if r.track == track and r.method == method]
      if len(filter_runs) > 0:
        x, mean, std = aggregate_mean_std(filter_runs, args.binning)
        ax.plot(x, mean, color=color, label=method.title())
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
    ax.legend()

def plot_mean_min_max(args, runs, axes):
  tracks = sorted(set([r.track for r in runs]))
  methods = sorted(set([r.method for r in runs]))
  if not type(axes)==np.ndarray:
    axes = [axes]
  for i, (track, ax) in enumerate(zip(tracks, axes)):
    ax.set_title(track.title())
    ax.set_xlabel(args.xlabel if args.xlabel else args.xaxis)
    if i <= 0:  # show y label only on first row
      ax.set_ylabel(args.ylabel if args.ylabel else args.yaxis)
    for j, method in enumerate(methods):
      color = PALETTE[j]
      filter_runs = [r for r in runs if r.track == track and r.method == method]
      if len(filter_runs) > 0:
        x, mean, min, max = aggregate_mean_min_max(filter_runs, args.binning)
        ax.plot(x, mean, color=color, label=method.title())
        ax.fill_between(x, min, max, color=color, alpha=0.1)
    ax.legend()


def aggregate_max(runs):
  all_x = np.concatenate([r.x for r in runs])
  all_y = np.concatenate([r.y for r in runs])
  all_logs = np.concatenate([[r.logdir for _ in r.y] for r in runs])
  order = np.argsort(all_y)
  all_x, all_y, all_logs = all_x[order], all_y[order], all_logs[order]
  return all_x, all_y, all_logs

def get_best_performing_models(args, runs, n_models=5):
  tracks = sorted(set([r.track for r in runs]))
  print("\nBest models per track")
  for i, track in enumerate(tracks):
    filter_runs = [r for r in runs if r.track == track]
    sorted_x, sorted_y, sorted_logs = aggregate_max(filter_runs)
    for x, y, log in zip(sorted_x[-n_models:], sorted_y[-n_models:], sorted_logs[-n_models:]):
      print(f'Track: {track}, {args.ylabel}: {y}, {args.xlabel}: {x}, Logdir: {log}')
    print()


def main(args):
  runs = load_runs(args)
  tracks = sorted(set([r.track for r in runs]))
  fig, axes = plt.subplots(2, len(tracks))    # 2 rows: mean +- std, mean btw min max
  plot_mean_std(args, runs, axes[0])
  plot_mean_min_max(args, runs, axes[1])
  args.outdir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = args.outdir / f'curves_{timestamp}.png'
  fig.tight_layout(pad=1.0)
  fig.savefig(filename)
  get_best_performing_models(args, runs, n_models=5)

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--xaxis', type=str, required=False, default='steps')
  parser.add_argument('--yaxis', type=str, required=True)
  parser.add_argument('--xlabel', type=str, default="")
  parser.add_argument('--ylabel', type=str, default="")
  parser.add_argument('--binning', type=int, default=10000)
  parser.add_argument('--tracks', nargs='+', type=str, default=['austria', 'columbia', 'treitlstrasse'])
  return parser.parse_args()


if __name__=='__main__':
  main(parse())