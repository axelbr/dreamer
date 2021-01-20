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

Run = collections.namedtuple('Run', 'track method seed x y')

def load_runs(args):
  runs = []
  for dir in args.indir:
    #files = list(dir.glob('**/*jsonl'))
    files = list(dir.glob('**/events*'))
    for file in files:
      try:
        track, method, seed = file.relative_to(dir).parts[:-1]  # path to jsonl file, e.g. columbia/h20/seed
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
        yy = [float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)]
        xx = [tensor.step for tensor in event_acc.Tensors(tag)]
        df = pd.DataFrame(list(zip(xx, yy)), columns=[args.xaxis, args.yaxis])
        x = df[args.xaxis].to_numpy()
        y = df[args.yaxis].to_numpy()
        runs.append(Run(track, method, seed, x, y))
        print(f'Track: {track}, method: {method}, seed: {seed}.')
      except ValueError as err:
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