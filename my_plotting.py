import argparse
import pathlib
import time
from typing import Tuple

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

ALL_TRACKS_DICT = {'austria': 'AUT', 'barcelona': 'BRC', 'columbia': 'COL',
                   'gbr': 'GBR', 'treitlstrasse': 'HALL',
                   'treitlstrasse_v2': 'HALL', 'treitlstrassev2': 'HALL'}
ALL_METHODS_DICT = {'dreamer': 'Dream', 'mpo': 'MPO', 'd4pg': 'D4PG', 'ppo': 'PPO', 'sac': 'SAC'}
DREAMER_CONFS = {}
Run = collections.namedtuple('Run', 'logdir train_track test_track method seed x y')

BEST_MFREE_PERFORMANCES = {'austria': {'d4pg': 0.38, 'mpo': 0.36, 'ppo': 0.36, 'sac': 0.13},
                           'columbia': {'d4pg': 2.06, 'mpo': 2.13, 'ppo': 2.07, 'sac': 1.73},
                           'treitlstrassev2': {'d4pg': 0.77, 'mpo': 0.30, 'ppo': 0.6, 'sac': 0.31}}
def process_logdir_name(logdir):
  splitted = logdir.split('_')
  if len(splitted) == 4:  # assume logdir: eval_algo_traintrack_timestamp
    _, algo, track, _ = splitted
    seed = -1
  elif len(splitted) == 6:    # assume logdir: track_algo_max_progress_seed_timestamp
    track, algo, _, _, seed, _ = splitted
    seed = int(seed)
  elif len(splitted) == 9:  # assume logdir: track_dreamer_max_progress_ArK_BlL_HH_seed_timestamp
    track, algo, _, _, action_repeat, batch_len, horizon, seed, _ = splitted
    action_repeat = int(''.join(filter(str.isdigit, action_repeat)))  # keep param value
    batch_len = int(''.join(filter(str.isdigit, batch_len)))
    horizon = int(''.join(filter(str.isdigit, horizon)))
    seed = int(seed)
    algo = f'{algo} (AR{action_repeat}, BL{batch_len}, H{horizon})'
  elif len(splitted) == 10:  # assume logdir: track_dreamer_method_max_progress_ArK_BlL_HH_seed_timestamp
    track, algo, method, _, _, action_repeat, batch_len, horizon, seed, _ = splitted
    action_repeat = int(''.join(filter(str.isdigit, action_repeat)))  # keep param value
    batch_len = int(''.join(filter(str.isdigit, batch_len)))
    horizon = int(''.join(filter(str.isdigit, horizon)))
    seed = int(seed)
    algo = f'{algo} + {method}'   #
  else:
    raise NotImplementedError(f'cannot parse {logdir}')
  if "dreamer" in algo:
    if not algo in DREAMER_CONFS.keys():
      DREAMER_CONFS[algo] = f"dreamer{len(DREAMER_CONFS.keys())+1}"
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
    print(f'Loading runs from {dir}', end='')
    files = list(dir.glob('**/events*'))
    for file in files:
      try:
        filepath = file.relative_to(dir).parts[:-1]  # path to jsonl file, e.g. columbia/h20/seed
        train_track, method, seed = process_filepath(filepath)
        if not train_track in args.tracks:
          continue
        if not any([m in method for m in args.methods]):
          continue
        event_acc = EventAccumulator(str(file), size_guidance={'scalars': 100000, 'tensors': 100000})  # max number of items to keep
        event_acc.Reload()
        if args.type=='train':
          if args.tag in event_acc.Tags()['tensors']:
            tag = args.tag
            y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
            x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
          elif args.tag + "_mean" in event_acc.Tags()['tensors']:
            tag = args.tag + "_mean"  # todo, remove it with new experiment runs, use only progress_mean
            y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
            x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
          elif args.tag in event_acc.Tags()['scalars']:
            tag = args.tag
            y = np.array([float(scalar.value) for scalar in event_acc.Scalars(tag)])
            x = np.array([int(scalar.step) for scalar in event_acc.Scalars(tag)])
          elif args.tag + "_mean" in event_acc.Tags()['scalars']:
            tag = args.tag + "_mean"  # todo, remove it with new experiment runs, use only progress_mean
            y = np.array([float(scalar.value) for scalar in event_acc.Scalars(tag)])
            x = np.array([int(scalar.step) for scalar in event_acc.Scalars(tag)])
          else:
            continue
          if 'sac' in method or 'ppo' in method:    # to be consistent with logs collected with sb3
            x = x * 4
          runs.append(Run(filepath, train_track, train_track, method, seed, x, y))  # in this case, train track = test track
          print('.', end='')
        else:
          for test_track in ALL_TRACKS_DICT.keys():
            if not test_track in args.tracks:
              continue
            if f'{test_track}/{args.tag}' in event_acc.Tags()['tensors']:
              tag = f'{test_track}/{args.tag}'
            else:
              continue
            y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
            x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
            runs.append(Run(filepath, train_track, test_track, method, seed, x, y))
            print('.', end='')
      except ValueError as err:
        print(f'Error {file}: {err}')
        continue
      except NotImplementedError as err:
        print(f'Error {file}: {err}')
        continue
    print()
  print_stat(runs)
  return runs

def print_stat(runs):
  train_tracks = sorted(set([r.train_track for r in runs]))
  test_tracks = sorted(set([r.test_track for r in runs]))
  methods = sorted(set([r.method for r in runs]))
  for train_track in train_tracks:
    for test_track in test_tracks:
      for method in methods:
        filter_runs = [r for r in runs if r.train_track == train_track and
                       r.test_track == test_track and r.method == method]
        print(f'Train Track: {train_track}, Test Track: {test_track}, Method: {method}: {len(filter_runs)} experiment')


def aggregate_mean_std(runs: list, binning: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  all_x = np.concatenate([r.x for r in runs])
  all_y = np.concatenate([r.y for r in runs])
  order = np.argsort(all_x)
  all_x, all_y = all_x[order], all_y[order]
  reducer = lambda y: (np.nanmean(np.array(y)), np.nanstd(np.array(y)))
  if binning is None:
    binned_x = [np.max(all_x)]
  else:
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
  return np.array(binned_x), np.array(binned_mean), np.array(binned_mean)-np.array(binned_std), np.array(binned_mean)+np.array(binned_std)

def aggregate_mean_min_max(runs, binning=None):
  # if `binning` is None, aggregate over all the values
  all_x = np.concatenate([r.x for r in runs])
  all_y = np.concatenate([r.y for r in runs])
  order = np.argsort(all_x)
  all_x, all_y = all_x[order], all_y[order]
  reducer = lambda y: (np.nanmean(np.array(y)), np.nanmin(np.array(y)), np.nanmax(np.array(y)))
  if binning is None:
    binned_x = [np.max(all_x)]
  else:
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


def plot_filled_curve(args, runs, axes, aggregator):
  tracks = sorted(set([r.train_track for r in runs]))
  methods = sorted(set([r.method for r in runs]))
  if not type(axes)==np.ndarray:
    axes = [axes]
  for i, (track, ax) in enumerate(zip(tracks, axes)):
    if args.show_labels:
      ax.set_title(ALL_TRACKS_DICT[track].upper())
      ax.set_xlabel(args.xlabel)
      if i <= 0:  # show y label only on first row
        ax.set_ylabel(args.ylabel if args.ylabel else args.tag)
    track_runs = [r for r in runs if r.train_track == track]
    for j, method in enumerate(methods):
      color = PALETTE[j]
      filter_runs = [r for r in track_runs if r.method == method]
      if len(filter_runs) > 0:
        x, mean, min, max = aggregator(filter_runs, args.binning)
        min = np.where(min>0, min, 0)
        ax.plot(x, mean, color=color, label=method.upper())
        ax.fill_between(x, min, max, color=color, alpha=0.1)
    # plot baselines
    min_x = np.min(np.concatenate([r.x for r in track_runs]))
    max_x = np.max(np.concatenate([r.x for r in track_runs]))
    for j, (value, name) in enumerate(zip(args.hbaseline_values, args.hbaseline_names)):
      color = 'red'
      ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dotted', label=name.upper())
    if args.show_mfree_baselines:
      for j, (name, value) in enumerate(BEST_MFREE_PERFORMANCES[track].items()):
        color = PALETTE[len(methods) + len(args.hbaseline_values) + j]
        ax.hlines(y=value, xmin=min_x, xmax=max_x, color=color, linestyle='dashed', label=name.upper())


def aggregate_max(runs):
  all_x = np.concatenate([r.x for r in runs])
  all_y = np.concatenate([r.y for r in runs])
  all_logs = np.concatenate([[r.logdir for _ in r.y] for r in runs])
  order = np.argsort(all_y)
  all_x, all_y, all_logs = all_x[order], all_y[order], all_logs[order]
  return all_x, all_y, all_logs

def get_best_performing_models(args, n_models=5):
  runs = load_runs(args)
  tracks = sorted(set([r.train_track for r in runs]))
  print("\nBest models per track")
  for i, track in enumerate(tracks):
    filter_runs = [r for r in runs if r.track == track]
    sorted_x, sorted_y, sorted_logs = aggregate_max(filter_runs)
    for x, y, log in zip(sorted_x[-n_models:], sorted_y[-n_models:], sorted_logs[-n_models:]):
      print(f'Track: {track}, {args.ylabel}: {y}, {args.xlabel}: {x}, Logdir: {log}')
    print()


def plot_train_figures(args, outdir):
  runs = load_runs(args)
  tracks = sorted(set([r.train_track for r in runs]))
  outdir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  for aggregator, fn in zip(['mean_std', 'mean_minmax'], [aggregate_mean_std, aggregate_mean_min_max]):
    fig, axes = plt.subplots(1, len(tracks), figsize=(3*len(tracks), 3))  # 2 rows: mean +- std, mean btw min max
    plot_filled_curve(args, runs, axes, aggregator=fn)
    if args.legend:
      handles, labels = axes[-1].get_legend_handles_labels()
      fig.legend(handles, labels, loc='lower center', ncol=len(labels), framealpha=1.0)
    filename = f'curves_' + '_'.join(tracks) + f'_{aggregator}_{timestamp}.png'
    fig.tight_layout(pad=1.0)
    fig.savefig(outdir / filename)
    print(f"[Info] Written {outdir / filename}")

def plot_test_figures(args, outdir):
  runs = load_runs(args)
  train_tracks = sorted(set([r.train_track for r in runs]))
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  outdir.mkdir(parents=True, exist_ok=True)
  for aggregator, fn in zip(['mean_std', 'mean_minmax'], [aggregate_mean_std, aggregate_mean_min_max]):
    fig, axes = plt.subplots(1, len(train_tracks), figsize=(3 * len(train_tracks), 3))  # 2 rows: mean +- std, mean btw min max
    for j, train_track in enumerate(train_tracks):
      filter_runs = [r for r in runs if r.train_track == train_track]
      ax = axes[j]
      plot_error_bar(args, filter_runs, ax, aggregator=fn)
      filename = outdir / f'curves_{aggregator}_{timestamp}.png'
      fig.tight_layout(pad=1.0)
      fig.savefig(filename)

def plot_error_bar(args, runs, ax, aggregator):
  train_track = sorted(set([r.train_track for r in runs]))
  assert len(train_track)==1
  train_track = train_track[0]
  test_tracks = sorted(set([r.test_track for r in runs]))
  means, n_errors, p_errors, colors = [], [], [], []
  for j, test_track in enumerate(test_tracks):
    filter_runs = [r for r in runs if r.train_track == train_track and r.test_track == test_track]
    if len(filter_runs) > 0:
      x, mean, min, max = aggregator(filter_runs, None)
      means.append(mean[0])
      n_errors.append(mean[0]-min[0])
      p_errors.append(max[0]-mean[0])
      colors.append('green' if ALL_TRACKS_DICT[train_track]==ALL_TRACKS_DICT[test_track] else 'red')
  xpos = np.arange(len(test_tracks))
  ax.bar(xpos, means, yerr=np.array([n_errors, p_errors]), align='center', alpha=0.5, color=colors, ecolor='black', capsize=10)
  ax.set_xticks(xpos)
  ax.set_xticklabels([ALL_TRACKS_DICT[track] for track in test_tracks])
  ax.set_title(f'Train {train_track.title()}')
  ax.set_ylabel(args.ylabel)
  ax.set_ylim(0, 1.1)


def main(args):
  init = time.time()
  assert len(args.hbaseline_names) == len(args.hbaseline_values)
  outdir = args.outdir / f'{args.type}'
  if args.type == 'train':
    args.tag = 'test/progress'
    plot_train_figures(args, outdir)
  elif args.type == 'test':
    args.tag = 'progress'     # it will be composed as f'{track}/progress'
    plot_test_figures(args, outdir)
  elif args.type == 'stat':
    args.tag = 'test/progress'
    get_best_performing_models(args, n_models=5)
  else:
    raise NotImplementedError(f'not implemented type = {args.type}')
  print(f"\n[Info] Elapsed Time: {time.time()-init:.3f} seconds")


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--type', type=str, required=True, choices=['train', 'test', 'stat'])
  parser.add_argument('--xlabel', type=str, default="")
  parser.add_argument('--ylabel', type=str, default="")
  parser.add_argument('--binning', type=int, default=15000)
  parser.add_argument('--legend', action='store_true')
  parser.add_argument('--show_labels', action='store_true')
  parser.add_argument('--show_mfree_baselines', action='store_true')
  parser.add_argument('--tracks', nargs='+', type=str, default=ALL_TRACKS_DICT.keys())
  parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
  parser.add_argument('--hbaseline_names', nargs='+', type=str, default=[])
  parser.add_argument('--hbaseline_values', nargs='+', type=float, default=[])
  return parser.parse_args()


if __name__=='__main__':
  main(parse())