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


def load_runs(args, file_parsers, tag):
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
        if args.type == 'train':
          if tag in event_acc.Tags()['tensors']:
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
          for test_track in LONG_TRACKS_DICT.keys():
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
  return runs


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
      colors.append('green' if SHORT_TRACKS_DICT[train_track]==SHORT_TRACKS_DICT[test_track] else 'red')
  xpos = np.arange(len(test_tracks))
  ax.bar(xpos, means, yerr=np.array([n_errors, p_errors]), align='center', alpha=0.5, color=colors, ecolor='black', capsize=10)
  ax.set_xticks(xpos)
  ax.set_xticklabels([SHORT_TRACKS_DICT[track] for track in test_tracks])
  ax.set_title(f'TRAIN {LONG_TRACKS_DICT[train_track]}')
  ax.set_ylabel(args.ylabel)
  ax.set_ylim(0, 1.1)


def main(args):
  init = time.time()
  assert len(args.hbaseline_names) == len(args.hbaseline_values)
  outdir = args.outdir / f'{args.type}'
  if args.type == 'train':
    args.tag = 'test/progress_mean'
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
  parser.add_argument('--tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
  parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
  parser.add_argument('--hbaseline_names', nargs='+', type=str, default=[])
  parser.add_argument('--hbaseline_values', nargs='+', type=float, default=[])
  return parser.parse_args()


if __name__=='__main__':
  main(parse())