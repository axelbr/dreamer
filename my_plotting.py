import argparse
import pathlib
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import collections
import warnings

PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')

Run = collections.namedtuple('Run', 'track method seed x y')

def load_runs(args):
  runs = []
  for dir in args.indir:
    files = list(dir.glob('**/*jsonl'))
    for file in files:
      try:
        track, method, seed = file.relative_to(dir).parts[:-1]  # path to jsonl file, e.g. columbia/h20/seed
        with file.open() as f:
          df = pd.DataFrame([json.loads(line) for line in f.readlines()])
        df = df[[args.xaxis, args.yaxis]].dropna()
        x = df[args.xaxis].to_numpy()
        y = df[args.yaxis].to_numpy()
        runs.append(Run(track, method, seed, x, y))
        print(f'Track: {track}, method: {method}, seed: {seed}.')
      except ValueError as err:
        print(f'Error {file}: {err}')
        continue
  return runs


def aggregate(runs, binning):
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




def main(args):
  runs = load_runs(args)
  tracks = sorted(set([r.track for r in runs]))
  methods = sorted(set([r.method for r in runs]))
  fig, axes = plt.subplots(1, len(tracks))
  for i, (track, ax) in enumerate(zip(tracks, axes)):
    ax.set_title(track.title())
    ax.set_xlabel(args.xaxis)
    if i<=0:    # show y label only on first row
      ax.set_ylabel(args.yaxis)
    for j, method in enumerate(methods):
      color = PALETTE[j]
      filter_runs = [r for r in runs if r.track==track and r.method==method]
      if len(filter_runs)>0:
        x, mean, std = aggregate(filter_runs, args.binning)
        if j==1:
          mean = mean - .2
        ax.plot(x, mean, color=color, label=method)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
    ax.legend()
  args.outdir.mkdir(parents=True, exist_ok=True)
  filename = args.outdir / 'curves.png'
  fig.savefig(filename)

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--xaxis', type=str, required=True)
  parser.add_argument('--yaxis', type=str, required=True)
  parser.add_argument('--binning', type=int, default=10000)
  return parser.parse_args()


if __name__=='__main__':
  main(parse())