import argparse
import time
import pathlib
from datetime import datetime

from plotting.aggregators import MeanStd, MeanMinMax
from plotting.log_parsers import EvaluationParser
from plotting.structs import LONG_TRACKS_DICT, ALL_METHODS_DICT, SHORT_TRACKS_DICT, PALETTE
from plotting.utils import load_runs
import matplotlib.pyplot as plt
import numpy as np


def plot_error_bar(args, runs, ax, aggregator):
  train_track = sorted(set([r.train_track for r in runs]))
  assert len(train_track)==1
  train_track = train_track[0]
  methods = sorted(set([r.method for r in runs]))
  bar_width = 0.25
  for i, method in enumerate(methods):
      means, n_errors, p_errors, colors, ecolors, test_tracks = [], [], [], [], [], []
      for j, test_track in enumerate(args.tracks):
        filter_runs = [r for r in runs if r.train_track == train_track and r.test_track == test_track and r.method == method]
        if len(filter_runs) > 0:
          x, mean, min, max = aggregator(filter_runs, None)
          means.append(mean[0])
          n_errors.append(mean[0]-min[0])
          p_errors.append(max[0]-mean[0])
          colors.append('red' if 'mpo' in method else PALETTE[i])       # force 'red' mpo for consistency other plots
          ecolors.append((1.0, 0.0, 0.0) if SHORT_TRACKS_DICT[train_track]==SHORT_TRACKS_DICT[test_track] else (0.0, 0.0, 0.0))
          test_tracks.append(test_track)
      xpos = np.arange(1, len(test_tracks)+1)
      ax.bar(xpos + i*bar_width, means, bar_width, yerr=np.array([n_errors, p_errors]), align='center', alpha=0.8,
             color=colors, edgecolor=ecolors, capsize=5, label=method.upper())
  mins = xpos-bar_width/2
  ax.set_xticks((mins + (mins + len(methods)*bar_width))/2)
  ax.set_xticklabels([SHORT_TRACKS_DICT[track] for track in test_tracks])
  ax.set_title(f'TRAIN {LONG_TRACKS_DICT[train_track]}'.upper())
  ax.set_ylabel(args.ylabel)
  ax.set_ylim(0, 1.1)
  # keep only axis, remove top/right
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

def main(args):
  tag = "progress"
  args.ylabel = args.ylabel if args.ylabel!="" else tag
  runs = load_runs(args, [EvaluationParser()], tag=tag, eval_mode=True)
  train_tracks = sorted(set([r.train_track for r in runs]))
  args.outdir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  for aggregator, fn in zip(['mean_minmax'], [MeanMinMax()]):
      fig, axes = plt.subplots(1, len(train_tracks), figsize=(4 * len(train_tracks), 3))
      # todo move loop on train tracks in plot error bar
      for i, (train_track, ax) in enumerate(zip(train_tracks, axes)):
        filter_runs = [r for r in runs if r.train_track == train_track]
        plot_error_bar(args, filter_runs, ax, aggregator=fn)
      if args.legend:
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), framealpha=1.0, handletextpad=0.1)
      filename = f'eval_' + '_'.join(train_tracks) + f'_{aggregator}_{timestamp}.png'
      fig.tight_layout(pad=2.5)
      fig.savefig(args.outdir / filename)
      print(f"[Info] Written {args.outdir / filename}")

def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--xlabel', type=str, default="")
  parser.add_argument('--ylabel', type=str, default="")
  parser.add_argument('--binning', type=int, default=15000)
  parser.add_argument('--legend', action='store_true')
  parser.add_argument('--show_labels', action='store_true')
  parser.add_argument('--tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
  parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
  parser.add_argument('--first_n_models', type=int, required=None, default=2)
  return parser.parse_args()


if __name__=='__main__':
  init = time.time()
  main(parse())
  print(f"\n[Info] Elapsed Time: {time.time()-init:.3f} seconds")