import argparse
import pathlib
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from plotting.aggregators import MeanMinMax
from plotting.log_parsers import DreamerParser
from plotting.utils import load_runs

from plotting.structs import ALL_METHODS_DICT, LONG_TRACKS_DICT

def plot_performance(args, tag, runs, axes, metric="max"):
  fn = MeanMinMax()
  tracks = sorted(set([r.train_track for r in runs]))
  methods = sorted(set([r.method for r in runs]), key=lambda met: int(met.split("_")[-1]))
  xmethods = [int(met.split("_")[-1]) for met in methods]
  if not type(axes) == np.ndarray:
    axes = [axes]
  for i, (track, ax) in enumerate(zip(tracks, axes)):
    if args.show_labels:
      ax.set_title(LONG_TRACKS_DICT[track].upper())
      ax.set_xlabel(args.xlabel)
      if i <= 0:  # show y label only on first row
        ax.set_ylabel(args.ylabel if args.ylabel else tag)
    track_runs = [r for r in runs if r.train_track == track]
    best_performances_means, best_performances_stds = [], []
    for j, method in enumerate(methods):
      filter_runs = [r for r in track_runs if r.method == method]
      if len(filter_runs) > 0:
        maxes = [np.max(r.y) for r in filter_runs]
        mean, std = np.nanmean(maxes), np.nanstd(maxes)
        best_performances_means.append(mean)
        best_performances_stds.append(std)
    best_performances_means = np.stack(best_performances_means)
    best_performances_stds = np.stack(best_performances_stds)
    ax.plot(xmethods, best_performances_means, c='k')
    ax.errorbar(xmethods, best_performances_means, yerr=best_performances_stds, fmt='.k', capsize=2)
    ax.set_xticks(xmethods)
    ax.set_xticklabels(xmethods)
    ax.set_ylim(0.25, 1.5)
    # keep only axis, remove top/right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main(args):
  assert len(args.hbaseline_names) == len(args.hbaseline_values)
  tag = "test/progress"   # load_run will check for `tag` or `tag_mean`
  runs = load_runs(args, [DreamerParser(gby_parameter='horizon')], tag=tag)
  tracks = sorted(set([r.train_track for r in runs]))
  args.outdir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  fig, axes = plt.subplots(1, len(tracks), figsize=(3 * len(tracks), 3))
  plot_performance(args, tag, runs, axes, metric='max')
  if args.legend:
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), framealpha=1.0)
  filename = f'curves_' + '_'.join(tracks) + f'_best_performance_{timestamp}.png'
  fig.tight_layout(pad=1.0)
  fig.savefig(args.outdir / filename)
  print(f"[Info] Written {args.outdir / filename}")


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--xlabel', type=str, default="")
  parser.add_argument('--ylabel', type=str, default="")
  parser.add_argument('--legend', action='store_true')
  parser.add_argument('--show_labels', action='store_true')
  parser.add_argument('--tracks', nargs='+', type=str, default=LONG_TRACKS_DICT.keys())
  parser.add_argument('--methods', nargs='+', type=str, default=ALL_METHODS_DICT.keys())
  parser.add_argument('--hbaseline_names', nargs='+', type=str, default=[])
  parser.add_argument('--hbaseline_values', nargs='+', type=float, default=[])
  return parser.parse_args()


if __name__=='__main__':
  init = time.time()
  main(parse())
  print(f"\n[Info] Elapsed Time: {time.time()-init:.3f} seconds")