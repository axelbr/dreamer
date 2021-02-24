import collections
import warnings
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plotting.structs import PALETTE, SHORT_TRACKS_DICT, LONG_TRACKS_DICT, ALL_METHODS_DICT, BEST_MFREE_PERFORMANCES

DREAMER_CONFS = {}

Run = collections.namedtuple('Run', 'logdir train_track test_track method seed x y')

def parse_file(dir, file, file_parsers):
    filepath = file.relative_to(dir).parts[:-1][0]
    for file_parser in file_parsers:
        try:
            train_track, method, seed = file_parser(filepath)
            return train_track, method, seed
        except Exception as ex:     # if a parser fails, try to use the next one
            continue
    raise Exception()   # if all the parsers fail, then raise expection



def check_track(track, tracks):
    return track in tracks

def check_method(method, methods):
    return any([m in method for m in methods])

def load_runs(args, file_parsers, tag):
  runs = []
  for dir in args.indir:
    print(f'Loading runs from {dir}', end='')
    files = list(dir.glob('**/events*'))
    for file in files:
        try:
            train_track, method, seed = parse_file(dir, file, file_parsers)
        except Exception as ex:
            warnings.warn(f"cannot parse {file}")
            continue
        if not check_track(train_track, args.tracks) or not check_method(method, args.methods):
          continue
        try:
            event_acc = EventAccumulator(str(file), size_guidance={'scalars': 100000, 'tensors': 100000})  # max number of items to keep
            event_acc.Reload()
        except ValueError as err:
            print(f'Error {file}: {err}')
            continue
        except Exception as err:
            print(f'Error {file}: {err}')
            continue
        if tag in event_acc.Tags()['tensors']:
          y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
          x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
        elif tag + "_mean" in event_acc.Tags()['tensors']:
          y = np.array([float(tf.make_ndarray(tensor.tensor_proto)) for tensor in event_acc.Tensors(tag)])
          x = np.array([tensor.step for tensor in event_acc.Tensors(tag)])
        elif tag in event_acc.Tags()['scalars']:
          y = np.array([float(scalar.value) for scalar in event_acc.Scalars(tag)])
          x = np.array([int(scalar.step) for scalar in event_acc.Scalars(tag)])
        elif tag + "_mean" in event_acc.Tags()['scalars']:
          y = np.array([float(scalar.value) for scalar in event_acc.Scalars(tag)])
          x = np.array([int(scalar.step) for scalar in event_acc.Scalars(tag)])
        else:
          continue
        if 'sac' in method or 'ppo' in method:    # because we cannot scale steps in sb3 based on action_repeat
          x = x * 4
        runs.append(Run(file, train_track, train_track, method, seed, x, y))  # in this case, train track = test track
        print('.', end='')
    print()
  return runs

def aggregate_max(runs):
    all_x = np.concatenate([r.x for r in runs])
    all_y = np.concatenate([r.y for r in runs])
    all_logs = np.concatenate([[r.logdir for _ in r.y] for r in runs])
    order = np.argsort(all_y)
    all_x, all_y, all_logs = all_x[order], all_y[order], all_logs[order]
    return all_x, all_y, all_logs


def plot_filled_curve(args, runs, axes, aggregator):
    tracks = sorted(set([r.train_track for r in runs]))
    methods = sorted(set([r.method for r in runs]))
    if not type(axes) == np.ndarray:
        axes = [axes]
    for i, (track, ax) in enumerate(zip(tracks, axes)):
        if args.show_labels:
            ax.set_title(LONG_TRACKS_DICT[track].upper())
            ax.set_xlabel(args.xlabel)
            if i <= 0:  # show y label only on first row
                ax.set_ylabel(args.ylabel if args.ylabel else args.tag)
        track_runs = [r for r in runs if r.train_track == track]
        for j, method in enumerate(methods):
            color = PALETTE[j]
            filter_runs = [r for r in track_runs if r.method == method]
            if len(filter_runs) > 0:
                x, mean, min, max = aggregator(filter_runs, args.binning)
                min = np.where(min > 0, min, 0)
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
