from dreamer import tools
from dreamer.environments import wrappers

def make_async_env(config, writer, prefix, datadir, store, gui=False):
    return wrappers.Async(
        ctor=lambda: make_env(config, writer, prefix, datadir, store=store, gui=gui),
        strategy=config.parallel
    )

def make_env(config, writer, prefix, datadir, store, gui=False):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, (64, 64), grayscale=False,
        life_done=True, sticky_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'racecar':
    if gui:
      env = wrappers.SingleRaceCarWrapper(id='A', name=task + '_Gui-v0')
    else:
      env = wrappers.SingleRaceCarWrapper(id='A', name=task + '-v0')

    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  env = wrappers.RewardObs(env)
  return env