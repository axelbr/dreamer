import atexit
import functools
import sys
import threading
import traceback

import gym
import racecar_gym
import numpy as np
from PIL import Image

envs = {}


class RaceCarWrapper:
  def __init__(self, track, prefix, id, rendering=False):
    from racecar_gym.envs.multi_agent_race import MultiAgentScenario, MultiAgentRaceEnv
    from racecar_gym.tasks import register_task
    from racecar_gym.tasks.progress_based import MaximizeProgressTask, MaximizeProgressMaskObstacleTask
    env_id = track
    if env_id not in envs.keys():
      register_task("maximize_progress", MaximizeProgressTask)
      register_task("maximize_progress_obstacle", MaximizeProgressMaskObstacleTask)
      scenario = MultiAgentScenario.from_spec(f"scenarios/{track}.yml", rendering=rendering)
      envs[env_id] = MultiAgentRaceEnv(scenario=scenario)
    self._mode = "grid" if prefix=="test" else "random"
    self._env = envs[env_id]
    self._id = id     # main agent id, for rendering?
    self.agent_ids = list(self._env.observation_space.spaces.keys())   # multi-agent ids
    self.n_agents = len(self.agent_ids)

  @property
  def observation_space(self):
    assert 'speed' not in self._env.observation_space.spaces[self._id]
    spaces = {}
    for id, obss in self._env.observation_space.spaces.items():
      agent_space = {}
      for obs_name, obs_space in obss.spaces.items():
        agent_space[obs_name] = obs_space
        agent_space['speed'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
      spaces[id] = gym.spaces.Dict(agent_space)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    action_space = self._env.action_space
    flat_action_space = dict()
    for id, act in self._env.action_space.spaces.items():
      flat_action_space[id] = gym.spaces.Box(np.append(act['motor'].low, action_space[self._id]['steering'].low),
                                             np.append(act['motor'].high, action_space[self._id]['steering'].high))
    return flat_action_space

  def step(self, actions):
    actions = {i: {'motor': actions[i][0], 'steering': actions[i][1]} for i in self.agent_ids}
    obs, reward, done, info = self._env.step(actions)
    for id in self.agent_ids:
      obs[id]['speed'] = np.linalg.norm(info[id]['velocity'][:3])
      if 'low_res_camera' in obs[id]:
        obs[id]['image'] = obs[id]['low_res_camera']
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset(mode=self._mode)
    for id in self.agent_ids:
      obs[id]['speed'] = 0.0
      if 'low_res_camera' in obs[id]:
        obs[id]['image'] = obs[id]['low_res_camera']
    return obs

  def render(self, **kwargs):
    return self._env.render(**kwargs)

  def close(self):
    self._env.close()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    dones = {id: False for id in self._env.agent_ids}
    total_rewards = {id: 0.0 for id in self._env.agent_ids}
    current_step = 0
    while current_step < self._amount and not any(dones.values()):
      obs, rewards, dones, info = self._env.step(action)
      total_rewards = {id: total_rewards[id] + rewards[id] for id in self._env.agent_ids}
      current_step += 1
    return obs, total_rewards, dones, info

  def render(self, **kwargs):
    return self._env.render(**kwargs)


class ReduceActionSpace:

  def __init__(self, env, low, high):
    self._env = env
    self._low = np.array(low)
    self._high = np.array(high)

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    original = {id: (action[id] + 1) / 2 * (self._high - self._low) + self._low for id in self._env.agent_ids}
    return self._env.step(original)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, rewards, dones, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      dones = {id: True for id in self._env.agent_ids}
      self._step = None
    return obs, rewards, dones, info

  def reset(self, **kwargs):
    self._step = 0
    return self._env.reset(**kwargs)


class Render:

  def __init__(self, env, callbacks=None):
    self._env = env
    self._callbacks = callbacks or ()
    self._reset_videos_dict()

  def _reset_videos_dict(self):
    self._videos = {'birds_eye-A': []}
    for agent_id in self._env.agent_ids:
      self._videos[f'follow-{agent_id}'] = []

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obss, reward, dones, info = self._env.step(action)
    for k in self._videos.keys():
      mode, agent = k.split('-')
      frame = self._env.render(mode=mode, agent=agent)
      self._videos[k].append(frame)
    if any(dones.values()):
      for callback in self._callbacks:
        callback(self._videos)
    return obss, reward, dones, info

  def reset(self):
    obs = self._env.reset()
    for k in self._videos.keys():
      mode, agent = k.split('-')
      frame = self._env.render(mode=mode, agent=agent)
      self._videos[k] = [frame]
    return obs


class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episodes = [None for _ in env.agent_ids]      # in multi-agent: store 1 episode for each agent

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obss, reward, dones, info = self._env.step(action)
    obss = {id: {k: self._convert(v) for k, v in obs.items()} for id, obs in obss.items()}
    transition = obss.copy()
    for i, id in enumerate(obss.keys()):
      transition[id]['action'] = action[id]
      transition[id]['reward'] = reward[id]
      transition[id]['discount'] = info.get('discount', np.array(1 - float(dones[id])))
      transition[id]['progress'] = info[id]['lap'] + info[id]['progress']
      self._episodes[i].append(transition[id])
    if any(dones.values()):
      episodes = [{k: [t[k] for t in episode] for k in episode[0]} for episode in self._episodes]
      episodes = [{k: self._convert(v) for k, v in episode.items()} for episode in episodes]
      for callback in self._callbacks:
        callback(episodes)
    return obss, reward, dones, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    for i, id in enumerate(obs.keys()):
      transition[id]['action'] = np.zeros(self._env.action_space[id].shape)
      transition[id]['reward'] = 0.0
      transition[id]['discount'] = 1.0
      transition[id]['progress'] = 666.0
      self._episodes[i] = [transition[id]]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class NormalizeObservations:
  def __init__(self, env, config):
    self._config = config
    self._env = env
    self._mask = {obs_type: np.logical_and(
      np.isfinite(env.observation_space[obs_type].low),
      np.isfinite(env.observation_space[obs_type].high)
    ) for obs_type in env.observation_space.spaces.keys()}
    self._low = {obs_type: np.where(self._mask, env.observation_space[obs_type].low, -1)
                 for obs_type in env.observation_space.spaces.keys()}
    self._high = {obs_type: np.where(self._mask, env.observation_space[obs_type].high, 1)
                  for obs_type in env.observation_space.spaces.keys()}

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = {}
    for obs in self._env.observation_space.spaces.keys():
      if obs == self._config.obs_type:
        low = np.where(self._mask[obs], -np.ones_like(self._low[obs]), self._low[obs])
        high = np.where(self._mask[obs], np.ones_like(self._low[obs]), self._high[obs])
        space[obs] = gym.spaces.Box(low, high, dtype=np.float32)
      else:
        space[obs] = self._env.observation_space.spaces[obs]
    return gym.spaces.Dict(space)

  @property
  def original_observation_space(self):
    return self._env.observation_space

  def step(self, action):
    original_obs, reward, done, info = self._env.step(action)
    obs = original_obs
    # normalize observations in [-.5, +.5]
    obs_type = self._config.obs_type
    obs[obs_type] = (original_obs[obs_type] - self._low[obs_type]) / (self._high[obs_type] - self._low[obs_type]) - 0.5
    obs[obs_type] = np.where(self._mask[obs_type], original_obs[obs_type], obs[obs_type])
    return obs, reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
      np.isfinite(env.action_space.low),
      np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  @property
  def original_action_space(self):
    return gym.spaces.Box(self._low, self._high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class ObsDict:

  def __init__(self, env, key='obs'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    # if not np.allclose(reference, action):
    #  raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs