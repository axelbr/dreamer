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

class SingleForkedRaceCarWrapper:
  def __init__(self, name, prefix, id, rendering=False):
    from racecar_gym.envs.forked_multi_agent_race import ForkedMultiAgentRaceEnv, MultiAgentRaceEnv
    from racecar_gym.envs.multi_agent_race import MultiAgentScenario
    from racecar_gym.tasks import Task, register_task
    from racecar_gym.tasks.progress_based import MaximizeProgressTask, MaximizeProgressMaskObstacleTask
    if name not in envs.keys():
      register_task("maximize_progress", MaximizeProgressTask)
      scenario = MultiAgentScenario.from_spec(f"scenarios/{name}.yml", rendering=rendering)
      if prefix == "prefill":
        env = ForkedMultiAgentRaceEnv(scenario=scenario, mode='random')
        env = TimeLimit(env, 500)  # prefill with many shorter episodes
        self._mode = "random"
      elif prefix == "train":
        env = ForkedMultiAgentRaceEnv(scenario=scenario, mode='random')
        self._mode = "random"
      elif prefix == "test":
        env = ForkedMultiAgentRaceEnv(scenario=scenario, mode='grid')
        self._mode = "grid"
      else:
        raise NotImplementedError(f'prefix {prefix} not implemented')
      envs[name + "_" + prefix] = env
    self._env = envs[name + "_" + prefix]
    self._agent_ids = list(self._env.observation_space.spaces.keys())
    self._id = id


  @property
  def observation_space(self):
    space = self._env.observation_space[self._id]
    return space

  @property
  def action_space(self):
    action_space = self._env.action_space
    return gym.spaces.Box(
      np.append(action_space[self._id]['motor'].low, action_space[self._id]['steering'].low),
      np.append(action_space[self._id]['motor'].high, action_space[self._id]['steering'].high)
    )

  def step(self, action):
    actions = dict([(a, {'motor': 0.0, 'steering': 0}) for a in self._agent_ids])
    actions[self._id] = {'motor': action[0], 'steering': action[1]}
    obs, reward, done, info = self._env.step(actions)
    if 'low_res_camera' in obs[self._id]:
      obs[self._id]['image'] = obs[self._id]['low_res_camera']
    return obs[self._id], reward[self._id], done[self._id], info[self._id]

  def reset(self):
    obs = self._env.reset()
    if 'low_res_camera' in obs[self._id]:
      obs[self._id]['image'] = obs[self._id]['low_res_camera']
    return obs[self._id]

  def render(self, **kwargs):
    return self._env.render(**kwargs)


class SingleRaceCarWrapper:

  def __init__(self, name, id, size=(100,)):
    if name not in envs.keys():
      scenario = racecar_gym.MultiAgentScenario.from_spec('scenarios/austria.yml', rendering=False)
      envs[name] = racecar_gym.MultiAgentRaceEnv(scenario=scenario)
    self.env = envs[name]
    self._agent_ids = list(self.env.observation_space.spaces.keys())
    self._size = size
    self._id = id

  @property
  def observation_space(self):
    space = self.env.observation_space[self._id]
    return space

  @property
  def action_space(self):
    action_space = self.env.action_space
    return gym.spaces.Box(
      np.append(action_space[self._id]['motor'].low, action_space[self._id]['steering'].low),
      np.append(action_space[self._id]['motor'].high, action_space[self._id]['steering'].high)
    )

  def step(self, action):
    actions = dict([(a, {'motor': (0, 0), 'steering': 0}) for a in self._agent_ids])
    actions[self._id] = {'motor': (action[0], action[1]), 'steering': action[2]}
    obs, reward, done, info = self.env.step(actions)
    if 'low_res_camera' in obs[self._id]:
      obs[self._id]['image'] = obs[self._id]['low_res_camera']
    return obs[self._id], reward[self._id], done[self._id], info[self._id]

  def reset(self):
    obs = self.env.reset()
    if 'low_res_camera' in obs[self._id]:
      obs[self._id]['image'] = obs[self._id]['low_res_camera']
    return obs[self._id]


class ProcgenWrapper(gym.Wrapper):

  def __init__(self, env, seed=None):
    super().__init__(env)
    self._random = np.random.RandomState(seed)

  def step(self, action):
    obs, reward, done, info = super().step(action)
    return dict(image=obs), reward, done, info

  def reset(self, **kwargs):
    return dict(image=super().reset())


class PyBullet:

  def __init__(self, name, size=(320, 240), camera=None):
    if name not in envs.keys():
      envs[name] = gym.make(name)
    self._env = envs[name]
    self._env.render()
    self._env.reset()
    self._size = size

  @property
  def observation_space(self):
    return gym.spaces.Dict({'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)})

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    _, reward, done, info = self._env.step(action)
    obs = self._observe()
    return obs, reward, done, info

  def reset(self):
    self._env.reset()
    return self._observe()

  def _observe(self):
    return dict(image=self._env.render(mode='rgb_array', width=64, height=64))


class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
        -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
      0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:

  LOCK = threading.Lock()

  def __init__(
          self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
          life_done=False, sticky_actions=True):
    import gym
    version = 0 if sticky_actions else 4
    name = ''.join(word.title() for word in name.split('_'))
    with self.LOCK:
      self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
    self._action_repeat = action_repeat
    self._size = size
    self._grayscale = grayscale
    self._noops = noops
    self._life_done = life_done
    self._lives = None
    shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
    self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
    self._random = np.random.RandomState(seed=None)

  @property
  def observation_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.reset()
    noops = self._random.randint(1, self._noops + 1)
    for _ in range(noops):
      done = self._env.step(0)[2]
      if done:
        with self.LOCK:
          self._env.reset()
    self._lives = self._env.ale.lives()
    if self._grayscale:
      self._env.ale.getScreenGrayscale(self._buffers[0])
    else:
      self._env.ale.getScreenRGB2(self._buffers[0])
    self._buffers[1].fill(0)
    return self._get_obs()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      _, reward, done, info = self._env.step(action)
      total_reward += reward
      if self._life_done:
        lives = self._env.ale.lives()
        done = done or lives < self._lives
        self._lives = lives
      if done:
        break
      elif step >= self._action_repeat - 2:
        index = step - (self._action_repeat - 2)
        if self._grayscale:
          self._env.ale.getScreenGrayscale(self._buffers[index])
        else:
          self._env.ale.getScreenRGB2(self._buffers[index])
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self):
    if self._action_repeat > 1:
      np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
    image = np.array(Image.fromarray(self._buffers[0]).resize(
      self._size, Image.BILINEAR))
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    return {'image': image}


class Collect:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
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


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = {id: True for id in self._env.action_space.spaces.keys()}
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self, **kwargs):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info

  def render(self, **kwargs):
    return self._env.render(**kwargs)


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

class ReduceActionSpace:

  def __init__(self, env, low, high):
    self._env = env
    self._mask = np.logical_and(
      np.isfinite(env.action_space.low),
      np.isfinite(env.action_space.high))
    self._low = np.array(low)
    self._high = np.array(high)

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


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


class GapFollowerWrapper:
  def __init__(self, action_space):
    from agents.gap_follower import GapFollower
    self._gf = GapFollower()
    # for action normalization
    self._mask = np.logical_and(
      np.isfinite(action_space.low),
      np.isfinite(action_space.high))
    self._low = np.where(self._mask, action_space.low, -1)
    self._high = np.where(self._mask, action_space.high, 1)

  def action(self, observation, **kwargs):
    original = self._gf.action(observation)
    action = 2 * (original - self._low) / (self._high - self._low) - 1
    return action


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


class SpeedObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'speed' not in spaces
    spaces['speed'] = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['speed'] = np.linalg.norm(info['velocity'])
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['speed'] = 0.0
    return obs

  def render(self, **kwargs):
    return self._env.render(**kwargs)


class Async:
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    if strategy != 'none':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=self._worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except ConnectionResetError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _worker(self, ctor, conn):
    try:
      env = ctor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError(f'Received message of unknown type {message}')
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error in environment process: {stacktrace}')
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()
