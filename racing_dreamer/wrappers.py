import gym
import numpy as np

class SingleRaceCarWrapper:

  def __init__(self, name, id, size=(100,), rendering: bool = True):
    import racecar_gym
    scenario = racecar_gym.MultiAgentScenario.from_spec(
        path=f'scenario/{name}.yml',
        rendering=rendering
    )
    self.env = racecar_gym.MultiAgentRaceCarEnv(scenario=scenario)
    self._agent_ids = list(self.env.observation_space.spaces.keys())
    self._size = size
    self._id = id
    self._last_reward = 0


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
    actions[self._id] = action
    obs, reward, done, info = self.env.step(actions)
    if reward[self._id] != self._last_reward:
      self._last_reward = reward[self._id]
      print(reward[self._id])
    if 'low_res_camera' in obs[self._id]:
      obs[self._id]['image'] = obs[self._id]['low_res_camera']
    return obs[self._id], reward[self._id], done[self._id], info[self._id]

  def reset(self):
    obs = self.env.reset()
    self._last_reward = 0
    if 'low_res_camera' in obs[self._id]:
      obs[self._id]['image'] = obs[self._id]['low_res_camera']
    return obs[self._id]