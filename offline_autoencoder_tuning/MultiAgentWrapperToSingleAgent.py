import gym
import numpy as np

class MultiAgentWrapperToSingleAgent(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Box(
            np.append([2, 0.5], env.action_space['A']['steering'].low),
            np.append(env.action_space['A']['motor'].high, env.action_space['A']['steering'].high))
        self.observation_space = env.observation_space['A']['lidar']

    def step(self, action):
        action = {'A': {'motor': (action[0], action[1]), 'steering': action[2]}}
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

    def render(self, mode='human'):
        pass