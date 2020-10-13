import gym
import tensorflow as tf
from rlephant import Episode, ReplayStorage, Transition

from racing_dreamer.baseline import GapFollower
from racing_dreamer.wrappers import SingleRaceCarWrapper


def train():
    pass

def collect_pretraining_data(filename: str, timesteps: int, env):
    storage = ReplayStorage(filename=filename)
    t = 0
    agent = GapFollower()
    while t < timesteps:
        obs = env.reset()
        done = False
        episode = Episode()
        episode_steps = 0
        while not done:
            action = agent.action(obs)
            obs, reward, done, info = env.step(action)
            episode.append(Transition.from_tuple((obs, action, reward, done)))
            done = done or t >= timesteps or episode_steps >= 4000
            t += 1
            episode_steps += 1
            print(f'{t}/{timesteps}')
        storage.save(episode)



if __name__ == '__main__':
    scenario = 'austria_single'
    env = SingleRaceCarWrapper(name=scenario, id='A', rendering=False)
    collect_pretraining_data(f'pretraining_{scenario}.h5', timesteps=40_000, env=env)