import os
from time import sleep
from datetime import datetime

from agents.gap_follower import GapFollower
from racecar_gym import SingleAgentScenario
import rlephant
from racecar_gym.envs import SingleAgentRaceEnv
import wrappers

config_dir, output_dir = "conf", "out"
config_env_files = ['random_starts_austria.yml']
sim_rendering = False
assert(len(config_env_files) < 2 or not sim_rendering), "cause by PyBullet re-init of env's GUIs, set rendering=False"

n_episodes_x_track = 2000
n_obs_per_trace = 500                  # e.g. 100 lidar acquisitions = 4 seconds

def create_h5_filepath(config_file, episodes=n_episodes_x_track, max_obs=n_obs_per_trace, out_dir=output_dir):
    prefix = "dataset_single_agent_austria"
    config_details = "{}episodes_{}steps".format(episodes, max_obs)
    track_name = config_file.split(".")[:-1][0]
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    return os.path.join(out_dir, "{}_{}_{}.h5".format(prefix, track_name, config_details, timestamp))

def create_wrapped_env(config_file, conf_dir=config_dir, rendering=sim_rendering):
    scenario = SingleAgentScenario.from_spec(path=os.path.join(conf_dir, config_file), rendering=rendering)
    env = SingleAgentRaceEnv(scenario=scenario)
    env = wrappers.ActionRepeat(env, 4)
    return env

def debug(i_episode, n_chars_debug_line=25):
    print(".", end=" ")
    if i_episode > 0 and i_episode % n_chars_debug_line == 0:
        print()

for config_file in config_env_files:
    env = create_wrapped_env(config_file)
    agent = GapFollower()
    storage = rlephant.ReplayStorage(create_h5_filepath(config_file))

    print("\n[Info] Simulation {} traces for {} seconds".format(n_episodes_x_track, n_obs_per_trace * 0.04))
    for i_episode in range(n_episodes_x_track):
        debug(i_episode)
        episode = rlephant.Episode()
        obs = env.reset(mode='random')
        done = False
        track_observations = []
        for i_acq in range(n_obs_per_trace):
            action = agent.action(obs)
            action = {'motor': (action[0], action[1]), 'steering': action[2]}
            obs, rewards, done, states = env.step(action)
            if done:
                break
            if sim_rendering:
                sleep(0.001)
            # here: store transition only every delta_lidar/delta_sim
            transition = rlephant.Transition(
                observation={'lidar': obs['lidar']},
                action=action,
                reward=rewards,
                done=done)
            episode.append(transition)
            if done:
                break
        # here: either complete episode (done) or collected nr observations
        storage.save(episode)

    env.close()
    print()
