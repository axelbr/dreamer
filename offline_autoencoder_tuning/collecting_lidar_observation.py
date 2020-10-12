import os
from time import sleep
from datetime import datetime
from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_race_car_env import MultiAgentRaceCarEnv
from baselines.gap_follower import GapFollower
from offline_autoencoder_tuning.MultiAgentWrapperToSingleAgent import MultiAgentWrapperToSingleAgent as EnvWrapper
import rlephant

config_dir, output_dir = "conf", "out"
config_env_files = ['random_starts_austria.yml', 'random_starts_track1.yml']
sim_rendering = False
assert(len(config_env_files) < 2 or not sim_rendering), "cause by PyBullet re-init of env's GUIs, set rendering=False"

n_episodes_x_track = 2000
n_obs_per_trace = 1000                  # e.g. 100 lidar acquisitions = 4 seconds
delta_lidar, delta_sim = 0.04, 0.01     # lidar's rate is 40ms, sim step is 10ms
assert(delta_lidar % delta_sim == 0)    # TODO: check this

def create_h5_filepath(config_file, episodes=n_episodes_x_track, max_obs=n_obs_per_trace, out_dir=output_dir):
    prefix = "dataset"
    config_details = "{}episodes_{}maxobs".format(episodes, max_obs)
    track_name = config_file.split(".")[:-1][0]
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    return os.path.join(out_dir, "{}_{}_{}.h5".format(prefix, track_name, config_details, timestamp))

def create_wrapped_env(config_file, conf_dir=config_dir, rendering=sim_rendering):
    scenario = MultiAgentScenario.from_spec(path=os.path.join(conf_dir, config_file), rendering=rendering)
    env = MultiAgentRaceCarEnv(scenario=scenario)
    return EnvWrapper(env)

def debug(i_episode, n_chars_debug_line=25):
    print(".", end=" ")
    if i_episode > 0 and i_episode % n_chars_debug_line == 0:
        print()

for config_file in config_env_files:
    env = create_wrapped_env(config_file)
    agent = GapFollower()
    storage = rlephant.ReplayStorage(create_h5_filepath(config_file))

    done = False
    env.reset()
    print("\n[Info] Simulation {} traces for {} seconds".format(n_episodes_x_track, n_obs_per_trace * delta_lidar))
    for i_episode in range(n_episodes_x_track):
        debug(i_episode)
        episode = rlephant.Episode()

        track_observations = []
        for i_acq in range(n_obs_per_trace):
            for i_step in range(int(delta_lidar / delta_sim)):
                if i_acq == 0 and i_step == 0:      # first state of each track
                    done = False
                    obs = env.reset()
                action = agent.action(obs['A'])
                obs, rewards, dones, states = env.step(action)
                done = any(dones.values())
                if done:
                    break
                if sim_rendering:
                    sleep(0.001)
            # here: store transition only every delta_lidar/delta_sim
            transition = rlephant.Transition(
                observation={'lidar': obs['A']['lidar']},
                action={'action': action},
                reward=rewards['A'],
                done=done)
            episode.append(transition)
            if done:
                break
        # here: either complete episode (done) or collected nr observations
        storage.save(episode)

    env.close()
    print()
