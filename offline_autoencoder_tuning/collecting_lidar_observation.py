import os
from time import sleep
from datetime import datetime
from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_race_car_env import MultiAgentRaceCarEnv
from baselines.gap_follower import GapFollower
from offline_autoencoder_tuning.MultiAgentWrapperToSingleAgent import MultiAgentWrapperToSingleAgent as EnvWrapper

out_dir = "out"
config_env_files = ['random_starts_austria.yml', 'random_starts_track1.yml']
sim_rendering = False
assert(len(config_env_files) < 2 or not sim_rendering), "cause by PyBullet re-init of env's GUIs, set rendering=False"

n_traces_per_track = 100
n_obs_per_trace = 100  # e.g. 100 lidar acquisitions = 4 seconds
n_step_per_lidar_obs = 4    # lidar's rate is 40ms, sim step is 10ms, then lidar acq changes every 4 frames
sim_step = 0.01

n_chars_debug_line = 25
for config_file in config_env_files:
    scenario = MultiAgentScenario.from_spec(path=config_file, rendering=sim_rendering)
    env = MultiAgentRaceCarEnv(scenario=scenario)
    print(env.action_space)
    env = EnvWrapper(env)
    agent = GapFollower()

    done = False
    env.reset()

    output = ""
    print("\n[Info] Simulation {} traces for {} seconds".format(n_traces_per_track,
                                                                n_obs_per_trace * n_step_per_lidar_obs * sim_step))
    for i_track in range(n_traces_per_track):
        print(".", end=" ")
        if i_track > 0 and i_track % n_chars_debug_line == 0:
            print()
        track_observations = []
        for i_acq in range(n_obs_per_trace):
            for i_step in range(n_step_per_lidar_obs):
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
            # save acquisition for offline dataset
            # append the trace observations to output
            dataset_entry = "{},{},{},{}\n".format(config_file, i_track, i_acq,
                                                    ','.join(["{}".format(acq) for acq in obs['A']['lidar']]))
            output += dataset_entry

            if done:
                break


    # write output file
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    out_filename = "dataset_from_{}_{}.txt".format(config_file.split(".")[:-1][0], timestamp)
    with open(os.path.join(out_dir, out_filename), 'w') as out:
        out.write(output)
    env.close()
    print()
