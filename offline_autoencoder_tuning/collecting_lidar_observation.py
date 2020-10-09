from time import sleep

from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_race_car_env import MultiAgentRaceCarEnv

config_env_files = ['random_starts_austria.yml', 'random_starts_track1.yml']
sim_rendering = False
assert(len(config_env_files) < 2 or not sim_rendering), "cause by PyBullet issue on re-initialize env's GUI, please set rendering=False"

n_traces_per_track = 10
n_step_per_trace = 1
for config_file in config_env_files:
    scenario = MultiAgentScenario.from_spec(path=config_file, rendering=sim_rendering)
    env = MultiAgentRaceCarEnv(scenario=scenario)

    done = False
    obs = env.reset()
    for _ in range(n_traces_per_track):
        obs = env.reset()
        for _ in range(n_step_per_trace):
            action = env.action_space.sample()
            obs, rewards, dones, states = env.step(action)
            done = any(dones.values())
            sleep(0.01)
    env.close()