from racecar_gym import Task, register_task
import math

class MinimizeSectionTime(Task):

    def __init__(self, timelimit: float = 20.0, checkpoints: int = 10):
        self._time_limit = timelimit
        self._checkpoint_count = checkpoints
        self._last_checkpoint = None
        self._last_timestep = None
        self._current_lap = None

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        checkpoint = math.floor(agent_state['progress'] * self._checkpoint_count)
        lap = agent_state['lap']
        timestep = agent_state['time']

        if self._last_checkpoint is None:
            self._last_checkpoint = checkpoint
            self._current_lap = lap
            self._last_timestep = timestep
            reward = 0.0
        elif math.fabs(self._last_checkpoint - checkpoint) > 0:
            self._last_checkpoint = checkpoint
            self._current_lap = lap
            reward = 1.0
        else:
            reward = 0.0 # self._last_timestep - timestep

        self._last_timestep = timestep
        return reward

    def done(self, agent_id, state) -> bool:
        return state[agent_id]['time'] >= self._time_limit

    def reset(self):
        self._last_checkpoint = None
        self._last_timestep = None
        self._current_lap = None

register_task(name='minimize_section_time', task=MinimizeSectionTime)