from dreamer import Dreamer
import numpy as np
from typing import Dict, Tuple


class RacingDreamer:

    def __init__(self, checkpoint_file: str, actspace, obspace):
        self._checkpoint = checkpoint_file
        self._dreamer = Dreamer(config=None, datadir=None, actspace=actspace, obspace=obspace, writer=None)
        # load checkpoint
        if (checkpoint_file).exists():
            self._dreamer.load(checkpoint_file)
            print('Load checkpoint.')

    def preprocess_lidar(self, scan):
        # Step 1: clip values in simulated sensors' ranges
        min_range, max_range = 0.0, 15.0
        lidar = np.clip(scan, min_range, max_range)
        # Step 2: normalize lidar ranges in 0, 1
        lidar = (lidar - min_range) / (max_range - min_range)
        return lidar

    def action(self, observation: Dict[str, np.ndarray], state=None) -> Tuple[float, float]:
        scan = observation['lidar'][0]
        proc_ranges = self.preprocess_lidar(scan)
        min_index, max_index = 0, 1080
        embed = proc_ranges[min_index:max_index]

        if state is None:
            latent = self._dreamer._dynamics.initial(len(observation["lidar"]))
            action = np.zeros((1, 2), 'float')
        else:
            latent, action = state
        latent, _ = self._dreamer._dynamics.obs_step(latent, action, embed)
        feat = self._dynamics.get_feat(latent)
        action = self._actor(feat).mode()
        state = (latent, action)

        return action, state

    def __call__(self, observation: Dict[str, np.ndarray]) -> Tuple[float, float]:
        return self.action(observation)
