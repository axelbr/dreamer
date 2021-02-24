from abc import abstractmethod
from typing import Tuple


class Parser():
    @abstractmethod
    def __call__(self, logdir) -> Tuple[str, str, int]:
        pass

class ModelFreeParser(Parser):
    def __call__(self, logdir):
        splitted = logdir.split('_')
        if len(splitted) == 6:  # assume logdir: track_dreamer_max_progress_seed_timestamp
            track, algo, _, _, seed, _ = splitted
            base_algo = ''
        else:
            raise NotImplementedError(f'cannot parse {logdir}')
        seed = int(seed)
        method = algo
        return track, method, seed


class EvaluationParser(Parser):
    def __call__(self, logdir):
        # format: eval_algorithm_trainingtrack_timestamp
        splitted = logdir.split('_')
        assert len(splitted) == 4
        _, algo, track, _ = splitted
        seed = -1
        return track, algo, seed

class DreamerParser(Parser):
    parameters = {'horizon': 'H', 'batch_length': 'Bl', 'action_repeat': 'Ar'}

    def __init__(self, gby_parameter=None):
        """
        Return the parsed experiment name, grouping by a parameter (e.g. horizon, batch_length)
        :param gby_parameter: parameter for group by. If `None` gby method
        """
        assert gby_parameter is None or gby_parameter in self.parameters.keys()
        self._param = gby_parameter

    def __call__(self, logdir):
        splitted = logdir.split('_')
        if len(splitted) == 9:  # assume logdir: track_dreamer_max_progress_ArK_BlL_HH_seed_timestamp
            track, algo, _, _, action_repeat, batch_len, horizon, seed, _ = splitted
            base_algo = algo
        elif len(splitted) == 10:  # assume logdir: track_dreamer_method_max_progress_ArK_BlL_HH_seed_timestamp
            track, algo, _, _, method, action_repeat, batch_len, horizon, seed, _ = splitted
            base_algo = f'{algo}+{method}'
        else:
            raise NotImplementedError(f'cannot parse {logdir}')
        current_params = {
            'horizon': int(''.join(filter(str.isdigit, horizon))),
            'batch_length': int(''.join(filter(str.isdigit, batch_len))),
            'action_repeat': int(''.join(filter(str.isdigit, action_repeat)))
        }
        seed = int(seed)
        # create method based on group-by parameter
        if self._param is not None:
            method = f'{base_algo}_{self._param}_{current_params[self._param]}'
        else:
            method = f'{base_algo}'
        return track, method, seed
