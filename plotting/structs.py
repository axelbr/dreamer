PALETTE = 10 * (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    '#fdbf6f')

LONG_TRACKS_DICT = {'austria': 'AUSTRIA', 'barcelona': 'BARCELONA', 'columbia': 'COLUMBIA',
                    'gbr': 'GREATBRITAIN', 'treitlstrasse': 'LECTURE HALL',
                    'treitlstrasse_v2': 'LECTURE HALL', 'treitlstrassev2': 'LECTURE HALL'}
SHORT_TRACKS_DICT = {'austria': 'AUT', 'barcelona': 'BRC', 'columbia': 'COL',
                     'gbr': 'GBR', 'treitlstrasse': 'HALL',
                     'treitlstrasse_v2': 'HALL', 'treitlstrassev2': 'HALL'}
ALL_METHODS_DICT = {'dream': 'Dream', 'mpo': 'MPO', 'd4pg': 'D4PG', 'ppo': 'PPO', 'sac': 'SAC'}
BEST_MFREE_PERFORMANCES = {'austria': {'d4pg': 0.38, 'mpo': 0.36, 'ppo': 0.36, 'sac': 0.36},
                           'columbia': {'d4pg': 2.06, 'mpo': 2.13, 'ppo': 2.09, 'sac': 1.97},
                           'treitlstrassev2': {'d4pg': 0.77, 'mpo': 0.69, 'ppo': 0.66, 'sac': 0.30}}
BEST_DREAMER_PERFORMANCES = {'austria': {'dreamer': 1.31},
                             'columbia': {'dreamer': 2.23},
                             'treitlstrassev2': {'dreamer': 2.00}}