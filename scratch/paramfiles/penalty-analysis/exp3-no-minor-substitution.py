import importlib.util
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

main_func = 'EXPERIMENT_3'  # 'BAYES_OPT' | 'EXPERIMENT_3' | 'PRECOMPUTE' | 'MULTI_OBJECTIVE_BAYES_OPT'

seed = 0xBE_BAD_BAE

# Sensitivity analysis weights
penalty_weights = [1, 1, 0, 1, 1]

# Experiment 3 timing
simulation_time = {'warm_up': 7 * 6 * 3,
                   'horizon': 7 * 6 * 4,
                   'cool_down': 0}

# Anticpation for usability
anticipation = [True, False, False]
# No anticipation
# anticipation = False
# Anticipation and forecasting
# anticipation = [True, True, True]

pop_phen_configs = dict(dummy='data/bloodgroup_frequencies/ABD_old_dummy_demand.tsv')

# Experiment 3
dummy_demand = dict()
# For precompute
# dummy_demand = dict(excess_supply=0)

inventory = {
    'appointments': stats.randint(33, 34),
    'units_per_appointment': stats.randint(10, 11),
    'stock': stats.randint(3500, 3501),
    'starting_inventory': 30_000 - 3500,
    'scd_requests_ratio': 330 / 3500,
    # 'initial_age_dist': 'data/inventory/initial_age_distribution.tsv',
}

rules = ['Extended']

replications = 100
cpus = 10

exp2 = None

stock_measurement = {
    'watched_antigens': np.array([114688, 114688, 114688, 114688, 114688, 114688, 114688, 114688, 31744, 128, 64, 192, 31936]),

    'watched_phenotypes': np.array([0, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 21504, 0, 0, 0, 21504]),

    'watched_phenotypes_names':  ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+', 'R0', 'Fya-', 'Fyb-', 'Fya-_Fyb-', 'R0_Fya-_Fyb-'],
}


forecasting = {
    'units_days': 1,
    'requests_days': 3,
    'units_shows': 1,
    'requests_shows': 1
}
pre_compute_folder = 'out/experiments/exp3/precompute/new/'

ab_datafile = 'BSCSimulator/experiments/bayes_alloAb_frequencies.tsv'

bayes_opt = {
    'init_points_count': 5,
    'num_iterations': 5,
    'replications': replications,
    'num_objectives': 2,
}

solver = 'maxflow'

constraints = {
    'max_age': 35,
    'max_young_blood': 14,
    'yb_constraint': True,
    'substitution_weight_equal': False
}
