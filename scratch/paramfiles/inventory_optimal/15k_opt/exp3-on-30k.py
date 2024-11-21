import importlib.util
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# 'BAYES_OPT' | 'EXPERIMENT_3' | 'PRECOMPUTE' | 'MULTI_OBJECTIVE_BAYES_OPT'
main_func = 'EXPERIMENT_3'

seed = 0xBE_BAD_BAE

# Optimised weights
penalty_weights = [0.478391, 0.003616, 0.057091, 0.452611, 0.008291]
# penalty_weights = [0.512, 0, 0, 0.481, 0.007]
# penalty_weights = [0, 1, 0, 1, 0.05]
# Naive weights
# penalty_weights = None

# Experiment 2 timing
# simulation_time = {'warm_up': 0,
#                    'horizon': 1,
#                    'cool_down': 0}

# Experiment 3 timing
simulation_time = {'warm_up': 7 * 6 * 3,
                   'horizon': 7 * 6 * 4,
                   'cool_down': 0}

# Anticpation for usability
# anticipation = [True, False, False]
# No anticipation
# anticipation = False
# Anticipation and forecasting
anticipation = [True, False, False]

pop_phen_configs = dict(
    dummy='data/bloodgroup_frequencies/ABD_old_dummy_demand.tsv')

# Experiment 2
# dummy_demand = {'dummy_data': None, 'excess_supply': 0}
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
pre_compute_folder = 'out/experiments/exp3/precompute/20230711-14-53/'

ab_datafile = 'BSCSimulator/experiments/bayes_alloAb_frequencies.tsv'

bayes_opt = {
    'init_points_count': 5,
    'num_iterations': 5,
    'replications': replications,
    'num_objectives': 2,
    'iteration_chunks': 5,
    'gp_mean': None,
    'variable_names': ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood'],
    'objective_name': ['alloimmunisations'],
    'objective_names': ['alloimmunisations', 'scd_shortages', 'O_level']
}

solver = 'maxflow'  # 'pot' or 'ortools' or ('maxflow' or 'ortools-maxflow)

constraints = {
    'max_age': 35,
    'max_young_blood': 14,
    'yb_constraint': True,
    # 'substitution_weight_equal': True,
    'substitution_weight_equal': False
}

computation_times = True
