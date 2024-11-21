import importlib.util
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# What to run in main
main_func = 'EXPERIMENT_3'  # 'BAYES_OPT' | 'EXPERIMENT_3' | 'PRECOMPUTE' | 'MULTI_OBJECTIVE_BAYES_OPT'

seed = 0xBE_BAD_BAE

# Optimised weights
# penalty_weights = [0.578, 0, 0, 0.061, 0.361]
# Naive weights
penalty_weights = None

# Experiment 2 timing
# simulation_time = {'warm_up': 0,
#                    'horizon': 1,
#                    'cool_down': 0}

# Experiment 3 timing
simulation_time = {'warm_up': 7 * 6 * 4,
                   'horizon': 7 * 6 * 5,
                   'cool_down': 0}

# Anticpation for usability
# anticipation = [True, False, False]
# No anticipation
# anticipation = False
# Anticipation and forecasting
anticipation = [True, False, False]

pop_phen_configs = dict()

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
    'initial_age_dist': 'data/inventory/initial_age_distribution.tsv',
}

rules = ['Extended']

replications = 1
cpus = 1

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

ab_datafile = 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv'

bayes_opt = {
    'init_points_count': 5,
    'num_iterations': 5,
    'replications': replications,
    'num_objectives': 3,
    'iteration_chunks': 5,
    'gp_mean': None,
    'variable_names': ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood'],
    # For fixed variables use a dictionary with the variable name as key and the value as value
    # 'fixed_variables': {'young_blood': 0.0},
    'objective_name': ['alloimmunisations'],
    # Options for objective names: 'alloimmunisations', 'scd_shortages', 'O_level',
    # 'O_neg_level', 'O_pos_level', 'O_level',
    # 'D_subs_num_patients', 'ABO_subs_num_patients', 'ABOD_subs_num_patients'
    'objective_names': ['alloimmunisations', 'scd_shortages', 'O_level'],
    'objective_directions': ['MIN', 'MIN', 'MAX'],
    'normalized_kernel': True,
}

solver = 'maxflow' # 'pot' or 'ortools' or ('maxflow' or 'ortools-maxflow)

constraints = {
    'max_age': 35,
    'max_young_blood': 14,
    'yb_constraint': True,
    'substitution_weight_equal': True
}

computation_times = False

if len(sys.argv) > 1:
    try:
        print(f'Parameter file passed: {sys.argv[1]}')
        pfile_path = os.path.realpath(os.path.expanduser(sys.argv[1]))
        pfile = os.path.split(pfile_path)[1]
        spec = importlib.util.spec_from_file_location(pfile[:-3], pfile_path)
        imported_param = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_param)
        globals().update(imported_param.__dict__)
    except IndexError as e:
        logger.exception('No parameter file passed. Using default parameters.')
        print('No parameter file passed. Using default parameters.')
    except FileNotFoundError as e:
        logger.exception(f'Parameter file {pfile} not found. Using default parameters.')
        print(f'Parameter file {pfile} not found. Using default parameters.')
    except ImportError as e:
        logger.exception(f'Parameter file {pfile} is not a valid python file. Using default parameters.')
        print(f'Parameter file {pfile} is not a valid python file. Using default parameters.')
    except Exception as e:
        logger.exception(f'Error while importing parameter file {pfile}. Using default parameters.')
        print(f'Error while importing parameter file {pfile}. Using default parameters.')
