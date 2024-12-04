"""
Main module for running different experiments in the BSCSimulator.

This module sets up logging, parses command-line arguments, and runs the appropriate
experiment based on the configuration specified in the `parameters` module of the
`experiments` package.

Functions:
    main(): Entry point for the script. Determines which experiment to run based on
            the `param.main_func` value and executes it with the appropriate parameters.

Raises:
    Exception: Logs any exception that occurs during the execution of the main function.
    KeyboardInterrupt: Logs if the script is interrupted by the user.

Usage:
    Run this script from the command line as a module within the BSCSimulator package,
    passing any additional arguments to the main function.
"""

import datetime
import logging
import sys

from .experiments import (bayes_opt_tuning, exp3,
                          multi_objective_bayes_opt_tuning)
from .experiments import parameters as param
from .experiments import precompute_exp3

RANDOM_SEED = 20220228

now = datetime.datetime.now().strftime('%Y%m%d')
logging.basicConfig(filename=f'out/logs/hpc_bsc_{now}.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    print(f'A total of {len(sys.argv)} arguments were passed.\n')
    if param.main_func == 'EXPERIMENT_3':
        exp3(param.rules, seed=param.seed, replications=param.replications, cpus=param.cpus,
             pop_phen_configs=param.pop_phen_configs, anticipation=param.anticipation,
             weights=param.penalty_weights, **param.dummy_demand, exp2=param.exp2,
             **param.simulation_time, **param.stock_measurement,
             forecasting=param.forecasting, pre_compute_folder=param.pre_compute_folder,
             ab_datafile=param.ab_datafile, solver=param.solver, **param.constraints,
             **param.inventory, computation_times=param.computation_times)
    elif param.main_func == 'PRECOMPUTE':
        precompute_exp3(param.rules, seed=param.seed, cpus=param.cpus,
                        **param.dummy_demand, **param.simulation_time,
                        replications=param.replications, folder=param.pre_compute_folder,
                        **param.stock_measurement)
    elif param.main_func == 'BAYES_OPT':
        bayes_opt_tuning(**param.bayes_opt,
                         tuning_kwargs=dict(
                             cpus=param.cpus, seed=param.seed,
                             pop_phen_configs=param.pop_phen_configs,
                             anticipation=param.anticipation,
                             **param.dummy_demand, **param.simulation_time,
                             **param.stock_measurement, forecasting=param.forecasting,
                             pre_compute_folder=param.pre_compute_folder,
                             **param.constraints, **param.inventory,
                             ab_datafile=param.ab_datafile, solver=param.solver,)
                         )
    elif param.main_func == 'MULTI_OBJECTIVE_BAYES_OPT':
        multi_objective_bayes_opt_tuning(**param.bayes_opt,
                                         tuning_kwargs=dict(
                                             cpus=param.cpus, seed=param.seed,
                                             pop_phen_configs=param.pop_phen_configs,
                                             anticipation=param.anticipation,
                                             **param.dummy_demand, **param.simulation_time,
                                             **param.stock_measurement,
                                             forecasting=param.forecasting,
                                             pre_compute_folder=param.pre_compute_folder,
                                             **param.constraints, **param.inventory,
                                             ab_datafile=param.ab_datafile, solver=param.solver)
                                         )
    if len(sys.argv) > 2:
        print(*sys.argv[2:], sep='\n')
    print('\n------------------\n\n\n')
    return


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt):
        logger.exception('Exiting due to error.')
