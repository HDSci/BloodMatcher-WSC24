"""
This module contains functions for loading data and running simulation experiments.
These simulations are either one-off experiments or one of many iterations of a tuning process.
"""
import datetime
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import randint

from BSCSimulator.antigen import Antigens
from BSCSimulator.demand import Demand
from BSCSimulator.inventory import Inventory
from BSCSimulator.matching import MatchingArea
from BSCSimulator.simulator import SimulationManager
from BSCSimulator.supply import Supply
from BSCSimulator.util import (ANTIGENS, abd_usability,
                               dummy_population_phenotypes,
                               list_of_permutations, pad_abd_phenotypes,
                               population_phenotype)


logger = logging.getLogger(__name__)


def load_rule_sets(filename) -> dict:
    """Loads rules

    Reads a JSON file and returns the dictionary representing the object in JSON file.
    :param filename: Filepath to JSON file.
    :return dict:
    """
    with open(filename) as f:
        a = json.load(f)
    return a


def load_immunogenicity(filename='data/immune_risks/immunogenicity.tsv') -> pd.DataFrame:
    """Loads immunogenicity data

    :param str filename: Filepath to TSV file, defaults to 'data/immune_risks/immunogenicity.tsv'
    :return: Immunogenicity data as a pandas DataFrame
    """
    df = pd.read_csv(filename, sep='\t')
    return df


def load_alloantibodies(filename: str = 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv') -> pd.DataFrame:
    """Loads alloantibodies data

    :param filename: Filepath to TSV file, defaults to 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv'
    :type filename: str
    :return: Alloantibodies data as a pandas DataFrame
    :type return: pd.DataFrame
    """
    df = pd.read_csv(filename, sep='\t')
    return df


def load_initial_age_distribution(filename: str = 'data/inventory/inital_age_distribution.tsv') -> pd.DataFrame:
    """Loads initial age distribution data

    :param str filename: Filepath to TSV file, defaults to 'data/inventory/inital_age_distribution.tsv'
    :return DataFrame: Initial age distribution data as a pandas DataFrame
    """
    try:
        df = pd.read_csv(filename, sep='\t')
    except (FileNotFoundError, ValueError):
        logger.warning(
            f'Could not load initial age distribution data from {filename}.')
        df = None
    return df


def exp3(rules, anticipation=False, seed=0xBE_BAD_BAE, cpus=1, replications=60, weights=None, pop_phen_configs: dict = None, **kwargs):
    """Experiment 3
    
        Parameters
        ----------
        rules : list
            List of rules to apply in the simulation.
        anticipation : bool, optional
            Whether to use anticipation in the simulation. Default is False.
        seed : int, optional
            Seed for random number generation. Default is 0xBE_BAD_BAE.
        cpus : int, optional
            Number of CPUs to use for parallel simulations. Default is 1.
        replications : int, optional
            Number of replications to run. Default is 60.
        weights : list of float, optional
            Weights for cost function. Default is None.
        pop_phen_configs : dict, optional
            Population phenotype configurations. Default is None.
        **kwargs : dict
            Additional keyword arguments for customization.

        Other Parameters
        ----------------
        appointments : int, optional
            Number of appointments. Default is a random integer between 33 and 34.
        units_per_appointment : int, optional
            Units per appointment. Default is a random integer between 10 and 11.
        stock : int, optional
            Initial stock. Default is a random integer between 3500 and 3501.
        excess_supply : int, optional
            Excess supply. Default is 3500 - 33 * 10.
        warm_up : int, optional
            Warm-up period in days. Default is 7 * 6 * 4 (24 weeks).
        horizon : int, optional
            Horizon period in days. Default is 7 * 6 * 5 (30 weeks).
        cool_down : int, optional
            Cool-down period in days. Default is 0 (0 weeks).
        forecasting : str, optional
            Forecasting method. Default is None.
        solver : str, optional
            Solver to use. Default is 'maxflow'.
        yb_constraint : bool, optional
            Whether to apply young blood constraint. Default is True.
        substitution_weight_equal : bool, optional
            Whether to use equal substitution penalty. Default is True.
        pre_compute_folder : str, optional
            Folder for pre-computed data. Default is 'out/experiments/exp{exp}/precompute/'.
        watched_antigens : list, optional
            List of watched antigens. Default is np.array([114688]).
        watched_phenotypes : list, optional
            List of watched phenotypes. Default is np.array([0]).
        watched_phenotypes_names : list, optional
            Names of watched phenotypes. Default is ['O-'].
        max_age : int, optional
            Maximum age for inventory. Default is 35.
        starting_inventory : int, optional
            Starting inventory. Default is 30_000 - 3500.
        ab_datafile : str, optional
            File for alloantibody data. Default is 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv'.
        initial_age_dist : str, optional
            Initial age distribution. Default is None.
        computation_times : bool, optional
            Whether to record computation times. Default is False.
        scd_requests_ratio : float, optional
            Ratio of SCD requests. Default is 330/3500.
        dummy_data : list, optional
            Dummy data for extra demand. Default is None.

        Raises
        ------
        OSError
            If a unique output folder cannot be created after a specified number of attempts.

        Notes
        -----
        This function runs a six-week simulation experiment with various configurations and outputs the results to specified folders.
    """
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime('%Y%m%d-%H-%M')

    folder_clash = True
    folder_clash_count = 0
    folder_clash_max = 100
    folder_clash_rng = np.random.default_rng(int(start_time * 1000))
    while folder_clash and folder_clash_count < folder_clash_max:
        root_now = datetime.datetime.now()
        root_folder_date = root_now.strftime('%Y%m%d')
        root_folder_time = root_now.strftime('%H%M')
        exp: str = '3' if kwargs.get(
            'exp2', None) is None else kwargs.get('exp2', '2')

        folder = os.path.join(
            f'out/experiments/exp{exp}', root_folder_date, root_folder_time, '')
        folder = os.path.realpath(folder)
        if os.path.exists(folder):
            folder_clash_count += 1
            time.sleep(20 * folder_clash_rng.random())
        else:
            try:
                os.makedirs(folder, exist_ok=False)
                folder_clash = False
            except OSError:
                folder_clash_count += 1
                time.sleep(20 * folder_clash_rng.random())

    if folder_clash_count == folder_clash_max:
        raise OSError(
            f'Could not create unique output folder after {folder_clash_max} attempts.')

    print(f'\n###\nStarting Experiment {exp} at {start_datetime} with rules:')
    for rule in rules:
        print(rule)
    print()
    if pop_phen_configs is None:
        pop_phen_configs = dict()
    matching_rules = load_rule_sets(
        'BSCSimulator/experiments/matching_rules.json')['MATCHING_RULES']
    donor_data = population_phenotype(pop_phen_configs.get(
        'donor', 'data/bloodgroup_frequencies/blood_groups_donors.json'), 0.01)
    patient_data = population_phenotype(pop_phen_configs.get(
        'patient', 'data/bloodgroup_frequencies/blood_groups.json'), 1.0)
    non_scd_frequencies = dummy_population_phenotypes(pop_phen_configs.get(
        'dummy', 'data/bloodgroup_frequencies/ABD_dummy_demand.tsv'))
    dummy_data = pad_abd_phenotypes(non_scd_frequencies, len(ANTIGENS) - 3)
    dummy_data = kwargs.get('dummy_data', dummy_data)
    allo_ab_data = load_alloantibodies(kwargs.get(
        'ab_datafile', 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv'))
    data = (donor_data, patient_data)
    immuno = load_immunogenicity()
    init_age_dist = load_initial_age_distribution(
        kwargs.get('initial_age_dist', None))
    pre_compute_folder = os.path.realpath(kwargs.get(
        'pre_compute_folder', f'out/experiments/exp{exp}/precompute/'))

    for rule in rules:
        matching_antigens = matching_rules[rule]['antigen_set']

        appointments = kwargs.get('appointments', randint(33, 34))
        units_per_appointment = kwargs.get(
            'units_per_appointment', randint(10, 11))
        stock = kwargs.get('stock', randint(3500, 3501))
        _excess_supply = kwargs.get('excess_supply', 3500 - 33 * 10)

        # unpack warm_up, horizon, cool_down from kwargs if present else use defaults
        warm_up = kwargs.get('warm_up', 7 * 6 * 4)  # 4 weeks
        horizon = kwargs.get('horizon', 7 * 6 * 5)  # 5 weeks
        cool_down = kwargs.get('cool_down', 0)  # 0 weeks

        forecasting = kwargs.get('forecasting', None)

        if isinstance(anticipation, bool):
            _anticipation = [anticipation and rule == 'Extended'] * 3
        else:
            _anticipation = [antn and rule ==
                             'Extended' for antn in anticipation]

        Antigens.population_abd_usabilities = abd_usability(
            non_scd_frequencies.frequencies.to_numpy(),
            kwargs.get('scd_requests_ratio', 330/3500), 1.0)
        antigens = Antigens(ANTIGENS, rule=matching_antigens,
                            allo_Abs=allo_ab_data.values.flatten())
        antigens.allo_risk = immuno[antigens.antigen_index[3:]].to_numpy(
        ).flatten()

        def demand():
            return Demand(antigens, patient_data, appointments, units_per_appointment,
                          antigen_string=False, dummy_data=dummy_data, dummy_extra_demand=_excess_supply)

        def supply():
            return Supply(antigens, donor_data, stock, antigen_string=False)

        def matching():
            return MatchingArea(
                algo='transport', antigens=antigens, matching_rule=rule, anticipation=_anticipation[0],
                cost_weights=weights, solver=kwargs.get('solver', 'maxflow'),
                young_blood_constraint=kwargs.get('yb_constraint', True),
                substitution_penalty_parity=kwargs.get('substitution_weight_equal', True))

        def inventory():
            return Inventory(
                kwargs.get('max_age', 35),
                kwargs.get('starting_inventory', 30_000 - 3500),
                watched_antigens=kwargs.get(
                    'watched_antigens', np.array([114688])),
                watched_phenotypes=kwargs.get(
                    'watched_phenotypes', np.array([0])),
                start_age_dist=init_age_dist)

        pre_compute_folder = pre_compute_folder if _anticipation[1] or _anticipation[2] else None
        manager = SimulationManager(
            antigens, demand, supply, matching, inventory, warm_up, horizon, cool_down, replications, seed,
            pre_compute_folder, _anticipation[1], _anticipation[2], forecasting=forecasting)
        if cpus > 1 and replications > 1:
            manager.do_simulations_parallel(min(cpus, replications))
        else:
            manager.do_simulations()
        manager.statistics()

        pad = [0, 0, 0]
        allo_padded = np.hstack(([pad, pad], np.vstack(manager.allo)))
        scd_short_padded = np.full(
            (2, len(ANTIGENS)), np.array(manager.scd_shorts)[:, None])
        all_short_padded = np.full(
            (2, len(ANTIGENS)), np.array(manager.all_shorts)[:, None])

        index = ['mismatch_avg', 'mismatch_stderr', 'allo_avg', 'allo_stderr', 'subs_avg', 'subs_stderr',
                 'short_avg', 'short_stderr', 'all_short_avg', 'all_short_stderr']
        stacked = np.vstack((*manager.mismatches, allo_padded,
                            *manager.subs, scd_short_padded, all_short_padded))
        df = pd.DataFrame(stacked, columns=ANTIGENS, index=index)

        now = datetime.datetime.now()

        file = os.path.join(folder, rule + now.strftime('%H-%M_output.tsv'))

        df.to_csv(file, sep='\t')
        print(f'Output written to {file}')

        stocks = np.hstack(manager.stocks)
        stock_cols = kwargs.get('watched_phenotypes_names', ['O-']) + ['total']
        full_stock_cols = stock_cols + ['_se_' + a for a in stock_cols]
        df2 = pd.DataFrame(stocks, columns=full_stock_cols)

        file2 = os.path.join(folder, rule + now.strftime('%H-%M_stocks.tsv'))

        df2.to_csv(file2, sep='\t')

        cols = [' to '.join(com) for com in list_of_permutations(
            [('O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+')] * 2)]
        df3 = pd.DataFrame(manager.abo_cm, columns=cols)
        file3 = os.path.join(folder, rule + now.strftime('%H-%M_abocm.tsv'))
        df3.to_csv(file3, sep='\t')

        df3_1 = pd.DataFrame(manager.abod_mm, columns=cols)
        file3_1 = os.path.join(
            folder, rule + now.strftime('%H-%M_abodmm_subs.tsv'))
        df3_1.to_csv(file3_1, sep='\t')

        df3_2_cols = ['D_substitutions',
                      'ABO_substitutions', 'ABOD_substitutions']
        df3_2 = pd.DataFrame(manager.pats_subs, columns=df3_2_cols)
        file3_2 = os.path.join(
            folder, rule + now.strftime('%H-%M_abodmm_pats_subs.tsv'))
        df3_2.to_csv(file3_2, sep='\t')

        file4 = os.path.join(
            folder, rule + now.strftime('%H-%M_failures.json'))

        with open(file4, 'w+') as f4:
            json.dump(manager.failures, f4, indent=2)

        objectives = manager.objs
        obj_cols = ['alloimmunisations',
                    'scd_shortages', 'expiries', 'all_shortages']
        obj_stock_cols = ['O_neg_level', 'O_pos_level', 'O_level']
        obj_mm_cols = ['D_subs_num_patients',
                       'ABO_subs_num_patients', 'ABOD_subs_num_patients']
        df_obj = pd.DataFrame(
            objectives, columns=obj_cols+obj_stock_cols+obj_mm_cols)
        file5 = os.path.join(
            folder, rule + now.strftime('%H-%M_objectives.tsv'))
        df_obj.to_csv(file5, sep='\t', index=False)

        file6 = os.path.join(
            folder, rule + now.strftime('%H-%M_age_distributions.npz'))
        age_distributions = dict(
            total_age_dist=manager.ages[0], total_age_dist_stderr=manager.ages[1])
        array_names = kwargs.get('watched_phenotypes_names', ['O-'])
        for i, name in enumerate(array_names):
            age_distributions.update(
                {name: manager.phen_ages[0][i], name + '_stderr': manager.phen_ages[1][i]})
        age_distributions.update(
            {'age_dist_given_to_scd': manager.scd_ages[0],
             'age_dist_given_to_scd_stderr': manager.scd_ages[1]})
        np.savez(file6, **age_distributions)

        record_computation_times = kwargs.get('computation_times', False)
        if record_computation_times:
            file7 = os.path.join(
                folder, rule + now.strftime('%H-%M_computation_times.tsv'))
            np.savetxt(file7, manager.computation_times, delimiter='\t')

        end_time = time.time()
        elapsed_mins = (end_time - start_time) / 60

        print(f'\nThe elapsed time so far is {elapsed_mins: .1f} minute(s).')

    print(f'\nThe output folder is at {folder}')


def precompute_exp3(
        rules, anticipation=False, seed=0xBE_BAD_BAE, cpus=1, pop_phen_configs: dict = None, replications=200,
        folder=None, **kwargs):
    """Pre-computation for Experiment 3

        Parameters
        ----------
        rules : list
            List of rules to be applied in the experiment.
            For pre-computation, only one need be provided - it does not matter which one.
        anticipation : bool, optional
            Redundant for pre-computation.
            Flag to indicate if anticipation or used, by default False.
        seed : int, optional
            Random seed for reproducibility, by default 0xBE_BAD_BAE.
        cpus : int, optional
            Number of CPUs to use for computation, by default 1.
        pop_phen_configs : dict, optional
            Configuration dictionary for population phenotypes, by default None.
        replications : int, optional
            Number of replications for the simulation, by default 200.
        folder : str, optional
            Output folder path to save pre-computed data. 
            By default None, in which case the new folder is created with the current timestamp.
        **kwargs : dict
            Additional keyword arguments for configuration.
            
        Other Parameters
        ----------------
        appointments : int, optional
            Number of SCD appointments per day, by default randint(33, 34).
        units_per_appointment : int, optional
            Number of units per SCD appointment, by default randint(10, 11).
        stock : int, optional
            Daily supply of blood units, by default randint(3500, 3501).
        excess_supply : int, optional
            Non-SCD demand, by default 3500 - 33 × 10.
        warm_up : int, optional
            Warm-up period in days, by default 7 × 6 × 4 (24 weeks).
        horizon : int, optional
            Horizon period in days, by default 7 × 6 × 5 (30 weeks).
        cool_down : int, optional
            Cool-down period in days, by default 0 (0 weeks).
        starting_inventory : int, optional
            Starting inventory, by default 30_000 - 3500.
        scd_requests_ratio : float, optional
            Ratio of SCD requests to total requests, by default 330/3500.

        Returns
        -------
        None
    """
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime('%Y%m%d-%H-%M')
    print(f'\n###\Pre-comuputing Experiment 3 at {start_datetime} with rules:')
    for rule in rules:
        print(rule)
    print()
    if pop_phen_configs is None:
        pop_phen_configs = dict()
    matching_rules = load_rule_sets(
        'BSCSimulator/experiments/matching_rules.json')['MATCHING_RULES']
    donor_data = population_phenotype(pop_phen_configs.get(
        'donor', 'data/bloodgroup_frequencies/blood_groups_donors.json'), 0.01)
    patient_data = population_phenotype(pop_phen_configs.get(
        'patient', 'data/bloodgroup_frequencies/blood_groups.json'), 1.0)
    non_scd_frequencies = dummy_population_phenotypes(pop_phen_configs.get(
        'dummy', 'data/bloodgroup_frequencies/ABD_dummy_demand.tsv'))
    dummy_data = pad_abd_phenotypes(non_scd_frequencies, len(ANTIGENS) - 3)
    allo_ab_data = load_alloantibodies()
    data = (donor_data, patient_data)
    immuno = load_immunogenicity()
    init_age_dist = load_initial_age_distribution(
        kwargs.get('initial_age_dist', None))
    folder = folder if folder is not None else os.path.realpath(
        f'out/experiments/exp3/precompute/{start_datetime}')
    folder = os.path.realpath(folder)
    os.makedirs(folder, exist_ok=True)

    for rule in rules:
        matching_antigens = matching_rules[rule]['antigen_set']

        appointments = kwargs.get('appointments', randint(33, 34))
        units_per_appointment = kwargs.get(
            'units_per_appointment', randint(10, 11))
        stock = kwargs.get('stock', randint(3500, 3501))
        _excess_supply = kwargs.get('excess_supply', 3500 - 33 * 10)

        # unpack warm_up, horizon, cool_down from kwargs if present else use defaults
        warm_up = kwargs.get('warm_up', 7 * 6 * 4)  # 4 weeks
        horizon = kwargs.get('horizon', 7 * 6 * 5)  # 5 weeks
        cool_down = kwargs.get('cool_down', 0)  # 0 weeks

        _anticipation = anticipation and rule == 'Extended'

        Antigens.population_abd_usabilities = abd_usability(
            non_scd_frequencies.frequencies.to_numpy(),
            kwargs.get('scd_requests_ratio', 330/3500), 1.0)
        antigens = Antigens(ANTIGENS, rule=matching_antigens,
                            allo_Abs=allo_ab_data.values.flatten())
        antigens.allo_risk = immuno[antigens.antigen_index[3:]].to_numpy(
        ).flatten()

        def demand(): return Demand(antigens, patient_data, appointments, units_per_appointment,
                                    antigen_string=False, dummy_data=dummy_data, dummy_extra_demand=_excess_supply)

        def supply(): return Supply(antigens, donor_data, stock, antigen_string=False)

        def matching(): return MatchingArea(algo='transport', antigens=antigens, matching_rule=rule,
                                            anticipation=_anticipation)

        def inventory(): return Inventory(kwargs.get('max_age', 35),
                                          kwargs.get(
                                              'starting_inventory', 30_000 - 3500),
                                          watched_antigens=kwargs.get(
                                              'watched_antigens', np.array([114688])),
                                          watched_phenotypes=kwargs.get(
                                              'watched_phenotypes', np.array([0])),
                                          start_age_dist=init_age_dist)

        manager = SimulationManager(antigens, demand, supply, matching, inventory,
                                    warm_up, horizon, cool_down, replications, seed,
                                    precompute_outfolder=folder)
        manager.do_precompute(cpus)

        end_time = time.time()
        elapsed_mins = (end_time - start_time) / 60

        print(f'\nThe precompute took {elapsed_mins: .1f} minute(s).')
        break

    print(f'\nThe output folder is at {folder}')


def tuning(rule='Extended', seed=0xBE_BAD_BAE, replications=10, cpus=10, weights: np.ndarray = None, pop_phen_configs: dict = None, num_objectives=1, anticipation=True, **kwargs):
    """
    Tuning of the simulation parameters.

    Parameters
    ----------
    rule : str, optional
        The matching rule to use, by default 'Extended'.
    seed : int, optional
        The seed to use, by default 0xBE_BAD_BAE.
    replications : int, optional
        The number of replications to perform, by default 10.
    cpus : int, optional
        The number of CPUs to use, by default 10.
    weights : np.ndarray, optional
        Weights for the total penalty/cost function components, by default None.
    pop_phen_configs : dict, optional
        Population phenotype configurations, by default None.
    num_objectives : int, optional
        Number of objectives being optimised, by default 1.
    anticipation : bool or list of bool, optional
        Whether to use anticipation in the matching, by default True.
    **kwargs : dict
        Additional keyword arguments.
        
    Other Parameters
    ----------------
    objectives_names : list, optional
        Names of the objectives
    folder : str, optional
        Folder to save the output from simulation replications
    substitution_weight_equal : bool, optional
        Whether to force both substitution penalties weights to be equal

    Returns
    -------
    float or np.ndarray
        The mean value(s) of the objective(s) considered.
        
    Notes
    -----
    For other possible keyword arguments, see the `exp3` function.
    """
    start_datetime = datetime.datetime.now().strftime('%Y%m%d-%H-%M')
    root_now = datetime.datetime.now()
    root_folder_date = root_now.strftime('%Y%m%d')
    root_folder_time = root_now.strftime('%H%M')
    exp: str = '3' if kwargs.get(
        'exp2', None) is None else kwargs.get('exp2', '2')
    print(f'\n###\Tuning evaluation at {start_datetime}')
    if pop_phen_configs is None:
        pop_phen_configs = dict()
    matching_rules = load_rule_sets(
        'BSCSimulator/experiments/matching_rules.json')['MATCHING_RULES']
    donor_data = population_phenotype(pop_phen_configs.get(
        'donor', 'data/bloodgroup_frequencies/blood_groups_donors.json'), 0.01)
    patient_data = population_phenotype(pop_phen_configs.get(
        'patient', 'data/bloodgroup_frequencies/blood_groups.json'), 1.0)
    non_scd_frequencies = dummy_population_phenotypes(
        pop_phen_configs.get('dummy', 'data/bloodgroup_frequencies/ABD_dummy_demand.tsv'))
    dummy_data = pad_abd_phenotypes(non_scd_frequencies, len(ANTIGENS) - 3)
    dummy_data = kwargs.get('dummy_data', dummy_data)
    allo_ab_data = load_alloantibodies(kwargs.get(
        'ab_datafile', 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv'))
    data = (donor_data, patient_data)
    immuno = load_immunogenicity()
    init_age_dist = load_initial_age_distribution(
        kwargs.get('initial_age_dist', None))
    pre_compute_folder = os.path.realpath(kwargs.get(
        'pre_compute_folder', f'out/experiments/exp{exp}/precompute/'))

    matching_antigens = matching_rules['Extended']['antigen_set']

    if weights is None:
        weights = np.ones(5)

    appointments = kwargs.get('appointments', randint(33, 34))
    units_per_appointment = kwargs.get(
        'units_per_appointment', randint(10, 11))
    stock = kwargs.get('stock', randint(3500, 3501))
    _excess_supply = kwargs.get('excess_supply', 3500 - 33 * 10)

    # unpack warm_up, horizon, cool_down from kwargs if present else use defaults
    warm_up = kwargs.get('warm_up', 7 * 6 * 4)  # 4 weeks
    horizon = kwargs.get('horizon', 7 * 6 * 5)  # 5 weeks
    cool_down = kwargs.get('cool_down', 0)  # 0 weeks

    forecasting = kwargs.get('forecasting', None)

    if isinstance(anticipation, bool):
        _anticipation = [anticipation and rule == 'Extended'] * 3
    else:
        _anticipation = [antn and rule == 'Extended' for antn in anticipation]

    Antigens.population_abd_usabilities = abd_usability(
        non_scd_frequencies.frequencies.to_numpy(),
        kwargs.get('scd_requests_ratio', 330/3500), 1.0)
    antigens = Antigens(ANTIGENS, rule=matching_antigens,
                        allo_Abs=allo_ab_data.values.flatten())
    antigens.allo_risk = immuno[antigens.antigen_index[3:]].to_numpy(
    ).flatten()

    def demand():
        return Demand(antigens, patient_data, appointments, units_per_appointment,
                      antigen_string=False, dummy_data=dummy_data, dummy_extra_demand=_excess_supply)

    def supply():
        return Supply(antigens, donor_data, stock, antigen_string=False)

    def matching():
        return MatchingArea(
            algo='transport', antigens=antigens, matching_rule=rule, anticipation=_anticipation[0],
            cost_weights=weights, solver=kwargs.get('solver', 'maxflow'),
            young_blood_constraint=kwargs.get('yb_constraint', True),
            substitution_penalty_parity=kwargs.get('substitution_weight_equal', True))

    def inventory():
        return Inventory(
            kwargs.get('max_age', 35),
            kwargs.get('starting_inventory', 30_000 - 3500),
            watched_antigens=kwargs.get(
                'watched_antigens', np.array([114688])),
            watched_phenotypes=kwargs.get('watched_phenotypes', np.array([0])),
            start_age_dist=init_age_dist)

    pre_compute_folder = pre_compute_folder if _anticipation[1] or _anticipation[2] else None
    manager = SimulationManager(antigens, demand, supply, matching, inventory,
                                warm_up, horizon, cool_down, replications, seed,
                                pre_compute_folder,  _anticipation[1],
                                _anticipation[2], forecasting=forecasting)
    if cpus > 1 and replications > 1:
        manager.do_simulations_parallel(min(cpus, replications))
    else:
        manager.do_simulations()
    manager.statistics()

    pad = [0, 0, 0]
    allo_padded = np.hstack(([pad, pad], np.vstack(manager.allo)))
    scd_short_padded = np.full(
        (2, len(ANTIGENS)), np.array(manager.scd_shorts)[:, None])
    all_short_padded = np.full(
        (2, len(ANTIGENS)), np.array(manager.all_shorts)[:, None])

    index = ['mismatch_avg', 'mismatch_stderr', 'allo_avg', 'allo_stderr', 'subs_avg', 'subs_stderr',
             'short_avg', 'short_stderr', 'all_short_avg', 'all_short_stderr']
    stacked = np.vstack((*manager.mismatches, allo_padded,
                        *manager.subs, scd_short_padded, all_short_padded))
    df = pd.DataFrame(stacked, index=index, columns=ANTIGENS)

    now = datetime.datetime.now()
    folder = kwargs.get('folder', os.path.join(
        'out/experiments/exp3/tuning', root_folder_date, root_folder_time, ''))
    folder = os.path.realpath(folder)
    os.makedirs(folder, exist_ok=True)

    file = os.path.join(folder, rule + now.strftime('%d_%H-%M_output.tsv'))

    df.to_csv(file, sep='\t')
    print(f'Output written to {file}')

    obj_cols = ['alloimmunisations',
                'scd_shortages', 'expiries', 'all_shortages']
    obj_stock_cols = ['O_neg_level', 'O_pos_level', 'O_level']
    obj_mm_cols = ['D_subs_num_patients',
                   'ABO_subs_num_patients', 'ABOD_subs_num_patients']
    df_obj = pd.DataFrame(manager.objs, columns=obj_cols +
                          obj_stock_cols+obj_mm_cols)
    file5 = os.path.join(
        folder, rule + now.strftime('%d_%H-%M_objectives.tsv'))
    df_obj.to_csv(file5, sep='\t', index=False)

    objectives_to_use = kwargs.get('objectives_names', obj_cols)[
        :num_objectives]
    objectives_values = df_obj[objectives_to_use].mean(axis=0).values

    if num_objectives == 1:
        return objectives_values[0]
    else:
        return objectives_values
