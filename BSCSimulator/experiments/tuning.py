""" Bayesian optimization of the simulator parameters using emukit."""

import datetime
import os
import time
from typing import Dict, List, Tuple, Union

import GPy
import numpy as np
import pandas as pd
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.core.loop import UserFunctionWrapper
from emukit.core.loop.stopping_conditions import \
    FixedIterationsStoppingCondition
from emukit.model_wrappers import GPyModelWrapper

from BSCSimulator.experiments.allo_incidence import tuning
from BSCSimulator.experiments.bayesopt import (
    MultiObjectiveBayesianOptimizationLoop, NormMatern52, TargetExtractorFunction)


class Evaluator:
    """Class to evaluate the simulator for a given set of parameters.

    :param replications: number of replications
    """

    def __init__(self, replications: int = 10, tuning_kwargs: Dict = None):
        self.replications = replications
        if tuning_kwargs is None:
            tuning_kwargs = dict()
        self.tuning_kwargs = tuning_kwargs

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the simulator for a given set of parameters.

        :param X: numpy array of shape (n_points, n_params)
        :return: numpy array of shape (n_points, 1)
        """
        fitness = np.zeros((X.shape[0], 1))
        for i, x in enumerate(X):
            alloimmunisations = tuning(weights=x, replications=self.replications, **self.tuning_kwargs)
            fitness[i] = alloimmunisations.sum()
        return fitness


class MOEvaluator:
    """Class to evaluate the simulator on a given set of parameters
    for multiple objectives.

    :param replications: number of replications
    :param num_objectives: number of objectives
    """

    def __init__(self, replications: int = 10, num_objectives: int = 2,
                 objective_directions: List = None, tuning_kwargs: Dict = None,
                 all_variables: Dict = None, fixed_variables: Dict = None,
                 variable_names: List = None):
        self.replications = replications
        self.num_objectives = num_objectives
        if objective_directions is None:
            self.objective_directions = ['MIN'] * num_objectives
        else:
            self.objective_directions = objective_directions
        directions_dict = {'MIN': 1, 'MAX': -1}
        self.directions_multipliers = np.array([directions_dict[d] for d in self.objective_directions])
        if tuning_kwargs is None:
            tuning_kwargs = dict()
        self.tuning_kwargs = tuning_kwargs
        self.all_variables = all_variables
        # self.num_variables = len(all_variables)
        self.fixed_variables = fixed_variables
        # self.fixed_variables_mask = fixed_variables_mask
        self.variable_names = variable_names

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the simulator for a given set of parameters.

        :param X: numpy array of shape (n_points, n_params)
        :return: numpy array of shape (n_points, n_objectives)
        """
        fitness = np.zeros((X.shape[0], self.num_objectives))
        for i, x in enumerate(X):
            if self.fixed_variables is not None:
                _x = np.zeros(len(self.all_variables))
                for p, v in zip(self.variable_names, x):
                    _x[self.all_variables[p]] = v
                for p, v in self.fixed_variables.items():
                    _x[self.all_variables[p]] = v
                x = _x
            objectives = tuning(weights=x, replications=self.replications,
                                num_objectives=self.num_objectives, **self.tuning_kwargs)
            fitness[i] = objectives * self.directions_multipliers
        return fitness


def bayes_opt_tuning(
        init_points_count=25, num_iterations=50, X_init: np.ndarray = None, Y_init: np.ndarray = None,
        replications=10, variable_names: List = None, objective_name: List = None,
        gp_mean: Union[float, int, GPy.core.mapping.Mapping] = None, iteration_chunks: int = 5,
        objective_directions: List = None, normalized_kernel: bool = True,
        tuning_kwargs: Dict = None, fixed_variables: Dict = None):
    """Perform Bayesian optimization of the simulator parameters.

    :param int init_points_count: number of initial points
    :param int num_iterations: number of iterations
    :param numpy.ndarray X_init: initial points
    :param numpy.ndarray Y_init: initial evaluations
    :param int replications: number of replications
    :param list variable_names: names of the variables to optimize
    :param list objective_name: name of the objective
    :param gp_mean: mean of the GP. Can be a float, an int or a GPy `Mapping`
    :param int iteration_chunks: number of times the GP should be replaced with a fresh GP object
    :param list objective_directions: maximisation or minimisation for the objective
    :param bool normalized_kernel: whether to use the normalized Matern52 kernel
    :param dict tuning_kwargs: kwargs for tuning function
    :param dict fixed_variables: fixed variables
    :return: None
    """
    folder_clash = True
    folder_clash_count = 0
    folder_clash_max = 100
    folder_clash_rng = np.random.default_rng(int(time.time() * 1000))
    while folder_clash and folder_clash_count < folder_clash_max:
        root_now = datetime.datetime.now()
        root_folder_date = root_now.strftime('%Y%m%d')
        root_folder_time = root_now.strftime('%H%M')
        folder = os.path.join('out/experiments/exp3/tuning', root_folder_date, root_folder_time, '')
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
        raise OSError(f'Could not create unique folder for tuning results after {folder_clash_max} attempts.')
        
    var_obj_names = ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood', 'alloimmunisations']
    if variable_names is None:
        var_names = var_obj_names[:-1]
    else:
        var_names = variable_names
    if objective_name is None:
        objective_name = var_obj_names[-1:]
    tuning_kwargs.update({'folder': folder, 'objectives_names': objective_name})
    # fixed_variables_mask = [name in fixed_variables.keys() for name in var_obj_names[:-1]]
    all_variables_indices = {name: i for i, name in enumerate(var_obj_names[:-1])}
    evaluator = MOEvaluator(replications, num_objectives=1, objective_directions=objective_directions,
                            tuning_kwargs=tuning_kwargs,
                            all_variables=all_variables_indices, fixed_variables=fixed_variables,
                            variable_names=var_names)
    space = ParameterSpace([ContinuousParameter(w, 0, 1) for w in var_names])

    lhs_design = LatinDesign(space)
    # init_points_count = 25
    if X_init is None:
        X_init = lhs_design.get_samples(init_points_count)
        Y_init = evaluator.evaluate(X_init)
    func = UserFunctionWrapper(evaluator.evaluate)

    if isinstance(gp_mean, (int, float)):
        mean_function = GPy.mappings.Constant(space.dimensionality, 1, gp_mean)
    elif isinstance(gp_mean, GPy.core.mapping.Mapping):
        mean_function = gp_mean
    elif gp_mean is None:
        mean_function = None
    else:
        mean_function = None
        
    iteration_chunks = min(iteration_chunks, num_iterations)
    iteration_chunk_size = num_iterations // iteration_chunks
    remaining_iterations = num_iterations % iteration_chunks
    
    if remaining_iterations > 0:
        iteration_chunks_list = [iteration_chunk_size] * iteration_chunks + [remaining_iterations]
    else:
        iteration_chunks_list = [iteration_chunk_size] * iteration_chunks
    
    if normalized_kernel:
        kernel = NormMatern52(space.dimensionality, variance=1.0, ARD=True)
    else:
        kernel = GPy.kern.Matern52(space.dimensionality, variance=1.0, ARD=True)
        
    X = X_init
    Y = Y_init
    for i in iteration_chunks_list:
        noise_var = Y.var() * 0.01
        gpy_model = GPy.models.GPRegression(X, Y, kernel, mean_function=mean_function,
                                            normalizer=True, noise_var=noise_var)
        gpy_model.optimize()
        model = GPyModelWrapper(gpy_model, n_restarts=5)
        gpy_model.likelihood.constrain_bounded(1e-9, noise_var, warning=False)
        gpy_model.kern.lengthscale.constrain_bounded(1e-9, 1e6, warning=False)

        bo_loop = BayesianOptimizationLoop(model=model, space=space)
        # num_iterations = 50
        stopping_condition = FixedIterationsStoppingCondition(i)

        bo_loop.run_loop(func, stopping_condition)
        # results = bo_loop.get_results()
        X = bo_loop.loop_state.X
        Y = bo_loop.loop_state.Y
    
    results = bo_loop.get_results()

    results_df = pd.DataFrame(np.atleast_2d(
        np.hstack((results.minimum_location.flatten(), [results.minimum_value * evaluator.directions_multipliers[0]]))),
                              columns=var_names + objective_name)

    points = np.hstack((bo_loop.loop_state.X, bo_loop.loop_state.Y * evaluator.directions_multipliers[0]))
    points_df = pd.DataFrame(points, columns=var_names + objective_name)

    prefix = datetime.datetime.now().strftime('%d_%H-%M')
    results_df.to_csv(os.path.join(folder, prefix + 'tuning_results.tsv'), index=False, sep='\t')
    points_df.to_csv(os.path.join(folder, prefix + 'tuning_points.tsv'), index=False, sep='\t')


def unpack_previous_evaluations(
        files: Union[List, str, Tuple],
        var_obj_names: List = ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood',
                               'alloimmunisations']) -> Dict:
    """Unpack the previous evaluations from the tuning points file.

    :param files: path to the tuning points file or list of paths
    :param list var_obj_names: list of variable and objective names
    :return dict: dict with keys 'X_init' and 'Y_init'
    """
    if isinstance(files, str):
        results_df = pd.read_csv(files, sep='\t')
        X_init = results_df[var_obj_names[:-1]].to_numpy()
        Y_init = results_df[var_obj_names[-1:]].to_numpy()
    elif isinstance(files, (list, tuple)):
        all_X_init = []
        all_Y_init = []
        for file in files:
            results_df = pd.read_csv(file, sep='\t')
            all_X_init.append(results_df[var_obj_names[:-1]].to_numpy())
            all_Y_init.append(results_df[var_obj_names[-1:]].to_numpy())
        X_init = np.vstack(all_X_init)
        Y_init = np.vstack(all_Y_init)
    result = {'X_init': X_init, 'Y_init': Y_init}
    return result


def multi_objective_bayes_opt_tuning(
        init_points_count=25, num_iterations=50, X_init: np.ndarray = None, Y_init: np.ndarray = None,
        replications=10, num_objectives=2, variable_names: List = None, objective_names: List = None,
        objective_directions: List = None, iteration_chunks : int = 5, normalized_kernel: bool = True,
        tuning_kwargs: Dict = None, fixed_variables: Dict = None):
    """Multi-objective Bayesian optimization for tuning.

    :param int init_points_count: number of initial points
    :param int num_iterations: number of iterations
    :param ndarray X_init: initial points as 2D numpy array 
    :param ndarray Y_init: initial evaluations as 2D numpy array
    :param int replications: number of replications
    :param int num_objectives: number of objectives
    :param list variable_names: names of variables
    :param list objective_names: names of objectives
    :param list objective_directions: maximisation or minimisation for each objective
    :param int iteration_chunks: number of times the GP should be replaced with a fresh GP object
    :param bool normalized_kernel: whether to use the normalized Matern52 kernel
    :param dict tuning_kwargs: kwargs for tuning
    :param dict fixed_variables: fixed variables
    :return: None    
    """
    folder_clash = True
    folder_clash_count = 0
    folder_clash_max = 100
    folder_clash_rng = np.random.default_rng(int(time.time() * 1000))
    while folder_clash and folder_clash_count < folder_clash_max:
        root_now = datetime.datetime.now()
        root_folder_date = root_now.strftime('%Y%m%d')
        root_folder_time = root_now.strftime('%H%M')
        folder = os.path.join('out/experiments/exp3/tuning', root_folder_date, root_folder_time, '')
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
        raise OSError(f'Could not create unique folder for tuning results after {folder_clash_max} attempts.')
    
    tuning_kwargs.update({'folder': folder})
    
    var_obj_names = ['immunogenicity', 'usability', 'substitutions',
                     'fifo', 'young_blood', 'alloimmunisations', 'shortages']
    if variable_names is None:
        var_names = var_obj_names[:-num_objectives]
    else:
        var_names = variable_names
    if objective_names is None:
        obj_names = var_obj_names[-num_objectives:]
    else:
        obj_names = objective_names[:num_objectives]
    tuning_kwargs.update({'objectives_names': obj_names})
    all_variables_indices = {name: i for i, name in enumerate(var_obj_names[:5])}
    evaluator = MOEvaluator(replications, num_objectives, objective_directions, tuning_kwargs,
                            all_variables=all_variables_indices, fixed_variables=fixed_variables,
                            variable_names=var_names)
    # obj_index_names = {name: i for i, name in enumerate(obj_names)}
    space = ParameterSpace([ContinuousParameter(w, 0, 1) for w in var_names])

    lhs_design = LatinDesign(space)
    if X_init is None:
        X_init = lhs_design.get_samples(init_points_count)
        Y_init = evaluator.evaluate(X_init)
    func = UserFunctionWrapper(evaluator.evaluate)

    iteration_chunks = min(iteration_chunks, num_iterations)
    iteration_chunk_size = num_iterations // iteration_chunks
    remaining_iterations = num_iterations % iteration_chunks
    
    if remaining_iterations > 0:
        iteration_chunks_list = [iteration_chunk_size] * iteration_chunks + [remaining_iterations]
    else:
        iteration_chunks_list = [iteration_chunk_size] * iteration_chunks

    if normalized_kernel:
        kernel = NormMatern52(space.dimensionality, variance=1.0, ARD=True)
    else:
        kernel = GPy.kern.Matern52(space.dimensionality, variance=1.0, ARD=True)
    
    extractor = TargetExtractorFunction(random_seed=20230501, num_objectives=num_objectives, lower_anchor=0)
    
    X = X_init
    Y_scalarized = extractor.mock_call(Y_init, 20230501)
    Y = Y_init
    for i in iteration_chunks_list:
        is_same = Y_scalarized.min(axis=0) == Y_scalarized.max(axis=0)
        if is_same.any():
            # Multiply first element by 1.001 and the last by 0.999
            # to prevent errors in GP normalizer
            Y_scalarized[0, is_same] *= 1.001
            Y_scalarized[-1, is_same] *= 0.999           
        noise_var = Y_scalarized.var() * 0.01
        gpy_model = GPy.models.GPRegression(
            X, Y_scalarized, kernel,
            normalizer=True, noise_var=noise_var)
        gpy_model.optimize()
        model = GPyModelWrapper(gpy_model, n_restarts=5)
        gpy_model.likelihood.constrain_bounded(1e-9, 1e-6, warning=False)
        gpy_model.kern.lengthscale.constrain_bounded(1e-9, 1e6, warning=False)

        bo_loop = MultiObjectiveBayesianOptimizationLoop(space, X, Y, model, targets_extractor=extractor)
        stopping_condition = FixedIterationsStoppingCondition(i)

        bo_loop.run_loop(func, stopping_condition)
        X = bo_loop.loop_state.X
        Y_scalarized = extractor.mock_call(bo_loop.loop_state.Y, 20230501)
        Y = bo_loop.loop_state.Y

    results = bo_loop.get_results()

    results_df = pd.DataFrame(
        np.hstack((results.pareto_front_X, results.pareto_front_Y * evaluator.directions_multipliers)),
        columns=var_names + obj_names)

    points = np.hstack((bo_loop.loop_state.X, bo_loop.loop_state.Y * evaluator.directions_multipliers))
    points_df = pd.DataFrame(points, columns=var_names + obj_names)

    prefix = datetime.datetime.now().strftime('%d_%H-%M')
    results_df.to_csv(os.path.join(folder, prefix + 'tuning_results.tsv'), index=False, sep='\t')
    points_df.to_csv(os.path.join(folder, prefix + 'tuning_points.tsv'), index=False, sep='\t')
