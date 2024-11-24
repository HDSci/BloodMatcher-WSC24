"""Custom multi-objective Bayesian optimization algorithm using emukit."""


import logging
from typing import Callable

import numpy as np
import GPy
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.acquisitions.log_acquisition import \
    LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import \
    LocalPenalizationPointCalculator
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable, IModel
from emukit.core.loop import (FixedIntervalUpdater, OuterLoop,
                              SequentialPointCalculator)
from emukit.core.loop.loop_state import LoopState, create_loop_state
from emukit.core.optimization import (AcquisitionOptimizerBase,
                                      GradientAcquisitionOptimizer)
from emukit.core.parameter_space import ParameterSpace
from jmetal.core.quality_indicator import HyperVolume
from jmetal.util.ranking import FastNonDominatedRanking

from BSCSimulator.util import normalize

from .individuals import Individual

_log = logging.getLogger(__name__)


class MultiObjectiveBayesianOptimizationLoop(OuterLoop):
    def __init__(
        self,
        space: ParameterSpace,
        X_init: np.ndarray,
        Y_init: np.ndarray,
        model: IModel,
        acquisition: Acquisition = None,
        update_interval: int = 1,
        batch_size: int = 1,
        acquisition_optimizer: AcquisitionOptimizerBase = None,
        targets_extractor: Callable = None,
        extra_objectives_names: dict = None
    ):
        """
        Custom emukit class that implements a loop for multi-objective Bayesian optimization
        using the ParEGO algorithm.

        :param space: Input space where the optimization is carried out.
        :param X_init: Initial design points.
        :param Y_init: Initial observations.
        :param model: The model that will approximate the scalarized objective function.
        :param acquisition: The acquisition function that will be used to collect new points (default, EI).
                            If batch size is greater than one, this acquisition must output positive values only.
        :param update_interval: Number of iterations between optimization of model hyper-parameters. Defaults to 1.
        :param batch_size: How many points to evaluate in one interation of the optimization loop. Defaults to 1.
        :param acquisition_optimizer: Optimizer selecting next evaluation points by maximizing acquisition.
                                      Gradient based optimizer is used if None. Defaults to None.
        :param targets_extractor: Function that scalarizes the outputs using an augmented weighted Tchebycheff approach
                                  to retrain the model.
        :param extra_objectives_names: Names of the extra objectives as defined in the UserFunctionWrapper object.
                                       Dictionary with keys as the names of the objectives
                                       and values as the indices of Y_init.
        """

        self.model = model

        if acquisition is None:
            acquisition = ExpectedImprovement(model)

        model_updaters = FixedIntervalUpdater(
            model, update_interval, targets_extractor)

        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        if batch_size == 1:
            _log.info('Batch size is 1, using SequentialPointCalculator')
            candidate_point_calculator = SequentialPointCalculator(
                acquisition, acquisition_optimizer)
        else:
            _log.info(
                f'Batch size is {batch_size}, using LocalPenalizationPointCalculator')
            log_acquisition = LogAcquisition(acquisition)
            candidate_point_calculator = LocalPenalizationPointCalculator(
                log_acquisition, acquisition_optimizer, model, space, batch_size)

        loop_state = create_loop_state(X_init, Y_init)

        super().__init__(candidate_point_calculator, model_updaters, loop_state)

    def get_results(self):
        return MultiObjectiveBayesianOptimizationResults(self.loop_state)


class MultiObjectiveBayesianOptimizationResults:
    def __init__(self, loop_state: LoopState):
        """
        Custom emukit class that implements a container for the results of the multi-objective Bayesian optimization
        loop. It takes as input the loop state and computes some results, e.g. the Pareto front.

        :param loop_state: The loop state in its current form.
        """
        self.loop_state = loop_state
        self.ideal_solution = np.min(loop_state.Y, axis=0)
        self.nadir_solution = np.max(loop_state.Y, axis=0)
        self.hv_ref_point = self.nadir_solution + 0.01 * \
            (self.nadir_solution - self.ideal_solution)
        self._pareto_ranker = FastNonDominatedRanking()
        self.pareto_front_X, self.pareto_front_Y = self._compute_pareto_front(
            loop_state)
        self.hypervolume = self._compute_hypervolume(loop_state)

    def _compute_pareto_front(self, loop_state: LoopState = None):
        if loop_state is None:
            loop_state = self.loop_state

        X = loop_state.X
        Y = loop_state.Y
        C = np.zeros((X.shape[0], 1))
        x_bounds = np.hstack(
            (np.zeros((X.shape[1], 1)), np.ones((X.shape[1], 1))))
        c_bounds = np.array([[-np.inf, np.inf]])

        solutions = Individual.create_individual_solutions(
            x_bounds, X, Y, C, c_bounds)
        self._pareto_ranker.compute_ranking(solutions)
        pareto_front_sols = self._pareto_ranker.ranked_sublists[0]
        self._pareto_front = np.vstack(
            [ind.to_numpy() for ind in pareto_front_sols])
        return self._pareto_front[:, :X.shape[1]], self._pareto_front[:, X.shape[1]:]

    def _compute_hypervolume(self, loop_state: LoopState = None):
        if loop_state is None:
            loop_state = self.loop_state

        front = self.get_ranked_lists(0)
        self._hv = HyperVolume(self.hv_ref_point.tolist())
        return self._hv.compute([front[i].objectives for i in range(len(front))])

    def get_ranked_lists(self, rank=None):
        """
        Returns the ranked lists of solutions, ordered by dominance rank.
        Each list contains solutions that are non-dominated by the solutions in that list
        and subsequent lists in the sequence.

        :param rank: The rank of the list to return. If None, returns all lists.
        :return: The ranked lists of solutions.        
        """
        if rank is None:
            return self._pareto_ranker.ranked_sublists
        rank = min(rank, len(self._pareto_ranker.ranked_sublists))
        return self._pareto_ranker.ranked_sublists[rank]


class TargetExtractorFunction:
    def __init__(
            self, random_seed: int = 0, num_objectives: int = 2, rho: float = 0.05, lower_anchor: int = None,
            upper_anchor: int = None):
        """
        Class that implements a function that scalarizes the outputs using an augmented weighted Tchebycheff approach
        to retrain the model.

        :param random_seed: The random seed so that the random weights can be reproduced.
        :param num_objectives: The number of objectives.
        :param rho: The parameter that controls the weight of the augmented objective.
        :param lower_anchor: The lower anchor of the objective functions.
                                So, e.g., if the objective values are to be normalised between 0 and 1,
                                ``lower_anchor`` will be mapped to 0.
                                If ``None``, the minimum value of the objective values is used.
        :param upper_anchor: The upper anchor of the objective functions.
                                So, e.g., if the objective values are to be normalised between 0 and 1,
                                ``upper_anchor`` will be mapped to 1.
                                If ``None``, the maximum value of the objective values is used.
        """
        self.random_seed = random_seed
        self.num_objectives = num_objectives
        self._rng = np.random.default_rng(random_seed)
        self.lower_anchor = lower_anchor
        self.upper_anchor = upper_anchor
        self.rho = rho

    def __call__(self, loop_state: LoopState):
        """
        Function that scalarizes the outputs using an augmented weighted Tchebycheff approach
        to retrain the model.

        :param loop_state: The loop state in its current form.
        :return: The scalarized outputs.
        """
        Y = loop_state.Y
        if Y.shape[1] != self.num_objectives:
            raise ValueError(
                f'Number of objectives in loop state ({Y.shape[1]}) does not match '
                f'the number of objectives in the target extractor ({self.num_objectives})')
        # Draw random weight vector
        weights = random_weight_vector(k=self.num_objectives, rng=self._rng)
        # Normalise samples
        Y_norm = normalize(Y, 'maxmin', lower=0, upper=1, anchor_lower=self.lower_anchor,
                           anchor_upper=self.upper_anchor)
        # Weight samples
        Y_weighted = Y_norm * weights
        # Scalarise samples using augmented Tchebycheff
        Y_scalarised = np.max(Y_weighted, axis=1) + \
            self.rho * Y_weighted.sum(axis=1)
        return Y_scalarised[:, None]

    def mock_call(self, Y: np.ndarray, random_seed: int = 0):
        """
        Function that scalarizes the outputs using an augmented weighted Tchebycheff approach
        to retrain the model.

        :param ndarray Y: The Y values to be scalarized.
        :return ndarray: The scalarized outputs.
        """
        if Y.shape[1] != self.num_objectives:
            raise ValueError(
                f'Number of objectives in Y values ({Y.shape[1]}) does not match '
                f'the number of objectives in the target extractor ({self.num_objectives})')
        # Draw random weight vector
        weights = random_weight_vector(
            k=self.num_objectives, rng=np.random.default_rng(random_seed))
        # Normalise samples
        Y_norm = normalize(Y, 'maxmin', lower=0, upper=1, anchor_lower=self.lower_anchor,
                           anchor_upper=self.upper_anchor)
        # Weight samples
        Y_weighted = Y_norm * weights
        # Scalarise samples using augmented Tchebycheff
        Y_scalarised = np.max(Y_weighted, axis=1) + \
            self.rho * Y_weighted.sum(axis=1)
        return Y_scalarised[:, None]


def random_weight_vector(k: int = 2, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generates a random vector of numbers in [0,1] based on number of ``k`` objectives to weight.

    :param k: Number of objectives (2 or 3)
    :param rng: Random number generator
    :return: 1-D numpy array (vector) of positive elements which sum up to 1
    """
    if rng is None:
        rng = np.random
    s = 10 if k == 2 else 4
    if k == 1:
        return np.array([1.])
    elif k > 3 or k < 1:
        raise ValueError

    _sum = 0
    _lambda = []
    while len(_lambda) < k:
        try:
            _draw = rng.integers(s + 1)
        except AttributeError:
            _draw = rng.randint(s + 1)
        if _draw + _sum <= s:
            _sum += _draw
            _lambda.append(_draw)
        if _sum == s and len(_lambda) < k:
            _lambda.append(0)
        if len(_lambda) + 1 == k:
            _lambda.append(s - _sum)
            _sum = s
    _lambda = np.array(_lambda) / s
    return _lambda


class NormMatern52(GPy.kern.Matern52):
    """Normalized Matern52 kernel variant.

    This kernel is a Matern52 kernel with the points normalized by the sum of the absolute values of the coordinates.
    If the sum of the absolute values is zero, i.e., the point is the origin,
    the point is replaced by a point with all coordinates equal to 1.

    For example, the input point (0.5, 0.5) is the same as the point (1, 1) or (π, π).
    """

    def __init__(self, input_dim, variance=1, lengthscale=None, ARD=False, active_dims=None, name='NormalizedMatern52'):
        super().__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def _unscaled_dist(self, X, X2=None):
        X_abs_sum = np.abs(X).sum(axis=1)
        _X = X.copy()
        _X[X_abs_sum == 0] = 1
        norm_X = _X / np.abs(_X).sum(axis=1, keepdims=True)
        if X2 is not None:
            X2_abs_sum = np.abs(X2).sum(axis=1)
            _X2 = X2.copy()
            _X2[X2_abs_sum == 0] = 1
            norm_X2 = _X2 / np.abs(_X2).sum(axis=1, keepdims=True)
        else:
            norm_X2 = None
        return super()._unscaled_dist(norm_X, norm_X2)
