"""This module contains the ``Individual`` class which can be used by jMetalPy quality indicators."""

from typing import List
import numpy as np
from numpy import ndarray


class Individual:
    def __init__(self, x_bounds: ndarray, x: ndarray, y: ndarray, c: ndarray):
        """
        ``Solution``-like object which can be used by jMetalPy quality indicators.
        Shape of arrays is such that:

        - n_vars = number of variables
        - n_solutions = number of sampled/visited points
        - n_obj = number of objectives
        - n_cons = number of black-box constraints

        :param x_bounds: Numpy array of shape (n_vars, 2) containing lower and upper bounds on decision variables.
        :param x: Numpy array of shape (n_vars,) representing the X values.
        :param y: Numpy array of shape (n_obj,) representing the objective function values - Y values.
        :param c: Numpy array of shape (n_cons,) representing the values of the constraints.
        """

        self.number_of_variables = x.shape[0]
        self.number_of_objectives = y.shape[0]
        self.number_of_constrains = c.shape[0]
        self.number_of_constraints = self.number_of_constrains
        self.lower_bound = x_bounds[:, 0].tolist()
        self.upper_bound = x_bounds[:, 1].tolist()

        self.variables = x.tolist()
        self.objectives = y.tolist()
        self.constraints = c.tolist()
        self.attributes = {}

    @classmethod
    def create_individual_solutions(cls, x_bounds: ndarray, x: ndarray, y: ndarray, c: ndarray,
                                    c_bounds: ndarray, remove_infeasible=False):
        """
        Creates a list of solutions which can then be used by jMetalPy quality indicators.
        Shape of arrays is such that:

        - n_vars = number of variables
        - n_solutions = number of sampled/visited points
        - n_obj = number of objectives
        - n_cons = number of black-box constraints
        For ``x_bounds`` and ``c_bounds``, columns 0 and 1 are lower and upper bounds respectively.

        :param x_bounds: Numpy array of shape (n_vars, 2) containing lower and upper bounds on decision variables.
        :param x: Numpy array of shape (n_solutions, n_vars) representing the sampled points.
        :param y: Numpy array of shape (n_solutions, n_obj) representing the objective function values for each sampled point.
        :param c: Numpy array of shape (n_solutions, n_cons) representing the values of the constraints.
        :param c_bounds: Numpy array of shape (n_cons, 2) containing lower and upper bounds on constraint values.
        :param remove_infeasible: skips/removes infeasible solutions if set to True
        :return: List of Individual.
        """
        solutions: List[Individual] = []

        for xval, yval, cval in zip(x, y, c):
            c_low = cval - c_bounds[:, 0]
            c_upp = c_bounds[:, 1] - cval
            c_all = np.hstack((c_low, c_upp))
            if remove_infeasible and bool(np.all(c_all >= 0)) is False:
                continue

            ind = cls(x_bounds, xval, yval, c_all)
            solutions.append(ind)

        return solutions

    def to_numpy(self, include_constraints=False):
        """
        Converts the ``Individual`` object to a numpy array.

        :param include_constraints: If set to True, the constraints are also included in the returned array.
        :return: Numpy array of shape (1, n_vars + n_obj) if ``include_constraints`` is False, else (1, n_vars + n_obj + n_cons).
        """
        if include_constraints:
            return np.hstack((self.variables, self.objectives, self.constraints))[None, :]
        else:
            return np.hstack((self.variables, self.objectives))[None, :]

    def __copy__(self):
        ones = np.ones(shape=(1, 2))
        new_one = Individual(ones, ones, ones, ones)
        new_one.__dict__.update(self.__dict__.copy())
        return new_one

    def __str__(self):
        return f"Individual(x={self.variables}, y={self.objectives}, c={self.constraints})"

    def __repr__(self):
        return self.__str__()
