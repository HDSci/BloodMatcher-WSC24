"""
The experiments package contains modules and functions for running experiments on the BSCSimulator,
utilizing the components of BSCSimulator that are outside of this package.

The main modules in this package are:
    - allo_incidence: for running one-off simulations of matching rules;
    - tuning: for optimising the parameters of matching rules using Bayesian optimisation.
"""
from .allo_incidence import exp3, precompute_exp3
from .allo_incidence import tuning as allo_incidence_tuning
from .tuning import (bayes_opt_tuning, multi_objective_bayes_opt_tuning,
                     unpack_previous_evaluations)
