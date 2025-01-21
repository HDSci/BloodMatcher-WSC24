# Blood Supply Chain Simulator & Extended Blood Matching for Sickle Cell Patients


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14343039.svg)](https://doi.org/10.5281/zenodo.14343039)

This repository contains the software implementation and source code for our paper titled ["Optimization Of Extended Red Blood Cell Matching In Transfusion Dependent Sickle Cell Patients"](https://ieeexplore.ieee.org/document/10838863):

> F. B. Oyebolu, M. Chion, M. L. Wemelsfelder, S. Trompeter, N. Gleadall and W. J. Astle, "Optimization of Extended Red Blood Cell Matching in Transfusion Dependent Sickle Cell Patients," 2024 Winter Simulation Conference (WSC), Orlando, FL, USA, 2024, pp. 1023-1034, doi: 10.1109/WSC63780.2024.10838863.



## Contents & Structure of the Repository

The repository contains:
1. The implementation of a discrete-event simulation model for evaluating the performance of different blood matching rules in a blood supply chain for transfusion-dependent sickle cell patients.
    This can be found in the [`BSCSimulator`](BSCSimulator) package.
2. A Google OR-Tools-based network solver for solving the day-to-day minimum cost flow problem for allocating red blood cell (RBC) units to requests.
    This can be found in the [`mincostflow.pyx`](BSCSimulator/mincostflow.pyx) file.
3. A set of experiments for evaluating the performance of different blood matching rules under different scenarios.
    - The structure of experiments are defined in the [`experiments`](BSCSimulator/experiments) package.
    - Specifically, functions in [`allo_incidence.py`](BSCSimulator/experiments/allo_incidence.py).
4. Bayesian optimization implementation for optimizing the policy parameters of the matching rules.
    - [`bayesopt.py`](BSCSimulator/experiments/bayesopt.py) contains the implementation of the Bayesian optimization algorithms.
    - [`tuning.py`](BSCSimulator/experiments/tuning.py) contains the functions for tuning the policy parameters of the matching rules.
5. Scripts for generating plots and tables from the simulation results, as well as a Jupyter notebook for visualizing and analyzing the results.
    - [`convenience_functions.py`](scratch/convenience_functions.py) contains the functions for generating plots and tables.
    - [`20240229.ipynb`](scratch/notebooks/20240229.ipynb) is the Jupyter notebook for visualizing and analyzing the results.


## Installation

This software was developed primarily on Linux systems and using Python 3.8.
Pyenv and pyenv-virtualenv were used to manage the Python virtual environments for this project and the dependencies are listed in the [`requirements.txt`](requirements.txt) file.

However, please note:
- We cannot guarantee that this software will work straight out-of-the-box on Windows (especially with the [Cython extensions](#cython-extensions)). We recommend using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/about), which is a solution we have used for parts of the development.
- Python added some features to type hinting between versions 3.8 and 3.9, e.g., being able to use `list[int]` instead of `typing.List[int]`. See [this StackOverflow post for an explanation](https://stackoverflow.com/questions/39458193/using-list-tuple-etc-from-typing-vs-directly-referring-type-as-list-tuple-etc/39458225#39458225). The type hints in this code _may_ contain a mix of these styles, and if you are using Python 3.8, you may get a `TypeError: 'type' object is not subscriptable` error. Either upgrade to Python 3.9 or hunt down the type hints and change them to the older style. (Sorry about that! Some of the documentation and type hints were written after an upgrade to a newer Python version.)
- The Jupyter notebook used a different virtual environment, the requirements for which are listed in the [`scratch/notebooks/`](scratch/notebooks) directory.


### Cython Extension(s)

After installing the dependencies from [`requirements.txt`](requirements.txt), make sure to [Cythonize](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#compilation-using-setuptools) the [`mincostflow.pyx`](BSCSimulator/mincostflow.pyx) file which will enable the OR-Tools network solver.
This is done by running the [`setup.py`](setup.py) script with arguments as shown below:

```bash

python setup.py build_ext --inplace

```

## Execution

### Experiments

To run the simulator, execute the following command:

```bash

python -m BSCSimulator.main [parameterfile]

```

If a parameter file is not provided, the simulator will use the default parameters defined in the default file, [`BSCSimulator/experiments/parameters.py`](BSCSimulator/experiments/parameters.py).
Also, if a parameter is not provided in the optional parameter file, the simulator will use the default value defined in the default file.
However, this is not 'recursive' to parameters that are dictionaries. If you want to change a single value in a dictionary, you will need to provide the entire dictionary (with the other key-value pairs you want to keep the same) in the parameter file.

The parameter files used for the experiments in our paper are under the [`scratch/paramfiles`](scratch/paramfiles) directory. For the matching rules with forecasting (i.e., **E1**), you will need to run the precomputation of supply/demand beforehand. The parameter file for the precomputation is under the [`scratch/paramfiles/precomputation`](scratch/paramfiles/precomputation) directory.
For completeness:
- Results from Section 5.1 are from running the parameter files in `matching-rules/`.
- Results from Section 5.2 are from running the parameter files in `penalty-analysis/`.
- Results from Section 5.3 are from running the parameter files in `tuning/`, then updating the parameter files in `inventory_optimal/` with the tuned parameters and running those files.

Output will be written to an `out` directory on the top level of the repository.
Of this, results are written generally to a date-stamped directory, e.g., `out/experiments/exp3/20240228/`, but usually the program will print out the exact directory it is writing to.
Error logs are written to `out/logs/`.

### Analysis

As previously mentioned, the Jupyter notebook [`20240229.ipynb`](scratch/notebooks/20240229.ipynb) contains the code for visualizing and analyzing the results of the experiments.

Do note that we used an IDE that allowed us to run the notebooks such that it seemed like it was on the top level of the repository. If you cannot find that setting, you will have to either adjust the paths in the notebook or move the notebook to the top level of the repository.
Similarly, the notebook is currently set up to write figures to specific file paths. You may need to adjust these paths to your own, or comment out the lines that write to files, or make sure the directories exist.
It generally writes to one directory with path `scratch/figures/Enlil`.


## General Notes & Assumptions

### Antigens

The antigens are represented as integers, with each bit representing a different antigen: 0 if the antigen is not present, 1 if it is present. The bits are ordered as follows: A, B, D, C, c, E, e, K, k, Fya, Fyb, Jka, Jkb, M, N, S, s. For example, the antigen profile `0b00000000000000001` represents a patient that is positive only for the s antigen.

### Alloantibodies

When generating the alloantibody profiles, we first generate a 'mask' that is a boolean array of length 14 (all antigens bar the major antigens A, B, and D). `True` means they could have the alloantibody, `False` means they do not. This is generated by taking a random number between 0 and 1, and if it is less than the probability of having the alloantibody, we set the corresponding element in the mask to `True`.
This array is passed around to various objects and functions, and is stored in the precomputed data. However, from the description of its generation, it is clear that the value in the mask is not conditioned on the absence of the corresponding antigen in the patient's profile.
It is only when we are checking for compatibility in the `transport_matching` function ([roughly here](https://github.com/HDSci/BloodMatcher-WSC24/blob/main/BSCSimulator/matching.py#L288)) that we make sure that the patient can only have an alloantibody if they do not have the corresponding antigen and make the conversion to their _actual_ alloantibody profile.
So to be safe, always force this conversion if you want to read the alloantibody profile.

### Demand

Each request is represented as an integer numpy array with four elements. In order, these are: a unique identifier for the request, the patient's phenotype (antigen profile), the number of units requested, and the time the request is due.

### Supply

Each donor is represented as an integer numpy array with three elements. In order, these are: a unique identifier for the donor, the donor's phenotype, and the date/time the unit was donated. This representation also extends to how the inventory is stored.
