import numpy as np
import pandas as pd


ANTIGENS = ["A", "B", "D", "C", "c", "E", "e", "K", "k", "Fya", "Fyb", "Jka", "Jkb", "M", "N", "S", "s"]


def antigen_distribution_from_files(filenames):
    """Reads antigen distributions from files and returns them as a list of numpy arrays.
    
    :param list filenames: list of filenames of tsv files with antigen distributions
    :return list: list of numpy arrays with antigen distributions
    """
    antigens = []
    for filename in filenames:
        array = pd.read_csv(filename, sep='\t').to_numpy()
        array[:, -2:] = array[:, -2:] / (array[:, -2:]).sum(axis=0)
        antigens.append(array)
    return antigens


def bloodgroup_frequency(antigens_info: dict, pop_dist: np.ndarray):
    """
    Calculates the frequency of all blood phenotypes from the antigen distributions.
    Returns a matrix of phenotypes and a vector of frequencies.
    
    :param dict antigens_info: dictionary with keys for blood group systems and values for filenames, antigens, and which antigens to include
    :param numpy.ndarray pop_dist: vector of fractions of White and Black population
    :return tuple: tuple of numpy.ndarray with phenotypes and frequencies
    """
    blood_groups = antigens_info.keys()
    antigen_filenames = [antigens_info.get(bg).get('filename') for bg in blood_groups]
    antigens_incl = [antigens_info.get(bg).get('include') for bg in blood_groups]
    antigen_list = [np.array(antigens_info.get(bg).get('antigens'))[antigens_incl[i]] for i, bg in enumerate(blood_groups)]
    antigen_list = list(np.hstack(antigen_list))
    antigen_dists = antigen_distribution_from_files(antigen_filenames)
    
    phens = np.array([])
    freq = np.array([])
    
    for k, antigen_dist in enumerate(antigen_dists):
        if not any(antigens_incl[k]):
            continue
        
        mat = antigen_dist[:, np.append(antigens_incl[k], [True, True])]
        unique_rows = np.unique(mat[:, :-2], axis=0)
        freq_vec = np.zeros([np.size(unique_rows, 0), 2])
        
        excl_vec = [True] * len(unique_rows)
        for l in range(len(unique_rows)):
            log_vec = np.all(mat[:, :-2] == unique_rows[l, :], axis=1)
            freq_vec[l, :] = sum(mat[log_vec, -2:])
            # exclude bloodgroups that do not occur (occur with frequency 0)
            if sum(freq_vec[l, :]) == 0:
                excl_vec[l] = False
        
        unique_rows = unique_rows[excl_vec]
        freq_vec = freq_vec[excl_vec]
        
        if np.size(phens, 0) == 0:
            phens = unique_rows
            freq = freq_vec
        else:
            phens = np.concatenate((np.kron(phens, np.ones([np.size(unique_rows, 0), 1])), np.kron(
                np.ones([np.size(phens, 0), 1]), unique_rows)), axis=1)
            freq = np.kron(freq, np.ones([np.size(freq_vec, 0), 1])) * np.kron(np.ones([np.size(freq, 0), 1]), freq_vec)
    
    return phens, freq.dot(pop_dist), antigen_list
