import os
from typing import List, Union

import numpy as np
import pandas as pd

DEFAULT_ANTIGEN_ORDER = ('A', 'B', 'D', 'C', 'c', 'E',
                         'e', 'K', 'k', 'Fya', 'Fyb')

DEFAULT_VECTOR_LENGTH = 2 ** 5


class Antigens:
    population_frequencies = None
    population_abd_usabilities = None

    def __init__(self, antigens: List, antigen_order=None, alloimmunisation_risk=None, rule=None,
                 allo_Abs=None) -> None:
        self.antigens = antigens
        self.antigen_order = DEFAULT_ANTIGEN_ORDER if antigen_order is None else antigen_order
        self.vector_length = len(antigens)
        assert set(self.antigen_order).issubset(set(self.antigens))
        self.mask = 2 ** self.vector_length - 1
        self.reference = self._setup_antigen_dict(
            antigens, self.antigen_order, self.vector_length)
        self.antigen_index = list(self.reference.keys())
        self.allo_risk = np.ones(
            self.vector_length - 3) if alloimmunisation_risk is None else alloimmunisation_risk
        self.matching_antigens = [a for a in self.antigen_index if a in rule]
        self.major_mask = [a in ['A', 'B', 'D'] for a in self.antigen_index]
        self.minor_mask = [a not in ['A', 'B', 'D']
                           and a in rule for a in self.antigen_index]
        rh_kell = ('C', 'c', 'E', 'e', 'K')
        self.rhkell_mask = [a in rh_kell for a in self.antigen_index]
        self.alloantibody_freqs = allo_Abs
        self._setup_convert_to_binarray()

    def _setup_antigen_dict(self, antigens, antigen_order=DEFAULT_ANTIGEN_ORDER, length=DEFAULT_VECTOR_LENGTH):
        power = int(length - 1)
        ant_dict = dict()
        _antigens = list(antigen_order) + antigens
        for antigen in _antigens:
            if antigen in ant_dict:
                continue
            ant_dict.update({antigen: 2 ** power})
            power -= 1
        return ant_dict

    def _setup_convert_to_binarray(self):
        self._binarrays = self._convert_to_binarray(
            [i for i in range(2 ** self.vector_length)])

    def convert_to_int(self, antigens: List[str], base: str = 'neg') -> int:
        if base == 'neg':
            num = 0
            for antigen in antigens:
                a = antigen.strip('+')
                num += self.reference.get(a, 0)
            return num
        elif base == 'pos':
            num = 2 ** self.vector_length - 1
            for antigen in antigens:
                a = antigen.strip('-')
                num -= self.reference.get(a, 0)
            return num
        else:
            raise ValueError(
                f'Parameter base must be "neg" or "pos" only, got "{base}" instead.')

    def convert_to_binarray(self, a: Union[int, List[int]]) -> np.ndarray:
        return self._binarrays[a]

    def _convert_to_binarray(self, a):
        if isinstance(a, int):
            return np.array(binarray(a, self.vector_length))
        else:
            return np.array([binarray(i, self.vector_length) for i in a])

    def convert_to_symbols(self, a, abo=True):
        array = np.atleast_2d(self.convert_to_binarray(a))
        array = (array == 1)[:, :len(self.antigen_index)]
        antigen_index = np.array([sym + '+' for sym in self.antigen_index])
        symbols = [list(antigen_index[row]) for row in array]
        if not abo:
            return symbols
        for person in symbols:
            abod = ''
            a_or_b = False
            if 'A+' in person:
                person.remove('A+')
                abod += 'A'
                a_or_b = True
            if 'B+' in person:
                person.remove('B+')
                abod += 'B'
                a_or_b = True
            if not a_or_b:
                abod += 'O'
            if 'D+' in person:
                abod += '+'
                person.remove('D+')
            else:
                abod += '-'
            person.insert(0, abod)
        return symbols

    def convert_to_full_symbols(self, a, abo=True):
        array = np.atleast_2d(self.convert_to_binarray(a))[
            :, :len(self.antigen_index)]
        num_phens = len(array)
        ant_pos_template = [s + '+' for s in self.antigen_index]
        ant_neg_template = [s + '-' for s in self.antigen_index]
        ant_pos_template = np.vstack((ant_pos_template,) * num_phens)
        ant_neg_template = np.vstack((ant_neg_template,) * num_phens)
        symbols = np.full(array.shape, '', np.dtype(('U', 6)))
        symbols[array == 1] = ant_pos_template[array == 1]
        symbols[array == 0] = ant_neg_template[array == 0]
        if not abo:
            return symbols
        abd_symbols = np.full((array.shape[0], 3), '', np.dtype(('U', 6)))
        abd_template = np.vstack((['A', 'B'],) * num_phens)
        dpos_template = np.full((num_phens, 1), '+', np.dtype(('U', 6)))
        dneg_template = np.full((num_phens, 1), '-', np.dtype(('U', 6)))
        abd_symbols[:, :2][array[:, :2] == 1] = abd_template[array[:, :2] == 1]
        abd_symbols[:, 2:][array[:, 2:3] ==
                           1] = dpos_template[array[:, 2:3] == 1]
        abd_symbols[:, 2:][array[:, 2:3] ==
                           0] = dneg_template[array[:, 2:3] == 0]
        abd_symbols = np.char.add(np.char.add(
            abd_symbols[:, 0:1], abd_symbols[:, 1:2]), abd_symbols[:, 2:3])
        abd_symbols[abd_symbols == '+'] = 'O+'
        abd_symbols[abd_symbols == '-'] = 'O-'
        symbols = np.hstack((abd_symbols, symbols[:, 3:]))
        return symbols

    def binarray_to_int(self, phen_array):
        pows = np.arange(len(self.antigen_index) - 1, -1, -1)
        bin_pows = 2 ** pows
        ints = phen_array.dot(bin_pows[:, None])
        return ints

    @classmethod
    def read_population_frequencies(cls, input_file='BSCSimulator/population_phenotype_frequencies.tsv'):
        input_file = os.path.realpath(input_file)
        cls.population_frequencies = pd.read_csv(input_file, sep='\t')


def binarray(a: int, w: int = 8):
    b = [int(i) for i in f'{a:0{w}b}']
    return b


def count_set_bits(n: int):
    count = 0
    while (n):
        n &= (n - 1)
        count += 1
    return count


def bnot(a: int, m: int = 0):
    if m == 0:
        m = 2 ** a.bit_length() - 1
    return ~a & m


def not_compatible(a: int, b: int, m: int = 0):
    return bnot(a, m) & b


def counts_to_cumulative(counts: np.ndarray):
    mask = counts > 0
    non_zero_counts = counts[mask]
    return np.cumsum(non_zero_counts, dtype=np.int32)


def get_set_bits_indexes(a: np.ndarray, cum_counts: Union[List[int], np.ndarray]):
    a = a == 1
    antigens = np.array([np.arange(a.shape[1]) for _ in range(a.shape[0])])
    unpartitioned_antigens = antigens[a]
    partitioned_antigens = np.split(unpartitioned_antigens, cum_counts)
    return partitioned_antigens


if __name__ == "__main__":
    people_phen = [np.random.randint(0, 2 ** 7) for _ in range(5)]
    people_counts = np.array([count_set_bits(i) for i in people_phen])
    people_cum_counts = counts_to_cumulative(people_counts)
    people_bin_arrs = np.array([binarray(i, 7) for i in people_phen])
    people_bit_indices = get_set_bits_indexes(
        people_bin_arrs, people_cum_counts)
