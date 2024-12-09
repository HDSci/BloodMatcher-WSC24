import os
from typing import List, Union

import numpy as np
import pandas as pd

DEFAULT_ANTIGEN_ORDER = ('A', 'B', 'D', 'C', 'c', 'E',
                         'e', 'K', 'k', 'Fya', 'Fyb')

DEFAULT_VECTOR_LENGTH = 2 ** 5


class Antigens:
    '''
    A class to represent antigens and perform various operations related to them.
    Attributes:
        population_frequencies (pd.DataFrame): Population frequencies of antigens.
        population_abd_usabilities (pd.DataFrame): Usabilities of antigens in the population.
    
    Methods:
        
        __init__(antigens: List, antigen_order=None, alloimmunisation_risk=None, rule=None, allo_Abs=None) -> None:
            Initializes the Antigens object with the given parameters.
        _setup_antigen_dict(antigens, antigen_order=DEFAULT_ANTIGEN_ORDER, length=DEFAULT_VECTOR_LENGTH) -> dict:
        _setup_convert_to_binarray() -> None:
            Sets up the binary arrays for the antigen by converting a range of integers to their corresponding binary representations.
        convert_to_binarray(a: Union[int, List[int]]) -> np.ndarray:
            Converts an integer or a list of integers to a binary array representation.
        _convert_to_binarray(a) -> np.ndarray:
            Converts an integer or a list of integers to a binary array representation.
        binarray_to_int(phen_array) -> np.ndarray | int:
            Converts a binary array to an integer or an array of integers.
        read_population_frequencies(cls, input_file='BSCSimulator/population_phenotype_frequencies.tsv') -> None:
            Reads population frequencies from a file and stores them in the class attribute.
    '''
    population_frequencies = None
    population_abd_usabilities = None

    def __init__(self, antigens: List, antigen_order=None, alloimmunisation_risk=None, rule=None, allo_Abs=None) -> None:
        """
        Initialize an Antigen object.

        Parameters:
            antigens (List): A list of antigens.
            antigen_order (List, optional): The order of antigens. Defaults to DEFAULT_ANTIGEN_ORDER.
            alloimmunisation_risk (List, optional): The risk of alloimmunisation. Defaults to an array of ones.
            rule (List, optional): A list of rules for matching antigens.
            allo_Abs (List, optional): Frequencies of alloantibodies.

        Attributes:
            antigens (List): A list of antigens.
            antigen_order (List): The order of antigens.
            vector_length (int): The length of the antigen vector.
            mask (int): A bitmask for the antigen vector.
            reference (dict): A dictionary mapping antigens to their indices.
            antigen_index (List): A list of antigen indices.
            allo_risk (List): The risk of alloimmunisation.
            matching_antigens (List): A list of antigens that match the given rule.
            major_mask (List): A list indicating major antigens ('A', 'B', 'D').
            minor_mask (List): A list indicating minor antigens not in ['A', 'B', 'D'] but in the rule.
            rhkell_mask (List): A list indicating Rh and Kell antigens ('C', 'c', 'E', 'e', 'K').
            alloantibody_freqs (List): Frequencies of alloantibodies.
        """
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
        """
        Sets up a dictionary mapping antigens to unique power-of-two values.

        Args:
            antigens (list): A list of antigens to be included in the dictionary.
            antigen_order (list, optional): A list of antigens that defines the order of precedence.
                Defaults to DEFAULT_ANTIGEN_ORDER.
            length (int, optional): The length of the vector, which determines the highest power of two.
                Defaults to DEFAULT_VECTOR_LENGTH.

        Returns:
            out (dict): A dictionary where keys are antigens and values are unique power-of-two integers.
        """
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
        """
        Sets up the binary arrays for the antigen by converting a range of integers
        (from 0 to 2^vector_length - 1) to their corresponding binary representations.

        This method initializes the `_binarrays` attribute with the result of the
        `_convert_to_binarray` method, which takes a list of integers and converts
        each integer to its binary representation.

        The length of the binary arrays is determined by the `vector_length` attribute.
        """
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
        """
        Convert an integer or a list of integers to a binary array.
        
        Looks up the binary array representation of the input integer(s) in the precomputed
        binary arrays and returns the corresponding binary array(s).

        Parameters:
            a (int or list of int): An integer or a list of integers to be converted.

        Returns:
            binarray (ndarray): A numpy array representing the binary form of the input.
        """
        return self._binarrays[a]

    def _convert_to_binarray(self, a):
        """
        Convert an integer or a list of integers to a binary array representation.

        Parameters:
            a (int or list of int): An integer or a list of integers to be converted.

        Returns:
            binarray (ndarray): A binary array representation of the input integer(s).
        """
        if isinstance(a, int):
            return np.array(binarray(a, self.vector_length))
        else:
            return np.array([binarray(i, self.vector_length) for i in a])

    def binarray_to_int(self, phen_array) -> np.ndarray | int:
        """
        Convert a binary array to an integer or an array of integers.

        This method takes a binary array and converts it to its corresponding integer
        representation. If the input is a 2D array, it returns an array of integers.

        Args:
            phen_array (ndarray):
                A binary array where each row represents a binary number.

        Returns:
            array (ndarray or int):
                An integer or an array of integers corresponding to the binary input.
        """
        pows = np.arange(len(self.antigen_index) - 1, -1, -1)
        bin_pows = 2 ** pows
        ints = phen_array.dot(bin_pows[:, None])
        return ints

    @classmethod
    def read_population_frequencies(cls, input_file='BSCSimulator/population_phenotype_frequencies.tsv'):
        input_file = os.path.realpath(input_file)
        cls.population_frequencies = pd.read_csv(input_file, sep='\t')


def binarray(a: int, w: int = 8) -> List[int]:
    """
    Convert an integer to a binary array representation.

    Args:
        a (int): The integer to be converted.
        w (int, optional): The width of the binary representation. Defaults to 8.

    Returns:
        List[int]: A list of integers representing the binary digits of the input integer.
    """
    b = [int(i) for i in f'{a:0{w}b}']
    return b


def bnot(a: int, m: int = 0) -> int:
    """
    Perform a bitwise NOT operation on an integer with an optional mask.

    Args:
        a (int): The integer to perform the bitwise NOT operation on.
        m (int, optional): The mask to apply after the NOT operation. If not provided or set to 0,
                           the mask will be set to cover all bits of `a`.

    Returns:
        int: The result of the bitwise NOT operation, masked by `m`.
    """
    if m == 0:
        m = 2 ** a.bit_length() - 1
    return ~a & m


def not_compatible(a: int, b: int, m: int = 0) -> int:
    """
    Determines if two phenotypes (represented by integers) are not compatible.

    Args:
        a (int): The first phenotype - from the recipient.
        b (int): The second phenotype - from the donor.
        m (int, optional): The mask to apply to the first integer.
            Needs to be the largest integer that can be used in the phenotype representation.
            I.e., (2 ^ num_antigens) - 1.
            Defaults to 0.

    Returns:
        int: Zero if the phenotypes are compatible, otherwise a non-zero integer.
    """
    return bnot(a, m) & b


if __name__ == "__main__":
    pass
