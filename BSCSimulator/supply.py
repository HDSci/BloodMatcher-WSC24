import numpy as np


class Supply:
    """
    A class to represent the supply of units with specific antigens.

    Attributes:
    -----------
    antigens : list or None
        A list of antigens or None.
    data : DataFrame
        A DataFrame containing supply choices and their probabilities.
    num_units_rv : scipy.stats.rv_discrete or None
        A random variable distribution for the number of units.
    antigen_string : bool
        A flag indicating if antigens are represented as strings.
    current_date : int
        The current date in the simulation.
    _unit_id_ticker : int
        A ticker for generating unique unit IDs.
    num_units_dist : scipy.stats.rv_discrete or None
        A random variable distribution for the number of units.
    total_donated_units : int
        The running total number of donated units.
    _supply_choices : ndarray
        An array of supply choices.
    _supply_probabilities : ndarray
        An array of probabilities corresponding to the supply choices.

    Methods:
    --------
    tick():
        Advances the current date by one.
    _supply(rng=None, units=1) -> ndarray:
        Generates a supply of units based on the given random number generator and number of units.
    supply(num_units=None, rng=None):
        Supplies the specified number of units, generating unique IDs and timestamps for each unit.
    _add_to_total_supply(units):
        Adds the supplied units to the total donated units count.
    """

    def __init__(self, antigens=None, data=None, num_units_rv=None, antigen_string=True):
        self.antigens = antigens
        self.data = data
        self.current_date = 0
        self._unit_id_ticker = 0
        self.num_units_dist = num_units_rv
        self.antigen_string = antigen_string
        self.total_donated_units = 0
        self._supply_choices = self.data.iloc[:, 0].to_numpy()
        self._supply_probabilities = self.data.iloc[:, 1].to_numpy()

    def tick(self):
        self.current_date += 1

    def _supply(self, rng=None, units=1) -> np.ndarray:
        choices = self._supply_choices
        probabilities = self._supply_probabilities
        rng = np.random if rng is None else rng
        result = rng.choice(choices, p=probabilities, size=units)
        return np.array(result)

    def supply(self, num_units=None, rng=None):
        """
        Generate a supply of units with unique IDs and associated phenotypes.

        Parameters:
            num_units (int, optional): The number of units to supply. If None, the number of units
                                    will be determined by `self.num_units_dist` if it is not None,
                                    otherwise it defaults to 1.
            rng (Generator, optional): A random number generator instance to be used
                                                for generating random values.

        Returns:
        result (ndarray): A 2D array where each row represents a unit with columns for unit ID,
                          phenotype, and the current date.
        """
        if num_units is None and self.num_units_dist is None:
            num_units = 1
        elif num_units is None and self.num_units_dist is not None:
            num_units = self.num_units_dist.rvs(random_state=rng)

        phenos = self._supply(rng, num_units)[:, None]
        ids = np.arange(self._unit_id_ticker,
                        self._unit_id_ticker + num_units)[:, None]
        times = np.full(num_units, self.current_date, dtype=int)[:, None]
        result = np.hstack((ids, phenos, times))
        self._unit_id_ticker += num_units
        self._add_to_total_supply(result)
        return result

    def _add_to_total_supply(self, units):
        self.total_donated_units += len(units)


if __name__ == '__main__':
    pass
