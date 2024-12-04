import numpy as np


class Demand:
    """
    A class to represent the demand for blood units.

    Parameters
    ----------
    antigens : object, optional
        Antigens data (default is None).
    data : pandas.DataFrame, optional
        DataFrame containing demand choices and probabilities (default is None).
    num_requests_rv : scipy.stats.rv_discrete, optional
        Random variable for the number of requests (default is None).
    num_units_rv : scipy.stats.rv_discrete, optional
        Random variable for the number of units (default is None).
    antigen_string : bool, optional
        Whether to return antigen as a string (default is True).
    dummy_data : DataFrame, optional
        DataFrame containing dummy demand data (default is None).
        I.e., non-SCD demand.
    dummy_extra_demand : int, optional
        Extra dummy demand to be added (default is 0).
    """

    def __init__(self, antigens=None, data=None, num_requests_rv=None, num_units_rv=None, antigen_string=True,
                 dummy_data=None, dummy_extra_demand=0):
        self.antigens = antigens
        self.data = data
        self.current_date = 0
        self._request_id_ticker = 0
        self.num_requests_dist = num_requests_rv
        self.num_units_dist = num_units_rv
        self.antigen_string = antigen_string
        self.total_requested_units = 0
        self._demand_choices = self.data.iloc[:, 0].to_numpy()
        self._demand_probabilities = self.data.iloc[:, 1].to_numpy()
        self.dummy_demand = self._setup_dummy_demand(
            dummy_data, dummy_extra_demand)

    def tick(self):
        """
        Increment the current date by one.
        """
        self.current_date += 1

    def _demand(self, rng=None, units=None) -> np.ndarray:
        """
        Generate a demand phenotype based on the given random number generator and units.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator (default is numpy.random).
        units : int, optional
            Number of units (default is None).

        Returns
        -------
        numpy.ndarray
            Array containing the generated demand phenotype and number of units requested.
        """
        choices = self._demand_choices
        probabilities = self._demand_probabilities
        rng = np.random if rng is None else rng
        result = rng.choice(choices, p=probabilities)
        if self.antigen_string:
            return np.array(result)
        elif units is None and self.num_units_dist is None:
            return np.array(result), 1
        elif units is None and self.num_units_dist is not None:
            return np.array(result), self.num_units_dist.rvs(random_state=rng)

    def demand(self, num_requests=None, rng=None):
        """
        Generate multiple demands.

        Parameters
        ----------
        num_requests : int, optional
            Number of requests (default is None).
        rng : numpy.random.Generator, optional
            Random number generator (default is numpy.random).

        Returns
        -------
        tuple
            Tuple containing the generated demands and alloantibodies mask.
        """
        if num_requests is None and self.num_requests_dist is None:
            num_requests = 1
        elif num_requests is None and self.num_requests_dist is not None:
            num_requests = self.num_requests_dist.rvs(random_state=rng)

        result = ((self._request_id_ticker + i, *self._demand(rng),
                  self.current_date) for i in range(num_requests))
        result = tuple(result)
        self._request_id_ticker += num_requests
        self._add_to_total_demand(result)
        result = self._inject_dummy_demand(result, False)
        alloabs_mask = self._allo_antibodies(num_requests, rng)
        return result, alloabs_mask

    def _add_to_total_demand(self, demands):
        """
        Add the generated demands to the total requested units.

        Parameters
        ----------
        demands : ndarray or list or tuple
            Array containing the generated demands.
        """
        demands = np.atleast_2d(demands)
        if demands.shape[1] == 3:
            units = len(demands)
        elif demands.shape[1] == 4:
            units = demands[:, 2].sum()
        self.total_requested_units += units

    def _setup_dummy_demand(self, data, sum_demand):
        """
        Setup dummy demand based on the given data and sum demand.

        Parameters
        ----------
        data : DataFrame
            DataFrame containing dummy demand data.
        sum_demand : int
            Total dummy demand to be added.

        Returns
        -------
        ndarray
            Array containing the dummy demand.
        """
        if data is None or sum_demand < 1:
            return None
        units = np.round(data.iloc[:, 1].to_numpy() * sum_demand).astype(int)
        if units.sum() < sum_demand:
            units[0] += sum_demand - units.sum()
        elif units.sum() > sum_demand:
            excess = units.sum() - sum_demand
            i = 1
            while excess > 0:
                if units[-i] > 1:
                    units[-i] -= 1
                    excess -= 1
                i = (i + 1) % len(units)
        phens = data.iloc[:, 0].to_numpy()
        ids = np.full((len(data), 1), -1, dtype=int)
        times = ids + 2
        result = np.hstack((ids, phens[:, None], units[:, None], times))
        return result

    def _inject_dummy_demand(self, generated_demand, add_to_tot_demand=False):
        """
        Inject dummy demand into the generated demand.

        Parameters
        ----------
        generated_demand : numpy.ndarray
            Array containing the generated demand.
        add_to_tot_demand : bool, optional
            Whether to add the dummy demand to the total requested units (default is False).

        Returns
        -------
        result: ndarray
            Array containing the combined demand.
        """
        if self.dummy_demand is None:
            return generated_demand
        self.dummy_demand[:, 3] = self.current_date
        self.total_requested_units += self.dummy_demand[:, 2].sum() if add_to_tot_demand else 0
        result = np.vstack(
            (np.atleast_2d(generated_demand), self.dummy_demand))
        return result

    def _allo_antibodies(self, num_requests, rng):
        """
        Generate alloantibodies mask based on the given number of requests and random number generator.

        Parameters
        ----------
        num_requests : int
            Number of requests.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        allo_mask: ndarray
            Array containing the alloantibodies mask.
        """
        freqs = self.antigens.alloantibody_freqs
        prob = rng.uniform(size=(num_requests, len(freqs)))
        allo_mask = prob < freqs
        if self.dummy_demand is not None:
            dummy_allos = np.full((len(self.dummy_demand), len(freqs)), False)
            allo_mask = np.vstack((allo_mask, dummy_allos))
        return allo_mask


if __name__ == '__main__':
    pass
