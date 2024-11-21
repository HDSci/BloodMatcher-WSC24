import numpy as np


class Demand:

    # TODO: Refactor to take in locations and the demand for each patient group in each location.
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
        self.dummy_demand = self._setup_dummy_demand(dummy_data, dummy_extra_demand)

    def tick(self):
        self.current_date += 1

    def _demand(self, rng=None, units=None) -> np.ndarray:
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
        if num_requests is None and self.num_requests_dist is None:
            num_requests = 1
        elif num_requests is None and self.num_requests_dist is not None:
            num_requests = self.num_requests_dist.rvs(random_state=rng)

        result = ((self._request_id_ticker + i, *self._demand(rng), self.current_date) for i in range(num_requests))
        result = tuple(result)
        self._request_id_ticker += num_requests
        self._add_to_total_demand(result)
        result = self._inject_dummy_demand(result, False)
        alloabs_mask = self._allo_antibodies(num_requests, rng)
        return result, alloabs_mask

    def _add_to_total_demand(self, demands):
        demands = np.atleast_2d(demands)
        if demands.shape[1] == 3:
            units = len(demands)
        elif demands.shape[1] == 4:
            units = demands[:, 2].sum()
        self.total_requested_units += units

    def _setup_dummy_demand(self, data, sum_demand):
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
        if self.dummy_demand is None:
            return generated_demand
        self.dummy_demand[:, 3] = self.current_date
        self.total_requested_units += self.dummy_demand[:, 2].sum() if add_to_tot_demand else 0
        result = np.vstack((np.atleast_2d(generated_demand), self.dummy_demand))
        return result

    def _allo_antibodies(self, num_requests, rng):
        freqs = self.antigens.alloantibody_freqs
        prob = rng.uniform(size=(num_requests, len(freqs)))
        allo_mask = prob < freqs
        if self.dummy_demand is not None:
            dummy_allos = np.full((len(self.dummy_demand), len(freqs)), False)
            allo_mask = np.vstack((allo_mask, dummy_allos))
        return allo_mask


if __name__ == '__main__':
    pass
