import numpy as np


class Supply:

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
