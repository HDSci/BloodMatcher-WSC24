import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

DEFAULT_SHELF_LIFE = 35


class Inventory:

    def __init__(
            self, shelf_life: int = DEFAULT_SHELF_LIFE, starting_inventory: int = 0, watched_antigens: np.ndarray = None,
            watched_phenotypes: np.ndarray = None, start_age_dist: pd.DataFrame = None) -> None:
        """Initialise the inventory.

        Inventory of blood units.

        :param int shelf_life: The shelf life of the units in the inventory, in days.
        :param int starting_inventory: The initial number of units in the inventory.
        :param ndarray watched_antigens: 1-D integer array of
        masks selecting the antigens in phenotype combinations to watch
        :param ndarray watched_phenotypes: 1-D integer array of
        phenotype combinations to watch
        :param DataFrame start_age_dist: A pandas DataFrame with
        columns 'age' and 'freq' containing the starting age distribution of the inventory.
        If provided, the inventory will be initialised with units of the ages specified
        in the 'age' column, with the frequency of each age specified in the 'freq' column.
        If not provided, the inventory will be initialised with units of age 1.
        """
        self.shelf_life = shelf_life
        # Columns: ID, antigen vector, date bled
        self.store = np.empty(shape=(0, 3), dtype=int)
        self.expired = np.empty(shape=(0, 3), dtype=int)
        self._start_date = 0
        self.current_date = self._start_date
        self.init_stock = starting_inventory
        self.stock_levels = []
        self.watched_antigens = watched_antigens
        self.watched_phenotypes = watched_phenotypes
        self.age_distribution = np.empty(
            shape=(0, self.shelf_life + 1), dtype=int)
        self._contains_phenotype_combo = None
        self.pheno_age_dist = [[] for _ in range(len(self.watched_phenotypes))]
        if start_age_dist is None:
            self.start_age_dist = False
        else:
            self.start_ages = start_age_dist.iloc[:, 0].to_numpy()
            self.start_age_freqs = start_age_dist.iloc[:, 1].to_numpy()
            self.start_age_dist = True

    def tick(self):
        """Increment the current date by one day."""
        self.current_date += 1

    def add_to_store(self, units: ArrayLike):
        """Add units to the inventory.

        :param ndarray units: rows of units to add to the inventory.
        Columns: ID, antigen vector, date bled
        """
        _units = np.atleast_2d(units)
        if _units.size <= 0:
            return
        self.store = np.vstack((self.store, _units))

    def remove_from_store(self, units: ArrayLike):
        """Remove units from the inventory.

        :param ndarray units: rows of units in the inventory to remove
        """
        units = np.atleast_2d(units)
        _not_i = np.isin(self.store[:, 0], units[:, 0],
                         assume_unique=True, invert=True)
        self.store = self.store[_not_i, :]

    def units_younger_than(self, age: int):
        """Return units younger than a given age.

        :param int age: age in days
        :return ndarray: units younger than age
        """
        i = self.store[:, 2] > self.current_date + 1 - age
        return self.store[i, :]

    def units_older_than(self, age: int):
        """Return units older than a given age.

        :param int age: age in days
        :return ndarray: units older than age
        """
        i = self.store[:, 2] < self.current_date + 1 - age
        return self.store[i, :]

    def remove_expired_units(self):
        """Remove expired units from the inventory.

        Assumes to be called at the end of the day.
        So, units at the end of their shelf life are removed
        and recorded in the expired units - `self.expired`.
        """
        expired = self.units_older_than(self.shelf_life - 1)
        if expired.size == 0:
            return
        self.expired = np.vstack((self.expired, expired))
        self.store = self.units_younger_than(self.shelf_life)

    def initialise_inventory(self, supply, rng):
        """Initialise the inventory with units from the supply.

        :param Supply supply: supply of blood units
        :param Generator rng: random number generator
        """
        stock = supply(self.init_stock, rng)
        if self.start_age_dist:
            dates = self._starting_age_distribution(rng)
            stock[:, 2] = dates
        self.add_to_store(stock)

    def warmup_clear(self):
        """Clear the expired units from the warmup period."""
        self.expired = np.empty(shape=(0, 3), dtype=int)

    def measure_stock(self):
        """Measure the stock levels and age distribution of the inventory."""
        self.measure_stock_levels()
        self.measure_age_distribution()

    def measure_stock_levels(self, supply: np.ndarray = None, watched_antigens: np.ndarray = None, watched_phenotypes: np.ndarray = None) -> None:
        """Measure stock levels of watched phenotypes.

        Measures and then stores the stock levels for strategeic antigen combinations.
        These stock levels are measured as a percentage of the total stock of the supply.

        :param ndarray supply: 1-D integer array of antigen phenotypes of the supply
        :param ndarray antigen_combinations: 1-D integer array of masks selecting the antigens in phenotype combinations to watch
        :param ndarray watched_phenotypes: 1-D integer array of phenotype combinations to watch
        """
        if watched_antigens is None:
            watched_antigens = self.watched_antigens
        if watched_phenotypes is None:
            watched_phenotypes = self.watched_phenotypes
        if supply is None:
            supply = self.store[:, 1]
        contains_phenotype_combo = supply & watched_antigens[:, None] == watched_phenotypes[:, None]
        stock_levels = contains_phenotype_combo.sum(axis=1)
        total_stock = len(supply)
        stock_levels = stock_levels / total_stock
        self.stock_levels.append(np.append(stock_levels, total_stock))
        self._contains_phenotype_combo = contains_phenotype_combo

    def measure_age_distribution(self, supply: np.ndarray = None, shelf_life: int = None, for_phenotypes=True) -> None:
        """Measure the age distribution of the inventory.

        Measures and then stores the age distribution of the inventory.
        The age distribution is measured as the number of units of each age.

        :param ndarray supply: 1-D integer array of origin dates of stored units
        :param int shelf_life: maximum shelf life of units
        """
        if supply is None:
            supply = self.current_date - self.store[:, 2] + 1
        if shelf_life is None:
            shelf_life = self.shelf_life
        age_distribution = np.bincount(supply, minlength=shelf_life+1)
        self.age_distribution = np.vstack(
            (self.age_distribution, age_distribution))

        if for_phenotypes and self._contains_phenotype_combo is not None:
            contains_phenotype_combo = self._contains_phenotype_combo
            pheno_age_dist = [np.bincount(
                supply[combo], minlength=shelf_life+1) for combo in contains_phenotype_combo]
            for combo, age_dist in zip(self.pheno_age_dist, pheno_age_dist):
                combo.append(age_dist)
        self._contains_phenotype_combo = None

    def _starting_age_distribution(self, rng: np.random.Generator) -> np.ndarray:
        """Generate the starting age distribution of the inventory.

        :param Generator rng: random number generator
        :return: 1-D integer array of origin dates of stored units
        """
        ages = rng.choice(self.start_ages, size=self.init_stock,
                          p=self.start_age_freqs)
        ages[ages > self.shelf_life] = self.shelf_life
        ages[ages < 2] = 2
        dates = self.current_date - ages + 1
        return dates

    def mean_O_type_stock(self, start: int = 1, end: int = np.inf) -> np.ndarray:
        """Return the mean stock of O type blood units.

        Measures the mean levels of both O- and O+ blood units
        as a percentage of the total stock of the supply
        over the specified period in the simulation.

        Assumes that the first two watched phenotypes are O- and O+.

        :param int start: start date of period
        :param int end: end date of period
        :return ndarray: mean stock of O type blood units
        """
        start_index = max(start - 1, 0)
        end_index = min(end, len(self.stock_levels))
        stock_levels = np.array(self.stock_levels)[start_index:end_index, :2]
        mean_stock_levels = stock_levels.mean(axis=0)
        if mean_stock_levels.size == 0:
            mean_stock_levels = np.zeros(2)
        elif mean_stock_levels.size == 1:
            mean_stock_levels = np.append(mean_stock_levels, 0)
        return mean_stock_levels
