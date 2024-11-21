"""Classes for matching platform/area and algorithms."""


import warnings

import numpy as np
import ot

from .antigen import Antigens, bnot, not_compatible
from .metrics import fifo_discount, immunogenicity, major_antigen_substitution, sum_of_substitutions, young_blood_penalty
from .mincostflow import mincostflow, maxflow_mincost
from .util import list_of_permutations


class MatchingArea:

    def __init__(
            self, algo=None, antigens: Antigens = None, matching_rule='ABOD', anticipation=False, cost_weights=None,
            solver='POT', young_blood_constraint=True, substitution_penalty_parity=True) -> None:
        self.current_date = 0
        # ID demand, supply, date, type demand, supply
        self.matches = np.empty((0, 5), dtype=int)
        self.pending_requests = np.empty(
            shape=(0, 5), dtype=int)  # 5th column = matched or not
        self.copy_of_inventory = None
        self.matching_algo = algo
        self.matching_rule = matching_rule
        self.antigens = antigens
        self._todays_matches = None
        self.matches_costs = None
        self.immediately_unmet_requests = np.empty(shape=(0, 6), dtype=int)
        self.num_matches = 0
        self.abo_cm_combos = None
        self.abo_cm_counts = 0
        self.scd_shortages = 0
        self.all_shortages = 0
        self.pr_allo_abs = np.empty(
            shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        self.anticipation = anticipation
        self.forecast_units = np.empty(shape=(0, 3), dtype=int)
        self.forecast_requests = np.empty(
            shape=(0, 5), dtype=int)  # 5th column = matched or not
        self.fr_allo_abs = np.empty(
            shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        if cost_weights is None:
            cost_weights = np.array([1, 1, 1, 1, 1])
        elif np.abs(cost_weights).sum() == 0:
            cost_weights = np.array([1, 1, 1, 1, 1])
        if substitution_penalty_parity:
            # Equal weight for major antigen substitution and sum of substitutions
            cost_weights[2] = cost_weights[1]
        _cost_weights = cost_weights / np.abs(cost_weights).sum()
        self.transport_matching_weights = _cost_weights
        self.ages_given_to_scd = np.empty(shape=(0, 35+1), dtype=int)
        self.solver = solver
        self.yb_constraint = young_blood_constraint
        self.abod_mm_combos = None
        self.abod_mm_counts = 0
        self.abo_mm_combos = None
        self.abo_mm_counts = 0
        self.d_mm_combos = None
        self.d_mm_counts = 0
        self.abod_mm_pat_counts = 0
        self.abo_mm_pat_counts = 0
        self.d_mm_pat_counts = 0

    def tick(self):
        self.current_date += 1

    def track_unmatched_requests(self):
        if self.pending_requests.size == 0:
            return
        time = np.full(len(self.pending_requests), self.current_date)[:, None]
        unmet_requests = np.hstack((self.pending_requests, time))
        self.immediately_unmet_requests = np.vstack(
            (self.immediately_unmet_requests, unmet_requests))

    def remove_matched_requests(self):
        # TODO: Parameterize whether partially satisfied requests are removed.
        # Currently they are removed if at least a unit of blood supplied
        i = self.pending_requests[:, 4] < 0
        self.pending_requests = self.pending_requests[i]
        self.pr_allo_abs = self.pr_allo_abs[i]

    def get_inventory(self, inventory):
        store = inventory.store.copy()
        self.copy_of_inventory = store

    def receive_new_requests(self, demand, abs_mask=None):
        demand = np.atleast_2d(demand)
        if demand.size <= 0:
            return
        if demand.shape[1] < self.pending_requests.shape[1]:
            padding_shape = (
                demand.shape[0], self.pending_requests.shape[1] - demand.shape[1])
            padding = np.zeros(padding_shape, dtype=int)
            demand = np.hstack((demand, padding))
        self.pending_requests = np.vstack((self.pending_requests, demand))
        if abs_mask is None:
            abs_mask = np.full((len(demand), self.pr_allo_abs.shape[1]), False)
        self.pr_allo_abs = np.vstack((self.pr_allo_abs, abs_mask))

    def get_forecasts(self, units, requests):
        self.forecast_units = units.copy()
        # TODO: This needs to be handled in a better way
        padding_shape = (requests[0].shape[0], 1)
        padding = np.zeros(padding_shape, dtype=int)
        self.forecast_requests = np.hstack((requests[0], padding))
        self.fr_allo_abs = requests[1].copy()

    def matching_algorithm(self):
        if self.matching_algo is None or self.matching_algo == 'default':
            return self.default_matching()
        elif self.matching_algo == 'transport':
            return self.transport_matching()

    def default_matching(self):
        matches = []
        free_inventory = np.full(len(self.copy_of_inventory), True)
        inv_index = np.arange(len(free_inventory))
        for i, request in enumerate(self.pending_requests):
            antigen_compatibility = bnot(
                request[1], self.antigens.mask) & self.copy_of_inventory[:, 1]
            # Minus 3 for ABD antigens
            compatibility = antigen_compatibility < 2 ** (
                self.antigens.vector_length - 3)
            avail_and_compat = compatibility & free_inventory
            if not avail_and_compat.any():
                continue
            num_matched_units = avail_and_compat.sum()
            self.pending_requests[i, 4] = num_matched_units
            ind = inv_index[avail_and_compat][:num_matched_units]
            matched_units = self.copy_of_inventory[avail_and_compat][:num_matched_units]
            free_inventory[ind] = False
            matches.extend(
                [[request[0], matched_unit[0], self.current_date, request[1], matched_unit[1]] for matched_unit in
                 matched_units])
        self._todays_matches = np.array(matches)
        return matches

    # @jit(nopython=False)
    def transport_matching(self, shelf_life=35, max_young_blood=14, solver=None):
        matches = []
        if self.pending_requests.size == 0:
            self._todays_matches = np.array(matches)
            return matches

        if solver is None:
            solver = self.solver
        reqs = self.pending_requests
        reqs_ab_mask = self.pr_allo_abs
        units = self.copy_of_inventory
        num_f_units = 0
        if self.anticipation and self.forecast_units.size > 0:
            units = np.vstack((units, self.forecast_units))
            num_f_units = len(self.forecast_units)
        num_f_reqs = 0
        if self.anticipation and self.forecast_requests.size > 0:
            reqs = np.vstack((reqs, self.forecast_requests))
            reqs_ab_mask = np.vstack((reqs_ab_mask, self.fr_allo_abs))
            num_f_reqs = len(self.forecast_requests)
        reqs_phen = self.antigens.convert_to_binarray(reqs[:, 1])
        units_phen = self.antigens.convert_to_binarray(units[:, 1])
        num_units = len(units_phen)
        num_reqs = len(reqs_phen)
        units_hist = np.ones(num_units, np.int64)
        reqs_hist = reqs[:, 2].astype(np.int64)
        sum_reqs = reqs_hist.sum()
        # Alloantibodies
        reqs_abs = np.ones(
            (reqs_phen.shape[0], reqs_phen.shape[1] - 3), dtype=int)
        abs_idx = (reqs_phen[:, 3:] == 0) & reqs_ab_mask
        reqs_abs[abs_idx] = 0
        if self.matching_rule == 'ABOD':
            reqs_abo_abs_phens = np.hstack((reqs_phen[:, :3], reqs_abs))
        else:
            reqs_abo_abs_phens = np.hstack((reqs_phen[:, :8], reqs_abs[:, 5:]))
        reqs_abo_abs = self.antigens.binarray_to_int(reqs_abo_abs_phens)

        # Calculate components of cost function
        scd_patients = reqs[:, 0] >= 0
        major = self.antigens.major_mask
        minor = self.antigens.minor_mask
        if self.matching_rule != 'ABOD':
            non_rh_kell_allo_normalised = self.antigens.allo_risk[minor[3:]]
            non_rh_kell_allo_normalised[:5] = 0
            if self.matching_rule == 'Limited':
                imm = 0
            else:
                imm = immunogenicity(
                    units_phen[:, minor], reqs_phen[:, minor], non_rh_kell_allo_normalised)
            subst = sum_of_substitutions(
                units_phen[:, minor], reqs_phen[:, minor], self.antigens.allo_risk[minor[3:]])
        if self.anticipation:
            units_abod_ints = units[:, 1] >> self.antigens.vector_length - 3
            reqs_abod_ints = reqs[:, 1] >> self.antigens.vector_length - 3
            usab_diff = major_antigen_substitution(units_abod_ints, reqs_abod_ints, self.anticipation,
                                                   self.antigens.population_abd_usabilities)
            fifo = fifo_discount(shelf_life + units[:, 2] - self.current_date, shelf_life,
                                 reqs[:, 3] - self.current_date,
                                 scd_patients * self.yb_constraint)  # If YB constraint is False, then FIFO is applied to SCD patients
            old_blood = young_blood_penalty(self.current_date - units[:, 2], max_young_blood,
                                            reqs[:, 3] - self.current_date, scd_patients)
        else:
            usab_diff = major_antigen_substitution(
                units_phen[:, major], reqs_phen[:, major])
            # If YB constraint is False, then FIFO is applied to SCD patients
            fifo = fifo_discount(shelf_life + units[:, 2] - self.current_date,
                                 scd_pat=scd_patients * self.yb_constraint)
            old_blood = young_blood_penalty(
                self.current_date - units[:, 2], scd_pat=scd_patients)
        antigen_compatibility = not_compatible(
            reqs_abo_abs, units[:, 1][None, :], self.antigens.mask)
        abod_incompat_indices = antigen_compatibility > 0
        incompat_indices = usab_diff < 0
        usab_diff[incompat_indices] = 0
        abod_incompat = np.zeros(usab_diff.shape)
        abod_incompat[abod_incompat_indices] = 1e32

        # TODO: Separate FIFO penalty from FIFO constraint
        # TODO: Separate Young blood penalty from Young blood constraint
        core_fifo = np.abs(fifo) <= 1
        w = self.transport_matching_weights
        w_3_fifo = fifo * 1
        w_3_fifo[core_fifo] *= w[3]
        # w_3 = w[3] * core_fifo
        # w_3[w_3 == 0] = 1
        core_old_blood = old_blood <= young_blood_penalty(
            np.array([max_young_blood - 1]), max_young_blood)[0, 0]
        w_4_old_blood = old_blood * self.yb_constraint
        w_4_old_blood[core_old_blood] *= w[4]
        # w_4 = w[4] * core_old_blood
        # w_4[w_4 == 0] = 1

        if self.matching_rule != 'ABOD':
            cost_matrix = w[0] * imm + w[2] * subst + w[1] * \
                usab_diff + abod_incompat - w_3_fifo + w_4_old_blood
        else:
            cost_matrix = usab_diff + abod_incompat - fifo + old_blood

        # If there are no abod compatible units, a request is given something.
        # Need to force matcher to give a dummy unit in that case.
        # Do this by adding dummy units even when supply > demand
        # Let's say a buffer of 101% of demand
        if solver.lower() != 'maxflow' and solver.lower() != 'ortools-maxflow':
            buff = int(np.ceil(sum_reqs * 1.01))
            dummy_units = np.full(num_reqs, 1e16)
            cost_matrix = np.hstack((cost_matrix, dummy_units[:, None]))
            units_hist = np.append(units_hist, buff)
            sum_units = num_units + buff
            if sum_reqs > sum_units:
                dummy_units = np.full(num_reqs, 1e16)
                cost_matrix = np.hstack((cost_matrix, dummy_units[:, None]))
                units_hist = np.append(units_hist, sum_reqs - sum_units)
            elif sum_units > sum_reqs:
                dummy_reqs = np.full(num_units + 1, 0)
                cost_matrix = np.vstack((cost_matrix, dummy_reqs))
                reqs_hist = np.append(reqs_hist, sum_units - sum_reqs)
        num_t_units = num_units - num_f_units
        num_t_reqs = num_reqs - num_f_reqs

        cm = cost_matrix.copy()
        bindex_b = np.abs(cm) > 1e15
        cost_matrix[bindex_b] = np.power(cost_matrix[bindex_b], 2/5)
        bindex_s = np.abs(cost_matrix) <= 100
        cost_matrix[bindex_s] *= 1000

        if solver.lower() == 'pot':
            num_iters = [10_000_000, 20_000_000, 50_000_000, 150_000_000]
            attempts = len(num_iters)
            for att, itern in enumerate(num_iters):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            category=UserWarning,
                            message='Input histogram consists of integer.*'
                        )
                        plan = ot.emd(units_hist, reqs_hist,
                                      cost_matrix.T, numItermax=itern)
                    _plan = plan[:num_t_units, :num_t_reqs].T
                    assert not np.any(
                        _plan[abod_incompat_indices[:num_t_reqs, :num_t_units]] > 0)
                    break
                except AssertionError as e:
                    if att + 1 < attempts:
                        continue
                    else:
                        raise RuntimeError(
                            'Transport solver did not find optimum.') from e
        elif solver.lower() == 'ortools':
            # plan = self._ortools_mincostflow(units_hist, reqs_hist, cm.T, cost_matrix.T)
            plan = mincostflow(units_hist, reqs_hist, cm.T, cost_matrix.T)
            _plan = plan[:num_t_units, :num_t_reqs].T
            assert not np.any(
                _plan[abod_incompat_indices[:num_t_reqs, :num_t_units]] > 0)
        elif solver.lower() == 'ortools-maxflow' or solver.lower() == 'maxflow':
            plan = maxflow_mincost(units_hist, reqs_hist, cm.T, cost_matrix.T)
            _plan = plan[:num_t_units, :num_t_reqs].T
            assert not np.any(
                _plan[abod_incompat_indices[:num_t_reqs, :num_t_units]] > 0)
        else:
            raise ValueError(f'Unknown solver {solver}.')

        # _plan = plan[:num_units, :num_reqs].T
        i = np.arange(_plan.size).reshape(_plan.shape)
        i_match = i[_plan > 0]
        di = i_match % num_t_units
        pj = i_match // num_t_units

        times = np.full(len(di), self.current_date, np.int64)[:, None]
        reqs[:num_t_reqs, 4] = _plan.sum(axis=1)
        matches = np.hstack(
            (reqs[pj, 0:1], units[di, 0:1], times, reqs[pj, 1:2], units[di, 1:2]))
        self._todays_matches = matches

        # Measure crossmatching
        self._measure_abod_crossmatch(di, pj, units_phen, reqs_phen)

        # Measure mixed match substitutions
        self._measure_abod_mixed_match_subsititutions(
            di, pj, units_phen, reqs_phen)

        # Measure the age distribution of the units given to SCD patients
        self._measure_ages_given_to_scd(di, units, True)

        return matches

    def update_matches(self, remove_dummy_demand: bool = True) -> np.ndarray:
        """Update the matches with the matches from the current day.

        Adds the matches from the current day to the matches from previous days.
        Also updates the number of matches and shortages.

        :param bool remove_dummy_demand: Whether to remove the dummy demand from the matches.
        :return np.ndarray: The updated record of matches.
        """
        if self._todays_matches.size > 0:
            _todays_matches = self._todays_matches
            if remove_dummy_demand:
                _todays_matches = self._todays_matches[self._todays_matches[:, 0] >= 0]
            self.matches = np.vstack((self.matches, _todays_matches))
            self.num_matches += len(_todays_matches)
            self.scd_shortages += self.pending_requests[self.pending_requests[:, 0]
                                                        >= 0, 2].sum() - len(_todays_matches)
            self.all_shortages += self.pending_requests[:,
                                                        2].sum() - len(self._todays_matches)
        else:
            self.scd_shortages += self.pending_requests[self.pending_requests[:, 0] >= 0, 2].sum(
            )
            self.all_shortages += self.pending_requests[:, 2].sum()
        return self.matches

    def warmup_clear(self):
        # ID demand, supply, date, type demand, supply
        self.matches = np.empty((0, 5), dtype=int)
        self.pending_requests = np.empty(
            shape=(0, 5), dtype=int)  # 5th column = matched or not
        self.pr_allo_abs = np.empty(
            shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)
        self.immediately_unmet_requests = np.empty(shape=(0, 6), dtype=int)
        self.num_matches = 0
        self.scd_shortages = 0
        self.all_shortages = 0
        self.abod_mm_combos = None
        self.abod_mm_counts = 0
        self.abo_mm_combos = None
        self.abo_mm_counts = 0
        self.d_mm_combos = None
        self.d_mm_counts = 0
        self.abod_mm_pat_counts = 0
        self.abo_mm_pat_counts = 0
        self.d_mm_pat_counts = 0

    def clear_forecasts(self):
        self.forecast_units = np.empty(shape=(0, 3), dtype=int)
        self.forecast_requests = np.empty(
            shape=(0, 5), dtype=int)  # 5th column = matched or not
        self.fr_allo_abs = np.empty(
            shape=(0, len(self.antigens.alloantibody_freqs)), dtype=bool)

    def push_update_to_inventory(self, inventory):
        if self._todays_matches.size <= 0:
            return
        i = np.isin(
            self.copy_of_inventory[:, 0], self._todays_matches[:, 1], assume_unique=True)
        _matched_units = self.copy_of_inventory[i]
        inventory.remove_from_store(_matched_units)

    def measure_mismatches(self, demand_phens, supply_phens):
        mismatched = supply_phens > demand_phens
        return mismatched

    def measure_cumulative_alloimmunisation(self, mismatched):
        cum_alloimmunisations = mismatched[:, 3:].sum(
            axis=0) * self.antigens.allo_risk
        return cum_alloimmunisations

    def measure_substitutions(self, demand_phens, supply_phens):
        subs = supply_phens < demand_phens
        return subs

    def _measure_abod_crossmatch(self, i, j, units_phen, reqs_phen, remove_dummy_demand=True):
        if remove_dummy_demand:
            i = i[self._todays_matches[:, 0] >= 0]
            j = j[self._todays_matches[:, 0] >= 0]

        dons = units_phen[i, :3]
        pats = reqs_phen[j, :3]
        phens_joined = np.hstack((dons, pats))
        if self.abo_cm_combos is None:
            combs = np.array(list_of_permutations([(0, 1)] * 6))
            self.abo_cm_combos = combs
        else:
            combs = self.abo_cm_combos
        cm_count = np.zeros(len(combs))
        unique, counts = np.unique(phens_joined, axis=0, return_counts=True)
        for i, u in enumerate(unique):
            cm_count[(combs == u).all(axis=1)] += counts[i]
        self.abo_cm_counts += cm_count

    def _measure_abod_mixed_match_subsititutions(
            self, i: np.ndarray, j: np.ndarray, units_phen: np.ndarray, reqs_phen: np.ndarray):
        # Remove non-SCD patients
        i = i[self._todays_matches[:, 0] >= 0]
        j = j[self._todays_matches[:, 0] >= 0]

        # Measure number of times an abod/abo/d substitution was done
        dons = units_phen[i, :3]
        pats = reqs_phen[j, :3]
        # Concatenate ABOD phenotypes of donors with patients'
        abod_phens_joined = np.hstack((dons, pats))
        # Instantiate arrays that define combinations of substitutions
        if self.abod_mm_combos is None:
            abod_combs = np.array(list_of_permutations([(0, 1)] * 6))
            abo_combs = np.array(list_of_permutations([(0, 1)] * 4))
            d_combs = np.array(list_of_permutations([(0, 1)] * 2))
            self.abod_mm_combos = abod_combs
            self.abo_mm_combos = abo_combs
            self.d_mm_combos = d_combs
        else:
            abod_combs = self.abod_mm_combos
            abo_combs = self.abo_mm_combos
            d_combs = self.d_mm_combos
        # Instantiate arrays that count the number of substitutions
        abod_mm_count = np.zeros(len(abod_combs))
        abo_mm_count = np.zeros(len(abo_combs))
        d_mm_count = np.zeros(len(d_combs))
        # Count the number of substitutions for each unique combination
        unique, counts = np.unique(
            abod_phens_joined, axis=0, return_counts=True)
        for k, u in enumerate(unique):
            abod_mm_count[(abod_combs == u).all(axis=1)] += counts[k]
            abo_mm_count[(abo_combs == u[[0, 1, 3, 4]]
                          ).all(axis=1)] += counts[k]
            d_mm_count[(d_combs == u[[2, 5]]).all(axis=1)] += counts[k]
        self.abod_mm_counts += abod_mm_count
        self.abo_mm_counts += abo_mm_count
        self.d_mm_counts += d_mm_count

        # Measure how many patients received an abod/abo/d substitution
        unique_patients = np.unique(j)
        abod_mm_pat_count = 0
        abo_mm_pat_count = 0
        d_mm_pat_count = 0
        for k in unique_patients:
            pat_d_type = reqs_phen[k, 2]
            pat_abo_type = reqs_phen[k, :2].dot([2, 1])
            units_given_indices = k == j
            units_given_d_type = dons[units_given_indices, 2]
            units_given_abo_type = dons[units_given_indices, :2].dot([2, 1])
            d_mm_pat_count += np.any(units_given_d_type < pat_d_type) * 1
            abo_mm_pat_count += np.any(units_given_abo_type < pat_abo_type) * 1
            abod_mm_pat_count += np.any((units_given_d_type < pat_d_type)
                                        & (units_given_abo_type < pat_abo_type)) * 1
        self.abod_mm_pat_counts += abod_mm_pat_count
        self.abo_mm_pat_counts += abo_mm_pat_count
        self.d_mm_pat_counts += d_mm_pat_count

    def _measure_ages_given_to_scd(self, i, units, remove_dummy_demand=True):
        if remove_dummy_demand:
            i = i[self._todays_matches[:, 0] >= 0]
        ages = self.current_date - units[i, 2] + 1
        ages_hist = np.bincount(ages, minlength=35+1)
        self.ages_given_to_scd = np.vstack((self.ages_given_to_scd, ages_hist))
