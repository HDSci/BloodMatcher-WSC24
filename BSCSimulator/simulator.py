import os
import time
import traceback
from typing import List

import numpy as np
from multiprocess import Pool
from tqdm import tqdm


class SimulationManager:

    def __init__(self, antigens, demand, supply, matching, inventory, warm_up, horizon, cool_down,
                 replications: int = 100, seed=0xBADBEEF, precomputed_infolder=None, future_demand=False,
                 future_supply=False, forecasting=None, precompute_outfolder=None) -> None:
        self.n = replications
        self.antigens = antigens
        self.demand = demand
        self.supply = supply
        self.matching = matching
        self.inventory = inventory
        self.warm_up = warm_up
        self.horizon = horizon
        self.cool_down = cool_down
        self.simulations: List[Simulation] = []
        self.seed = seed
        self.seeds = []
        self.failures = {'num': 0, 'seeds': []}
        self.precompute_infolder = precomputed_infolder
        self.forecast_demand = future_demand
        self.forecast_supply = future_supply
        self.forecasting = forecasting
        self.precompute_outfolder = precompute_outfolder

    def do_simulations(self):
        _seed = self.seed
        for i in tqdm(range(self.n)):
            if i != 0:
                _seed = _seed + 17
            self.seeds.append(_seed)
            rng = np.random.default_rng(_seed)
            # TODO: Set up Location or have it set up Supply & Demand.
            clocks = [self.demand(), self.supply(),
                      self.matching(), self.inventory()]
            timing = [self.warm_up, self.horizon, self.cool_down]
            rngs = {k: np.random.default_rng(_seed)
                    for k in ['supply', 'demand']}
            sim = Simulation(self.antigens, *clocks, *timing, clocks, rng, self.forecast_demand,
                             self.forecast_supply, self.forecasting, rngs=rngs)
            sim.computed_vars_file = None if self.precompute_infolder is None else os.path.join(
                self.precompute_infolder, f'{i:05d}_{_seed:#X}_randvars.npz')
            sim.simulate()
            self.simulations.append(sim)
        return self.simulations

    def do_simulations_parallel(self, workers):
        _seed = self.seed
        _to_simulate = []
        for i in range(self.n):
            if i != 0:
                _seed = _seed + 17
            self.seeds.append(_seed)
            rng = np.random.default_rng(_seed)
            # TODO: Set up Location or have it set up Supply & Demand.
            clocks = [self.demand(), self.supply(),
                      self.matching(), self.inventory()]
            timing = [self.warm_up, self.horizon, self.cool_down]
            rngs = {k: np.random.default_rng(_seed)
                    for k in ['supply', 'demand']}
            sim = Simulation(self.antigens, *clocks, *timing, clocks, rng, self.forecast_demand,
                             self.forecast_supply, self.forecasting, rngs=rngs)
            sim.computed_vars_file = None if self.precompute_infolder is None else os.path.join(
                self.precompute_infolder, f'{i:05d}_{_seed:#X}_randvars.npz')
            _to_simulate.append(sim)

        p = Pool(workers)
        _simulated = list(tqdm(p.imap(simulate, _to_simulate,
                          chunksize=min(30, self.n // workers)), total=self.n))
        p.close()
        p.terminate()
        self.simulations = _simulated

        return self.simulations

    def _pre_compute(self, i, seed):
        rng = np.random.default_rng(seed)
        # TODO: Set up Location or have it set up Supply & Demand.
        clocks = [self.demand(), self.supply(),
                  self.matching(), self.inventory()]
        timing = [self.warm_up, self.horizon, self.cool_down]
        rngs = {k: np.random.default_rng(seed) for k in ['supply', 'demand']}
        sim = Simulation(self.antigens, *clocks, *
                         timing, clocks, rng, rngs=rngs)
        sim.pre_compute_random_vars()
        time.sleep(0.001)

        filename = f"{i:05d}_{seed:#X}_randvars"
        filename = os.path.join(self.precompute_outfolder, filename)
        np.savez_compressed(filename, **sim.precomputed_vars)
        del sim
        return filename

    def do_precompute(self, workers=1):
        _seed = self.seed
        args = []
        if workers > 1:
            for i in range(self.n):
                if i != 0:
                    _seed = _seed + 17
                self.seeds.append(_seed)
                args.append((i, _seed))

            p = Pool(workers)
            outs = p.starmap(self._pre_compute, args, self.n // workers)
            p.close()
            p.terminate()
        else:
            for i in tqdm(range(self.n)):
                if i != 0:
                    _seed = _seed + 17
                self.seeds.append(_seed)
                outs = self._pre_compute(i, _seed)
                # rng = np.random.default_rng(_seed)
                # clocks = [self.demand(), self.supply(), self.matching(), self.inventory()]
                # timing = [self.warm_up, self.horizon, self.cool_down]
                # sim = Simulation(self.antigens, *clocks, *timing, clocks, rng)
                # sim.pre_compute_random_vars()
                # self.simulations.append(sim)
        return self.simulations

    def statistics(self):
        mismatch = []
        allo = []
        subs = []
        scd_shorts = []
        stocks = []
        abo_cm = []
        abod_mm = []
        fails = []
        objs = []
        ages = []
        phen_ages = []
        scd_ages = []
        all_shorts = []
        pats_subs = []
        comp_times = []
        for i, sim in enumerate(self.simulations):
            if sim.failed:
                fails.append(self.seeds[i])
                continue
            sim.final_statistics()
            mismatch.append(sim.stats['mismatches'])
            allo.append(sim.stats['cum_allo'])
            subs.append(sim.stats['substitutions'])
            scd_shorts.append(sim.stats['scd_shortages'])
            stocks.append(sim.stats['stocks'])
            abo_cm.append(sim.stats['abo_cm'])
            abod_mm.append(sim.stats['abod_mm'])
            pats_subs.append(sim.stats['pats_mm_counts'])
            objs.append(
                (sim.stats['cum_allo'].sum(),
                 sim.stats['scd_shortages'],
                 sim.stats['expiries'],
                 sim.stats['all_shortages'],
                 sim.stats['o_type_stocks'][0],     # O-
                 sim.stats['o_type_stocks'][1],     # O+
                 sim.stats['o_type_stocks'][2],     # O- plus O+
                 sim.stats['pats_mm_counts'][0],    # D
                 sim.stats['pats_mm_counts'][1],    # ABO
                 sim.stats['pats_mm_counts'][2],    # ABOD
                 )
            )
            ages.append(sim.stats['stocks_age'])
            phen_ages.append(sim.stats['stocks_pheno_age'])
            scd_ages.append(sim.stats['scd_unit_ages'])
            all_shorts.append(sim.stats['all_shortages'])
            comp_times.append(sim.computation_time)
        n = len(mismatch)
        self.mismatches = np.mean(mismatch, axis=0), np.std(
            mismatch, axis=0) / np.sqrt(n)
        self.allo = np.mean(allo, axis=0), np.std(allo, axis=0) / np.sqrt(n)
        self.subs = np.mean(subs, axis=0), np.std(subs, axis=0) / np.sqrt(n)
        self.scd_shorts = np.mean(scd_shorts), np.std(scd_shorts) / np.sqrt(n)
        self.stocks = np.mean(stocks, axis=0), np.std(
            stocks, axis=0) / np.sqrt(n)
        self.abo_cm = np.mean(abo_cm, axis=0), np.std(
            abo_cm, axis=0) / np.sqrt(n)
        self.abod_mm = np.mean(abod_mm, axis=0), np.std(
            abod_mm, axis=0) / np.sqrt(n)
        self.pats_subs = np.mean(pats_subs, axis=0), np.std(
            pats_subs, axis=0) / np.sqrt(n)
        self.objs = np.array(objs)
        self.ages = np.mean(ages, axis=0), np.std(ages, axis=0) / np.sqrt(n)
        self.phen_ages = np.mean(phen_ages, axis=0), np.std(
            phen_ages, axis=0) / np.sqrt(n)
        self.scd_ages = np.mean(scd_ages, axis=0), np.std(
            scd_ages, axis=0) / np.sqrt(n)
        self.all_shorts = np.mean(all_shorts), np.std(all_shorts) / np.sqrt(n)
        self.computation_times = np.array(comp_times)
        self.failures.update({'num': self.n - n, 'seeds': fails})


class Simulation:

    def __init__(self, antigens, demand, supply, matching, inventory, warm_up, horizon, cool_down,
                 clocks=None, rng=None, forecast_demand=False, forecast_supply=False, forecasting=dict(),
                 rngs=dict()) -> None:
        self.clocks = clocks if clocks is not None else [
            demand, supply, matching, inventory]
        self.warm_up = warm_up
        self.horizon = horizon
        self.cool_down = cool_down
        self.rng = rng if rng is not None else np.random.default_rng()
        self.antigens = antigens
        self.demand = demand
        self.supply = supply
        self.matching = matching
        self.inventory = inventory
        self.time = 0
        self.stats = None
        self.failed = False
        self.precomputed_vars = None
        self.computed_vars_file = None
        self._loaded_vars = None
        self._loaded_vars_strides = None
        self._forecast_demand = forecast_demand
        self._forecast_supply = forecast_supply
        forecasting = forecasting if isinstance(forecasting, dict) else dict()
        self.forecast_supply_days = forecasting.get('units_days', 1)
        self.forecast_supply_shows = forecasting.get('units_shows', 1)
        self.forecast_demand_days = forecasting.get('requests_days', 1)
        self.forecast_demand_shows = forecasting.get('requests_shows', 1)
        self._forecast_rng = np.random.default_rng(20220228)
        self.rngs = rngs
        self.computation_time = 0

    def simulate(self):
        start_comp_time = time.time()
        try:
            self._open_loaded_vars()
            while self.time < self.horizon:
                self.tick()
                if self.time == 1:
                    self.inventory.initialise_inventory(
                        self.supply.supply, self.rngs.get('supply', self.rng))
                donated_units = self.supply.supply(
                    rng=self.rngs.get('supply', self.rng))
                self.inventory.add_to_store(donated_units)
                self.inventory.measure_stock()

                # TODO: This interface can stay the same in the line below
                # But the returned array `new_requests` array should have
                # two extra columns: one for location and one for patient type
                # The same applies to `donated_units` array a few lines above
                new_requests, abs_mask = self.demand.demand(
                    rng=self.rngs.get('demand', self.rng))
                self.matching.receive_new_requests(new_requests, abs_mask)
                self.matching.get_inventory(self.inventory)
                forecasts = self.get_n_days_forecast()
                forecasts = self._randomise_forecast(forecasts)
                if forecasts is not None:
                    self.matching.get_forecasts(*forecasts)
                self.matching.matching_algorithm()
                self.matching.update_matches()
                self.matching.push_update_to_inventory(self.inventory)
                self.matching.remove_matched_requests()
                self.matching.clear_forecasts()

                # np.savez('scratch/manual_tests/forecasting/last_day_forecast.npz',
                #          donated_units=donated_units, new_requests=new_requests, abs_mask=abs_mask)
                self.matching.track_unmatched_requests()
                self.inventory.remove_expired_units()
                if self.warm_up == self.time:
                    self.demand.total_requested_units = 0
                    self.matching.warmup_clear()
                    self.inventory.warmup_clear()
        except RuntimeError as e:
            self.failed = True
            traceback.print_exception(e)
        finally:
            self._close_loaded_vars()
        self.computation_time = time.time() - start_comp_time

    def tick(self):
        self.time += 1
        for clock in self.clocks:
            clock.tick()

    def final_statistics(self):
        """
        Uses historical matches to measure mismatches, alloimmunisations, and substitutions.
        Assigns the results to `self.stats` dictionary.

        Full list of statistics:
        - mismatches: total number of mismatches per antigen
        - cum_allo: cumulative expected alloimmunisation per antigen
        - substitutions: mean number of substitutions per antigen
        - scd_shortages: number of shortages for SCD patients
        - stocks: inventory levels for major blood groups and watched phenotypes
        - abo_cm: number of units from each major blood group (ABOD) given to each major blood group
        - expiries: number of units expired
        - stocks_age: age distribution of the inventory over the simulation
        - stocks_pheno_age: age distribution of the inventory over the simulation for each watched phenotype
        - scd_unit_ages: age distribution of units given to SCD patients
        - all_shortages: number of shortages for all patients (SCD + dummy demand)
        - o_type_stocks: inventory levels for O-, O+, and O- plus O+
        - abod_mm: number of units from each major blood group (ABOD) given to each major blood group (SCD only)
        - pats_mm_counts: number of patients that received at least one D/ABO/ABOD substitution
        """
        matched_demand = self.antigens.convert_to_binarray(
            self.matching.matches[:, 3])
        matched_supply = self.antigens.convert_to_binarray(
            self.matching.matches[:, 4])
        mismatches = self.matching.measure_mismatches(
            matched_demand, matched_supply)
        cum_allo = self.matching.measure_cumulative_alloimmunisation(
            mismatches)
        substitutions = self.matching.measure_substitutions(
            matched_demand, matched_supply)
        scd_shortages = self.demand.total_requested_units - self.matching.num_matches
        all_shortages = self.matching.all_shortages
        stocks = np.array(self.inventory.stock_levels)
        o_type_stocks = self.inventory.mean_O_type_stock(
            start=self.warm_up, end=self.horizon - self.cool_down)
        o_type_stocks = np.append(o_type_stocks, o_type_stocks.sum())
        stocks_age = self.inventory.age_distribution
        stocks_pheno_age = [np.array(dist)
                            for dist in self.inventory.pheno_age_dist]
        expiries = self.inventory.expired.shape[0]
        abo_cm = self.matching.abo_cm_counts
        scd_unit_ages = self.matching.ages_given_to_scd
        abod_mm = self.matching.abod_mm_counts
        pats_mm_counts = [self.matching.d_mm_pat_counts,
                          self.matching.abo_mm_pat_counts,
                          self.matching.abod_mm_pat_counts]
        self.stats = dict(mismatches=mismatches.sum(axis=0), cum_allo=cum_allo,
                          substitutions=substitutions.mean(axis=0),
                          scd_shortages=scd_shortages,
                          stocks=stocks, abo_cm=abo_cm,
                          expiries=expiries, stocks_age=stocks_age,
                          stocks_pheno_age=stocks_pheno_age,
                          scd_unit_ages=scd_unit_ages,
                          all_shortages=all_shortages, o_type_stocks=o_type_stocks,
                          abod_mm=abod_mm, pats_mm_counts=pats_mm_counts)

    def pre_compute_random_vars(self):
        starting_inventory = None
        units = []
        requests = []
        requests_Abs = []
        strides = []
        while self.time < self.horizon:
            self.tick()
            if self.time == 1:
                self.inventory.initialise_inventory(
                    self.supply.supply, self.rngs.get('supply', self.rng))
                starting_inventory = self.inventory.store
            donated_units = self.supply.supply(
                rng=self.rngs.get('supply', self.rng))
            units.append(donated_units)
            new_requests, abs_mask = self.demand.demand(
                rng=self.rngs.get('demand', self.rng))
            requests.append(new_requests)
            requests_Abs.append(abs_mask)
            strides.append((len(donated_units), len(new_requests)))
        self.precomputed_vars = {'start_inventory': np.vstack(starting_inventory), 'units': np.vstack(units),
                                 'requests': np.vstack(requests), 'requests_Abs': np.vstack(requests_Abs),
                                 'strides': np.vstack(strides)}

    def _open_loaded_vars(self):
        if self.computed_vars_file is None:
            return
        temp = np.load(self.computed_vars_file)
        if temp['units'].shape[0] < 1.3e6:
            self._loaded_vars = {k: v.copy() for k, v in temp.items()}
            temp.close()
        else:
            self._loaded_vars = temp
        self._loaded_vars_strides = self._loaded_vars['strides'].cumsum(axis=0)

    def _close_loaded_vars(self):
        if self._loaded_vars is not None:
            del self._loaded_vars_strides
            try:
                self._loaded_vars.close()
            except AttributeError:
                pass
            finally:
                self._loaded_vars = None

    def get_n_days_forecast(self):
        if self._loaded_vars is None or self.time >= self.horizon:
            return
        n_units = self.forecast_supply_days
        n_requests = self.forecast_demand_days

        start_units = self.time - 1
        end_units = min(self.horizon - 1, self.time - 1 + n_units)
        slice_start_units = self._loaded_vars_strides[start_units]
        slice_end_units = self._loaded_vars_strides[end_units]
        units = self._loaded_vars['units'][slice_start_units[0]:slice_end_units[0], :]

        start_requests = self.time - 1
        end_requests = min(self.horizon - 1, self.time - 1 + n_requests)
        slice_start_requests = self._loaded_vars_strides[start_requests]
        slice_end_requests = self._loaded_vars_strides[end_requests]
        requests = self._loaded_vars['requests'][slice_start_requests[1]:slice_end_requests[1], :]
        requests_Abs = self._loaded_vars['requests_Abs'][slice_start_requests[1]:slice_end_requests[1], :]
        return units, (requests, requests_Abs)

    def _randomise_forecast(self, forecast):
        if forecast is None:
            return forecast

        units, (requests, requests_Abs) = forecast
        u_shows = min(1, self.forecast_supply_shows)
        if u_shows < 1:
            tot = int(len(units) / u_shows)
            u_no_shows = tot - len(units)
            random_extra_units = self._forecast_rng.choice(
                self._loaded_vars['units'][:, 1], u_no_shows, replace=True)
            dates = np.arange(np.min(units[:, -1]), np.max(units[:, -1]) + 1)
            dates = np.repeat(dates, int(
                len(random_extra_units) / len(dates) + 1))[:len(random_extra_units)]
            ids = np.arange(
                np.max(units[:, 0]) + 1, np.max(units[:, 0]) + 1 + len(random_extra_units))
            random_extra_units = np.hstack(
                (ids[:, None], random_extra_units[:, None], dates[:, None]))
            show_units = np.vstack((units, random_extra_units))
        else:
            show_units = units
        show_reqs = requests
        show_Abs = requests_Abs

        if not self._forecast_supply:
            show_units = np.empty((0, units.shape[1]), dtype=units.dtype)
        if not self._forecast_demand:
            show_reqs = np.empty((0, requests.shape[1]), dtype=requests.dtype)
            show_Abs = np.empty(
                (0, requests_Abs.shape[1]), dtype=requests_Abs.dtype)
        return show_units, (show_reqs, show_Abs)


def simulate(sim: Simulation) -> Simulation:
    """Simulate a single replication of the simulation.

    Wrapper function for the `simulate` method of the `Simulation` class
    that allows for parallelisation.
    Inserts a 0.2 second pause before executing the simulation.

    :param Simulation sim: simulation to run
    return Simulation: the same simulation object
    """
    time.sleep(0.2)
    sim.simulate()
    return sim