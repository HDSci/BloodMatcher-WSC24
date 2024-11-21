import numexpr as ne
import numpy as np


def usability(supply, demand):
    assert len(demand.shape) == 2
    usability_norm = demand.size
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    usability = np.sum(demand_rotated >= supply, axis=(0, 2)) / usability_norm
    return usability


def phenotype_usability(supply, demand):
    assert len(demand.shape) == 2
    usability_norm = len(demand)
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    usability = np.all(demand_rotated >= supply, axis=2).sum(
        axis=0) / usability_norm
    return usability


def pop_phenotype_usability(abd_phenotypes, frequencies):
    usability = frequencies[abd_phenotypes.flatten()]
    return usability


def receivability(supply, demand):
    assert len(supply.shape) == 2
    receivability_norm = supply.size
    demand = np.atleast_2d(demand)
    supply_3d = supply[:, :, None]
    supply_rotated = np.swapaxes(supply_3d, 1, 2)
    receivability = np.sum(supply_rotated <= demand,
                           axis=(0, 2)) / receivability_norm
    return receivability


def sum_of_substitutions(supply, demand, weights=None):
    assert len(demand.shape) == 2
    weights = np.ones(demand.shape[1]) if weights is None else weights
    substitution_norm = weights.sum()
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    substitutions = ne.evaluate(
        'sum((demand_rotated > supply) * weights, axis=2)') / substitution_norm
    return substitutions


def immunogenicity(supply, demand, risk):
    assert len(demand.shape) == 2
    risk_norm = risk.sum()
    supply = np.atleast_2d(supply)
    demand_3d = demand[:, :, None]
    demand_rotated = np.swapaxes(demand_3d, 1, 2)
    immunogenicity = ne.evaluate(
        'sum((demand_rotated < supply) * risk, axis=2)') / risk_norm
    return immunogenicity


def opportunity_cost(supply, demand, normalised=True):
    Fsi = usability(supply, demand)
    Fsj = usability(demand, demand)
    Fsj = Fsj[:, None]
    oc = Fsi - Fsj
    if normalised:
        oc = oc / Fsi
    return oc


def major_antigen_substitution(supply, demand, population=False, pop_frequencies=None):
    if population:
        Fsi = pop_phenotype_usability(supply, pop_frequencies)
        Fsj = pop_phenotype_usability(demand, pop_frequencies)
    else:
        Fsi = phenotype_usability(supply, demand)
        Fsj = phenotype_usability(demand, demand)
    Fsj = Fsj[:, None]
    mas = Fsi - Fsj
    return mas


# TODO: Make calculation of age constraints separate from the core FIFO penalty
# I.e., calculating and constraining forecasted units so that they cannot be used
# earlier than they are produced should be done in a separate function.
def fifo_discount(remaining_shelf_life, max_life=35, reqs_dates=None, scd_pat=False):
    if reqs_dates is None:
        discount = _fifo_discount(remaining_shelf_life)
        # Penalise forecasted units so they cannot be used for today's requests
        discount[remaining_shelf_life > max_life] = -1e17
        discount = discount[None, :]
        if np.any(scd_pat):
            len_demand = len(scd_pat)
            discount = np.full((len_demand, discount.size), discount)
            discount[scd_pat] = 0
    else:
        _remaining_shelf_life = remaining_shelf_life[None,
                                                     :] - reqs_dates[:, None]
        discount = _fifo_discount(_remaining_shelf_life)
        allocating_to_the_past = (
            remaining_shelf_life - max_life)[None, :] > reqs_dates[:, None]
        will_expire = _remaining_shelf_life < 1
        # discount[_shelf_life > max_life] = -1e17
        discount[allocating_to_the_past] = -1e24
        discount[will_expire] = -1e17
        if np.any(scd_pat):
            discount[scd_pat] = 0
    return discount


def young_blood_penalty(unit_age, max_young_blood=14, reqs_dates=None, scd_pat=True):
    """Young blood/old blood penalties and constraints.

    Adds a constraint so that blood that is not 'young blood' cannot be used
    for SCD patients.
    Sets penalties and bonuses so that older 'young blood' is used first
    up 7 days old, then between 7 and 14 days old, the penalties exponentially increase.

    :param unit_age: age minus 1 of the unit in days.
    :param max_young_blood: maximum age of young blood in days.
    :param reqs_dates: dates of the requests.
    :param scd_pat: boolean array indicating which requests are for SCD patients.
    :return: penalty array.    
    """
    if reqs_dates is None:
        penalty = _young_blood_penalty(unit_age + 1)
        # Penalise non-young blood so it cannot be used
        penalty[unit_age + 1 > max_young_blood] = 1e17
        penalty = penalty[None, :]
        if not np.all(scd_pat):
            len_demand = len(scd_pat)
            penalty = np.full((len_demand, penalty.size), penalty)
            penalty[~scd_pat] = 0
    else:
        _unit_age = unit_age[None, :] + reqs_dates[:, None]
        penalty = _young_blood_penalty(_unit_age + 1)
        allocating_to_the_past = _unit_age < 0
        penalty[allocating_to_the_past] = 1e24
        will_be_old_blood = _unit_age + 1 > max_young_blood
        penalty[will_be_old_blood] = 1e17
        if not np.all(scd_pat):
            penalty[~scd_pat] = 0
    return penalty


def _fifo_discount(shelf_life):
    # discount = ne.evaluate('0.5 ** (shelf_life / 5)')
    max_abs = np.max(np.abs(shelf_life))
    poss_shelf_lifes = np.hstack(
        (np.arange(max_abs + 1), np.arange(-max_abs, 0)))
    shelf_life_lookups = 0.5 ** (poss_shelf_lifes / 5)
    discount = shelf_life_lookups[shelf_life.astype(int)]
    # discount = 0.5 ** (shelf_life / 5)
    return discount


def _young_blood_penalty(unit_age, a=7.686455, b=9.580724, c=0, d=1.1976):
    max_abs = np.max(np.abs(unit_age))
    poss_unit_ages = np.hstack(
        (np.arange(max_abs + 1), np.arange(-max_abs, 0)))
    unit_age_lookups = (-1/a * poss_unit_ages +
                        np.exp(poss_unit_ages - b) + c) * d
    penalty = unit_age_lookups[unit_age.astype(int)]
    return penalty
    # penalty = (-1/a * unit_age + np.exp(unit_age - b) + c) * d
    # penalty = ne.evaluate('(-1/a * shelf_life + exp(shelf_life - b) + c) * d')


if __name__ == "__main__":
    np.random.seed(20220228)
    extreme_phens = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    supply = np.vstack((extreme_phens, np.random.randint(0, 2, (13, 4))))
    demand = np.vstack((extreme_phens, np.random.randint(0, 2, (5, 4))))
    print('Supply:\n', supply)
    print('Demand:\n', demand)
    print()

    F_1 = usability(supply[0], demand)
    print(f'Usability of first supply unit: {F_1[0]}.\n')

    F_2 = usability(supply[1], demand)
    print(f'Usability of second supply unit: {F_2[0]}.\n')

    F_last = usability(supply[-1], demand)
    print(f'Usability of last supply unit: {F_last[0]}.\n')

    Fs = usability(supply, demand)
    print(f'\nUsability:\n{Fs}')

    Fd = receivability(supply, demand)
    print(f'\nReceivability:\n{Fd}')

    S = sum_of_substitutions(supply, demand)
    print(f'\nSubstitutions normalised:\n{S}')

    Imm = immunogenicity(supply, demand, np.array([2, 1, 5, 8]))
    print(f'\nImmunogenicity normalised:\n{Imm[:4, :4]}')

    Fsj = usability(demand, demand)
    print(f'\nUsability of demanded units:\n{Fsj}')

    Fsj_ex = Fsj[:, None]
    OC = (Fs - Fsj_ex) / Fs
    OC[OC < 0] = 10
    print(f'\nOpportunity Cost:\n{OC}')

    C = OC + Imm + S
    print(f'\nCost Matrix:\n{C}')
