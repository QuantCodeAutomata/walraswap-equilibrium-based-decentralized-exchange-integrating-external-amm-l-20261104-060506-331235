"""Experiment 3: Uniqueness Properties Under Weak Gross Substitutability (WGS).

Reproduces, numerically, Definition 4.1 (WGS), Lemma 4.2 (lattice property),
Theorem 4.3 (unique net trades across equilibria), Theorem 4.4 / Corollary
4.5 (the equilibrium set and its convexity), Proposition 4.6 / Definition
4.7 (the WGS test for pairwise supply functions g, tilde_g both decreasing)
and Corollary 4.9 (connectedness implies proportional equilibrium prices),
together with the Examples 2.6(IV) fact that continuous limit sell orders
satisfy WGS while continuous limit buy orders may not (tilde_g_buy is not
monotone).

Run: python -m exp.exp3_uniqueness
"""

from __future__ import annotations

import json

import numpy as np

from src.config import (
    EXP3_BUY_ORDER_PARAMS,
    EXP3_N_EQUILIBRIA_SEARCHES,
    EXP3_N_TOKENS,
    EXP3_SELL_ORDER_PARAMS,
    RNG,
)
from src.equilibrium import find_brouwer_fixed_point
from src.supply_functions import (
    ExtendedSupplyFunction,
    SumSupplyFunction,
    continuous_limit_buy_order,
    continuous_limit_sell_order,
    linear_pairwise_supply_function,
)
from src.uniqueness import (
    compute_u_ij,
    find_multiple_equilibria,
    is_wgs,
    lattice_min_max,
    pairwise_satisfies_wgs,
    verify_connectedness_implies_proportional,
    verify_convexity,
    verify_unique_net_trades,
)
from data.generate_synthetic import random_price_vector


def build_wgs_system(n: int, linear_params: list[tuple[float, float]]) -> tuple[SumSupplyFunction, list, np.ndarray]:
    """A connected cycle of linear pairwise orders g_{a,b}, which satisfy WGS
    since g_{a,b}(r) = -ar+b is decreasing and g_{a,b}(r)/r = -a+b/r is also
    decreasing for r>0 (Proposition 4.6)."""
    pairs = [(i, (i + 1) % n) for i in range(n)]
    agents = []
    components = []
    M = np.zeros(n)
    for (i, j), (a, b) in zip(pairs, linear_params * (len(pairs) // len(linear_params) + 1)):
        pf = linear_pairwise_supply_function(a, b)
        agents.append(ExtendedSupplyFunction(pf, sigma=[i, j], n=n))
        components.append(agents[-1])
        M[i] += b + 1e-9
        M[j] += a + 1e-9
    return SumSupplyFunction(components), agents, M


def main() -> dict:
    rng = np.random.default_rng(RNG.integers(0, 2**31))

    linear_params = [(1.0, 4.0), (2.0, 5.0), (0.8, 3.0)]
    Phi, agents, M = build_wgs_system(EXP3_N_TOKENS, linear_params)

    # Definition 4.1: WGS test on random comparable price pairs p <= q, p_i=q_i.
    price_pairs = []
    for _ in range(50):
        p = random_price_vector(EXP3_N_TOKENS, rng, low=0.5, high=2.0)
        q = p.copy()
        idx_free = rng.choice(EXP3_N_TOKENS, size=EXP3_N_TOKENS - 1, replace=False)
        q[idx_free] += rng.uniform(0.0, 2.0, size=len(idx_free))
        price_pairs.append((p, q))
    wgs_holds = is_wgs(Phi, price_pairs)

    # Lemma 4.2: lattice property (purely algebraic min/max, always well
    # defined; checked as a sanity property of the construction).
    p_sample = random_price_vector(EXP3_N_TOKENS, rng)
    q_sample = random_price_vector(EXP3_N_TOKENS, rng)
    p_and_q, p_or_q = lattice_min_max(p_sample, q_sample)
    lattice_ok = bool(np.all(p_and_q <= p_sample + 1e-12) and np.all(p_or_q >= p_sample - 1e-12))

    # Theorem 4.3 / 4.4: find multiple equilibria, check unique net trades,
    # equilibrium-set ratio bounds U_ij, and convexity (Corollary 4.5).
    equilibria = find_multiple_equilibria(Phi, M, EXP3_N_EQUILIBRIA_SEARCHES, rng)
    unique_trades = verify_unique_net_trades(agents, equilibria) if len(equilibria) >= 2 else {
        "per_agent_spread": [], "unique_net_trades": None,
    }
    u_01 = compute_u_ij(equilibria, 0, 1) if equilibria else None
    convexity = verify_convexity(Phi, equilibria) if len(equilibria) >= 2 else {
        "linear_convex": None, "log_convex": None,
    }

    # Proposition 4.6 / Definition 4.7: pairwise WGS test on continuous
    # limit sell vs buy orders (Examples 2.6(IV)).
    r_grid = np.geomspace(1e-3, 1e3, 2000)
    sb, sr1, sr2 = EXP3_SELL_ORDER_PARAMS
    pf_sell = continuous_limit_sell_order(sb, sr1, sr2)
    sell_wgs = pairwise_satisfies_wgs(pf_sell, r_grid)

    db, br1, br2 = EXP3_BUY_ORDER_PARAMS
    pf_buy = continuous_limit_buy_order(db, br1, br2)
    buy_wgs = pairwise_satisfies_wgs(pf_buy, r_grid)

    # Corollary 4.9: connectedness implies proportional equilibrium prices.
    pair_trades = {(i, (i + 1) % EXP3_N_TOKENS): 1.0 for i in range(EXP3_N_TOKENS)}  # cycle: all nonzero
    proportionality = verify_connectedness_implies_proportional(EXP3_N_TOKENS, pair_trades, equilibria)

    # Theorem 4.4 non-trivial equilibrium set: a DISCONNECTED system of two
    # independent 2-token linear WGS orders on pairs (0,1) and (2,3). Within
    # each pair the ratio p1/p0 (resp. p3/p2) is pinned by that pair's own
    # equilibrium, but the price level between the two pairs is a free
    # parameter (no order trades across them), so E_phi is 1-dimensional:
    # distinct equilibria share identical net trades (Theorem 4.3) while
    # U_01 (within a connected pair) is tight and U_02 (across pairs) is
    # unbounded, and the disconnected graph need not yield proportional
    # prices (Corollary 4.9, converse direction).
    n_disc = 4
    a01, b01 = 1.0, 4.0
    a23, b23 = 2.0, 5.0
    pf01 = linear_pairwise_supply_function(a01, b01)
    pf23 = linear_pairwise_supply_function(a23, b23)
    agents_disc = [
        ExtendedSupplyFunction(pf01, sigma=[0, 1], n=n_disc),
        ExtendedSupplyFunction(pf23, sigma=[2, 3], n=n_disc),
    ]
    Phi_disc = SumSupplyFunction(agents_disc)
    M_disc = np.array([b01 + 1e-9, a01 + 1e-9, b23 + 1e-9, a23 + 1e-9])

    disc_equilibria = []
    for scale in np.geomspace(0.2, 5.0, 6):
        q0 = np.array([1.0, a01 / b01, scale, scale * a23 / b23])
        q0 = q0 / q0.sum()
        out = find_brouwer_fixed_point(Phi_disc, M_disc, q0=q0)
        if out["converged"]:
            disc_equilibria.append(out["p"])
    disc_unique_trades = verify_unique_net_trades(agents_disc, disc_equilibria) if len(disc_equilibria) >= 2 else None
    u_01_disc = compute_u_ij(disc_equilibria, 0, 1) if disc_equilibria else None
    u_02_disc = compute_u_ij(disc_equilibria, 0, 2) if disc_equilibria else None
    pair_trades_disc = {(0, 1): 1.0, (2, 3): 1.0}  # no (0,2)/(1,3) edge: graph is disconnected
    proportionality_disc = verify_connectedness_implies_proportional(n_disc, pair_trades_disc, disc_equilibria)

    results = {
        "definition_4_1_wgs_holds_for_linear_cycle": wgs_holds,
        "lemma_4_2_lattice_property_well_defined": lattice_ok,
        "n_equilibria_found": len(equilibria),
        "theorem_4_3_unique_net_trades": unique_trades,
        "theorem_4_4_u_01": u_01,
        "corollary_4_5_convexity": convexity,
        "proposition_4_6_sell_order_satisfies_wgs": sell_wgs,
        "proposition_4_6_buy_order_satisfies_wgs": buy_wgs,
        "buy_order_wgs_expected_false": not buy_wgs["satisfies_wgs"],
        "corollary_4_9_connectedness_implies_proportional": proportionality,
        "theorem_4_4_disconnected_system": {
            "n_equilibria_found": len(disc_equilibria),
            "unique_net_trades": disc_unique_trades,
            "u_01_within_connected_pair": u_01_disc,
            "u_02_across_disconnected_pairs": u_02_disc,
            "proportionality_across_pairs": proportionality_disc,
        },
    }

    with open("results/exp3_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
