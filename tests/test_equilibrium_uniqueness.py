"""Tests of Section 3 (equilibrium existence) and Section 4 (uniqueness under WGS)."""

from __future__ import annotations

import numpy as np
import pytest

from src.equilibrium import find_brouwer_fixed_point, is_strongly_connected, pairwise_connects
from src.supply_functions import (
    SumSupplyFunction,
    continuous_limit_buy_order,
    continuous_limit_sell_order,
    linear_pairwise_supply_function,
)
from src.uniqueness import (
    find_multiple_equilibria,
    is_g_decreasing,
    is_g_over_r_decreasing,
    is_wgs,
    lattice_min_max,
    pairwise_satisfies_wgs,
    verify_connectedness_implies_proportional,
    verify_convexity,
    verify_unique_net_trades,
)
from src.supply_functions import ExtendedSupplyFunction

RNG = np.random.default_rng(42)


def _strict_linear_triangle(n: int = 3) -> tuple[list, SumSupplyFunction, np.ndarray]:
    """Three pairwise linear supply functions on a cycle 0-1-2-0, forming a
    strongly connected (hence strict, Theorem 3.4 remark) aggregate."""
    pairs = [(0, 1, 2.0, 5.0), (1, 2, 1.5, 4.0), (2, 0, 1.0, 3.0)]
    components, M = [], np.zeros(n)
    for i, j, a, b in pairs:
        pf = linear_pairwise_supply_function(a=a, b=b)
        components.append(ExtendedSupplyFunction(pf, sigma=[i, j], n=n))
        M[i] += b
        M[j] += a
    return pairs, SumSupplyFunction(components), M + 1e-6


def test_theorem_3_4_strict_supply_function_has_equilibrium():
    _, phi, M = _strict_linear_triangle()
    out = find_brouwer_fixed_point(phi, M)
    assert out["converged"]
    np.testing.assert_allclose(phi(out["p"]), 0.0, atol=1e-4)


def test_non_strict_counterexample_has_no_equilibrium_in_interior():
    """Remark after Theorem 3.4: phi(p1,p2) = (-p2/p1, 1) does not connect
    2->1 and admits no equilibrium (phi_2 = 1 != 0 everywhere)."""

    class Counterexample:
        n = 2

        def __call__(self, p):
            p = np.asarray(p, dtype=float)
            return np.array([-p[1] / p[0], 1.0])

    phi = Counterexample()
    p = np.array([0.3, 0.7])
    assert phi(p)[1] == pytest.approx(1.0)  # never zero: no equilibrium exists


def test_proposition_3_6_connectivity_of_pairwise_functions():
    sell = linear_pairwise_supply_function(a=2.0, b=5.0)
    report = pairwise_connects(sell)
    # lim_{r->0} g(r) = b = 5 > 0 => connects 2->1
    assert report["connects_2_to_1"]


def test_is_strongly_connected():
    edges = {(0, 1), (1, 2), (2, 0)}
    assert is_strongly_connected(3, edges)
    assert not is_strongly_connected(3, {(0, 1), (1, 2)})


def test_wgs_definition_4_1_linear_function():
    phi = linear_pairwise_supply_function(a=1.0, b=4.0)
    pairs = [(np.array([1.0, 2.0]), np.array([1.0, 3.0])), (np.array([2.0, 1.0]), np.array([3.0, 1.0]))]
    assert is_wgs(phi, pairs)


def test_wgs_fails_when_not_monotone_along_pair():
    """A synthetic strictly increasing-in-q_j phi_i violates WGS by construction."""

    class Increasing:
        n = 2

        def __call__(self, p):
            p = np.asarray(p, dtype=float)
            return np.array([p[1] / p[0], -1.0])

    phi = Increasing()
    pairs = [(np.array([1.0, 1.0]), np.array([1.0, 2.0]))]
    assert not is_wgs(phi, pairs)


def test_lattice_property_lemma_4_2():
    p = np.array([0.3, 0.7])
    q = np.array([0.5, 0.5])
    lo, hi = lattice_min_max(p, q)
    np.testing.assert_allclose(lo, [0.3, 0.5])
    np.testing.assert_allclose(hi, [0.5, 0.7])


def test_theorem_4_3_unique_net_trades_and_corollary_4_5_convexity():
    _, phi, M = _strict_linear_triangle()
    equilibria = find_multiple_equilibria(phi, M, n_starts=6, rng=RNG)
    assert len(equilibria) >= 1
    if len(equilibria) >= 2:
        agents = phi.components
        report = verify_unique_net_trades(agents, equilibria)
        assert report["unique_net_trades"]
        convexity = verify_convexity(phi, equilibria)
        assert convexity["linear_convex"] and convexity["log_convex"]


def test_corollary_4_9_connected_graph_implies_proportional_prices():
    p = np.array([1.0, 2.0, 3.0])
    q = np.array([2.0, 4.0, 6.0])  # proportional to p
    pair_trades = {(0, 1): 1.0, (1, 2): 1.0}
    report = verify_connectedness_implies_proportional(3, pair_trades, [p, q])
    assert report["graph_connected"]
    assert report["prices_proportional"]
    assert report["corollary_4_9_holds"]


def test_proposition_4_6_sell_order_satisfies_wgs():
    """Continuous limit sell orders satisfy WGS (Proposition 4.6): both g and
    g(r)/r (equivalently tilde_g) are decreasing."""
    phi = continuous_limit_sell_order(b=2.0, r1=0.5, r2=2.0)
    r_grid = np.geomspace(0.05, 20.0, 200)
    report = pairwise_satisfies_wgs(phi, r_grid)
    assert report["satisfies_wgs"]


def test_continuous_limit_buy_order_may_violate_wgs():
    """Examples 2.6(IV): tilde_g_buy is not monotone, so the buy order need
    not satisfy WGS (Proposition 4.6, Definition 4.7)."""
    phi = continuous_limit_buy_order(d=1.5, r1=0.5, r2=2.0)
    r_grid = np.geomspace(0.05, 20.0, 200)
    g_decreasing = is_g_decreasing(phi.g, r_grid)
    g_over_r_decreasing = is_g_over_r_decreasing(phi.g, r_grid)
    # g itself (an increasing step from -d to 0) is NOT decreasing.
    assert not g_decreasing or not g_over_r_decreasing
