"""Tests of Section 2 (supply functions and their pairwise characterization)."""

from __future__ import annotations

import numpy as np
import pytest

from src.supply_functions import (
    ExtendedSupplyFunction,
    PairwiseSupplyFunction,
    SumSupplyFunction,
    continuous_limit_buy_g,
    continuous_limit_buy_order,
    continuous_limit_sell_g,
    continuous_limit_sell_order,
    function_maximizing_amm_new_reserves,
    function_maximizing_amm_supply,
    linear_g,
    linear_pairwise_supply_function,
    restrict_supply_function,
)

RNG = np.random.default_rng(0)


def _random_prices(n: int, size: int) -> list[np.ndarray]:
    return [RNG.uniform(0.1, 10.0, size=n) for _ in range(size)]


def test_linear_pairwise_supply_function_satisfies_definition_2_1():
    """Definition 2.1: homogeneity (0), Walras's law (1), boundedness (2)."""
    phi = linear_pairwise_supply_function(a=3.0, b=7.0)
    prices = _random_prices(2, 20)
    report = phi.verify_conditions(prices, bound=7.0)
    assert report["homogeneity"]
    assert report["walras_law"]
    assert report["bounded_above"]
    # condition (a) of Proposition 2.4: lim_{r->0} g(r) = b >= 0
    assert phi.limit_at_zero() == pytest.approx(7.0, abs=1e-4)
    # condition (b): lim_{r->inf} g(r)/r = -a <= 0
    assert phi.limit_over_r_at_infinity() == pytest.approx(-3.0, abs=1e-2)


def test_sum_of_supply_functions_is_a_supply_function_proposition_2_2():
    phi1 = linear_pairwise_supply_function(a=1.0, b=4.0)
    phi2 = linear_pairwise_supply_function(a=2.0, b=3.0)
    phi_sum = SumSupplyFunction([phi1, phi2])
    prices = _random_prices(2, 20)
    report = phi_sum.verify_conditions(prices, bound=7.0)
    assert report["homogeneity"] and report["walras_law"] and report["bounded_above"]
    p = np.array([2.0, 5.0])
    np.testing.assert_allclose(phi_sum(p), phi1(p) + phi2(p))


def test_pairwise_characterization_equation_1():
    """phi_1(p) = g(p2/p1), phi_2(p) = -(p1/p2) g(p2/p1) (Proposition 2.4)."""
    g = linear_g(a=2.0, b=5.0)
    phi = PairwiseSupplyFunction(g=g)
    p = np.array([3.0, 4.0])
    r = p[1] / p[0]
    expected_phi1 = g(r)
    expected_phi2 = -(p[0] / p[1]) * expected_phi1
    out = phi(p)
    assert out[0] == pytest.approx(expected_phi1)
    assert out[1] == pytest.approx(expected_phi2)


def test_continuous_limit_sell_order_satisfies_supply_function_conditions():
    """Examples 2.6(IV): the continuous limit sell order is piecewise linear
    and bounded, and satisfies Definition 2.1."""
    phi = continuous_limit_sell_order(b=2.0, r1=0.5, r2=2.0)
    g = continuous_limit_sell_g(2.0, 0.5, 2.0)
    assert g(np.array([0.1]))[0] == pytest.approx(2.0)
    assert g(np.array([5.0]))[0] == pytest.approx(0.0)
    mid = g(np.array([1.25]))[0]
    assert 0.0 < mid < 2.0
    prices = _random_prices(2, 20)
    report = phi.verify_conditions(prices, bound=2.0)
    assert report["homogeneity"] and report["walras_law"] and report["bounded_above"]


def test_continuous_limit_buy_order_may_violate_wgs_but_is_still_a_supply_function():
    """Examples 2.6(IV): the buy order is a valid supply function (its
    boundedness and Walras's law hold); its WGS status is checked in
    test_uniqueness.py."""
    phi = continuous_limit_buy_order(d=1.5, r1=0.5, r2=2.0)
    g = continuous_limit_buy_g(1.5, 0.5, 2.0)
    assert g(np.array([0.1]))[0] == pytest.approx(0.0)
    assert g(np.array([5.0]))[0] == pytest.approx(-1.5)
    prices = _random_prices(2, 20)
    report = phi.verify_conditions(prices, bound=0.0)
    assert report["homogeneity"] and report["walras_law"]


def test_conjugate_function_is_involutive_definition_2_5():
    """tilde_tilde_g = g (the conjugate tilde_g_{a,b} = g_{b,a})."""
    phi = linear_pairwise_supply_function(a=3.0, b=7.0)
    r = np.geomspace(0.01, 100.0, 50)
    g_vals = phi.g(r)
    g_double_tilde = phi.conjugate().conjugate().g(r)
    np.testing.assert_allclose(g_double_tilde, g_vals, atol=1e-8)
    # tilde_g_{a,b} = g_{b,a}: tilde_g(r) should equal linear_g(b, a)(r).
    g_ba = linear_g(a=7.0, b=3.0)
    np.testing.assert_allclose(phi.conjugate().g(r), g_ba(r), atol=1e-8)


def test_function_maximizing_amm_examples_2_8():
    """The function-maximizing AMM's supply function is linear g_{y/2, x/2},
    and its optimal reserves u, v solve the stated first-order conditions."""
    x, y = 100.0, 50.0
    phi = function_maximizing_amm_supply(x, y)
    r = np.geomspace(0.01, 100.0, 20)
    expected = linear_g(a=y / 2.0, b=x / 2.0)(r)
    np.testing.assert_allclose(phi.g(r), expected)

    p = np.array([2.0, 3.0])
    u, v = function_maximizing_amm_new_reserves(x, y, p)
    # Budget balance: p1(x-u) + p2(y-v) = 0.
    assert p[0] * (x - u) + p[1] * (y - v) == pytest.approx(0.0, abs=1e-8)
    # u, v maximize uv given the budget line (first-order condition p1 u = p2 v).
    assert p[0] * u == pytest.approx(p[1] * v, abs=1e-6)


def test_extension_to_n_tokens_lemma_2_11_and_restriction_proposition_2_10():
    """A pairwise supply function on tokens (0, 2) extended to n=4 tokens is
    zero on the other coordinates, and restricting it back recovers the
    original pairwise function (Proposition 2.10)."""
    base = linear_pairwise_supply_function(a=1.0, b=5.0)
    ext = ExtendedSupplyFunction(base, sigma=[0, 2], n=4)
    assert ext.is_supported_on() == {0, 2}
    p = np.array([2.0, 3.0, 4.0, 1.0])
    out = ext(p)
    assert out[1] == pytest.approx(0.0)
    assert out[3] == pytest.approx(0.0)
    expected_sub = base(p[[0, 2]])
    assert out[0] == pytest.approx(expected_sub[0])
    assert out[2] == pytest.approx(expected_sub[1])

    restricted = restrict_supply_function(ext, index_set=[0, 2])
    np.testing.assert_allclose(restricted(p[[0, 2]]), expected_sub, atol=1e-8)


def test_invalid_price_vector_raises():
    phi = linear_pairwise_supply_function(a=1.0, b=1.0)
    with pytest.raises(ValueError):
        phi(np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        phi(np.array([0.0, 1.0]))


def test_linear_g_requires_nonnegative_parameters():
    with pytest.raises(ValueError):
        linear_g(a=-1.0, b=1.0)
