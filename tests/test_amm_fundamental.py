"""Tests of Section 5 (external AMMs) and Section 6 (the Fundamental Theorem)."""

from __future__ import annotations

import numpy as np

from src.amm import (
    ConstantProductAMM,
    ExternalAMMEdge,
    G_f_supply_function,
    SystemOfExternalAMMs,
    g_f_supply_function,
    verify_proposition_5_4,
    verify_proposition_5_8,
)
from src.fundamental_theorem import (
    TradeOrderEdge,
    build_connectivity_graph,
    find_equilibrium,
    is_connected,
    verify_remark_6_5,
    verify_theorem_6_4,
)
from src.supply_functions import continuous_limit_sell_order

RNG = np.random.default_rng(7)


def test_constant_product_amm_closed_forms():
    amm = ConstantProductAMM(a=100.0, b=200.0)
    r = np.array([0.5, 1.0, 1.9])
    # G_f(r) = b - sqrt(a b r); g_f(r) = sqrt(a b r) - a r (Examples, Section 5).
    np.testing.assert_allclose(amm.G(r), amm.b - np.sqrt(amm.a * amm.b * r), rtol=1e-10)
    np.testing.assert_allclose(amm.g(r), np.sqrt(amm.a * amm.b * r) - amm.a * r, rtol=1e-10)


def test_proposition_5_4_full_report():
    amm = ConstantProductAMM(a=50.0, b=120.0)
    r_grid = np.geomspace(1e-4, 100.0, 300)
    report = verify_proposition_5_4(amm, r_grid)
    assert report["g_le_G_everywhere"]
    assert report["G_is_wgs"]
    assert not report["g_is_wgs"]
    assert report["surplus_nonnegative"]
    assert report["G_connects_2_to_1"]
    assert not report["G_connects_1_to_2"]


def test_surplus_matches_closed_form():
    amm = ConstantProductAMM(a=10.0, b=40.0)
    r = np.geomspace(1e-3, 10.0, 50)
    s = amm.surplus(r)
    expected = (np.sqrt(amm.b) - np.sqrt(amm.a * r)) ** 2 / r
    np.testing.assert_allclose(s, expected, rtol=1e-8)
    assert np.all(s >= -1e-9)


def test_g_f_is_not_a_supply_function_wgs_but_G_f_is():
    amm = ConstantProductAMM(a=20.0, b=80.0)
    g_pf = g_f_supply_function(amm)
    G_pf = G_f_supply_function(amm)
    r_grid = np.geomspace(1e-3, 50.0, 100)
    assert np.all(np.diff(G_pf.g(r_grid)) <= 1e-9)  # G_f decreasing (WGS)
    assert not np.all(np.diff(g_pf.g(r_grid)) <= 1e-9)  # g_f not monotone


def _two_amm_three_token_system() -> SystemOfExternalAMMs:
    amm01 = ConstantProductAMM(a=100.0, b=100.0)
    amm12 = ConstantProductAMM(a=80.0, b=120.0)
    edges = [
        ExternalAMMEdge(name="amm01", token_in=0, token_out=1, amm=amm01),
        ExternalAMMEdge(name="amm12", token_in=1, token_out=2, amm=amm12),
    ]
    return SystemOfExternalAMMs(edges=edges)


def test_proposition_5_8_maximizes_p_dot_F():
    system = _two_amm_three_token_system()
    p = np.array([1.0, 1.2, 0.8])
    report = verify_proposition_5_8(system, p, rng=RNG)
    assert all(report.values())


def test_theorem_6_4_balance_and_nonnegative_surplus():
    n = 3
    amm_system = _two_amm_three_token_system()
    orders = [
        TradeOrderEdge("o01", 0, 1, continuous_limit_sell_order(b=3.0, r1=0.3, r2=2.0)),
        TradeOrderEdge("o12", 1, 2, continuous_limit_sell_order(b=3.0, r1=0.3, r2=2.0)),
        TradeOrderEdge("o20", 2, 0, continuous_limit_sell_order(b=3.0, r1=0.3, r2=2.0)),
    ]
    edges = build_connectivity_graph(n, orders, amm_system)
    assert is_connected(n, edges)

    result = find_equilibrium(n, orders, amm_system)
    assert result["converged"]
    p = result["p"]

    report = verify_theorem_6_4(n, orders, amm_system, p)
    assert report["surplus_nonnegative"]
    assert report["balance_matches_surplus"]
    assert all(report["strict_matches_sign"].values())


def test_remark_6_5_external_swaps_maximize_surplus_value():
    n = 3
    amm_system = _two_amm_three_token_system()
    orders = [
        TradeOrderEdge("o01", 0, 1, continuous_limit_sell_order(b=3.0, r1=0.3, r2=2.0)),
        TradeOrderEdge("o12", 1, 2, continuous_limit_sell_order(b=3.0, r1=0.3, r2=2.0)),
        TradeOrderEdge("o20", 2, 0, continuous_limit_sell_order(b=3.0, r1=0.3, r2=2.0)),
    ]
    result = find_equilibrium(n, orders, amm_system)
    p = result["p"]
    report = verify_remark_6_5(n, orders, amm_system, p, rng=RNG)
    assert report["is_maximizer"]
