"""Tests of Section 7 (convex-optimization equilibrium), Section 8 (protocol
simulation) and Appendix A (non-differentiable AMM functions)."""

from __future__ import annotations

import numpy as np

from src.amm import ConstantProductAMM, ExternalAMMEdge, SystemOfExternalAMMs
from src.convex_optimization import (
    H_linear,
    Z_linear,
    implied_Phi,
    incidence_matrix_B,
    solve_equilibrium_convex_program,
)
from src.nondiff_amm import PiecewiseLinearAMM, verify_lemma_A3, verify_proposition_A2
from src.protocol import GovernanceDAO, simulate_blocks


def test_incidence_matrix_lemma_7_2():
    B = incidence_matrix_B(3, [(0, 1), (1, 2), (2, 0)])
    assert B.shape == (3, 3)
    # Columns sum to zero: each pair contributes +1/-1 across two tokens.
    np.testing.assert_allclose(B.sum(axis=0), 0.0)


def test_H_Z_duality_lemma_7_1():
    """Z(t) should equal the direct supremum over u of {-t ln u - H(u)}
    for linear g_{a,b}, verifying the closed form in Lemma 7.1.2."""
    a, b, t = 2.0, 5.0, 1.5
    z_closed = Z_linear(a, b, t)
    u_grid = np.geomspace(1e-4, 1e3, 200000)
    obj = -t * np.log(u_grid) - H_linear(a, b, u_grid)
    assert abs(z_closed - obj.max()) < 5e-3


def test_theorem_7_3_convex_program_recovers_equilibrium():
    n = 3
    pairs = [(0, 1), (1, 2), (2, 0)]
    linear_params = {(0, 1): (2.0, 5.0), (1, 2): (1.5, 4.0), (2, 0): (1.0, 3.0)}
    out = solve_equilibrium_convex_program(n, pairs, linear_params)
    assert out["status"] in ("optimal", "optimal_inaccurate")
    assert out["objective_value"] < 1e-3
    Phi = implied_Phi(n, pairs, linear_params, out["p"])
    np.testing.assert_allclose(Phi, 0.0, atol=1e-2)


def test_piecewise_linear_amm_matches_constant_product_in_limit():
    """A fine piecewise-linear discretization of a smooth concave function
    should approximate the constant-product closed form (Appendix A vs
    Section 5), confirming the generalized h_f/g_f/G_f agree with the
    differentiable case in the absence of liquidity gaps."""
    a, b = 50.0, 100.0
    cp_amm = ConstantProductAMM(a=a, b=b)
    x_grid = np.linspace(0.0, 5000.0, 20000)
    f_vals = cp_amm.f(x_grid)
    slopes = np.diff(f_vals) / np.diff(x_grid)
    slopes = np.minimum.accumulate(np.clip(slopes, 0, None))  # enforce concavity against rounding
    pw = PiecewiseLinearAMM(breakpoints=x_grid, slopes=slopes)
    r_test = np.array([0.3, 0.6, 1.0])
    np.testing.assert_allclose(pw.G(r_test), cp_amm.G(r_test), atol=1.0)


def test_piecewise_linear_amm_with_liquidity_gap_proposition_A2():
    """A concentrated-liquidity pool with a gap (zero slope segment) still
    satisfies the monotonicity, continuity and surplus properties of
    Proposition A.2."""
    breakpoints = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    slopes = np.array([5.0, 2.0, 0.0, 0.0])  # liquidity thins to a gap on [20, 40)
    amm = PiecewiseLinearAMM(breakpoints=breakpoints, slopes=slopes)
    r_grid = np.geomspace(0.01, 10.0, 400)
    report = verify_proposition_A2(amm, r_grid)
    assert report["h_decreasing"]
    assert report["surplus_nonnegative"]
    assert abs(report["lim_g_at_0"]) < 1e-6
    assert abs(report["lim_G_at_0"] - report["expected_M"]) < 1e-3


def test_lemma_A3_unique_maximizer():
    breakpoints = np.array([0.0, 10.0, 20.0, 30.0])
    slopes = np.array([4.0, 2.0, 0.5])
    amm = PiecewiseLinearAMM(breakpoints=breakpoints, slopes=slopes)
    x_grid = np.linspace(0.0, 30.0, 3000)
    report = verify_lemma_A3(amm, r=1.0, x_grid=x_grid)
    assert report["is_maximizer"]


def test_protocol_simulation_produces_valid_batches():
    n = 4
    amm_system = SystemOfExternalAMMs(
        edges=[
            ExternalAMMEdge("a01", 0, 1, ConstantProductAMM(a=100.0, b=100.0)),
            ExternalAMMEdge("a12", 1, 2, ConstantProductAMM(a=80.0, b=120.0)),
            ExternalAMMEdge("a23", 2, 3, ConstantProductAMM(a=60.0, b=90.0)),
            ExternalAMMEdge("a30", 3, 0, ConstantProductAMM(a=70.0, b=70.0)),
        ]
    )
    dao = GovernanceDAO(
        w_token_supply=1_000_000.0,
        allowlist_upvote_threshold=0.1,
        tau=0.003,
        gamma_1=0.5,
        gamma_2=0.3,
    )
    rng = np.random.default_rng(123)
    batches = simulate_blocks(n, amm_system, dao, n_blocks=5, orders_per_block=3, gas_price=1e-9, rng=rng)
    assert len(batches) == 5
    for batch in batches:
        assert np.all(batch.surplus_by_token >= -1e-6)
        assert batch.solver_trading_fee_payout <= batch.trading_fee_value + 1e-9
        assert batch.solver_surplus_payout <= max(np.dot(batch.equilibrium_price, batch.surplus_by_token), 0) + 1e-6
        assert batch.dao_trading_fee_revenue >= -1e-9
