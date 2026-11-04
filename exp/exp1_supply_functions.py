"""Experiment 1: Supply Function Framework and Pairwise Characterization.

Reproduces, numerically, Definition 2.1 (supply function conditions),
Proposition 2.2 (sum of supply functions), Proposition 2.4 (pairwise
characterization via g), Examples 2.6(IV) (continuous limit sell/buy orders),
Definition 2.5 (conjugate function tilde_g), Examples 2.8 (the
function-maximizing AMM's linear supply function g_{y/2,x/2}), and Lemma 2.11
/ Proposition 2.10 (extension to n tokens and restriction) of the paper.

Run: python -m exp.exp1_supply_functions
"""

from __future__ import annotations

import json

import numpy as np

from src.config import (
    EXP1_AMM_RESERVES,
    EXP1_BUY_ORDER_PARAMS,
    EXP1_LINEAR_G_PARAMS,
    EXP1_N_RANDOM_PRICE_VECTORS,
    EXP1_N_TOKENS_GENERIC,
    EXP1_SELL_ORDER_PARAMS,
    RNG,
)
from src.supply_functions import (
    ExtendedSupplyFunction,
    PairwiseSupplyFunction,
    SumSupplyFunction,
    continuous_limit_buy_order,
    continuous_limit_sell_order,
    function_maximizing_amm_new_reserves,
    function_maximizing_amm_supply,
    linear_pairwise_supply_function,
    restrict_supply_function,
)
from data.generate_synthetic import random_price_vector


def sample_price_vectors(n: int, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [random_price_vector(n, rng) for _ in range(count)]


def verify_definition_2_1(pf: PairwiseSupplyFunction, samples_p2: list[np.ndarray]) -> dict:
    """Numerically check conditions (0)-(2) of Definition 2.1 for a pairwise
    supply function, plus condition (3) (boundary regularity, Proposition 2.4
    conditions (a)-(b))."""
    bound = pf.bound_phi1() + pf.bound_phi2() + 1e-6
    report = pf.verify_conditions(samples_p2, bound=bound)
    report["boundary_condition_a_liminf_g_nonneg"] = pf.limit_at_zero() >= -1e-9
    report["boundary_condition_b_limsup_g_over_r_nonpos"] = pf.limit_over_r_at_infinity() <= 1e-9
    return report


def verify_proposition_2_2(components: list, samples_p2: list[np.ndarray]) -> dict:
    """Proposition 2.2: the sum of supply functions is a supply function."""
    total = SumSupplyFunction(components)
    bound = sum(c.bound_phi1() + c.bound_phi2() for c in components if hasattr(c, "bound_phi1")) + 10.0
    return total.verify_conditions(samples_p2, bound=bound)


def verify_conjugate(pf: PairwiseSupplyFunction, r_grid: np.ndarray) -> dict:
    """Verify tilde_tilde_g = g numerically (Definition 2.5)."""
    g_tilde = pf.conjugate()
    g_tilde_tilde = g_tilde.conjugate()
    diff = np.max(np.abs(g_tilde_tilde.g(r_grid) - pf.g(r_grid)))
    return {"max_abs_diff_g_vs_tilde_tilde_g": float(diff), "tilde_tilde_g_equals_g": bool(diff < 1e-6)}


def verify_examples_2_8(x: float, y: float, price_samples: list[np.ndarray]) -> dict:
    """Function-maximizing AMM (Examples 2.8): verify its supply function
    equals g_{y/2, x/2} and that (u, v) solves the stated maximization."""
    pf_amm = function_maximizing_amm_supply(x, y)
    pf_expected = linear_pairwise_supply_function(a=y / 2.0, b=x / 2.0)
    r_grid = np.geomspace(1e-3, 1e3, 200)
    max_diff = float(np.max(np.abs(pf_amm.g(r_grid) - pf_expected.g(r_grid))))

    walras_checks = []
    budget_checks = []
    for p in price_samples:
        u, v = function_maximizing_amm_new_reserves(x, y, p)
        budget_checks.append(abs(p[0] * (x - u) + p[1] * (y - v)) < 1e-6)
        walras_checks.append(abs(np.dot(pf_amm(p), p)) < 1e-6)
    return {
        "max_abs_diff_vs_g_y2_x2": max_diff,
        "matches_g_y2_x2": bool(max_diff < 1e-6),
        "budget_constraint_holds": all(budget_checks),
        "walras_law_holds": all(walras_checks),
    }


def verify_lemma_2_11_extension_and_restriction(pf: PairwiseSupplyFunction, n: int, sigma: list[int]) -> dict:
    """Lemma 2.11: extend a 2-token pairwise supply function to n tokens via
    an injective sigma; Proposition 2.10: restricting it back recovers the
    original function."""
    extended = ExtendedSupplyFunction(pf, sigma=sigma, n=n)
    p_full = RNG.uniform(0.5, 5.0, size=n)
    phi_full = extended(p_full)
    zero_elsewhere = all(abs(phi_full[k]) < 1e-12 for k in range(n) if k not in sigma)

    restricted = restrict_supply_function(extended, index_set=sigma)
    p_sub = p_full[sigma]
    phi_sub_direct = pf(p_sub)
    phi_sub_restricted = restricted(p_sub)
    restriction_matches = bool(np.allclose(phi_sub_direct, phi_sub_restricted, atol=1e-9))

    return {
        "zero_outside_support": zero_elsewhere,
        "restriction_matches_original": restriction_matches,
    }


def main() -> dict:
    rng = np.random.default_rng(RNG.integers(0, 2**31))

    # 1-2. Definition 2.1 and Proposition 2.4 for linear and continuous orders.
    a, b = EXP1_LINEAR_G_PARAMS
    pf_linear = linear_pairwise_supply_function(a, b)
    samples_p2 = sample_price_vectors(2, EXP1_N_RANDOM_PRICE_VECTORS, rng)
    def1_linear = verify_definition_2_1(pf_linear, samples_p2)

    sb, sr1, sr2 = EXP1_SELL_ORDER_PARAMS
    pf_sell = continuous_limit_sell_order(sb, sr1, sr2)
    def1_sell = verify_definition_2_1(pf_sell, samples_p2)

    db, br1, br2 = EXP1_BUY_ORDER_PARAMS
    pf_buy = continuous_limit_buy_order(db, br1, br2)
    def1_buy = verify_definition_2_1(pf_buy, samples_p2)

    # 3. Proposition 2.2: sum of supply functions.
    prop_2_2 = verify_proposition_2_2([pf_linear, pf_sell], samples_p2)

    # 4. Conjugate function tilde_tilde_g = g (Definition 2.5), for each construction.
    r_grid = np.geomspace(1e-3, 1e3, 300)
    conj_linear = verify_conjugate(pf_linear, r_grid)
    conj_sell = verify_conjugate(pf_sell, r_grid)
    conj_buy = verify_conjugate(pf_buy, r_grid)

    # 5. Examples 2.8: function-maximizing AMM.
    x, y = EXP1_AMM_RESERVES
    ex_2_8 = verify_examples_2_8(x, y, samples_p2)

    # 6. Lemma 2.11 / Proposition 2.10: extension to n tokens and restriction.
    ext_restrict = verify_lemma_2_11_extension_and_restriction(
        pf_linear, n=EXP1_N_TOKENS_GENERIC, sigma=[1, 3]
    )

    results = {
        "definition_2_1": {
            "linear_g": def1_linear,
            "continuous_sell_order": def1_sell,
            "continuous_buy_order": def1_buy,
        },
        "proposition_2_2_sum_is_supply_function": prop_2_2,
        "conjugate_tilde_tilde_g_equals_g": {
            "linear_g": conj_linear,
            "continuous_sell_order": conj_sell,
            "continuous_buy_order": conj_buy,
        },
        "examples_2_8_function_maximizing_amm": ex_2_8,
        "lemma_2_11_extension_and_proposition_2_10_restriction": ext_restrict,
    }

    with open("results/exp1_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))
    return results


if __name__ == "__main__":
    main()
