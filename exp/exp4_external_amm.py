"""exp_4: External AMM Virtual Agents Construction and Properties (paper Sec. 5).

Builds constant-product AMMs from ``EXP4_CP_RESERVES``, verifies Proposition
5.4 (properties of g_f, G_f and the surplus s_f) both numerically and against
the closed-form formulas given in Section 5, assembles systems of external
AMMs (Definition 5.5) over a shared token pair and over a token chain, checks
that the prescribed swaps h^{C,p} maximize the local AMM objective
(Proposition 5.8), and verifies that the virtual agent phi^c built from G_f is
a valid supply function (Definition 2.1). Numeric results are saved to
``results/exp4_metrics.json`` and a plot of g_f, G_f, s_f to
``results/exp4_gf_Gf_surplus.png``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import random_price_vector
from src.amm import (
    ConstantProductAMM,
    ExternalAMMEdge,
    G_f_supply_function,
    SystemOfExternalAMMs,
    verify_proposition_5_4,
    verify_proposition_5_8,
)
from src.config import (
    EXP4_CP_RESERVES,
    EXP4_R_GRID_MAX_FACTOR,
    EXP4_R_GRID_POINTS,
    RANDOM_SEED,
)

RESULTS_DIR = Path("results")


def _make_r_grid(amm: ConstantProductAMM) -> np.ndarray:
    """r_grid in (0, EXP4_R_GRID_MAX_FACTOR * f'(0)], avoiding r = 0 exactly."""
    eps = amm.f_prime_at_0 * 1e-4
    return np.linspace(eps, EXP4_R_GRID_MAX_FACTOR * amm.f_prime_at_0, EXP4_R_GRID_POINTS)


def closed_form_match(amm: ConstantProductAMM, r_grid: np.ndarray) -> bool:
    """Compare amm.g, amm.G, amm.surplus against the Section 5 closed forms
    for the no-fee constant-product AMM: g_f(r) = sqrt(abr) - ar and
    G_f(r) = b - sqrt(abr) for r < b/a (0 outside that domain, matching the
    implementation's h_f(r) = 0 there); surplus s(r) = (sqrt(b)-sqrt(ar))^2/r."""
    a, b = amm.a, amm.b
    below = r_grid < amm.f_prime_at_0
    g_closed = np.where(below, np.sqrt(a * b * r_grid) - a * r_grid, 0.0)
    G_closed = np.where(below, b - np.sqrt(a * b * r_grid), 0.0)
    s_closed = (np.sqrt(b) - np.sqrt(a * r_grid)) ** 2 / r_grid

    g_ok = np.allclose(amm.g(r_grid), g_closed, atol=1e-6)
    G_ok = np.allclose(amm.G(r_grid), G_closed, atol=1e-6)
    s_ok = np.allclose(amm.surplus(r_grid), s_closed, atol=1e-6)
    return bool(g_ok and G_ok and s_ok)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== exp_4: External AMM Virtual Agents (Section 5) ===")

    amms = [ConstantProductAMM(a=a, b=b) for a, b in EXP4_CP_RESERVES]

    # --- Step 2-3: Proposition 5.4 + closed-form checks per AMM ------------
    prop54_results: dict[str, dict] = {}
    closed_form_results: dict[str, bool] = {}
    r_grids: list[np.ndarray] = []
    for (a, b), amm in zip(EXP4_CP_RESERVES, amms):
        key = f"(a={a}, b={b})"
        r_grid = _make_r_grid(amm)
        r_grids.append(r_grid)

        prop54 = verify_proposition_5_4(amm, r_grid)
        prop54_results[key] = prop54

        cf_match = closed_form_match(amm, r_grid)
        closed_form_results[key] = cf_match

        print(f"\n[AMM {key}] f'(0)={amm.f_prime_at_0:.4f}, M={amm.M:.4f}")
        for name, val in prop54.items():
            print(f"    Prop 5.4 - {name}: {val}")
        print(f"    closed_form_match: {cf_match}")

    # --- Step 4: two systems of external AMMs ------------------------------
    edges_same_pair = [
        ExternalAMMEdge(name=f"amm{k}", token_in=0, token_out=1, amm=amm)
        for k, amm in enumerate(amms)
    ]
    system_same_pair = SystemOfExternalAMMs(edges=edges_same_pair)

    edges_chain = [
        ExternalAMMEdge(name=f"amm{k}", token_in=k, token_out=k + 1, amm=amm)
        for k, amm in enumerate(amms)
    ]
    system_chain = SystemOfExternalAMMs(edges=edges_chain)

    print(f"\nSystem (3 parallel pools, same pair 0->1): n_tokens={system_same_pair.n_tokens}")
    print(f"System (chain over 4 tokens, 0->1->2->3): n_tokens={system_chain.n_tokens}")

    # --- Step 5: chain system at a random price vector ---------------------
    p_chain = random_price_vector(system_chain.n_tokens, rng)
    swaps_chain = system_chain.external_swaps(p_chain)
    F_C_chain = system_chain.F_C(swaps_chain)
    surplus_chain = system_chain.surplus_per_amm(p_chain)
    prop58_chain = verify_proposition_5_8(system_chain, p_chain, n_probe=200, rng=rng)

    print(f"\nChain system random price vector p = {p_chain}")
    print(f"  external_swaps = {swaps_chain}")
    print(f"  F_C(swaps)     = {F_C_chain}")
    print(f"  surplus_per_amm = {surplus_chain}")
    for name, is_max in prop58_chain.items():
        print(f"  Prop 5.8 maximizer check [{name}]: {is_max}")

    # Same exercise on the 3-parallel-pools system, for completeness.
    p_same_pair = random_price_vector(system_same_pair.n_tokens, rng)
    swaps_same_pair = system_same_pair.external_swaps(p_same_pair)
    F_C_same_pair = system_same_pair.F_C(swaps_same_pair)
    surplus_same_pair = system_same_pair.surplus_per_amm(p_same_pair)
    prop58_same_pair = verify_proposition_5_8(system_same_pair, p_same_pair, n_probe=200, rng=rng)

    print(f"\nParallel-pools system random price vector p = {p_same_pair}")
    print(f"  external_swaps = {swaps_same_pair}")
    print(f"  F_C(swaps)     = {F_C_same_pair}")
    print(f"  surplus_per_amm = {surplus_same_pair}")
    for name, is_max in prop58_same_pair.items():
        print(f"  Prop 5.8 maximizer check [{name}]: {is_max}")

    # --- Step 6: virtual agent phi^c is a valid supply function ------------
    probe_amm = amms[0]
    pf = G_f_supply_function(probe_amm)
    sample_prices = [random_price_vector(2, rng) for _ in range(20)]
    virtual_agent_conditions = pf.verify_conditions(sample_prices=sample_prices, bound=probe_amm.M)

    print(f"\nVirtual agent phi^c (G_f of AMM {EXP4_CP_RESERVES[0]}) supply-function conditions:")
    for name, val in virtual_agent_conditions.items():
        print(f"    {name}: {val}")

    # --- Step 7: plots -------------------------------------------------------
    fig, axes = plt.subplots(1, len(amms), figsize=(6 * len(amms), 5), squeeze=False)
    for ax, (a, b), amm, r_grid in zip(axes[0], EXP4_CP_RESERVES, amms, r_grids):
        ax.plot(r_grid, amm.g(r_grid), label="$g_f(r)$")
        ax.plot(r_grid, amm.G(r_grid), label="$G_f(r)$")
        ax.plot(r_grid, amm.surplus(r_grid), label="$s_f(r)$")
        ax.axvline(amm.f_prime_at_0, color="gray", linestyle="--", linewidth=1, label="$f'(0)=b/a$")
        ax.set_title(f"Constant-product AMM (a={a}, b={b})")
        ax.set_xlabel("$r = p_{in}/p_{out}$")
        ax.set_ylabel("value")
        ax.legend()
    fig.suptitle("External AMM virtual-agent functions $g_f$, $G_f$, $s_f$ (Section 5)")
    fig.tight_layout()
    plot_path = RESULTS_DIR / "exp4_gf_Gf_surplus.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # --- Step 8: save metrics ------------------------------------------------
    metrics = {
        "proposition_5_4": prop54_results,
        "closed_form_match": closed_form_results,
        "chain_system": {
            "price_vector": p_chain,
            "external_swaps": swaps_chain,
            "F_C": F_C_chain,
            "surplus_per_amm": surplus_chain,
            "proposition_5_8_maximizer_check": prop58_chain,
        },
        "parallel_pools_system": {
            "price_vector": p_same_pair,
            "external_swaps": swaps_same_pair,
            "F_C": F_C_same_pair,
            "surplus_per_amm": surplus_same_pair,
            "proposition_5_8_maximizer_check": prop58_same_pair,
        },
        "virtual_agent_supply_function_conditions": virtual_agent_conditions,
    }
    metrics_path = RESULTS_DIR / "exp4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)

    # --- Step 9: concise pass/fail summary -----------------------------------
    # Expected boolean outcomes per Proposition 5.4: g_le_G_everywhere,
    # strict_inequality_below_fprime0, G_is_wgs, G_connects_2_to_1 and
    # surplus_nonnegative must hold (True); g_is_wgs and G_connects_1_to_2 are
    # expected to be False (g is *not* WGS, G connects 2->1 but not 1->2), so
    # they pass precisely when they equal False.
    expected_prop54 = {
        "g_le_G_everywhere": True,
        "strict_inequality_below_fprime0": True,
        "G_is_wgs": True,
        "g_is_wgs": False,
        "G_connects_2_to_1": True,
        "G_connects_1_to_2": False,
        "surplus_nonnegative": True,
    }
    print("\n=== Summary ===")
    for key, prop54 in prop54_results.items():
        all_ok = all(bool(prop54[name]) == expected for name, expected in expected_prop54.items())
        print(f"  Proposition 5.4 all checks pass (match expected True/False pattern) for {key}: {all_ok}")
    for key, ok in closed_form_results.items():
        print(f"  Closed-form match for {key}: {ok}")
    print(f"  Prop 5.8 (chain system) all pass: {all(prop58_chain.values())}")
    print(f"  Prop 5.8 (parallel-pools system) all pass: {all(prop58_same_pair.values())}")
    print(f"  Virtual agent phi^c is a valid supply function: {all(virtual_agent_conditions.values())}")
    print(f"\nSaved plot to {plot_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
