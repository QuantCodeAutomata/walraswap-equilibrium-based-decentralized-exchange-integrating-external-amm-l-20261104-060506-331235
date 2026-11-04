"""exp_6: Convex Optimization for Equilibrium Computation (paper Section 7).

Exercises the convex-program formulation of Theorem 7.3 (built on Lemma 7.1's
H/Z functions and Lemma 7.2's incidence matrix B) against the already-written
code in ``src/convex_optimization.py``, and cross-checks the resulting
equilibrium price against the independent Brouwer fixed-point construction of
Theorem 3.4 (``src/equilibrium.py``). No library code is modified here; this
script only calls the existing public API and records numeric/plot artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    EXP6_LINEAR_G,
    EXP6_N_TOKENS,
    EXP6_SIMPLEX_EPS,
    EXP6_SOLVER,
    RANDOM_SEED,
)
from src.convex_optimization import (
    H_linear,
    Z_linear,
    implied_Phi,
    solve_equilibrium_convex_program,
)
from src.equilibrium import find_brouwer_fixed_point
from src.supply_functions import (
    ExtendedSupplyFunction,
    SumSupplyFunction,
    linear_pairwise_supply_function,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("exp_6: Convex Optimization for Equilibrium Computation (Thm 7.3)")
    print("=" * 70)

    pairs = list(EXP6_LINEAR_G.keys())

    # -- Step 1: solve the convex program (Theorem 7.3) ---------------------
    result = solve_equilibrium_convex_program(
        EXP6_N_TOKENS, pairs, EXP6_LINEAR_G, solver=EXP6_SOLVER, eps=EXP6_SIMPLEX_EPS
    )
    status = result["status"]
    objective_value = float(result["objective_value"])
    p = np.asarray(result["p"], dtype=float)
    y = np.asarray(result["y"], dtype=float)

    print(f"\n[1] Solver status: {status}")
    print(f"    Objective value F(p*, y*) = {objective_value:.8f}")
    print(f"    p* = {p}")
    print(f"    y* = {y}")

    # -- Step 2: Theorem 7.3(2) -- market clearing at the optimum -----------
    Phi = implied_Phi(EXP6_N_TOKENS, pairs, EXP6_LINEAR_G, p)
    phi_clears = bool(np.allclose(Phi, 0.0, atol=1e-2))
    objective_near_zero = bool(abs(objective_value) < 1e-4)
    print(f"\n[2] Implied Phi(p*) = {Phi}")
    print(f"    Phi(p*) ~ 0 (market clearing, atol=1e-2): {'PASS' if phi_clears else 'FAIL'}")
    print(f"    objective_value ~ 0 (atol=1e-4): {'PASS' if objective_near_zero else 'FAIL'}")

    # -- Step 3: y_ij ~ p_i * g^{ij}(p_j/p_i) at the optimum -----------------
    y_matches_pg: dict[str, bool] = {}
    print("\n[3] Checking y_ij ~ p_i * g^{ij}(p_j/p_i):")
    for col, (i, j) in enumerate(pairs):
        a, b = EXP6_LINEAR_G[(i, j)]
        g_ij = -a * (p[j] / p[i]) + b
        expected = p[i] * g_ij
        ok = bool(np.isclose(y[col], expected, atol=1e-2))
        y_matches_pg[f"({i},{j})"] = ok
        print(
            f"    pair ({i},{j}): y={y[col]:.6f}, p_i*g^ij={expected:.6f} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )

    # -- Step 4: cross-check against the Brouwer fixed point (Thm 3.4) ------
    components = []
    upper_bounds = np.zeros(EXP6_N_TOKENS)
    for (i, j), (a, b) in EXP6_LINEAR_G.items():
        base = linear_pairwise_supply_function(a, b)
        comp = ExtendedSupplyFunction(base, sigma=[i, j], n=EXP6_N_TOKENS)
        components.append(comp)
        upper_bounds[i] += base.bound_phi1()
        upper_bounds[j] += base.bound_phi2()
    upper_bounds = upper_bounds + 1e-9  # avoid a zero bound

    Phi_agg = SumSupplyFunction(components)
    brouwer_result = find_brouwer_fixed_point(Phi_agg, upper_bounds.tolist())
    p_brouwer = np.asarray(brouwer_result["p"], dtype=float)

    p_cvx_norm = p / p.sum()
    p_brouwer_norm = p_brouwer / p_brouwer.sum()
    max_abs_diff = float(np.max(np.abs(p_cvx_norm - p_brouwer_norm)))
    brouwer_matches = bool(max_abs_diff < 0.05)

    print("\n[4] Cross-check against the Brouwer fixed point (Theorem 3.4):")
    print(f"    Brouwer converged: {brouwer_result['converged']} "
          f"(iterations={brouwer_result['iterations']}, "
          f"residual_norm={brouwer_result['residual_norm']:.3e})")
    print(f"    p (convex program, normalized)  = {p_cvx_norm}")
    print(f"    p (Brouwer fixed point, normalized) = {p_brouwer_norm}")
    print(f"    max abs diff = {max_abs_diff:.6f} -> "
          f"{'PASS' if brouwer_matches else 'FAIL'} (tol 0.05)")

    # -- Step 5: Lemma 7.1 Legendre-type relation for H_linear/Z_linear -----
    (i0, j0) = pairs[0]
    a0, b0 = EXP6_LINEAR_G[(i0, j0)]
    r_grid = np.linspace(0.1 * (b0 / a0), 0.9 * (b0 / a0), 5)
    H_vals = H_linear(a0, b0, r_grid)
    legendre_errors = []
    for r_val, H_val in zip(r_grid, H_vals):
        t_val = -a0 * r_val + b0  # t = g(r)
        Z_val = Z_linear(a0, b0, t_val)
        expected = -t_val * np.log(r_val) - H_val
        legendre_errors.append(abs(Z_val - expected))
    legendre_errors = np.array(legendre_errors)
    legendre_max_err = float(np.max(legendre_errors))
    legendre_ok = bool(legendre_max_err < 1e-6)

    print(f"\n[5] Lemma 7.1 Legendre relation check on pair ({i0},{j0}), (a,b)=({a0},{b0}):")
    print(f"    r grid = {r_grid}")
    print(f"    max |Z(t) - (-t ln r - H(r))| = {legendre_max_err:.3e} -> "
          f"{'PASS' if legendre_ok else 'FAIL'} (atol=1e-6)")

    # -- Step 6: plot ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    token_labels = [f"token {k}" for k in range(EXP6_N_TOKENS)]
    x = np.arange(EXP6_N_TOKENS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, p_cvx_norm, width, label="Convex program (Thm 7.3)")
    ax.bar(x + width / 2, p_brouwer_norm, width, label="Brouwer fixed point (Thm 3.4)")
    ax.set_xticks(x)
    ax.set_xticklabels(token_labels)
    ax.set_ylabel("Normalized equilibrium price $p_i / \\sum_k p_k$")
    ax.set_title("exp_6: Equilibrium price -- convex program vs Brouwer fixed point")
    ax.legend()
    fig.tight_layout()
    fig_path = RESULTS_DIR / "exp6_convex_program.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n[6] Saved plot to {fig_path}")

    # -- Step 7: save metrics ---------------------------------------------
    metrics = {
        "status": status,
        "objective_value": objective_value,
        "p": p.tolist(),
        "y": y.tolist(),
        "Phi": Phi.tolist(),
        "phi_clears_market": phi_clears,
        "objective_near_zero": objective_near_zero,
        "y_matches_p_times_g": y_matches_pg,
        "brouwer_check": {
            "p_convex_program_normalized": p_cvx_norm.tolist(),
            "p_brouwer_normalized": p_brouwer_norm.tolist(),
            "max_abs_diff": max_abs_diff,
            "matches": brouwer_matches,
            "brouwer_converged": bool(brouwer_result["converged"]),
            "brouwer_iterations": int(brouwer_result["iterations"]),
        },
        "legendre_relation_check": {
            "pair": [int(i0), int(j0)],
            "a": float(a0),
            "b": float(b0),
            "r_grid": r_grid.tolist(),
            "errors": legendre_errors.tolist(),
            "max_error": legendre_max_err,
            "passed": legendre_ok,
        },
    }
    metrics_path = RESULTS_DIR / "exp6_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[7] Saved metrics to {metrics_path}")

    # -- Step 8: summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary of checks:")
    print(f"  Market clearing Phi(p*) ~ 0:        {'PASS' if phi_clears else 'FAIL'}")
    print(f"  Objective value ~ 0:                 {'PASS' if objective_near_zero else 'FAIL'}")
    all_y_ok = all(y_matches_pg.values())
    print(f"  y_ij ~ p_i * g^ij(p_j/p_i) (all pairs): {'PASS' if all_y_ok else 'FAIL'}")
    print(f"  Brouwer fixed point cross-check:     {'PASS' if brouwer_matches else 'FAIL'}")
    print(f"  Lemma 7.1 Legendre relation:         {'PASS' if legendre_ok else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
