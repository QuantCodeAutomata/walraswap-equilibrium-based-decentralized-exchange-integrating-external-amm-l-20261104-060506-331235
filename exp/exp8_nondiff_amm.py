"""exp_8: Non-Differentiable AMM Functions -- Generalization to Concentrated
Liquidity Pools (Appendix A).

Builds a piecewise-linear, concave AMM function f with genuine kinks at its
breakpoints and a terminal liquidity gap (a zero-slope segment, modeling an
exhausted Uniswap V3 tick range). Numerically verifies Proposition A.2
(monotonicity/continuity of h_f, the surplus inequality, limit behaviour) and
Lemma A.3 (h_f(r) uniquely maximizes f(x) - r x) across several representative
r values, including one landing inside the liquidity gap. Also checks the
one-sided-derivative concavity condition (Definition A.1(d)) at every interior
breakpoint, and contrasts the non-differentiable case against a differentiable
constant-product AMM (Section 5) with a comparable f'(0).
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.amm import ConstantProductAMM, verify_proposition_5_4
from src.config import (
    EXP8_BREAKPOINTS,
    EXP8_R_GRID_POINTS,
    EXP8_SLOPES,
    RANDOM_SEED,
)
from src.nondiff_amm import PiecewiseLinearAMM, verify_lemma_A3, verify_proposition_A2


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Build the piecewise-linear, concave AMM function (Definition A.1).
    # ------------------------------------------------------------------
    amm = PiecewiseLinearAMM(
        breakpoints=np.array(EXP8_BREAKPOINTS, dtype=float),
        slopes=np.array(EXP8_SLOPES, dtype=float),
    )
    print("=== exp_8: Non-Differentiable AMM Functions (Appendix A) ===")
    print(f"breakpoints = {amm.breakpoints.tolist()}")
    print(f"slopes      = {amm.slopes.tolist()}")
    print(f"M (bound)   = {amm.M:.6f}")
    print(f"f'(0)       = {amm.f_prime_at_0:.6f}")

    # ------------------------------------------------------------------
    # 2. Proposition A.2: monotonicity of h_f, surplus inequality, limits.
    # ------------------------------------------------------------------
    r_grid = np.linspace(1e-6, amm.f_prime_at_0 * 1.2, EXP8_R_GRID_POINTS)
    prop_a2_result = verify_proposition_A2(amm, r_grid)

    print("\n--- Proposition A.2 checks ---")
    print(f"[{_status(prop_a2_result['h_decreasing'])}] h_f nonincreasing on r_grid")
    print(f"[{_status(prop_a2_result['surplus_nonnegative'])}] surplus s_f(r) >= 0 on r_grid")
    print(f"[{_status(prop_a2_result['strict_where_h_positive'])}] G_f(r) > g_f(r) where h_f(r) > 0")
    print(f"[{_status(prop_a2_result['identity_h_of_fprime_holds'])}] h_f(partial_+f(x_k)) = x_k at kinks")
    print(f"    lim_{{r->0}} G_f(r) = {prop_a2_result['lim_G_at_0']:.6f} (expected M = {prop_a2_result['expected_M']:.6f})")
    print(f"    lim_{{r->0}} g_f(r) = {prop_a2_result['lim_g_at_0']:.6f} (expected ~0)")

    # ------------------------------------------------------------------
    # 3. Lemma A.3: h_f(r) uniquely maximizes f(x) - r x, at several r values
    #    landing in different segments, including one inside the liquidity gap.
    # ------------------------------------------------------------------
    x_grid = np.linspace(0.0, amm.breakpoints[-1], 2000)
    distinct_slopes = amm.slopes  # [9.0, 5.0, 2.0, 0.6, 0.0], nonincreasing
    r_values = [
        (distinct_slopes[0] + distinct_slopes[1]) / 2.0,  # inside segment 1's slope range
        (distinct_slopes[1] + distinct_slopes[2]) / 2.0,  # inside segment 2's slope range
        (distinct_slopes[2] + distinct_slopes[3]) / 2.0,  # inside segment 3's slope range
        (distinct_slopes[3] + distinct_slopes[4]) / 2.0,  # strictly between last two distinct
        # slopes (0.6 and 0.0): lands exactly at the onset of the liquidity gap
    ]

    lemma_a3_results = []
    print("\n--- Lemma A.3 checks (q(x) = f(x) - r x maximized at x* = h_f(r)) ---")
    for r_val in r_values:
        res = verify_lemma_A3(amm, r=float(r_val), x_grid=x_grid)
        res["r"] = float(r_val)
        lemma_a3_results.append(res)
        print(
            f"[{_status(res['is_maximizer'])}] r={r_val:.4f}: x*={res['x_star']:.4f}, "
            f"q*={res['q_star']:.6f}, max over grid={res['q_grid_max']:.6f}"
        )

    # ------------------------------------------------------------------
    # 4. Concavity / kink structure at every interior breakpoint
    #    (Definition A.1(d)): left_derivative(x_k) >= right_derivative(x_k).
    # ------------------------------------------------------------------
    interior_breakpoints = amm.breakpoints[1:-1]
    kink_concavity_checks = []
    print("\n--- Kink structure at interior breakpoints (concavity, Definition A.1(d)) ---")
    for x_k in interior_breakpoints:
        left = float(amm.left_derivative(np.array([x_k]))[0])
        right = float(amm.right_derivative(np.array([x_k]))[0])
        ok = bool(left >= right - 1e-9)
        kink_concavity_checks.append(ok)
        print(f"[{_status(ok)}] x_k={x_k:.2f}: left_derivative={left:.4f} >= right_derivative={right:.4f}")

    # ------------------------------------------------------------------
    # 5. Comparison with a differentiable constant-product AMM (Section 5)
    #    with a comparable f'(0).
    # ------------------------------------------------------------------
    cp_a = 10.0
    cp_b = cp_a * amm.f_prime_at_0
    cp_amm = ConstantProductAMM(a=cp_a, b=cp_b)
    prop_5_4_result = verify_proposition_5_4(cp_amm, r_grid)

    print("\n--- Comparison constant-product AMM (Proposition 5.4) ---")
    print(f"a={cp_a}, b={cp_b:.4f}, f'(0)={cp_amm.f_prime_at_0:.4f} (piecewise f'(0)={amm.f_prime_at_0:.4f})")
    for key, val in prop_5_4_result.items():
        print(f"    {key}: {val}")

    # ------------------------------------------------------------------
    # 6. Plots.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_f, ax_h, ax_g, ax_G, ax_s = axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]
    axes[1, 2].axis("off")

    x_plot = np.linspace(0.0, amm.breakpoints[-1], 2000)
    ax_f.plot(x_plot, amm.f(x_plot), color="tab:blue", label="f(x) piecewise-linear AMM")
    ax_f.scatter(amm.breakpoints, amm.f(amm.breakpoints), color="tab:red", zorder=5, label="breakpoints x_k")
    ax_f.axhline(amm.M, color="gray", linestyle=":", label=f"M = {amm.M:.2f}")
    ax_f.set_title("f(x): kinks and liquidity-gap plateau")
    ax_f.set_xlabel("x (amount sold)")
    ax_f.set_ylabel("f(x) (amount received)")
    ax_f.legend()

    h_pw, h_cp = amm.h(r_grid), cp_amm.h(r_grid)
    ax_h.plot(r_grid, h_pw, color="tab:blue", label="h_f piecewise AMM")
    ax_h.plot(r_grid, h_cp, color="tab:orange", linestyle="--", label="h_f constant-product AMM")
    # h_f of the constant-product AMM diverges as r -> 0 (unbounded liquidity
    # in x), while the piecewise AMM's h_f is bounded by its last breakpoint;
    # clip the view to that domain so the step structure stays visible.
    ax_h.set_ylim(0, amm.breakpoints[-1] * 1.2)
    ax_h.set_title("h_f(r) (y-axis clipped to piecewise domain)")
    ax_h.set_xlabel("r")
    ax_h.set_ylabel("h_f(r)")
    ax_h.legend()

    g_pw, g_cp = amm.g(r_grid), cp_amm.g(r_grid)
    ax_g.plot(r_grid, g_pw, color="tab:blue", label="g_f piecewise AMM")
    ax_g.plot(r_grid, g_cp, color="tab:orange", linestyle="--", label="g_f constant-product AMM")
    ax_g.set_title("g_f(r) = r * h_f(r)")
    ax_g.set_xlabel("r")
    ax_g.set_ylabel("g_f(r)")
    ax_g.legend()

    G_pw, G_cp = amm.G(r_grid), cp_amm.G(r_grid)
    ax_G.plot(r_grid, G_pw, color="tab:blue", label="G_f piecewise AMM")
    ax_G.plot(r_grid, G_cp, color="tab:orange", linestyle="--", label="G_f constant-product AMM")
    ax_G.set_title("G_f(r) = f(h_f(r))")
    ax_G.set_xlabel("r")
    ax_G.set_ylabel("G_f(r)")
    ax_G.legend()

    s_pw, s_cp = amm.surplus(r_grid), cp_amm.surplus(r_grid)
    ax_s.plot(r_grid, s_pw, color="tab:blue", label="s(r) piecewise AMM")
    ax_s.plot(r_grid, s_cp, color="tab:orange", linestyle="--", label="s(r) constant-product AMM")
    # s(r) ~ M / r as r -> 0 for *both* AMMs (Proposition A.2(VI) / 5.4(1)), a
    # shared universal divergence, not a kink/gap-specific feature; a log
    # scale keeps this visible without crushing the rest of the curve.
    ax_s.set_yscale("log")
    ax_s.set_title("surplus s(r) (log scale)")
    ax_s.set_xlabel("r")
    ax_s.set_ylabel("s(r)")
    ax_s.legend()

    fig.suptitle("exp_8: Non-Differentiable (piecewise-linear) vs Differentiable (constant-product) AMM")
    fig.tight_layout()
    fig.savefig("results/exp8_piecewise_amm.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7. Save metrics.
    # ------------------------------------------------------------------
    # s(r) ~ M / r as r -> 0 for *both* AMMs (a shared, model-independent
    # divergence from Proposition A.2(VI) / 5.4(1)), so the raw max over the
    # full r_grid (which starts at 1e-6) would just compare that universal
    # singularity rather than any kink/gap-specific feature. The comparison
    # below instead uses a trimmed sub-grid that excludes the near-zero
    # neighborhood, giving a meaningful, finite surplus comparison.
    trim_mask = r_grid >= 0.1 * amm.f_prime_at_0
    max_surplus_piecewise = float(np.max(s_pw[trim_mask]))
    max_surplus_cp = float(np.max(s_cp[trim_mask]))
    comparison_summary = {
        "note": "max surplus computed on r >= 0.1 * f_prime_at_0 to exclude "
        "the shared r->0 singularity common to both AMMs",
        "max_surplus_piecewise_amm": max_surplus_piecewise,
        "max_surplus_constant_product_amm": max_surplus_cp,
        "cp_a": cp_a,
        "cp_b": cp_b,
        "cp_f_prime_at_0": cp_amm.f_prime_at_0,
    }

    metrics = {
        "amm_M": float(amm.M),
        "amm_f_prime_at_0": float(amm.f_prime_at_0),
        "proposition_A2": prop_a2_result,
        "lemma_A3_results": lemma_a3_results,
        "kink_concavity_checks": [bool(v) for v in kink_concavity_checks],
        "interior_breakpoints": [float(v) for v in interior_breakpoints],
        "proposition_5_4_comparison_amm": prop_5_4_result,
        "comparison_summary": comparison_summary,
    }

    with open("results/exp8_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ------------------------------------------------------------------
    # 8. Final concise summary.
    # ------------------------------------------------------------------
    all_checks = (
        [prop_a2_result["h_decreasing"], prop_a2_result["surplus_nonnegative"]]
        + [prop_a2_result["strict_where_h_positive"], prop_a2_result["identity_h_of_fprime_holds"]]
        + [res["is_maximizer"] for res in lemma_a3_results]
        + kink_concavity_checks
    )
    print("\n=== Summary ===")
    print(f"All {len(all_checks)} checks passed: {_status(all(all_checks))}")
    print(f"max surplus (piecewise AMM)        = {max_surplus_piecewise:.6f}")
    print(f"max surplus (constant-product AMM) = {max_surplus_cp:.6f}")
    print("Saved results/exp8_piecewise_amm.png and results/exp8_metrics.json")


if __name__ == "__main__":
    main()
