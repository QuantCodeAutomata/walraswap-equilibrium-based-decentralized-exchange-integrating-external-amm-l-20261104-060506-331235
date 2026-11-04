"""Experiment 5: Fundamental Theorem of Walraswap (Section 6, Theorem 6.4).

Builds a 3-token system of admissible trade orders (continuous limit sell
orders, Examples 2.6(IV)) together with a system of external constant-product
AMMs (Section 5), checks the connectivity condition (Definition 6.2), finds
the clearing-price vector via the Brouwer construction of Theorem 3.4 applied
to the aggregate supply function Phi = phi^O + phi^C (Remark 6.3), and
verifies Theorem 6.4(a)-(b) and Remark 6.5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.amm import ConstantProductAMM, ExternalAMMEdge, SystemOfExternalAMMs
from src.config import (
    EXP5_AMM_RESERVES,
    EXP5_N_TOKENS,
    EXP5_TRADE_ORDER_LINEAR_PARAMS,
    RNG,
)
from src.fundamental_theorem import (
    TradeOrderEdge,
    build_connectivity_graph,
    find_equilibrium,
    is_connected,
    verify_remark_6_5,
    verify_theorem_6_4,
)
from src.supply_functions import linear_pairwise_supply_function

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_system() -> tuple[list[TradeOrderEdge], SystemOfExternalAMMs]:
    """3-token example (Section 6): token-pair AMMs (0,1) and (1,2), plus
    linear pairwise trade orders (Examples 2.6(I)) connecting every pair of
    tokens in both directions, so that (O, C) is strongly connected
    (Definition 6.2)."""
    orders = [
        TradeOrderEdge(f"order_{i}_{j}", i, j, linear_pairwise_supply_function(*params))
        for (i, j), params in EXP5_TRADE_ORDER_LINEAR_PARAMS.items()
    ]
    amm_edges = [
        ExternalAMMEdge(f"amm_{i}_{j}", token_in=i, token_out=j, amm=ConstantProductAMM(a=a, b=b))
        for (i, j), (a, b) in EXP5_AMM_RESERVES.items()
    ]
    return orders, SystemOfExternalAMMs(edges=amm_edges)


def main() -> dict:
    n = EXP5_N_TOKENS
    orders, amm_system = build_system()

    edges = build_connectivity_graph(n, orders, amm_system)
    connected = is_connected(n, edges)
    assert connected, "Definition 6.2: the (O, C) graph must be strongly connected for the test to be meaningful"

    eq = find_equilibrium(n, orders, amm_system, q0=np.full(n, 1.0 / n))
    p = eq["p"]

    theorem_6_4 = verify_theorem_6_4(n, orders, amm_system, p)
    remark_6_5 = verify_remark_6_5(n, orders, amm_system, p, rng=RNG)

    results = {
        "n_tokens": n,
        "connectivity_edges": sorted(edges),
        "definition_6_2_connected": bool(connected),
        "equilibrium_price": p.tolist(),
        "fixed_point_residual": eq["residual"],
        "balance_matches_surplus": theorem_6_4["balance_matches_surplus"],
        "surplus_nonnegative": theorem_6_4["surplus_nonnegative"],
        "per_amm_surplus": theorem_6_4["per_amm_surplus"],
        "strict_matches_sign": theorem_6_4["strict_matches_sign"],
        "remark_6_5_is_maximizer": remark_6_5["is_maximizer"],
        "remark_6_5_value_at_h_Cp": remark_6_5["value_at_h_Cp"],
        "remark_6_5_max_probe_value": remark_6_5["max_probe_value"],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp5_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
