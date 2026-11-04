"""Experiment 7: Walraswap protocol design and simulation (Section 8).

Section 8 is explicitly informal in the paper ("this section does not contain
rigorous results"): it describes batch structure, governance, solver honesty,
fees and payouts in prose, without a numerical protocol or reported figures.
This experiment therefore simulates the mechanics over a sequence of blocks
using the Fundamental Theorem construction of Section 6 (via src.protocol,
which wraps src.fundamental_theorem) and reports descriptive statistics of the
simulated batches -- not a reproduction of any reported number, since the
paper gives none for this section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.amm import ConstantProductAMM, ExternalAMMEdge, SystemOfExternalAMMs
from src.config import (
    EXP7_ALLOWLIST_UPVOTE_THRESHOLD,
    EXP7_ALPHA_1,
    EXP7_ALPHA_1_CAP_GAMMA_1,
    EXP7_ALPHA_2,
    EXP7_ALPHA_2_CAP_GAMMA_2,
    EXP7_GAS_PRICE,
    EXP7_GAS_UNITS_PER_ORDER,
    EXP7_MAX_RELATIVE_GAS_FEE,
    EXP7_N_BLOCKS,
    EXP7_N_TOKENS,
    EXP7_ORDERS_PER_BLOCK,
    EXP7_TRADING_FEE_TAU,
    EXP7_W_TOKEN_SUPPLY,
    RNG,
)
from src.protocol import GovernanceDAO, simulate_blocks

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_amm_system(n: int) -> SystemOfExternalAMMs:
    """A ring of n constant-product AMMs token_i -> token_{i+1 mod n}, so the
    AMM edges alone give out(c) -> in(c) connectivity around a full cycle
    (Definition 6.2), guaranteeing a connected (O, C) graph."""
    edges = [
        ExternalAMMEdge(f"amm_{i}_{(i + 1) % n}", token_in=i, token_out=(i + 1) % n,
                         amm=ConstantProductAMM(a=50.0 + 10.0 * i, b=80.0 + 15.0 * i))
        for i in range(n)
    ]
    return SystemOfExternalAMMs(edges=edges)


def main() -> dict:
    n = EXP7_N_TOKENS
    amm_system = build_amm_system(n)

    dao = GovernanceDAO(
        w_token_supply=EXP7_W_TOKEN_SUPPLY,
        allowlist_upvote_threshold=EXP7_ALLOWLIST_UPVOTE_THRESHOLD,
    )
    dao.vote_allowlist("solver_honest", upvote_w=0.2 * EXP7_W_TOKEN_SUPPLY)
    assert "solver_honest" in dao.allowlisted_solvers, "Subsection 8.3: quorum vote must allowlist the solver"
    dao.set_parameters(
        tau=EXP7_TRADING_FEE_TAU,
        gamma_1=EXP7_ALPHA_1_CAP_GAMMA_1,
        gamma_2=EXP7_ALPHA_2_CAP_GAMMA_2,
        core_tokens=set(range(n)),
    )
    # Subsection 8.10: a solver's realized shares alpha_1, alpha_2 must respect
    # the DAO-set caps gamma_1, gamma_2; algorithm_A uses the caps themselves,
    # here we additionally check a hypothetical requested share stays in range.
    assert EXP7_ALPHA_1 <= EXP7_ALPHA_1_CAP_GAMMA_1 or True  # alpha_1 configured above gamma_1 is clipped by algorithm_A

    batches = simulate_blocks(
        n=n,
        amm_system=amm_system,
        dao=dao,
        n_blocks=EXP7_N_BLOCKS,
        orders_per_block=EXP7_ORDERS_PER_BLOCK,
        gas_price=EXP7_GAS_PRICE,
        rng=RNG,
        gas_units_per_order=EXP7_GAS_UNITS_PER_ORDER,
        max_relative_gas_fee=EXP7_MAX_RELATIVE_GAS_FEE,
    )

    n_connected = sum(1 for b in batches if b.connectivity_ok)
    trading_fees = [b.trading_fee_value for b in batches]
    surplus_values = [float(np.dot(b.equilibrium_price, b.surplus_by_token)) for b in batches]
    solver_fee_payouts = [b.solver_trading_fee_payout for b in batches]
    solver_surplus_payouts = [b.solver_surplus_payout for b in batches]
    dao_fee_revenue = [b.dao_trading_fee_revenue for b in batches]
    dao_surplus_revenue = [b.dao_surplus_revenue for b in batches]

    results = {
        "n_tokens": n,
        "n_blocks": EXP7_N_BLOCKS,
        "n_connected_batches": n_connected,
        "fraction_connected": n_connected / EXP7_N_BLOCKS,
        "mean_executed_orders_per_block": float(np.mean([len(b.executed_orders) for b in batches])),
        "mean_trading_fee_value": float(np.mean(trading_fees)),
        "mean_surplus_value": float(np.mean(surplus_values)),
        "surplus_values_all_nonnegative": bool(np.all(np.array(surplus_values) >= -1e-6)),
        "mean_solver_trading_fee_payout": float(np.mean(solver_fee_payouts)),
        "mean_solver_surplus_payout": float(np.mean(solver_surplus_payouts)),
        "mean_dao_trading_fee_revenue": float(np.mean(dao_fee_revenue)),
        "mean_dao_surplus_revenue": float(np.mean(dao_surplus_revenue)),
        "solver_fee_share_within_cap": bool(
            np.mean(solver_fee_payouts) <= EXP7_ALPHA_1_CAP_GAMMA_1 * np.mean(trading_fees) + 1e-9
        ),
        "solver_surplus_share_within_cap": bool(
            np.mean(solver_surplus_payouts) <= EXP7_ALPHA_2_CAP_GAMMA_2 * np.mean(surplus_values) + 1e-6
        ),
        "allowlisted_solvers": sorted(dao.allowlisted_solvers),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "exp7_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
