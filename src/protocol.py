"""Walraswap protocol simulation (Section 8).

Section 8 is explicitly described in the paper as informal protocol design,
not a set of theorems ("this section does not contain rigorous results").
This module simulates its mechanics -- batch construction via Theorem 6.4,
governance allowlisting, trading and gas fees, and solver payout -- as a
discrete-event system; it is a custom simulation, not a reproduction of a
numerical result, since the paper gives none for this section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.amm import ConstantProductAMM, ExternalAMMEdge, SystemOfExternalAMMs
from src.fundamental_theorem import (
    TradeOrderEdge,
    build_connectivity_graph,
    find_equilibrium,
    is_connected,
    verify_theorem_6_4,
)
from src.supply_functions import continuous_limit_sell_order


@dataclass
class GovernanceDAO:
    """Subsection 8.2/8.3: W token holders vote on the solver allowlist and
    protocol parameters (tau, gamma_1, gamma_2, core tokens)."""

    w_token_supply: float
    allowlist_upvote_threshold: float
    core_tokens: set[int] = field(default_factory=set)
    allowlisted_solvers: set[str] = field(default_factory=set)
    tau: float = 0.0
    gamma_1: float = 0.0
    gamma_2: float = 0.0

    def vote_allowlist(self, solver: str, upvote_w: float) -> bool:
        """A solver is allowlisted if its upvotes exceed the quorum fraction
        of total W supply (Subsection 8.3; the paper leaves the exact voting
        mechanism dispensable, so a simple quorum rule is used here)."""
        approved = upvote_w / self.w_token_supply >= self.allowlist_upvote_threshold
        if approved:
            self.allowlisted_solvers.add(solver)
        return approved

    def set_parameters(self, tau: float, gamma_1: float, gamma_2: float, core_tokens: set[int]) -> None:
        self.tau, self.gamma_1, self.gamma_2 = tau, gamma_1, gamma_2
        self.core_tokens = set(core_tokens)


@dataclass
class TradeOrder:
    """A trade order with a gas-fee specification (Subsection 8.8), on top of
    its pairwise WGS supply function (Definition 6.1)."""

    name: str
    token_i: int
    token_j: int
    supply_function: object
    gas_units: float
    max_relative_gas_fee: float
    value_of_sold_tokens: float  # used only to cap the gas fee, Subsection 8.8


def solver_select_viable_orders(
    orders: list[TradeOrder], gas_price: float
) -> tuple[list[TradeOrder], dict[str, float]]:
    """Subsection 8.8: the solver charges min(gas_units * gas_price,
    max_relative_gas_fee * value_of_sold_tokens) per order, and only includes
    an order in O' if this charge does not exceed its relative cap (i.e. the
    tabulated gas cost is within the order's own bound -- otherwise the order
    is not viable and is dropped from the batch)."""
    viable = []
    fees = {}
    for order in orders:
        tabulated_cost = order.gas_units * gas_price
        cap = order.max_relative_gas_fee * order.value_of_sold_tokens
        if tabulated_cost <= cap:
            viable.append(order)
            fees[order.name] = tabulated_cost
    return viable, fees


@dataclass
class Batch:
    block: int
    equilibrium_price: np.ndarray
    executed_orders: list[str]
    external_swaps: dict[str, float]
    surplus_by_token: np.ndarray
    trading_fee_value: float
    gas_fees: dict[str, float]
    solver_trading_fee_payout: float
    solver_surplus_payout: float
    dao_trading_fee_revenue: float
    dao_surplus_revenue: float
    connectivity_ok: bool


def algorithm_A(
    n: int,
    orders: list[TradeOrder],
    amm_system: SystemOfExternalAMMs,
    gas_price: float,
    dao: GovernanceDAO,
    block: int,
) -> Batch:
    """The deterministic batch-construction algorithm an honest solver must
    apply (Subsection 8.4): (1) collect disclosed orders, (2) drop orders
    whose gas fee exceeds their cap, (3) check connectivity (Definition 6.2),
    (4) compute the clearing price via Theorem 6.4, (5) compute external
    swaps h^{C,p}, (6) compute trading fees, gas fees and surplus, (7) split
    the solver payout under the DAO-set caps gamma_1, gamma_2 (Sec. 8.9-8.10).
    """
    viable_orders, gas_fees = solver_select_viable_orders(orders, gas_price)
    trade_edges = [TradeOrderEdge(o.name, o.token_i, o.token_j, o.supply_function) for o in viable_orders]

    edges = build_connectivity_graph(n, trade_edges, amm_system)
    connectivity_ok = is_connected(n, edges)

    result = find_equilibrium(n, trade_edges, amm_system)
    p = result["p"]

    verification = verify_theorem_6_4(n, trade_edges, amm_system, p)
    surplus_by_token = verification["surplus_by_token"]

    trading_fee_value = dao.tau * sum(o.value_of_sold_tokens for o in viable_orders)
    total_surplus_value = float(np.dot(p, surplus_by_token))

    alpha_1 = min(dao.gamma_1, dao.gamma_1)  # realized share capped at gamma_1 by construction
    alpha_2 = min(dao.gamma_2, dao.gamma_2)
    solver_trading_fee_payout = alpha_1 * trading_fee_value
    solver_surplus_payout = alpha_2 * total_surplus_value
    dao_trading_fee_revenue = trading_fee_value - solver_trading_fee_payout
    dao_surplus_revenue = total_surplus_value - solver_surplus_payout

    return Batch(
        block=block,
        equilibrium_price=p,
        executed_orders=[o.name for o in viable_orders],
        external_swaps=amm_system.external_swaps(p),
        surplus_by_token=surplus_by_token,
        trading_fee_value=trading_fee_value,
        gas_fees=gas_fees,
        solver_trading_fee_payout=solver_trading_fee_payout,
        solver_surplus_payout=solver_surplus_payout,
        dao_trading_fee_revenue=dao_trading_fee_revenue,
        dao_surplus_revenue=dao_surplus_revenue,
        connectivity_ok=connectivity_ok,
    )


def simulate_blocks(
    n: int,
    amm_system: SystemOfExternalAMMs,
    dao: GovernanceDAO,
    n_blocks: int,
    orders_per_block: int,
    gas_price: float,
    rng: np.random.Generator,
) -> list[Batch]:
    """Simulate ``n_blocks`` batches with randomly disclosed trade orders
    (Subsections 8.1.1, 8.4), each solved by algorithm_A."""
    batches = []
    for block in range(n_blocks):
        orders = []
        for k in range(orders_per_block):
            i, j = rng.choice(n, size=2, replace=False)
            b = float(rng.uniform(1.0, 5.0))
            r1, r2 = 0.3, 2.0
            value = float(rng.uniform(10.0, 100.0))
            orders.append(
                TradeOrder(
                    name=f"blk{block}_ord{k}",
                    token_i=int(i),
                    token_j=int(j),
                    supply_function=continuous_limit_sell_order(b, r1, r2),
                    gas_units=150000.0,
                    max_relative_gas_fee=0.02,
                    value_of_sold_tokens=value,
                )
            )
        batches.append(algorithm_A(n, orders, amm_system, gas_price, dao, block))
    return batches
