"""Batch exchange with external AMMs: the Fundamental Theorem (Section 6).

Custom implementation of Definition 6.1 (trade orders), Definition 6.2
(connectivity of (O, C)), Remark 6.3 (strictness of the aggregate),
Theorem 6.4 (clearing-price existence with nonnegative surplus) and
Remark 6.5 (optimality of the prescribed external swaps). Builds directly on
src.supply_functions, src.equilibrium and src.amm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.amm import SystemOfExternalAMMs
from src.equilibrium import find_brouwer_fixed_point
from src.supply_functions import ExtendedSupplyFunction, SumSupplyFunction, SupplyFunction


@dataclass
class TradeOrderEdge:
    """A trade order, viewed as a pairwise supply function on tokens (i, j),
    together with the directed connectivity edges it contributes
    (Definition 6.2 / Proposition 3.6)."""

    name: str
    token_i: int
    token_j: int
    supply_function: "object"  # src.supply_functions.PairwiseSupplyFunction


def build_connectivity_graph(
    n: int,
    orders: list[TradeOrderEdge],
    amm_system: SystemOfExternalAMMs,
) -> set[tuple[int, int]]:
    """Definition 6.2: the directed graph with an edge i->j whenever some
    trade order connects i->j (Proposition 3.6), or whenever some external
    AMM has in(c)=i, out(c)=j (an AMM's virtual agent G_c always connects
    out->in by Proposition 5.4(5), i.e. in token-index terms in(c) -> out(c)
    is NOT guaranteed, but out(c) -> in(c) is: G_c(r) is decreasing, Prop. 3.6
    with r = p_in/p_out, so it connects 2->1 in the local pair (in=1,out=2),
    i.e. out(c) -> in(c) in global indices)."""
    edges: set[tuple[int, int]] = set()
    for order in orders:
        pf = order.supply_function
        if pf.connects_2_to_1():
            edges.add((order.token_j, order.token_i))
        if pf.connects_1_to_2():
            edges.add((order.token_i, order.token_j))
    for e in amm_system.edges:
        # G_c connects out(c) -> in(c) (Proposition 5.4(5)): a positive price
        # of token in(c) can be sustained if token out(c) has positive price.
        edges.add((e.token_out, e.token_in))
    return edges


def is_connected(n: int, edges: set[tuple[int, int]]) -> bool:
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return nx.is_strongly_connected(g)


def aggregate_supply_function(
    n: int,
    orders: list[TradeOrderEdge],
    amm_system: SystemOfExternalAMMs,
) -> SupplyFunction:
    """Phi = phi^O + phi^C (Remark 6.3): the sum of all trade orders' supply
    functions and the aggregate virtual agent of the external AMM system."""
    components = [ExtendedSupplyFunction(o.supply_function, sigma=[o.token_i, o.token_j], n=n) for o in orders]
    components.append(amm_system.virtual_agent_supply_function(n))
    return SumSupplyFunction(components)


def trade_orders_supply_function(n: int, orders: list[TradeOrderEdge]) -> SupplyFunction:
    components = [ExtendedSupplyFunction(o.supply_function, sigma=[o.token_i, o.token_j], n=n) for o in orders]
    return SumSupplyFunction(components)


def upper_bounds_for_system(n: int, orders: list[TradeOrderEdge], amm_system: SystemOfExternalAMMs) -> np.ndarray:
    """An upper bound M_i for Phi_i, needed by the Brouwer map (Theorem 3.4):
    sum of each component's own upper bound on coordinate i. Trade orders
    built from bounded g (sell orders bounded by b, linear g_{a,b} bounded by
    b) contribute their b; G_f-based virtual agents are bounded by amm.M."""
    M = np.zeros(n)
    for o in orders:
        # phi_1(p) = g(p_j/p_i) is bounded above by bound_phi1 (numerical sup
        # of g) and phi_2(p) = tilde_g(p_i/p_j) by bound_phi2 = sup tilde_g
        # (Definition 2.5, Proposition 2.4), so each side of the pair gets
        # its own bound rather than an arbitrary numerical slack.
        M[o.token_i] += o.supply_function.bound_phi1() + 1e-9
        M[o.token_j] += o.supply_function.bound_phi2() + 1e-9
    from src.amm import G_f_supply_function

    for e in amm_system.edges:
        # G_c is decreasing with sup G_c(r) = M_c as r->0 (Proposition 5.4(1));
        # under the local axis order [out(c), in(c)] used by
        # SystemOfExternalAMMs.virtual_agent_supply_function, phi_local1 =
        # G_c(r) bounds phi_{out(c)} and phi_local2 = tilde(G_c)(r) bounds
        # phi_{in(c)} (numerical sup, as for trade orders above).
        pf = G_f_supply_function(e.amm)
        M[e.token_out] += pf.bound_phi1() + 1e-9
        M[e.token_in] += pf.bound_phi2() + 1e-9
    return M


def find_equilibrium(
    n: int,
    orders: list[TradeOrderEdge],
    amm_system: SystemOfExternalAMMs,
    q0: np.ndarray | None = None,
) -> dict:
    Phi = aggregate_supply_function(n, orders, amm_system)
    M = upper_bounds_for_system(n, orders, amm_system)
    return find_brouwer_fixed_point(Phi, M, q0=q0)


def verify_theorem_6_4(
    n: int,
    orders: list[TradeOrderEdge],
    amm_system: SystemOfExternalAMMs,
    p: np.ndarray,
) -> dict:
    """Theorem 6.4(a)-(b): at the clearing price p, for each token i the
    balance (phi^O(p) + F^C(h^{C,p}))_i equals sum of surpluses of AMMs with
    in(c)=i, and each surplus is nonnegative, strictly positive iff the price
    ratio r_c is below f_c'(0)."""
    phi_O = trade_orders_supply_function(n, orders)
    swaps = amm_system.external_swaps(p)
    F_C = amm_system.F_C(swaps)
    balance = phi_O(p) + F_C

    surplus_by_token = np.zeros(n)
    per_amm_surplus = amm_system.surplus_per_amm(p)
    strict_positive = {}
    for e in amm_system.edges:
        s = per_amm_surplus[e.name]
        surplus_by_token[e.token_in] += s
        r_c = p[e.token_in] / p[e.token_out]
        strict_positive[e.name] = bool(r_c < e.amm.f_prime_at_0)

    balance_matches = bool(np.allclose(balance, surplus_by_token, atol=1e-3))
    all_nonneg = bool(np.all(np.array(list(per_amm_surplus.values())) >= -1e-9))
    strict_matches_sign = {
        name: (per_amm_surplus[name] > 1e-9) == strict_positive[name] for name in per_amm_surplus
    }
    return {
        "balance": balance,
        "surplus_by_token": surplus_by_token,
        "balance_matches_surplus": balance_matches,
        "surplus_nonnegative": all_nonneg,
        "per_amm_surplus": per_amm_surplus,
        "strict_positive_expected": strict_positive,
        "strict_matches_sign": strict_matches_sign,
    }


def verify_remark_6_5(
    n: int,
    orders: list[TradeOrderEdge],
    amm_system: SystemOfExternalAMMs,
    p: np.ndarray,
    n_probe: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Remark 6.5: h^{C,p} maximizes p . (phi^O(p) + F^C(x)) over external
    swap vectors x >= 0. Since phi^O(p) does not depend on x, this reduces to
    maximizing p . F^C(x), which decouples per AMM (Proposition 5.8)."""
    if rng is None:
        rng = np.random.default_rng(1)
    swaps = amm_system.external_swaps(p)
    F_C_star = amm_system.F_C(swaps)
    value_star = float(np.dot(p, F_C_star))

    values_probe = []
    for _ in range(n_probe):
        probe_swaps = {name: max(0.0, v + rng.normal(scale=max(v, 1.0) * 0.3)) for name, v in swaps.items()}
        F_C_probe = amm_system.F_C(probe_swaps)
        values_probe.append(float(np.dot(p, F_C_probe)))

    return {
        "value_at_h_Cp": value_star,
        "max_probe_value": max(values_probe),
        "is_maximizer": bool(value_star >= max(values_probe) - 1e-6),
    }
