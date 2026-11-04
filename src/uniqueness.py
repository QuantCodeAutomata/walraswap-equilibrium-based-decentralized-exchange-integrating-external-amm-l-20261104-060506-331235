"""Uniqueness properties of equilibrium under WGS (Section 4).

Custom implementation of Definition 4.1 (WGS), Lemma 4.2 (lattice property),
Theorem 4.3 (unique net trades), Theorem 4.4 / Corollary 4.5 (equilibrium set
and its convexity), Proposition 4.6 / Definition 4.7 (WGS test for pairwise
supply functions), and Proposition 4.8 / Corollary 4.9 (price proportionality
under connectedness). These are bespoke equilibrium-theory constructs with no
off-the-shelf library equivalent; networkx is used only for the generic
graph-connectivity check of Corollary 4.9.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.equilibrium import find_brouwer_fixed_point
from src.supply_functions import PairwiseSupplyFunction, SupplyFunction


def is_wgs(
    phi: SupplyFunction,
    price_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-9,
) -> bool:
    """Definition 4.1 (weak gross substitutability): for all p <= q with
    p_i = q_i, phi_i(q) <= phi_i(p). Tested on supplied (p, q) pairs."""
    for p, q in price_pairs:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if not np.all(p <= q + 1e-12):
            raise ValueError("WGS test requires p <= q")
        phi_p, phi_q = phi(p), phi(q)
        equal_idx = np.where(np.isclose(p, q, atol=1e-12))[0]
        if not np.all(phi_q[equal_idx] <= phi_p[equal_idx] + tol):
            return False
    return True


def lattice_min_max(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lemma 4.2: if p, q are equilibria of a WGS supply function, so are
    p AND q = min(p,q) (coordinatewise) and p OR q = max(p,q)."""
    return np.minimum(p, q), np.maximum(p, q)


def compute_u_ij(equilibria: Sequence[np.ndarray], i: int, j: int) -> float:
    """Theorem 4.4: U_ij = sup over equilibria q of q_i / q_j, estimated from
    a (necessarily finite) sample of equilibrium price vectors."""
    ratios = [p[i] / p[j] for p in equilibria]
    return float(max(ratios))


def verify_convexity(
    phi: SupplyFunction,
    equilibria: Sequence[np.ndarray],
    n_combo: int = 10,
    residual_tol: float = 5e-2,
) -> dict:
    """Corollary 4.5: check that convex combinations of two equilibria are
    themselves (approximately) equilibria -- i.e. E_phi is convex -- and that
    the same holds for log E_phi (convex combinations in log-price space)."""
    if len(equilibria) < 2:
        raise ValueError("need at least two equilibria to test convexity")
    p, q = equilibria[0], equilibria[1]
    lambdas = np.linspace(0.0, 1.0, n_combo)
    max_resid_linear = 0.0
    max_resid_log = 0.0
    for lam in lambdas:
        mix = lam * p + (1 - lam) * q
        mix = mix / mix.sum()
        max_resid_linear = max(max_resid_linear, float(np.max(np.abs(phi(mix)))))

        log_mix = np.exp(lam * np.log(p) + (1 - lam) * np.log(q))
        log_mix = log_mix / log_mix.sum()
        max_resid_log = max(max_resid_log, float(np.max(np.abs(phi(log_mix)))))
    return {
        "max_residual_linear_combo": max_resid_linear,
        "max_residual_log_combo": max_resid_log,
        "linear_convex": max_resid_linear < residual_tol,
        "log_convex": max_resid_log < residual_tol,
    }


def find_multiple_equilibria(
    phi: SupplyFunction,
    upper_bounds: Sequence[float],
    n_starts: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Run the Brouwer fixed-point solver from several random starting points
    (Definition 3.1 / Theorem 3.4 construction) and collect the distinct
    equilibria found, used to probe Theorem 4.3 and Theorem 4.4 numerically."""
    n = phi.n
    results = []
    for _ in range(n_starts):
        q0 = rng.dirichlet(np.ones(n))
        out = find_brouwer_fixed_point(phi, upper_bounds, q0=q0)
        if out["converged"]:
            results.append(out["p"])
    # Deduplicate up to normalization / tolerance.
    unique = []
    for p in results:
        p_norm = p / p.sum()
        if not any(np.allclose(p_norm, u / u.sum(), atol=1e-3) for u in unique):
            unique.append(p)
    return unique


def verify_unique_net_trades(
    agents: Sequence[SupplyFunction],
    equilibria: Sequence[np.ndarray],
    tol: float = 5e-2,
) -> dict:
    """Theorem 4.3: for each agent (supply function) phi^a, the net trade
    phi^a(p) is the same across all equilibria p of the aggregate Phi."""
    per_agent_spread = []
    for agent in agents:
        trades = np.array([agent(p) for p in equilibria])
        spread = float(np.max(trades, axis=0).max() - np.min(trades, axis=0).min())
        per_agent_spread.append(spread)
    return {
        "per_agent_spread": per_agent_spread,
        "unique_net_trades": all(s < tol for s in per_agent_spread),
    }


# ---------------------------------------------------------------------------
# Proposition 4.6 / Definition 4.7: WGS test for pairwise supply functions
# ---------------------------------------------------------------------------

def is_g_decreasing(g: Callable[[np.ndarray], np.ndarray], r_grid: np.ndarray, tol: float = 1e-9) -> bool:
    values = g(r_grid)
    return bool(np.all(np.diff(values) <= tol))


def is_g_over_r_decreasing(g: Callable[[np.ndarray], np.ndarray], r_grid: np.ndarray, tol: float = 1e-9) -> bool:
    """Definition 4.7 (WGS function): g(r)/r is also decreasing -- equivalently
    tilde_g is decreasing, by the conjugate relation of Definition 2.5."""
    values = g(r_grid) / r_grid
    return bool(np.all(np.diff(values) <= tol))


def pairwise_satisfies_wgs(pf: PairwiseSupplyFunction, r_grid: np.ndarray) -> dict:
    """Proposition 4.6: a pairwise supply function satisfies WGS iff both g
    and tilde_g are decreasing, i.e. (Definition 4.7) g is decreasing and
    g(r)/r is decreasing."""
    g_decreasing = is_g_decreasing(pf.g, r_grid)
    g_over_r_decreasing = is_g_over_r_decreasing(pf.g, r_grid)
    g_tilde = pf.conjugate().g
    g_tilde_decreasing = is_g_decreasing(g_tilde, r_grid)
    return {
        "g_decreasing": g_decreasing,
        "g_over_r_decreasing": g_over_r_decreasing,
        "g_tilde_decreasing": g_tilde_decreasing,
        "satisfies_wgs": bool(g_decreasing and g_over_r_decreasing),
    }


# ---------------------------------------------------------------------------
# Proposition 4.8 / Corollary 4.9: connectedness and price proportionality
# ---------------------------------------------------------------------------

def trade_graph(n: int, pair_trades: dict[tuple[int, int], float], tol: float = 1e-9) -> "object":
    """Build the undirected trade graph (vertices = tokens, edges = pairs
    with nonzero trade at equilibrium), used to test Corollary 4.9."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for (i, j), value in pair_trades.items():
        if abs(value) > tol:
            g.add_edge(i, j)
    return g


def prices_are_proportional(p: np.ndarray, q: np.ndarray, tol: float = 1e-3) -> bool:
    """Corollary 4.9: two equilibrium price vectors are proportional iff
    p/q is constant across coordinates."""
    ratio = p / q
    return bool(np.max(ratio) - np.min(ratio) < tol * np.mean(ratio))


def verify_connectedness_implies_proportional(
    n: int,
    pair_trades: dict[tuple[int, int], float],
    equilibria: Sequence[np.ndarray],
) -> dict:
    import networkx as nx

    g = trade_graph(n, pair_trades)
    connected = nx.is_connected(g) if g.number_of_nodes() > 0 else False
    proportional = True
    if len(equilibria) >= 2:
        base = equilibria[0]
        proportional = all(prices_are_proportional(base, p) for p in equilibria[1:])
    return {
        "graph_connected": connected,
        "prices_proportional": proportional,
        "corollary_4_9_holds": (not connected) or proportional,
    }
