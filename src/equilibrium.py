"""Existence of equilibrium via Brouwer's fixed point theorem (Section 3).

Custom implementation of the proof of Theorem 3.4: no optimization library
implements this specific construction, so the Brouwer map rho is coded
directly from the paper and solved with a damped fixed-point iteration
(scipy offers no ready-made Brouwer-fixed-point routine; scipy.optimize is
used only as a generic root-finding fallback in tests).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.config import EXP2_BROUWER_DAMPING, EXP2_BROUWER_MAX_ITER, EXP2_BROUWER_TOL
from src.supply_functions import SupplyFunction


def simplex_project(q: np.ndarray) -> np.ndarray:
    """Project a nonnegative vector onto the simplex {q: q_i>=0, sum q_i=1}
    by renormalization (used only to repair rounding drift, not as a
    Euclidean projection)."""
    q = np.clip(q, 0.0, None)
    total = q.sum()
    if total <= 0:
        raise ValueError("cannot renormalize a zero vector onto the simplex")
    return q / total


def brouwer_map(phi: SupplyFunction, upper_bounds: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """rho_i(q) = q_i * (1 - (1/M_i) * phi_i(q/M)), from the proof of Theorem 3.4.

    ``upper_bounds`` M_i are the upper bounds of phi_i from Definition 2.1(2)
    (M_i >= phi_i(p) for every p in P_n).
    """
    M = np.asarray(upper_bounds, dtype=float)
    if np.any(M <= 0):
        raise ValueError("upper bounds M_i must be strictly positive")

    def rho(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        p = q / M
        # On the simplex boundary (q_i = 0) phi may be undefined (p_i = 0 is not
        # in P_n); the paper extends rho continuously to Delta (paper, proof of
        # Theorem 3.4). We approximate that continuous extension by evaluating
        # phi slightly inside the simplex when a coordinate is (numerically) zero.
        eps = 1e-12
        p_eval = np.where(p <= eps, eps, p)
        phi_val = phi(p_eval)
        out = q * (1.0 - phi_val / M)
        return out

    return rho


def find_brouwer_fixed_point(
    phi: SupplyFunction,
    upper_bounds: Sequence[float],
    q0: np.ndarray | None = None,
    max_iter: int = EXP2_BROUWER_MAX_ITER,
    damping: float = EXP2_BROUWER_DAMPING,
    tol: float = EXP2_BROUWER_TOL,
) -> dict:
    """Damped fixed-point iteration q_{k+1} = (1-damping) q_k + damping * rho(q_k),
    renormalized onto the simplex at every step (rho already preserves the sum
    exactly by Walras's law; renormalization only removes floating-point drift).

    Returns a dict with the fixed point ``q``, the equilibrium price vector
    ``p = q / M``, the number of iterations, and the residual ``Phi(p)``.
    """
    n = phi.n
    M = np.asarray(upper_bounds, dtype=float)
    rho = brouwer_map(phi, M)
    q = np.full(n, 1.0 / n) if q0 is None else simplex_project(np.asarray(q0, dtype=float))

    for k in range(max_iter):
        q_new = (1 - damping) * q + damping * rho(q)
        q_new = simplex_project(q_new)
        if np.max(np.abs(q_new - q)) < tol:
            q = q_new
            break
        q = q_new

    p = q / M
    p, residual = _polish_equilibrium(phi, p)
    return {
        "q": q,
        "p": p,
        "iterations": k + 1,
        "residual": residual,
        "residual_norm": float(np.max(np.abs(residual))),
        "converged": bool(np.max(np.abs(residual)) < 1e-6),
    }


def _polish_equilibrium(phi: SupplyFunction, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refine the damped fixed-point iterate into an accurate equilibrium.

    The map rho of the proof of Theorem 3.4 only certifies *existence* of a
    fixed point; the damped iteration above converges slowly (or stalls) once
    phi has steep regions (e.g. an AMM's G_c near its f'(0) kink), so its own
    residual can stay far from zero. By homogeneity (Definition 2.1(0)) and
    Walras's law (Definition 2.1(1), sum_i p_i phi_i(p) = 0), phi(p) = 0 is an
    (n-1)-dimensional system: fixing the scale p_0 = 1 and solving
    phi_1(p) = ... = phi_{n-1}(p) = 0 for p_1, ..., p_{n-1} automatically gives
    phi_0(p) = 0 too. This normalized system is solved with a standard
    root-finder (scipy.optimize.root) starting from the fixed-point iterate,
    which is far more robust numerically than the Brouwer map itself.
    """
    from scipy.optimize import root

    n = phi.n
    if n == 1:
        return p0, phi(p0)
    p0 = np.asarray(p0, dtype=float)
    scale = p0[0]
    x0 = p0[1:] / scale

    def residual_reduced(x: np.ndarray) -> np.ndarray:
        p = np.concatenate(([1.0], x))
        if np.any(p <= 0):
            return np.full(n - 1, 1e6)
        return phi(p)[1:]

    sol = root(residual_reduced, x0, method="hybr", tol=1e-12)
    p_candidate = np.concatenate(([1.0], sol.x))
    if np.all(p_candidate > 0):
        residual_candidate = phi(p_candidate)
        if np.max(np.abs(residual_candidate)) < np.max(np.abs(phi(p0))):
            return p_candidate / p_candidate.sum(), residual_candidate
    return p0, phi(p0)


# ---------------------------------------------------------------------------
# Connectivity (Definition 3.1) and Propositions 3.5, 3.6
# ---------------------------------------------------------------------------

def connectivity_graph_generic(
    phi: SupplyFunction,
    boundary_points: dict[tuple[int, int], np.ndarray],
    tol: float = 1e-9,
) -> set[tuple[int, int]]:
    """Definition 3.1: phi connects i -> j iff psi_i(p) < 0 whenever p_i = 0
    and p_j > 0. ``boundary_points`` maps (i, j) to a boundary price vector p
    with p_i = 0, p_j > 0 (and other coordinates > 0), used to test the edge.
    """
    edges = set()
    for (i, j), p in boundary_points.items():
        p = np.asarray(p, dtype=float)
        if not (p[i] == 0 and p[j] > 0):
            raise ValueError("boundary point must have p_i = 0 and p_j > 0")
        psi_i = phi(p)[i] * 1.0  # phi_i(p) itself; psi_i(p) = phi_i(p) p_i = 0 at the boundary,
        # Definition 3.1 in fact requires the *limiting* sign of phi_i as p_i -> 0+,
        # which we approximate with a small positive perturbation below.
        eps = 1e-9
        p_pert = p.copy()
        p_pert[i] = eps
        phi_i_pert = phi(p_pert)[i]
        if phi_i_pert < -tol:
            edges.add((i, j))
    return edges


def pairwise_connects(pf, eps: float = 1e-6) -> dict[str, bool]:
    """Proposition 3.6 applied to a PairwiseSupplyFunction pf acting on tokens
    (1, 2): returns whether phi connects 2->1 and 1->2."""
    return {
        "connects_2_to_1": pf.connects_2_to_1(eps),
        "connects_1_to_2": pf.connects_1_to_2(eps),
    }


def is_strongly_connected(n: int, edges: set[tuple[int, int]]) -> bool:
    """Whether the directed graph (Definition 3.1) with vertex set {0,...,n-1}
    and the given edges is strongly connected -- a sufficient condition (used
    in Theorem 3.4's remark) under which a strict aggregate supply function
    can be built by summing pairwise connectors."""
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return nx.is_strongly_connected(g)
