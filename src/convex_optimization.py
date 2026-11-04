"""Convex-optimization formulation of equilibrium computation (Section 7).

Custom implementation of Lemma 7.1 (H, Z and the convex perspective F),
Lemma 7.2 (the incidence matrix B and the constraint set S_n) and Theorem 7.3
(the convex program whose zero-objective minimizers are exactly the
equilibria). cvxpy is used as the convex-optimization *solver* once the
paper's objective and constraints are expressed exactly as written -- a
library default optimizer is not a substitute for the paper's H_ij/Z_ij
formulas, which are implemented from the closed form for linear g.
"""

from __future__ import annotations

from typing import Callable

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq

from src.config import EXP6_SIMPLEX_EPS, EXP6_SOLVER


def H_linear(a: float, b: float, r: np.ndarray) -> np.ndarray:
    """H(r) = a r - b ln(r) - a for g_{a,b}(r) = -a r + b (Lemma 7.1),
    obtained analytically from H(r) = -int_1^r g(s)/s ds."""
    r = np.asarray(r, dtype=float)
    return a * r - b * np.log(r) - a


def Z_linear(a: float, b: float, t: float) -> float:
    """Z(t) = sup_{u>0} {-t ln(u) - H(u)}, attained at u* with g(u*) = t
    (Lemma 7.1.2), i.e. u* = (b - t)/a, valid for t < b.

    Substituting u* into -t ln(u) - H(u) with H(r) = a r - b ln r - a gives
    Z(t) = (b - t) ln((b - t)/a) - (b - t) + a   for t < b
    (verified numerically against the direct supremum over u).
    """
    if t >= b:
        return float("inf")
    return float((b - t) * np.log((b - t) / a) - (b - t) + a)


def H_general(g: Callable[[np.ndarray], np.ndarray], r: np.ndarray, n_grid: int = 2000) -> np.ndarray:
    """H(r) = -int_1^r g(s)/s ds, evaluated by numerical quadrature for a
    general (non-linear) WGS function g (Lemma 7.1)."""
    from scipy.integrate import quad

    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.zeros_like(r)
    for idx, rv in enumerate(r):
        val, _ = quad(lambda s: g(np.array([s]))[0] / s, 1.0, rv, limit=200)
        out[idx] = -val
    return out


def Z_general(g: Callable[[np.ndarray], np.ndarray], t: float, u_lo: float = 1e-8, u_hi: float = 1e6) -> float:
    """Z(t) = sup_{u>0} {-t ln u - H(u)}; the maximizer solves g(u) = t
    (Lemma 7.1.2) since d/du[-t ln u - H(u)] = -t/u + g(u)/u = 0."""

    def root_fn(u):
        return float(g(np.array([u]))[0]) - t

    lo, hi = u_lo, u_hi
    f_lo, f_hi = root_fn(lo), root_fn(hi)
    if f_lo * f_hi > 0:
        # g may be monotone and never cross t within [lo, hi]: fall back to a
        # coarse grid search for the supremum directly.
        grid = np.geomspace(u_lo, u_hi, 4000)
        H_vals = H_general(g, grid)
        obj = -t * np.log(grid) - H_vals
        return float(np.max(obj))
    u_star = brentq(root_fn, lo, hi)
    H_star = H_general(g, np.array([u_star]))[0]
    return float(-t * np.log(u_star) - H_star)


def incidence_matrix_B(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Lemma 7.2: B has one column per ordered pair (i, j) with B_k^{ij} =
    delta_{ik} - delta_{jk}, so that y in S_n = {y : B y = 0} iff y_ij
    aggregates consistently across tokens (a discrete 'no money creation'
    constraint used to couple the per-pair variables y_ij)."""
    B = np.zeros((n, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        B[i, col] += 1.0
        B[j, col] -= 1.0
    return B


def solve_equilibrium_convex_program(
    n: int,
    pairs: list[tuple[int, int]],
    linear_params: dict[tuple[int, int], tuple[float, float]],
    solver: str = EXP6_SOLVER,
    eps: float = EXP6_SIMPLEX_EPS,
) -> dict:
    """Theorem 7.3: minimize F(p, y) = sum_{ij} [p_i H_ij(p_j/p_i) +
    p_i Z_ij(y_ij / p_i)] over p in the (open) simplex and y in S_n, using the
    closed-form H, Z of linear g_{a,b} (Lemma 7.1) and the matrix B of
    Lemma 7.2. The perspective p_i H_ij(p_j/p_i) is cvxpy's ``perspective``
    of the convex function H_ij, likewise for Z_ij.
    """
    p = cp.Variable(n, pos=True)
    y = cp.Variable(len(pairs))

    terms = []
    for col, (i, j) in enumerate(pairs):
        a, b = linear_params[(i, j)]
        # H_{ij}(r) = a r - b ln r - a is convex on r>0 (it is -H of a
        # concave-minus-log function); its perspective is
        # p_i H_ij(p_j/p_i) = a p_j - b p_i ln(p_j/p_i) - a p_i
        #                   = a p_j - b (p_i ln p_j - p_i ln p_i) - a p_i.
        # cvxpy's rel_entr(x, y) = x log(x/y) handles the perspective of
        # -ln directly: p_i ln(p_i/p_j) = rel_entr(p_i, p_j).
        term_H = a * p[j] - a * p[i] + b * cp.rel_entr(p[i], p[j])
        terms.append(term_H)

        # Z_{ij}(t) = (b - t) ln((b - t)/a) - (b - t) + a for t < b (see
        # Z_linear above). Its perspective p_i Z_ij(y_ij / p_i), with
        # s = p_i (b - y_ij/p_i) = b p_i - y_ij >= 0 at a feasible point:
        #   p_i Z(y_ij/p_i) = s ln(s/(a p_i)) - s + a p_i = rel_entr(s, a p_i) - s + a p_i.
        s = b * p[i] - y[col]
        term_Z = cp.rel_entr(s, a * p[i]) - s + a * p[i]
        terms.append(term_Z)

    objective = cp.Minimize(cp.sum(terms))
    B = incidence_matrix_B(n, pairs)
    constraints = [cp.sum(p) == 1, p >= eps, B @ y == 0]
    for col, (i, j) in enumerate(pairs):
        a, b = linear_params[(i, j)]
        constraints.append(y[col] <= b * p[i] - eps * eps)
        constraints.append(y[col] >= -1e6)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    p_opt = p.value
    y_opt = y.value
    return {
        "status": problem.status,
        "objective_value": float(problem.value),
        "p": p_opt,
        "y": y_opt,
    }


def implied_Phi(n: int, pairs: list[tuple[int, int]], linear_params: dict, p: np.ndarray) -> np.ndarray:
    """Phi_i(p) implied by summing g^{ij}(p_j/p_i) * sign convention
    (Equation (1)), used to check that the convex program's optimal p
    satisfies Phi(p) = 0 (Theorem 7.3(2))."""
    Phi = np.zeros(n)
    for (i, j) in pairs:
        a, b = linear_params[(i, j)]
        r = p[j] / p[i]
        g_ij = -a * r + b
        Phi[i] += g_ij
        Phi[j] += -(p[i] / p[j]) * g_ij
    return Phi
