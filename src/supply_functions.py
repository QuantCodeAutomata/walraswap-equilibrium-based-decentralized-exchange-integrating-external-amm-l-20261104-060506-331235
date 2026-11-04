"""Supply functions and their pairwise characterization (paper Section 2).

Custom implementation of Definition 2.1, Proposition 2.2, Proposition 2.4,
Definition 2.5, Examples 2.6 and 2.8, and Lemma 2.11 / Proposition 2.10.
No library implements this bespoke mechanism-design formalism, so every
construction here follows the paper's formulas directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.config import TOL_DEFAULT, TOL_LIMIT

ArrayLike = np.ndarray


def _as_price_vector(p: ArrayLike) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("price vector p must be one-dimensional")
    if np.any(p <= 0):
        raise ValueError("price vector p must lie in P_n = (R_>0)^n")
    return p


class SupplyFunction:
    """A supply function P_n -> R^n (Definition 2.1).

    Subclasses implement ``__call__(p) -> phi(p)``. This base class provides
    numerical checks of conditions (0)-(3) and of Walras's law / homogeneity,
    used throughout exp_1 and exp_2.
    """

    n: int

    def __call__(self, p: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def psi(self, p: ArrayLike) -> np.ndarray:
        """psi_i(p) = phi_i(p) * p_i, the net trade *value* vector (Def. 2.1(3))."""
        p = _as_price_vector(p)
        return self(p) * p

    # -- Definition 2.1 condition checks -----------------------------------
    def check_homogeneity(self, p: ArrayLike, lam: float, tol: float = TOL_DEFAULT) -> bool:
        """Condition (0): phi(lambda p) = phi(p) for all lambda > 0."""
        p = _as_price_vector(p)
        return bool(np.allclose(self(lam * p), self(p), atol=tol))

    def check_walras_law(self, p: ArrayLike, tol: float = TOL_DEFAULT) -> bool:
        """Condition (1): phi(p) . p = 0."""
        p = _as_price_vector(p)
        return bool(abs(np.dot(self(p), p)) < tol)

    def check_bounded_above(self, sample_prices: Sequence[ArrayLike], bound: float) -> bool:
        """Condition (2): each phi_i is bounded above by ``bound`` on the sample."""
        values = np.array([self(p) for p in sample_prices])
        return bool(np.all(values <= bound + 1e-9))

    def verify_conditions(
        self,
        sample_prices: Sequence[ArrayLike],
        bound: float,
        tol: float = TOL_DEFAULT,
    ) -> dict:
        """Numerically verify conditions (0) and (1) on a batch of price vectors,
        and condition (2) with the given upper bound. Condition (3) (boundary
        regularity) is checked separately per construction, since it requires a
        continuous extension to the boundary that is construction-specific.
        """
        homog_ok = all(self.check_homogeneity(p, 2.7, tol) for p in sample_prices)
        walras_ok = all(self.check_walras_law(p, tol) for p in sample_prices)
        bounded_ok = self.check_bounded_above(sample_prices, bound)
        return {"homogeneity": homog_ok, "walras_law": walras_ok, "bounded_above": bounded_ok}


class SumSupplyFunction(SupplyFunction):
    """Proposition 2.2: the sum of two (or more) supply functions is a supply function."""

    def __init__(self, components: Sequence[SupplyFunction]):
        if not components:
            raise ValueError("need at least one component supply function")
        self.components = list(components)
        self.n = components[0].n
        if any(c.n != self.n for c in components):
            raise ValueError("all components must act on the same number of tokens")

    def __call__(self, p: ArrayLike) -> np.ndarray:
        p = _as_price_vector(p)
        return np.sum([c(p) for c in self.components], axis=0)


# ---------------------------------------------------------------------------
# Proposition 2.4: pairwise supply functions via a single-variable function g
# ---------------------------------------------------------------------------

@dataclass
class PairwiseSupplyFunction(SupplyFunction):
    """phi_1(p) = g(p2/p1), phi_2(p) = -(p1/p2) g(p2/p1)  (Equation (1))."""

    g: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self):
        self.n = 2

    def __call__(self, p: ArrayLike) -> np.ndarray:
        p = _as_price_vector(p)
        if p.shape[0] != 2:
            raise ValueError("PairwiseSupplyFunction acts on P_2")
        r = p[1] / p[0]
        phi1 = self.g(r)
        phi2 = -(p[0] / p[1]) * phi1
        return np.array([phi1, phi2])

    def limit_at_zero(self, eps: float = TOL_LIMIT) -> float:
        """Numerical estimate of lim_{r->0} g(r), condition (a) of Prop. 2.4."""
        return float(self.g(eps))

    def limit_over_r_at_infinity(self, big: float = 1.0 / TOL_LIMIT) -> float:
        """Numerical estimate of lim_{r->inf} g(r)/r, condition (a) of Prop. 2.4."""
        return float(self.g(big) / big)

    def bound_phi1(self, r_grid: np.ndarray | None = None) -> float:
        """An upper bound for phi_1(p) = g(p2/p1) (Definition 2.1(2)). g need
        not be monotone (Examples 2.6(IV) notes that tilde_g_buy is not), so
        the bound is the numerical supremum of g over a wide log-spaced grid
        of r, rather than the boundary limit used for monotone constructions."""
        if r_grid is None:
            r_grid = np.geomspace(1e-6, 1e6, 4000)
        return float(max(np.max(self.g(r_grid)), 0.0))

    def bound_phi2(self, r_grid: np.ndarray | None = None) -> float:
        """An upper bound for phi_2(p) = -(p1/p2) g(p2/p1) = tilde_g(p1/p2)
        (Definition 2.5): the numerical supremum of tilde_g over a wide
        log-spaced grid of r (see ``bound_phi1``)."""
        if r_grid is None:
            r_grid = np.geomspace(1e-6, 1e6, 4000)
        return self.conjugate().bound_phi1(r_grid)

    def conjugate(self) -> "PairwiseSupplyFunction":
        """Definition 2.5: tilde_g(r) = -r * g(1/r)."""
        g = self.g

        def g_tilde(r):
            r = np.asarray(r, dtype=float)
            return -r * g(1.0 / r)

        return PairwiseSupplyFunction(g=g_tilde)

    def connects_2_to_1(self, eps: float = TOL_LIMIT) -> bool:
        """Proposition 3.6: phi connects 2->1 iff lim_{r->0} g(r) > 0."""
        return self.limit_at_zero(eps) > 0

    def connects_1_to_2(self, eps: float = TOL_LIMIT, big: float = 1.0 / TOL_LIMIT) -> bool:
        """Proposition 3.6: phi connects 1->2 iff lim_{r->0} tilde_g(r)
        = -lim_{r->inf} g(r)/r > 0."""
        return -self.limit_over_r_at_infinity(big) > 0


def linear_g(a: float, b: float) -> Callable[[np.ndarray], np.ndarray]:
    """g_{a,b}(r) = -a r + b, Examples 2.6(I), with a, b >= 0."""
    if a < 0 or b < 0:
        raise ValueError("linear_g requires a, b >= 0 (Examples 2.6(I))")

    def g(r):
        r = np.asarray(r, dtype=float)
        return -a * r + b

    return g


def linear_pairwise_supply_function(a: float, b: float) -> PairwiseSupplyFunction:
    """Build the pairwise supply function associated to g_{a,b} (Examples 2.6(I))."""
    return PairwiseSupplyFunction(g=linear_g(a, b))


# ---------------------------------------------------------------------------
# Examples 2.6(IV): continuous limit sell and buy orders
# ---------------------------------------------------------------------------

def continuous_limit_sell_g(b: float, r1: float, r2: float) -> Callable[[np.ndarray], np.ndarray]:
    """g_sell(r) = b for r<=r1, linear interpolation on [r1,r2], 0 for r>=r2."""
    if not (b > 0 and 0 < r1 < r2):
        raise ValueError("continuous_limit_sell_g requires b>0 and 0<r1<r2")

    def g(r):
        r = np.asarray(r, dtype=float)
        return np.piecewise(
            r,
            [r <= r1, (r > r1) & (r < r2), r >= r2],
            [lambda x: np.full_like(x, b, dtype=float),
             lambda x: b * (r2 - x) / (r2 - r1),
             lambda x: np.zeros_like(x, dtype=float)],
        )

    return g


def continuous_limit_buy_g(d: float, r1: float, r2: float) -> Callable[[np.ndarray], np.ndarray]:
    """g_buy(r) = 0 for r<=r1, linear interpolation on [r1,r2], -d for r>=r2."""
    if not (d > 0 and 0 < r1 < r2):
        raise ValueError("continuous_limit_buy_g requires d>0 and 0<r1<r2")

    def g(r):
        r = np.asarray(r, dtype=float)
        return np.piecewise(
            r,
            [r <= r1, (r > r1) & (r < r2), r >= r2],
            [lambda x: np.zeros_like(x, dtype=float),
             lambda x: -d * (x - r1) / (r2 - r1),
             lambda x: np.full_like(x, -d, dtype=float)],
        )

    return g


def continuous_limit_sell_order(b: float, r1: float, r2: float) -> PairwiseSupplyFunction:
    """Examples 2.6(IV): continuous limit sell order with parameters (b, r1, r2)."""
    return PairwiseSupplyFunction(g=continuous_limit_sell_g(b, r1, r2))


def continuous_limit_buy_order(d: float, r1: float, r2: float) -> PairwiseSupplyFunction:
    """Examples 2.6(IV): continuous limit buy order with parameters (d, r1, r2)."""
    return PairwiseSupplyFunction(g=continuous_limit_buy_g(d, r1, r2))


# ---------------------------------------------------------------------------
# Examples 2.8: function-maximizing AMM (Canidio & Fritsch), product objective
# ---------------------------------------------------------------------------

def function_maximizing_amm_supply(x: float, y: float) -> PairwiseSupplyFunction:
    """Supply function of the function-maximizing AMM with reserves x, y > 0
    maximizing u*v subject to p1(x-u) + p2(y-v) = 0 (Examples 2.8). The
    resulting supply function equals g_{y/2, x/2} (linear, Examples 2.6(I))."""
    if x <= 0 or y <= 0:
        raise ValueError("reserves x, y must be positive")
    return linear_pairwise_supply_function(a=y / 2.0, b=x / 2.0)


def function_maximizing_amm_new_reserves(x: float, y: float, p: ArrayLike) -> tuple[float, float]:
    """u, v maximizing uv s.t. p1(x-u)+p2(y-v)=0 (Examples 2.8 formula)."""
    p = _as_price_vector(p)
    u = x / 2.0 + (p[1] / p[0]) * (y / 2.0)
    v = (p[0] / p[1]) * (x / 2.0) + y / 2.0
    return float(u), float(v)


# ---------------------------------------------------------------------------
# Section 2.2: supply functions supported on a set of tokens (Lemma 2.11 /
# Proposition 2.10)
# ---------------------------------------------------------------------------

class ExtendedSupplyFunction(SupplyFunction):
    """Lemma 2.11: extend a supply function on m tokens to n >= m tokens by
    carrying indices through an injective map sigma: I_m -> I_n and setting
    phi_j = 0 for j not in the image of sigma."""

    def __init__(self, base: SupplyFunction, sigma: Sequence[int], n: int):
        m = base.n
        sigma = list(sigma)
        if len(sigma) != m:
            raise ValueError("sigma must have length equal to base.n")
        if len(set(sigma)) != m or any(not (0 <= s < n) for s in sigma):
            raise ValueError("sigma must be injective into {0, ..., n-1}")
        self.base = base
        self.sigma = sigma
        self.n = n

    def __call__(self, p: ArrayLike) -> np.ndarray:
        p = _as_price_vector(p)
        if p.shape[0] != self.n:
            raise ValueError(f"expected a price vector of length {self.n}")
        p_sub = p[self.sigma]
        phi_sub = self.base(p_sub)
        phi = np.zeros(self.n)
        phi[self.sigma] = phi_sub
        return phi

    def is_supported_on(self) -> set[int]:
        """Definition 2.9: the index set I on which this supply function is supported."""
        return set(self.sigma)


def restrict_supply_function(phi: SupplyFunction, index_set: Sequence[int]) -> Callable[[ArrayLike], np.ndarray]:
    """Proposition 2.10: restrict a supply function supported on ``index_set``
    (I = {i_1, ..., i_m}) to the m-token function tilde_phi(p) = pi(phi(p_hat)).
    ``p_hat`` fills the coordinates outside index_set with an arbitrary
    positive value (1.0), since phi does not depend on them by hypothesis."""
    index_set = list(index_set)

    def tilde_phi(p: ArrayLike) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        p_hat = np.ones(phi.n)
        p_hat[index_set] = p
        full = phi(p_hat)
        return full[index_set]

    return tilde_phi
