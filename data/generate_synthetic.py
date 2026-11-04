"""Synthetic inputs for the Walraswap experiments.

The paper's framework (supply functions, AMM-derived virtual agents,
equilibrium computation) is purely mathematical: every experiment runs on
constructed token reserves, price vectors and function parameters rather
than on observed market data. This module centralizes the random and
structured generation of those inputs so every experiment script draws them
the same way instead of inventing its own ad hoc generator.
"""

from __future__ import annotations

import numpy as np


def random_price_vector(n: int, rng: np.random.Generator, low: float = 0.1, high: float = 10.0) -> np.ndarray:
    """A random strictly positive price vector in P_n (paper Section 2)."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return rng.uniform(low, high, size=n)


def random_simplex_point(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random point of the (n-1)-simplex Delta, used for Brouwer-map tests
    (Theorem 3.4 proof)."""
    return rng.dirichlet(np.ones(n))


def random_boundary_price(n: int, i: int, j: int, rng: np.random.Generator) -> np.ndarray:
    """A boundary price vector with p_i = 0, p_j > 0 and other coordinates
    strictly positive, used to test the connectivity condition (Def. 3.1)."""
    p = rng.uniform(0.1, 10.0, size=n)
    p[i] = 0.0
    p[j] = rng.uniform(1.0, 10.0)
    return p


def random_cp_reserves(rng: np.random.Generator, low: float = 10.0, high: float = 500.0) -> tuple[float, float]:
    """Random reserve pair (a, b) for a constant-product AMM f(x)=bx/(a+x)
    (paper Section 5 example)."""
    return float(rng.uniform(low, high)), float(rng.uniform(low, high))


def random_linear_g_params(rng: np.random.Generator, a_high: float = 5.0, b_high: float = 10.0) -> tuple[float, float]:
    """Random (a, b) >= 0 for a linear g_{a,b} (Examples 2.6(I))."""
    return float(rng.uniform(0.1, a_high)), float(rng.uniform(1.0, b_high))


def random_sell_order_params(rng: np.random.Generator) -> tuple[float, float, float]:
    """Random (b, r1, r2) for a continuous limit sell order (Examples 2.6(IV))."""
    b = float(rng.uniform(1.0, 5.0))
    r1 = float(rng.uniform(0.2, 0.8))
    r2 = r1 + float(rng.uniform(0.5, 2.0))
    return b, r1, r2


def random_buy_order_params(rng: np.random.Generator) -> tuple[float, float, float]:
    """Random (d, r1, r2) for a continuous limit buy order (Examples 2.6(IV))."""
    return random_sell_order_params(rng)
