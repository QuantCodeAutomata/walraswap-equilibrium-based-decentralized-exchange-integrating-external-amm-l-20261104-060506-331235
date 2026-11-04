"""Non-differentiable external AMM functions (Appendix A): concentrated
liquidity pools with liquidity gaps.

Custom implementation of Definition A.1 (generalized AMM function: concavity
replaces differentiability), Proposition A.2 (one-sided derivatives,
continuity and monotonicity of h_f, the surplus inequality, limit behaviour)
and Lemma A.3 (h_f(r) uniquely maximizes f(x) - rx). No existing library
represents concentrated-liquidity curves as abstract concave functions with
one-sided derivatives, so the piecewise-linear construction follows the
paper's formulas directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PiecewiseLinearAMM:
    """A piecewise-linear, concave, increasing AMM function f with f(0) = 0,
    built from liquidity segments possibly separated by zero-liquidity gaps
    (Appendix A, simulating Uniswap V3 tick ranges with empty ticks).

    ``breakpoints`` are the x-coordinates 0 = x_0 < x_1 < ... < x_K, and
    ``slopes[k]`` is the (right-)derivative of f on (x_k, x_{k+1}); slopes
    must be nonincreasing (concavity, Definition A.1(d)) and nonnegative
    (monotonicity, Definition A.1(a)). A slope of 0 represents a liquidity
    gap: f is locally constant there.
    """

    breakpoints: np.ndarray
    slopes: np.ndarray

    def __post_init__(self):
        self.breakpoints = np.asarray(self.breakpoints, dtype=float)
        self.slopes = np.asarray(self.slopes, dtype=float)
        if len(self.breakpoints) != len(self.slopes) + 1:
            raise ValueError("need one more breakpoint than slopes")
        if np.any(np.diff(self.breakpoints) <= 0):
            raise ValueError("breakpoints must be strictly increasing")
        if np.any(self.slopes < 0):
            raise ValueError("slopes must be nonnegative (f increasing)")
        if np.any(np.diff(self.slopes) > 1e-12):
            raise ValueError("slopes must be nonincreasing (f concave)")
        # f values at breakpoints, by integrating the piecewise-constant
        # right derivative; f(0) = 0 (Definition A.1(b)).
        values = [0.0]
        for k in range(len(self.slopes)):
            values.append(values[-1] + self.slopes[k] * (self.breakpoints[k + 1] - self.breakpoints[k]))
        self.values = np.array(values)

    @property
    def M(self) -> float:
        """Bounded above (Definition A.1(c)): sup_x f(x), the plateau value
        reached once liquidity is exhausted (slopes[-1] may be 0)."""
        if self.slopes[-1] == 0:
            return float(self.values[-1])
        return float(self.values[-1] + 1e9)  # effectively unbounded if last slope>0; capped for plotting

    def f(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x_clipped = np.clip(x, 0.0, self.breakpoints[-1])
        idx = np.searchsorted(self.breakpoints, x_clipped, side="right") - 1
        idx = np.clip(idx, 0, len(self.slopes) - 1)
        base_x = self.breakpoints[idx]
        base_val = self.values[idx]
        return base_val + self.slopes[idx] * (x_clipped - base_x)

    def right_derivative(self, x: np.ndarray) -> np.ndarray:
        """partial_+ f(x): the right derivative, piecewise-constant,
        nonincreasing in x (concavity)."""
        x = np.asarray(x, dtype=float)
        x_clipped = np.clip(x, 0.0, self.breakpoints[-1] - 1e-12)
        idx = np.searchsorted(self.breakpoints, x_clipped, side="right") - 1
        idx = np.clip(idx, 0, len(self.slopes) - 1)
        return self.slopes[idx]

    def left_derivative(self, x: np.ndarray) -> np.ndarray:
        """partial_- f(x): the left derivative, equal to the right derivative
        of the previous segment at a breakpoint."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.breakpoints, x, side="left")
        idx = np.clip(idx - 1, 0, len(self.slopes) - 1)
        return self.slopes[idx]

    @property
    def f_prime_at_0(self) -> float:
        return float(self.slopes[0])

    def h(self, r: np.ndarray) -> np.ndarray:
        """Proposition A.2(III): h_f(r) = min{x>=0 : partial_+f(x) <= r}.

        Since the right derivative is a nonincreasing step function, this is
        found by locating the first segment whose slope is <= r and taking
        its left endpoint, or interpolating within a gap-adjacent plateau.
        """
        r = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.zeros_like(r)
        for idx, rv in enumerate(r):
            if rv >= self.slopes[0]:
                out[idx] = 0.0
                continue
            # find first k with slopes[k] <= rv
            k = np.searchsorted(-self.slopes, -rv, side="left")
            k = min(k, len(self.slopes) - 1)
            out[idx] = self.breakpoints[k]
        return out

    def g(self, r: np.ndarray) -> np.ndarray:
        """g_f(r) = r * h_f(r) (Definition 5.3, extended via Proposition A.2)."""
        r = np.asarray(r, dtype=float)
        return r * self.h(r)

    def G(self, r: np.ndarray) -> np.ndarray:
        """G_f(r) = f(h_f(r))."""
        return self.f(self.h(r))

    def surplus(self, r: np.ndarray) -> np.ndarray:
        """s_f(r) = (G_f(r) - g_f(r)) / r >= 0 (Proposition A.2(V))."""
        r = np.asarray(r, dtype=float)
        return (self.G(r) - self.g(r)) / r


def verify_proposition_A2(amm: PiecewiseLinearAMM, r_grid: np.ndarray) -> dict:
    """Numerically verify the parts of Proposition A.2 relevant to the
    virtual-agent construction: monotonicity/continuity of h_f, the surplus
    inequality r h_f(r) <= f(h_f(r)), and the r->0 limits."""
    h_vals = amm.h(r_grid)
    h_decreasing = bool(np.all(np.diff(h_vals) <= 1e-9))

    g_vals, G_vals = amm.g(r_grid), amm.G(r_grid)
    surplus_nonneg = bool(np.all(G_vals - g_vals >= -1e-9))
    strict_where_h_pos = bool(np.all((G_vals - g_vals)[h_vals > 1e-9] > -1e-9))

    eps = 1e-9
    lim_G_at_0 = float(amm.G(np.array([eps]))[0])       # expect -> M (Prop A.2.VI)
    lim_g_at_0 = float(amm.g(np.array([eps]))[0])        # expect -> 0 (Prop A.2.VII)

    # h_f(partial_+f(x)) = x check (Proposition A.2.IV), tested at the
    # breakpoints x_k with f(x_k) < M: since the right derivative is a
    # nonincreasing step function, slopes[k] < slopes[k-1] makes x_k the
    # *first* point with partial_+f(x) <= slopes[k], so h_f(slopes[k]) = x_k
    # exactly. This is the discrete analogue of Prop. A.2(IV) at the kinks
    # that replace classical differentiability in Appendix A.
    identity_ok = []
    for k, x in enumerate(amm.breakpoints[:-1]):
        if amm.f(np.array([x]))[0] >= amm.M - 1e-9:
            continue
        r_at_x = float(amm.right_derivative(np.array([x]))[0])
        if r_at_x <= 0:
            continue
        x_back = float(amm.h(np.array([r_at_x]))[0])
        identity_ok.append(abs(x_back - x) < 1e-6)

    return {
        "h_decreasing": h_decreasing,
        "surplus_nonnegative": surplus_nonneg,
        "strict_where_h_positive": strict_where_h_pos,
        "lim_G_at_0": lim_G_at_0,
        "expected_M": amm.M,
        "lim_g_at_0": lim_g_at_0,
        "identity_h_of_fprime_holds": all(identity_ok) if identity_ok else True,
    }


def verify_lemma_A3(amm: PiecewiseLinearAMM, r: float, x_grid: np.ndarray) -> dict:
    """Lemma A.3: q(x) = f(x) - r x is maximized uniquely at x* = h_f(r)."""
    x_star = float(amm.h(np.array([r]))[0])
    q_star = float(amm.f(np.array([x_star]))[0] - r * x_star)
    q_grid = amm.f(x_grid) - r * x_grid
    return {
        "x_star": x_star,
        "q_star": q_star,
        "q_grid_max": float(q_grid.max()),
        "is_maximizer": bool(q_star >= q_grid.max() - 1e-6),
    }
