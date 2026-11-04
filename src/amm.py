"""External AMMs and their associated supply functions (Section 5).

Custom implementation of Definition 5.1 (AMM function), Definition 5.2 (the
inverse h_f of f'), Definition 5.3 (g_f, G_f), Proposition 5.4, Definition
5.5 (system of external AMMs), Definition 5.6 (surplus s_c and virtual agents
phi^c), Definition 5.7 (F^C) and Proposition 5.8. This is the paper's own
construction; pyportfolioopt/skfolio solve portfolio allocation problems, not
this bespoke AMM-to-supply-function mapping, so everything is implemented
from the closed-form constant-product formulas given in Section 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.supply_functions import PairwiseSupplyFunction, SupplyFunction


@dataclass
class ConstantProductAMM:
    """Definition 5.1 example: f(x) = b x / (a + x), reserves (a, b) > 0, no fee.

    f is increasing, f(0) = 0, bounded above by M = b, continuously
    differentiable with f'(x) = ab/(a+x)^2 strictly decreasing, f'(0) = b/a.
    """

    a: float
    b: float

    def __post_init__(self):
        if self.a <= 0 or self.b <= 0:
            raise ValueError("constant-product reserves a, b must be positive")

    @property
    def M(self) -> float:
        """Upper bound of f (sup_x f(x) = b)."""
        return self.b

    @property
    def f_prime_at_0(self) -> float:
        return self.b / self.a

    def f(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.b * x / (self.a + x)

    def f_prime(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.a * self.b / (self.a + x) ** 2

    def h(self, r: np.ndarray) -> np.ndarray:
        """Definition 5.2: h_f(r) = (f')^{-1}(r) for r < f'(0), else 0.

        Closed form: h_f(r) = sqrt(ab/r) - a for 0 < r < b/a, else 0.
        """
        r = np.asarray(r, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inner = np.sqrt(self.a * self.b / r) - self.a
        return np.where(r < self.f_prime_at_0, np.maximum(inner, 0.0), 0.0)

    def g(self, r: np.ndarray) -> np.ndarray:
        """g_f(r) = r * h_f(r) = sqrt(abr) - a r  (Definition 5.3)."""
        r = np.asarray(r, dtype=float)
        h = self.h(r)
        return r * h

    def G(self, r: np.ndarray) -> np.ndarray:
        """G_f(r) = f(h_f(r)) = b - sqrt(abr) for r < f'(0), else M=b for r>=f'(0)
        collapses continuously since at r=f'(0), h=0 and f(0)=0... but the
        paper's closed form for r>=f'(0) uses h_f(r)=0 so G_f(r)=f(0)=0; the
        limit lim_{r->0} G_f(r) = M = b is the *left* boundary behaviour
        (Proposition 5.4(1)), not the value at r=f'(0)."""
        r = np.asarray(r, dtype=float)
        h = self.h(r)
        return self.f(h)

    def surplus(self, r: np.ndarray) -> np.ndarray:
        """Definition 5.6(a): s_f(r) = (G_f(r) - g_f(r)) / r
        = (b - 2 sqrt(abr) + ar)/r = (sqrt(b) - sqrt(a r))^2 / r >= 0."""
        r = np.asarray(r, dtype=float)
        return (np.sqrt(self.b) - np.sqrt(self.a * r)) ** 2 / r


def g_f_supply_function(amm: ConstantProductAMM) -> PairwiseSupplyFunction:
    """The (non-WGS) pairwise supply function built from g_f (Proposition 5.4(3))."""
    return PairwiseSupplyFunction(g=amm.g)


def G_f_supply_function(amm: ConstantProductAMM) -> PairwiseSupplyFunction:
    """The WGS pairwise supply function built from G_f (Proposition 5.4(3)-(4))."""
    return PairwiseSupplyFunction(g=amm.G)


def verify_proposition_5_4(amm: ConstantProductAMM, r_grid: np.ndarray) -> dict:
    """Numerically verify the five parts of Proposition 5.4 for a
    constant-product AMM on the given grid of r > 0 values."""
    eps = r_grid.min()
    big = r_grid.max()

    lim_g_at_0 = float(amm.g(np.array([eps]))[0])
    lim_g_over_r_at_inf = float(amm.g(np.array([big]))[0] / big)
    lim_G_at_0 = float(amm.G(np.array([eps]))[0])
    lim_G_over_r_at_inf = float(amm.G(np.array([big]))[0] / big)

    g_vals, G_vals = amm.g(r_grid), amm.G(r_grid)
    g_le_G = bool(np.all(g_vals <= G_vals + 1e-9))
    strict_below_fprime0 = r_grid < amm.f_prime_at_0
    strict_ok = bool(np.all(G_vals[strict_below_fprime0] - g_vals[strict_below_fprime0] > -1e-9))

    G_decreasing = bool(np.all(np.diff(G_vals) <= 1e-9))
    G_over_r_decreasing = bool(np.all(np.diff(G_vals / r_grid) <= 1e-9))
    g_is_wgs = bool(np.all(np.diff(g_vals) <= 1e-9))  # expected False

    pf_G = G_f_supply_function(amm)
    connects_2_1 = pf_G.connects_2_to_1()
    connects_1_2 = pf_G.connects_1_to_2()

    surplus_vals = amm.surplus(r_grid)
    surplus_nonneg = bool(np.all(surplus_vals >= -1e-9))

    return {
        "lim_g_at_0": lim_g_at_0,                 # expect ~0 (property 1)
        "lim_g_over_r_at_inf": lim_g_over_r_at_inf,  # expect ~0
        "lim_G_at_0": lim_G_at_0,                 # expect ~ amm.M
        "lim_G_over_r_at_inf": lim_G_over_r_at_inf,  # expect ~0
        "g_le_G_everywhere": g_le_G,               # property 2
        "strict_inequality_below_fprime0": strict_ok,
        "G_is_wgs": G_decreasing and G_over_r_decreasing,  # property 4
        "g_is_wgs": g_is_wgs,                      # property 4, expect False
        "G_connects_2_to_1": connects_2_1,         # property 5
        "G_connects_1_to_2": connects_1_2,         # property 5, expect False
        "surplus_nonnegative": surplus_nonneg,     # property 2 restated
    }


# ---------------------------------------------------------------------------
# Definition 5.5 / 5.6 / 5.7: systems of external AMMs and virtual agents
# ---------------------------------------------------------------------------

@dataclass
class ExternalAMMEdge:
    """One external AMM c, with in(c), out(c) token indices and its AMM function."""

    name: str
    token_in: int
    token_out: int
    amm: ConstantProductAMM


@dataclass
class SystemOfExternalAMMs:
    """Definition 5.5: a finite family C of external AMMs with in/out tokens."""

    edges: list[ExternalAMMEdge] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        idx = set()
        for e in self.edges:
            idx.add(e.token_in)
            idx.add(e.token_out)
        return (max(idx) + 1) if idx else 0

    def virtual_agent_supply_function(self, n: int) -> SupplyFunction:
        """Definition 5.6(d): phi^C = sum_c phi^c, the aggregate virtual agent."""
        from src.supply_functions import ExtendedSupplyFunction

        components = []
        for e in self.edges:
            pf = G_f_supply_function(e.amm)
            # PairwiseSupplyFunction(p) uses r = p[1]/p[0] and assigns g(r) to
            # local index 0: to get G_c(p_in/p_out) at out(c) and
            # -(p_out/p_in) G_c(p_in/p_out) at in(c) (Definition 5.6(c)), the
            # local axes must be ordered [out(c), in(c)], not [in(c), out(c)].
            components.append(ExtendedSupplyFunction(pf, sigma=[e.token_out, e.token_in], n=n))
        from src.supply_functions import SumSupplyFunction

        return SumSupplyFunction(components)

    def external_swaps(self, p: np.ndarray) -> dict[str, float]:
        """h^{C,p}: for each AMM c, the optimal swap size h_c(r_c), r_c = p_in/p_out
        (Proposition 5.8)."""
        swaps = {}
        for e in self.edges:
            r_c = p[e.token_in] / p[e.token_out]
            swaps[e.name] = float(e.amm.h(np.array([r_c]))[0])
        return swaps

    def F_C(self, swaps: dict[str, float]) -> np.ndarray:
        """Definition 5.7: F^C(x)_i = sum_{c: out(c)=i} f_c(x_c) - sum_{c: in(c)=i} x_c."""
        n = self.n_tokens
        out = np.zeros(n)
        for e in self.edges:
            x_c = swaps[e.name]
            out[e.token_out] += float(e.amm.f(np.array([x_c]))[0])
            out[e.token_in] -= x_c
        return out

    def surplus_per_amm(self, p: np.ndarray) -> dict[str, float]:
        surplus = {}
        for e in self.edges:
            r_c = p[e.token_in] / p[e.token_out]
            surplus[e.name] = float(e.amm.surplus(np.array([r_c]))[0])
        return surplus


def verify_proposition_5_8(system: SystemOfExternalAMMs, p: np.ndarray, n_probe: int = 50, rng=None) -> dict:
    """Proposition 5.8: h^{C,p} maximizes p . F^C(x) over x >= 0 componentwise
    (each AMM's swap decouples, so this reduces to maximizing
    p_out(c) f_c(x_c) - p_in(c) x_c over x_c >= 0 for every c)."""
    if rng is None:
        rng = np.random.default_rng(0)
    swaps = system.external_swaps(p)
    results = {}
    for e in system.edges:
        x_star = swaps[e.name]
        p_in, p_out = p[e.token_in], p[e.token_out]
        obj_star = p_out * e.amm.f(np.array([x_star]))[0] - p_in * x_star
        probe_x = rng.uniform(0.0, max(x_star * 3, 10.0), size=n_probe)
        probe_obj = p_out * e.amm.f(probe_x) - p_in * probe_x
        results[e.name] = bool(obj_star >= probe_obj.max() - 1e-6)
    return results
