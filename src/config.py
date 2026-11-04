"""Parameters used by every experiment, as module-level constants.

Each value cites the section/equation of "Walraswap: equilibrium-based
decentralized exchange integrating external AMM liquidity" (Yuhjtman) that
defines or motivates it, or states that the paper does not specify it. All
inputs of this replication are synthetic (token reserves, price vectors,
AMM parameters): the paper's framework is purely mathematical and uses no
market dataset.
"""

import numpy as np

# Reproducibility: the paper gives no explicit seed, since it offers no
# empirical/numerical protocol. One fixed seed is used throughout to make
# every experiment's synthetic construction deterministic.
RANDOM_SEED = 20261104

# ---------------------------------------------------------------------------
# Numerical tolerances (not specified in the paper; chosen to validate the
# analytic identities of Definition 2.1 etc. to float64 precision).
# ---------------------------------------------------------------------------
TOL_DEFAULT = 1e-8          # generic equality / inequality tolerance
TOL_LOOSE = 1e-5            # tolerance for quantities obtained by numerical
                            # root finding (e.g. Brouwer fixed point, solvers)
TOL_LIMIT = 1e-6            # step used to numerically approximate lim_{r->0}
                            # and lim_{r->inf} of a function g (paper Prop. 2.4)

# ---------------------------------------------------------------------------
# exp_1: Supply Function Framework and Pairwise Characterization (Sec. 2)
# ---------------------------------------------------------------------------
EXP1_N_RANDOM_PRICE_VECTORS = 200   # number of random p in P_n used to check
                                    # Definition 2.1 conditions numerically (not specified in the paper)
EXP1_N_TOKENS_GENERIC = 5           # n used for the generic (non-pairwise) checks of Def. 2.1 (not specified)
EXP1_SELL_ORDER_PARAMS = (2.0, 0.5, 2.0)   # (b, r1, r2) of a continuous limit sell order, Examples 2.6(IV)
EXP1_BUY_ORDER_PARAMS = (1.5, 0.5, 2.0)    # (d, r1, r2) of a continuous limit buy order, Examples 2.6(IV)
EXP1_AMM_RESERVES = (100.0, 50.0)   # (x, y) reserves of the function-maximizing AMM, Examples 2.8
EXP1_LINEAR_G_PARAMS = (3.0, 7.0)   # (a, b) of a linear g_{a,b}, Examples 2.6(I)

# ---------------------------------------------------------------------------
# exp_2: Equilibrium Existence via Brouwer Fixed Point (Sec. 3, Thm 3.4)
# ---------------------------------------------------------------------------
EXP2_N_TOKENS = 4                        # size of the strict supply function used for the existence test (not specified)
EXP2_BROUWER_MAX_ITER = 20000             # iteration cap for the damped fixed-point map rho (not specified in the paper)
EXP2_BROUWER_DAMPING = 0.5                # damping factor used for stable convergence of rho (not specified)
EXP2_BROUWER_TOL = 1e-10                  # stopping tolerance ||rho(q)-q||_inf
EXP2_LINEAR_G_POOL = [(1.0, 4.0), (2.0, 3.0), (0.5, 6.0)]  # (a,b) parameters of a
                                           # connected set of linear pairwise orders used to build a strict Phi

# ---------------------------------------------------------------------------
# exp_3: Uniqueness Properties Under WGS (Sec. 4)
# ---------------------------------------------------------------------------
EXP3_N_TOKENS = 4                   # size of the WGS system used to test Theorem 4.3/4.4 (not specified)
EXP3_N_EQUILIBRIA_SEARCHES = 6      # number of random starting simplex points used to locate distinct equilibria (not specified)
EXP3_SELL_ORDER_PARAMS = (2.0, 0.5, 2.0)   # reuse of Examples 2.6(IV) sell order for the WGS check
EXP3_BUY_ORDER_PARAMS = (1.5, 0.5, 2.0)    # reuse of Examples 2.6(IV) buy order for the non-WGS check

# ---------------------------------------------------------------------------
# exp_4: External AMM Virtual Agents (Sec. 5)
# ---------------------------------------------------------------------------
EXP4_CP_RESERVES = [(100.0, 200.0), (50.0, 400.0), (10.0, 10.0)]  # (a, b) reserve
                                           # pairs of constant-product AMMs f(x) = b x /(a + x), Sec. 5 example
EXP4_R_GRID_POINTS = 400                  # grid resolution used to plot g_f, G_f, surplus s_c (not specified)
EXP4_R_GRID_MAX_FACTOR = 5.0              # grid upper bound as a multiple of f'(0) (not specified)

# ---------------------------------------------------------------------------
# exp_5: Fundamental Theorem of Walraswap (Sec. 6, Thm 6.4)
# ---------------------------------------------------------------------------
EXP5_N_TOKENS = 3                         # 3-token example requested by the experiment description
# Token indices are 0-based (0, 1, 2) for the 3-token system.
EXP5_AMM_RESERVES = {(0, 1): (80.0, 120.0), (1, 2): (60.0, 90.0)}  # (a,b) reserves
                                           # of the two constant-product external AMMs (token i -> token j)
EXP5_TRADE_ORDER_LINEAR_PARAMS = {
    (0, 1): (1.0, 4.0),
    (1, 2): (1.0, 4.0),
    (0, 2): (1.0, 4.0),
}  # (a, b) of linear pairwise orders g_{a,b}, Examples 2.6(I), connecting every
   # token pair in both directions (a>0 gives tilde_g(0)>0, Proposition 3.6),
   # which continuous limit orders alone cannot since they are bounded and
   # hence satisfy lim_{r->inf} g(r)/r = 0 (so Definition 6.2 connectivity for
   # this 3-token example is not specified numerically in the paper)

# ---------------------------------------------------------------------------
# exp_6: Convex Optimization for Equilibrium Computation (Sec. 7, Thm 7.3)
# ---------------------------------------------------------------------------
EXP6_N_TOKENS = 3                 # 3-token system requested by the experiment description
# Token indices are 0-based (0, 1, 2) for the 3-token system.
EXP6_LINEAR_G = {
    (0, 1): (1.0, 4.0),
    (1, 2): (2.0, 3.0),
    (0, 2): (0.5, 6.0),
}                                  # (a, b) of linear g^{ij} = g_{a,b} for each pair, Examples 2.6(I)
EXP6_SIMPLEX_EPS = 1e-6            # lower bound imposed on p_i inside the open simplex (not specified)
EXP6_SOLVER = "CLARABEL"           # convex solver used by cvxpy (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_7: Walraswap Protocol Design and Simulation (Sec. 8)
# ---------------------------------------------------------------------------
EXP7_N_TOKENS = 4                   # tokens used in the protocol simulation (not specified)
EXP7_N_BLOCKS = 20                  # number of simulated block slots (not specified)
EXP7_ORDERS_PER_BLOCK = 6           # trade orders disclosed per block on average (not specified)
EXP7_TRADING_FEE_TAU = 0.003        # trading-fee rate tau of Eq. (5), Sec. 8.7 (illustrative value; not numerically specified)
EXP7_GAS_PRICE = 2.0e-5             # gas price expressed directly in token-value units
                                     # per gas unit, Sec. 8.8 (illustrative; the paper gives
                                     # no numeraire or gwei/token conversion, so gas cost
                                     # here is denominated directly in the same token-value
                                     # units as value_of_sold_tokens)
EXP7_GAS_UNITS_PER_ORDER = 25000.0  # tabulated gas units per order, Sec. 8.8 (illustrative; chosen,
                                     # together with EXP7_GAS_PRICE, so the tabulated gas cost (0.5
                                     # token-value units) is comparable to the per-order cap
                                     # max_relative_gas_fee * value_of_sold_tokens, so that the
                                     # gas-fee filtering of Subsection 8.8 is actually exercised)
EXP7_MAX_RELATIVE_GAS_FEE = 0.02    # per-order cap on gas_fee / value_of_sold_tokens, Sec. 8.8 (illustrative; not specified)
EXP7_ALPHA_1_CAP_GAMMA_1 = 0.10     # gamma_1: cap on the solver's share of trading fees, Sec. 8.10 (DAO keeps most fees -> small; not numerically specified)
EXP7_ALPHA_2_CAP_GAMMA_2 = 0.80     # gamma_2: cap on the solver's share of AMM surplus, Sec. 8.10 (can reach one; illustrative)
EXP7_ALPHA_1 = 0.10                 # solver's realized trading-fee share, must satisfy alpha_1 <= gamma_1
EXP7_ALPHA_2 = 0.50                 # solver's realized surplus share, must satisfy alpha_2 <= gamma_2
EXP7_W_TOKEN_SUPPLY = 1_000_000.0   # total W governance token supply used for the voting simulation (not specified)
EXP7_ALLOWLIST_UPVOTE_THRESHOLD = 0.05  # fraction of W supply required as upvotes for allowlisting, Sec. 8.3 (not specified)

# ---------------------------------------------------------------------------
# exp_8: Non-Differentiable AMM Functions (Appendix A)
# ---------------------------------------------------------------------------
EXP8_BREAKPOINTS = [0.0, 8.0, 16.0, 26.0, 40.0, 60.0]   # x-breakpoints of the
                            # piecewise-linear concave AMM function (cumulative
                            # amount sold), not specified in the paper
EXP8_SLOPES = [9.0, 5.0, 2.0, 0.6, 0.0]   # right-derivatives of f on each
                            # segment between consecutive breakpoints, strictly
                            # decreasing (Definition A.1(d), concavity) to
                            # create genuine kinks at tick boundaries -- each
                            # kink is a point of non-differentiability of the
                            # kind Appendix A's h_f via one-sided derivatives
                            # is built to handle. The final slope of 0 is a
                            # true liquidity gap: f plateaus at its bound M,
                            # simulating a Uniswap V3 concentrated-liquidity
                            # pool that runs out of liquidity in its tick
                            # range (not specified numerically in the paper)
EXP8_R_GRID_POINTS = 400    # grid resolution for plotting h_f, g_f, G_f (not specified)

RNG = np.random.default_rng(RANDOM_SEED)
