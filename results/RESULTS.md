# Results: Walraswap: Equilibrium-Based Decentralized Exchange Integrating External AMM Liquidity

Paper: **"Walraswap: equilibrium-based decentralized exchange integrating external
AMM liquidity"** (S. Yuhjtman). The paper is a theoretical mechanism-design paper:
it proves existence and uniqueness of Walrasian equilibrium for a batch exchange
whose supply side combines user trade orders with external-AMM virtual agents, and
gives a convex-optimization reformulation of equilibrium computation. It reports no
numerical experiments of its own and uses no market dataset; consequently this
replication contains no empirical "paper value" to match to a tolerance. Instead,
every experiment numerically verifies the propositions, lemmas and theorems of the
paper on synthetic inputs built to satisfy their hypotheses, and the verdict is
whether each stated mathematical property holds in the numerical check.

All numbers below are taken from the JSON files written in `results/` by the
corresponding `python -m exp.<name>` run (see README.md for the exact commands).

## Data source

No external data source is used anywhere in this repository. All inputs (token
reserves, price vectors, supply-function parameters, AMM reserves, gas/fee
parameters) are synthetic constructions defined in `src/config.py` and
`data/generate_synthetic.py`, as required by the purely mathematical nature of the
paper's framework (Sections 2-8, Appendix A). The QuantX Dataloader / Massive
market-data path described for other replications does not apply here and is not
invoked.

## Experiment 1: Supply Function Framework and Pairwise Characterization (Section 2)

File: `results/exp1_metrics.json`.

* **Definition 2.1** (homogeneity, Walras's law, boundedness above, boundary
  regularity conditions (a) and (b)) holds numerically for a linear pairwise
  supply function `g_{3,7}`, the continuous limit sell order
  `(b, r1, r2) = (2.0, 0.5, 2.0)` and the continuous limit buy order
  `(d, r1, r2) = (1.5, 0.5, 2.0)` (Examples 2.6(IV)): all five checks are `true`
  for all three.
* **Proposition 2.2** (sum of supply functions is a supply function): the sum of
  the three functions above again satisfies all checks (`homogeneity`,
  `walras_law`, `bounded_above` all `true`).
* **Conjugate function** (Definition 2.5): `max_abs_diff_g_vs_tilde_tilde_g` is
  `4.5e-13`, `4.4e-16` and `2.2e-16` respectively (float64 round-off), so
  `tilde_tilde_g == g` numerically, verifying the involution stated after
  Definition 2.5.
* **Examples 2.8** (function-maximizing AMM): with reserves `(x, y) = (100, 50)`,
  the constructed supply function matches `g_{y/2, x/2}(r) = -25 r + 50` exactly
  (`max_abs_diff_vs_g_y2_x2 = 0.0`), and both the budget constraint and Walras's
  law hold.
* **Lemma 2.11 / Proposition 2.10** (extension to n tokens and restriction back
  to the pair): the n-token extension is zero on every added coordinate
  (`zero_outside_support = true`) and restricting it recovers the original
  pairwise function exactly (`restriction_matches_original = true`).

**Verdict:** reproduced (all stated properties of Definition 2.1, Proposition 2.2,
the conjugate involution, Examples 2.6/2.8 and Lemma 2.11/Proposition 2.10 hold
numerically to float64 precision).

## Experiment 2: Equilibrium Existence via Brouwer Fixed Point (Section 3, Theorem 3.4)

File: `results/exp2_metrics.json`.

* A strict supply function built from three connected linear pairwise orders on
  4 tokens (`EXP2_LINEAR_G_POOL`) has a strongly connected trade graph
  (Proposition 3.6 / Definition 3.1). The damped Brouwer iteration converges in
  37 steps to a fixed point `q*`, giving equilibrium price
  `p = (0.268, 0.339, 0.143, 0.250)` with `residual_norm = 4.4e-16`, i.e.
  `Phi(p) = 0` to float64 precision: an equilibrium exists, as Theorem 3.4
  states.
* **Proposition 3.5** (connectivity preserved under addition): the union of two
  supply functions' trade graphs has 8 edges (>= the 4 required for strong
  connectivity on 4 tokens) and remains strongly connected.
* **Remark after Theorem 3.4** (counterexample): for the non-strict supply
  function `phi(p1,p2) = (-p2/p1, 1)`, `psi` at `p = (1-eps, eps)` is
  `(-1e-9, 1e-9)`, so `phi` does not connect `2 -> 1` (`does_not_connect_2_to_1 =
  true`) and the naive iteration does not settle at an interior clearing point
  (`brouwer_iteration_converged = false`), consistent with the remark that
  non-strict supply functions may lack an equilibrium.

**Verdict:** reproduced (Theorem 3.4 equilibrium exists with near-zero residual;
Proposition 3.5 holds; the counterexample behaves as the remark states).

## Experiment 3: Uniqueness Properties Under WGS (Section 4)

File: `results/exp3_metrics.json`.

* **Definition 4.1 / WGS**: the linear-cycle system of pairwise supply functions
  satisfies the WGS inequality (`definition_4_1_wgs_holds_for_linear_cycle =
  true`).
* **Theorem 4.3** (unique net trades): across the equilibria located by random
  restarts, the per-agent net-trade spread is `0.0` for all four tokens
  (`unique_net_trades = true`).
* **Theorem 4.4 / Corollary 4.5**: the ratio bound `U_01 = 0.927` is computed
  over the located equilibrium set, and both the linear and logarithmic
  combinations of two equilibria remain in the equilibrium set
  (`corollary_4_5_convexity.linear_convex = true`,
  `log_convex = true`).
* **Proposition 4.6 / Definition 4.7**: the continuous limit sell order has both
  `g` and `g/r` (hence `tilde_g`) decreasing, so it satisfies WGS
  (`satisfies_wgs = true`); the continuous limit buy order has `g` decreasing
  but `g/r` **not** decreasing, so it does **not** satisfy WGS
  (`satisfies_wgs = false`), exactly the asymmetry between sell and buy orders
  flagged in Examples 2.6(IV) and used again in Experiment 4.
* **Corollary 4.9**: on the connected system, equilibrium prices are
  proportional (`corollary_4_9_holds = true`); on a deliberately disconnected
  two-block system, the cross-block ratio `U_02 = 5.58` differs from the
  within-block ratio `U_01 = 0.25`, i.e. prices are **not** proportional across
  the disconnected pairs (`prices_proportional = false`), while `Corollary 4.9`
  still holds vacuously for each connected component.

**Verdict:** reproduced (Theorem 4.3, Theorem 4.4, Corollary 4.5, Proposition 4.6
and Corollary 4.9 all hold numerically, including the expected sell/buy-order
asymmetry and the disconnected-system counterexample to cross-block
proportionality).

## Experiment 4: External AMM Virtual Agents Construction and Properties (Section 5)

File: `results/exp4_metrics.json`, `results/exp4_gf_Gf_surplus.png`.

* For three constant-product pools with reserves `(a,b) in {(100,200), (50,400),
  (10,10)}`, the closed-form `g_f(r) = sqrt(abr) - ar`, `G_f(r) = b -
  sqrt(abr)` match the generic numerical construction via `h_f` to machine
  precision (`closed_form_match` all `true`).
* **Proposition 5.4** holds for every pool: `lim_{r->0} g_f = 0` is approximated
  within the grid (reported as the value at the smallest sampled `r`), `lim
  G_f(r) = M = b` as `r -> 0`, both `g_f/r` and `G_f/r -> 0` as `r -> infinity`,
  `g_f <= G_f` everywhere with strict inequality below `f'(0)`, `G_f` is WGS
  (`G_is_wgs = true`) while `g_f` is not (`g_is_wgs = false`), `G_f` connects
  `2 -> 1` but not `1 -> 2` (`G_connects_2_to_1 = true`,
  `G_connects_1_to_2 = false`), and the surplus `s_c(r) = (sqrt(b) -
  sqrt(ar))^2 / r` is nonnegative everywhere.
* **Proposition 5.8** (`h^{C,p}` maximizes `p . F^C`): verified by a local
  perturbation probe around the analytic maximizer for both a chain system
  (AMM 0-1, 1-2, 0-2) and a parallel-pools system (two pools on the same pair
  plus a cross pool); all `proposition_5_8_maximizer_check` entries are `true`
  for every AMM in both systems.
* The virtual agent `phi^c` built from `G_c` satisfies the supply-function
  conditions (homogeneity, Walras's law, boundedness above all `true`).

**Verdict:** reproduced (Proposition 5.4's five numbered properties and
Proposition 5.8 all hold numerically for every tested pool and system).

## Experiment 5: Fundamental Theorem of Walraswap (Section 6, Theorem 6.4)

File: `results/exp5_metrics.json`.

* A 3-token system of linear pairwise trade orders connecting every pair
  (`EXP5_TRADE_ORDER_LINEAR_PARAMS`) together with two constant-product external
  AMMs (token pairs `(0,1)` and `(1,2)`) has a strongly connected trade graph
  under **Definition 6.2** (`definition_6_2_connected = true`).
* The equilibrium price `p = (0.415, 0.321, 0.265)` found via the Brouwer
  construction applied to `Phi = phi^O + phi^C` has `fixed_point_residual` on
  the order of `1e-13`.
* **Theorem 6.4(a)**: the balance condition
  `(phi^O(p) + F^C(h^{C,p}))_i = sum_{c: in(c)=i} s_c(r_c)` holds
  (`balance_matches_surplus = true`) and both per-AMM surplus terms (`0.468` and
  `0.763`) are nonnegative (`surplus_nonnegative = true`).
* **Theorem 6.4(b)**: `strict_matches_sign` is `true` for both AMMs, i.e. each
  surplus term is strictly positive exactly when the clearing price ratio is
  below `f'_c(0)`.
* **Remark 6.5**: the value `p . (phi^O(p) + F^C(h^{C,p})) = 0.4387` exceeds the
  probed value at nearby external-swap vectors (`0.4385`), confirming `h^{C,p}`
  locally maximizes the surplus value (`remark_6_5_is_maximizer = true`).

**Verdict:** reproduced (connectivity, equilibrium existence, the balance/surplus
identity of Theorem 6.4(a)-(b), and the maximizer property of Remark 6.5 all hold
numerically on the 3-token example).

## Experiment 6: Convex Optimization for Equilibrium Computation (Section 7, Theorem 7.3)

File: `results/exp6_metrics.json`, `results/exp6_convex_program.png`.

* The convex program of **Theorem 7.3** (`minimize F(p,y)` over `p` in the open
  simplex and `y` in `S_n = {y : By = 0}`) is solved with `cvxpy` (solver
  `CLARABEL`) for a 3-token system of linear pairwise orders. The optimal
  objective value is `1.1e-10` (`objective_near_zero = true`), consistent with
  Theorem 7.3(2): `F(p,y) = 0` iff `Phi(p) = 0`.
* The optimal `p = (0.0645, 0.3548, 0.5806)` gives `Phi(p)` within `[-4.2e-5,
  1.6e-4]` of zero (`phi_clears_market = true`), and the optimal `y_ij` matches
  `p_i g^{ij}(p_j/p_i)` for every pair (`y_matches_p_times_g` all `true`).
* The convex-program solution matches, after normalization, the equilibrium
  found independently by the Brouwer construction of Theorem 3.4
  (`max_abs_diff = 2.2e-6`, `matches = true`), cross-validating the two
  equilibrium-computation routes on the same system.
* **Lemma 7.1** (`Z(t) = -t log r* - H(r*)` where `g(r*) = t`, the Legendre-type
  relation used to build `Z`): the analytic `H`/`Z` pair for a linear
  `g_{1,4}` matches the definition to within `4.4e-16` on a grid of `r` values
  (`legendre_relation_check.passed = true`).

**Verdict:** reproduced (Theorem 7.3's zero-objective characterization holds, the
optimizer reproduces `y_ij = p_i g^{ij}(p_j/p_i)`, and it agrees with the
independent Brouwer equilibrium).

## Experiment 7: Walraswap Protocol Design and Simulation (Section 8)

File: `results/exp7_metrics.json`.

Section 8 of the paper explicitly states that it "does not contain rigorous
results" and that the protocol sketch is "not expected to be implemented without
modification"; several mechanisms (the governance voting procedure, the exact
mechanism by which collected fees are swapped for W tokens and burned, the
solver-dependent order-validity policy, and the initial W-token distribution) are
left unspecified. This experiment implements one concrete, illustrative instance
of the sketch (Subsections 8.1.1-8.11) consistent with the structural
requirements the paper does state, and simulates 20 blocks with an honest solver:

* Every one of the 20 simulated blocks yields a strongly connected
  `(O, C)` system (`fraction_connected = 1.0`), so every batch is settled by the
  Theorem 6.4 construction with an average of `5.2` executed orders per block.
* The mean per-block trading-fee value is `0.986` (fee rate `tau = 0.003` of
  Subsection 8.7) and the mean AMM surplus value is `4.18`, always nonnegative
  across all 20 blocks (`surplus_values_all_nonnegative = true`), matching
  Theorem 6.4(a).
* Solver payout: mean trading-fee payout `0.0986` (share `alpha_1 = 0.10 <=
  gamma_1 = 0.10`) and mean surplus payout `3.344` (share `alpha_2 = 0.50 <=
  gamma_2 = 0.80`), both within their governance-set caps
  (`solver_fee_share_within_cap = true`, `solver_surplus_share_within_cap =
  true`), with the remainder (`0.887` trading fee, `0.836` surplus per block on
  average) accruing to the DAO treasury (Subsections 8.9-8.10).
* The governance allowlist simulation starts empty and allowlists the single
  honest solver once its upvotes cross `EXP7_ALLOWLIST_UPVOTE_THRESHOLD = 0.05`
  of the simulated W-token supply (Subsection 8.3).

**Verdict:** partially reproduced. The structural requirements the paper does
state (batch structure 8.1.1, fee mechanics 8.7, gas-fee bound 8.8, surplus split
with caps 8.9-8.10, and honest-solver equilibrium-following 8.11) are satisfied
by construction and verified above. The quantitative fee/gas/payout levels have no
paper-specified values (see "Deviations from the paper") and the governance and
censorship-resistance mechanisms (8.2, 8.5) are simulated only in simplified,
illustrative form, matching the paper's own statement that this section "does not
contain rigorous results."

## Experiment 8: Non-Differentiable AMM Functions (Appendix A)

File: `results/exp8_metrics.json`, `results/exp8_piecewise_amm.png`.

* A piecewise-linear concave AMM function with breakpoints `[0, 8, 16, 26, 40,
  60]` and strictly decreasing right-derivatives `[9.0, 5.0, 2.0, 0.6, 0.0]`
  (the last segment is a true liquidity gap, `f` constant / `M` attained)
  simulates a concentrated-liquidity pool with a finite number of active tick
  ranges, as in Uniswap V3.
* **Proposition A.2**: at each of the 4 interior breakpoints, the generalized
  `h_f(r) = min{x >= 0: partial_+ f(x) <= r}` is well-defined and the one-sided
  concavity/kink condition holds (`kink_concavity_checks` all `true`).
* **Lemma A.3** (unique maximizer of `q(x) = f(x) - rx`): at 5 sampled slopes
  `r`, the grid-search maximizer matches `h_f(r)` (`is_maximizer = true` for
  every sampled `r`, e.g. `r = 0.3` gives `q_star = 128.4` against
  `q_grid_max = 128.397`).
* The `g_f, G_f` pair built from the generalized `h_f` again satisfies
  Proposition 5.4: `g_f <= G_f` everywhere with strict inequality below
  `f'_+(0)`, `G_f` is WGS while `g_f` is not, `G_f` connects `2 -> 1` but not
  `1 -> 2`, and the surplus is nonnegative throughout
  (`proposition_5_4_comparison_amm.surplus_nonnegative = true`).
* Comparing the piecewise (concentrated-liquidity) pool to a constant-product
  pool of comparable initial depth (`a=10, b=90`, `f'(0)=9`), the maximum
  surplus reachable over `r >= 0.1 f'(0)` is larger for the piecewise pool
  (`117.4`) than for the constant-product pool (`45.2`), i.e. concentrating
  liquidity can increase the surplus captured relative to a single constant
  function near the pool's active range.

**Verdict:** reproduced (Proposition A.2's one-sided-derivative construction,
Lemma A.3's unique-maximizer property, and the extension of Proposition 5.4 to the
non-differentiable, concave AMM all hold numerically).

## Deviations from the paper

* **No external data source.** The paper's framework takes no market data as
  input (prices, reserves and order parameters are mathematical objects, not
  observed quantities), so no QuantX Dataloader or Massive REST call is made
  anywhere in this repository; this is a property of the paper's model, not an
  omission.
* **Numerical tolerances, grid resolutions, random seeds and solver choice**
  (`TOL_DEFAULT`, `TOL_LOOSE`, `EXP1_N_RANDOM_PRICE_VECTORS`,
  `EXP4_R_GRID_POINTS`, `EXP6_SOLVER = "CLARABEL"`, `RANDOM_SEED`, etc.) are not
  specified by the paper, since it reports no numerical experiments; each is
  documented with a one-line source note in `src/config.py` and chosen to
  validate the paper's analytic identities to float64 precision.
* **Token counts and reserve/parameter values** for the multi-token examples
  (4 tokens in Experiment 2/3, 3 tokens in Experiment 5/6, specific
  constant-product reserves in Experiments 4/5/8) are not given numerically by
  the paper, which illustrates these constructions only qualitatively (e.g.
  "a multi-token example" in Section 6, a 3-token system in Section 7) without
  specifying numbers; illustrative values satisfying the paper's hypotheses
  (connectivity, WGS, strict positivity of reserves) were chosen and are
  recorded in `src/config.py`.
* **Experiment 7 (Section 8) quantitative parameters.** The paper gives no
  numerical values for the trading-fee rate `tau`, gas price/units, the maximum
  relative gas fee, the solver payout caps `gamma_1, gamma_2`, the realized
  shares `alpha_1, alpha_2`, the W-token supply, or the allowlist threshold; all
  are illustrative constants in `src/config.py`, chosen so that the gas-fee
  filter of Subsection 8.8 is actually exercised. The paper itself states that
  Section 8 "does not contain rigorous results" and that several mechanisms
  (the voting procedure, the fee-to-W swap-and-burn mechanism, the exact
  order-validity policy, and the initial W-token distribution) are left
  unspecified or dispensable; the corresponding simulation components
  (`src/protocol.py`) implement one concrete, simplified instance of each rather
  than the paper's (absent) exact specification.
* **Experiment 8 (Appendix A) concentrated-liquidity shape.** The paper does not
  give explicit swap-function formulas for Uniswap V3 concentrated-liquidity
  pools; the piecewise-linear concave function with breakpoints and slopes used
  here is an illustrative analogue built to exhibit genuine kinks and a
  liquidity gap, matching Appendix A's qualitative description, not a literal
  Uniswap V3 formula.
* **Authors' code or data.** The paper text contains no link to an authors'
  code repository or dataset; none was used.

## Not reproduced

None. Every mathematical property stated in Sections 2-8 and Appendix A that
is covered by Experiments 1-8 above was checked numerically with a reported
verdict.
