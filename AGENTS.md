# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A numerical verification of "Walraswap: equilibrium-based decentralized
exchange integrating external AMM liquidity" (Sergio Andrés Yuhjtman). The
paper develops a Walrasian-equilibrium model of a decentralized-exchange
batch auction: trade orders are represented as *supply functions*
(Section 2), equilibrium existence is proved via Brouwer's fixed-point
theorem (Section 3), uniqueness properties under weak gross substitutability
are established (Section 4), external AMMs are folded into the equilibrium
problem as *virtual agents* (Section 5), the fundamental theorem of
Walraswap combines trade orders and external AMMs into one clearing
mechanism (Section 6), equilibrium computation is reformulated as a convex
program (Section 7), a protocol design is sketched (Section 8), and the AMM
construction is extended to non-differentiable, concave AMM functions such
as concentrated-liquidity pools (Appendix A).

The paper's framework is purely mathematical: every input (token reserves,
price vectors, supply-function parameters) is a synthetic mathematical
object, not observed market data, so this repository makes no external data
call.

## Layout

```
src/     flat library modules implementing the paper's mathematical objects
         (supply_functions.py, equilibrium.py, uniqueness.py, amm.py,
         fundamental_theorem.py, convex_optimization.py, protocol.py,
         nondiff_amm.py, config.py)
exp/     one script per experiment (exp1_supply_functions.py ... exp8_nondiff_amm.py),
         each calling only src/ functions and writing to results/
tests/   pytest unit tests of src/ components, on synthetic/analytic fixtures
results/ metrics (JSON), plots (PNG) and RESULTS.md, all written directly here
data/    data/generate_synthetic.py: synthetic input generators (random price
         vectors, simplex points, AMM reserve pairs, supply-function
         parameters) that experiments 1, 3 and 4 import for their randomized
         checks; no data_loader.py, since the paper's model takes no market
         data
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (no `market_data`-marked tests exist in this repository, since the
paper requires no market data; both commands run the same suite):

```bash
python -m pytest
python -m pytest -m "not market_data"
```

Experiments, run from the repository root, each writing its artifacts
directly into `results/`:

```bash
python -m exp.exp1_supply_functions
python -m exp.exp2_equilibrium
python -m exp.exp3_uniqueness
python -m exp.exp4_external_amm
python -m exp.exp5_fundamental_theorem
python -m exp.exp6_convex_optimization
python -m exp.exp7_protocol
python -m exp.exp8_nondiff_amm
```

or `./reproduce.sh` to run the test suite followed by all eight experiments.

## Conventions

* All experiment parameters live only in `src/config.py`, as module-level
  `UPPER_CASE` constants grouped by experiment, each with a one-line comment
  giving the paper section/equation it comes from, or stating that the
  paper does not specify it. No parameter is hardcoded elsewhere.
* Randomized-input generation shared across more than one experiment lives
  only under `data/generate_synthetic.py`. Experiments that only need a
  single seeded draw or a deterministic parameter grid (fixed by
  `src/config.py`) construct it inline instead of adding a one-off
  generator function used nowhere else.
* `tests/` holds every pytest file; it tests paper properties (signs,
  monotonicity, Walras's law, homogeneity, absence of look-ahead where
  applicable, equilibrium uniqueness) on small or analytic fixtures, not
  the specific numbers reported in `results/`.
* No API keys are needed anywhere in this repository.
* After a rerun, the experiment commands overwrite every file in `results/`
  (the JSON metrics and PNG plots). `results/RESULTS.md` is not regenerated
  automatically: its numbers and verdicts must be checked against the new
  result files and updated by hand whenever the code or configuration
  changes.
