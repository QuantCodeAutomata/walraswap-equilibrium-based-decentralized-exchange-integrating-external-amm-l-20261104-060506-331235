# Walraswap: Equilibrium-Based Decentralized Exchange Integrating External AMM Liquidity

This repository reproduces the mathematical framework and numerical experiments of
**"Walraswap: equilibrium-based decentralized exchange integrating external AMM
liquidity"** (S. Yuhjtman). The paper develops a Walrasian-equilibrium model of a
decentralized exchange batch auction in which trade orders are represented as
*supply functions* and external automated market makers (AMMs) are folded into the
same equilibrium problem through *virtual agents*. It proves existence (Section 3),
uniqueness properties under weak gross substitutability (Section 4), the
construction and properties of AMM virtual agents (Section 5), a fundamental
theorem for batch exchange with external AMMs (Section 6), a convex-optimization
reformulation of equilibrium computation (Section 7), sketches a protocol design
(Section 8), and extends the AMM construction to non-differentiable, concave AMM
functions (Appendix A).

The paper's framework is purely mathematical: every object used in the proofs
(token reserves, price vectors, supply-function parameters, AMM reserves) is
synthetic by construction, so **no external market dataset is required or used**.
All experiments are deterministic numerical verifications of the paper's
definitions, propositions, lemmas and theorems.

## Repository layout

```
src/     flat library modules implementing the paper's mathematical objects
exp/     one script per experiment, calling only src/ functions
tests/   pytest unit tests of src/ components (synthetic/analytic fixtures)
results/ metrics (JSON), plots (PNG) and RESULTS.md written by the exp/ scripts
data/    data/generate_synthetic.py: synthetic input generators (price vectors,
         AMM reserves) shared by the experiments; no market data loader is
         needed since the paper's model takes no market data as input
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or `.env` file are needed: this replication uses no external data
service.

## Running the tests

```bash
python -m pytest
```

There are no `market_data`-marked tests in this repository (the paper requires no
market data), so `python -m pytest -m "not market_data"` and `python -m pytest`
run the same suite.

## Running the experiments

Each experiment is runnable independently from the repository root and writes its
artifacts directly into `results/`:

```bash
python -m exp.exp1_supply_functions     # Sec. 2: supply function framework, Prop. 2.4, Examples 2.6/2.8
python -m exp.exp2_equilibrium          # Sec. 3: Brouwer fixed point, Theorem 3.4
python -m exp.exp3_uniqueness           # Sec. 4: WGS, Theorem 4.3/4.4, Corollary 4.5/4.9
python -m exp.exp4_external_amm         # Sec. 5: AMM virtual agents, Proposition 5.4/5.8
python -m exp.exp5_fundamental_theorem  # Sec. 6: Theorem 6.4, batch exchange with external AMMs
python -m exp.exp6_convex_optimization  # Sec. 7: convex program, Theorem 7.3
python -m exp.exp7_protocol             # Sec. 8: Walraswap protocol simulation
python -m exp.exp8_nondiff_amm          # Appendix A: non-differentiable (concentrated-liquidity) AMMs
```

Or run `./reproduce.sh` to run the test suite followed by all eight experiments in
sequence.

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for the comparison of every
experiment's numerical output against the paper's stated properties, verdicts, and
the "Deviations from the paper" / "Not reproduced" sections.
