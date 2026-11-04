#!/usr/bin/env bash
# Runs the test suite and every experiment of the Walraswap replication,
# writing all artifacts into results/.
set -uo pipefail
cd "$(dirname "$0")"

echo "=== Running test suite ==="
python -m pytest || exit 1

for exp in exp1_supply_functions exp2_equilibrium exp3_uniqueness \
           exp4_external_amm exp5_fundamental_theorem exp6_convex_optimization \
           exp7_protocol exp8_nondiff_amm; do
  echo "=== Running exp.${exp} ==="
  python -m "exp.${exp}" || exit 1
done

echo "=== Done. See results/RESULTS.md for the comparison to the paper. ==="
