#!/usr/bin/env bash
# Build the paper data archive from benchmark output folders.
#   bash tools/make_data_bundle.sh $B genie_paper_data.tar.gz
# Contains CSVs, paper_data/, and per run row.json + step2_hamiltonian.pkl.
# Excludes GQE training outputs and all other pickles.
set -euo pipefail
B="$1"; OUT="${2:-genie_paper_data.tar.gz}"
DIRS="n2_curve n2_dz h_chain h_chain_cmp blind fe2bpym"
LIST=$(mktemp)
cd "$B"
for d in $DIRS; do
  [ -d "$d" ] || { echo "skip $d (missing)"; continue; }
  for f in results.csv results_agg.csv selection_baselines.csv regime_diagnostics.csv router.txt; do
    [ -f "$d/$f" ] && echo "$d/$f" >> "$LIST"
  done
  [ -d "$d/paper_data" ] && find "$d/paper_data" -type f >> "$LIST"
  find "$d/runs" -maxdepth 2 -name row.json >> "$LIST" 2>/dev/null || true
  find "$d/runs" -maxdepth 3 -path "*/results/step2_hamiltonian.pkl" >> "$LIST" 2>/dev/null || true
  find "$d/runs" -maxdepth 3 -path "*/results/gqe_determinants.pkl" >> "$LIST" 2>/dev/null || true
done
echo "$(wc -l < "$LIST") files"
tar -czf "$OLDPWD/$OUT" -T "$LIST"
rm -f "$LIST"
ls -lh "$OLDPWD/$OUT"
