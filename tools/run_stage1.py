#!/usr/bin/env python
"""
Stage 1 driver -- the classical control.

Goes in: tools/run_stage1.py

Runs CIPSI with no quantum input up to the sizes that matter, and compares
against three things already measured:

    the oracle bound   (best possible at that size, from stage 0)
    GQE               (what the quantum pipeline actually achieved)
    chemical accuracy (1.6 mHa)

WHAT THE ANSWER MEANS
---------------------
If CIPSI from scratch lands near the oracle at N=2439, then classical selection
already solves this system and the quantum sampler is not contributing -- which
is worth knowing before building anything else, and is a publishable result in
its own right.

If CIPSI stalls well short of the oracle, there is room for a better proposal
distribution, and the quantum seed has something to add.

Usage:
    python tools/run_stage1.py                # LiH then ScH
    python tools/run_stage1.py --system ScH --threads 24
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GOLDEN = REPO / "tests" / "regression" / "golden"

# Sizes to compare at. 144 is ScH's N_chem from stage 0; 2439 is the number of
# distinct determinants GQE found across its whole run.
TARGETS = {"LiH": [2, 4, 8, 34], "ScH": [144, 423, 2439], "N2": [16]}

# Oracle errors at those sizes, measured in stage 0 (mHa).
ORACLE_MHA = {
    "LiH": {2: 0.5378, 4: 0.3635, 8: 0.1587, 34: 0.0000},
    "ScH": {144: 1.1606, 423: 0.2036, 2439: 0.0045},
}

GQE_MHA = {"ScH": 20.9, "LiH": 0.36}


def load_reference(system):
    from tests.regression.reference_values import SYSTEMS, Tier

    ref = SYSTEMS[system]
    energy, tier = ref["energies"]["DMET_CASCI"]
    if tier is not Tier.DETERMINISTIC:
        raise RuntimeError(f"{system} DMET_CASCI is {tier}, not DETERMINISTIC")
    return energy


def run_system(system, threads=1):
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle
    try:
        from quenais.quantum import det_analysis as da
        from quenais.quantum import det_expansion as dx
    except ImportError:
        import det_analysis as da
        import det_expansion as dx

    e_ref = load_reference(system)
    mol = load_from_dmet_pickle(
        str(GOLDEN / system / "step2_hamiltonian.pkl"), num_threads=threads)
    space = da.DeterminantSpace(mol.norb, mol.nelec)

    print(f"\n{'='*72}\n{system}   {space}\n{'='*72}")

    rows = []
    for target in TARGETS[system]:
        if target > space.ndet:
            continue
        print(f"\n-- CIPSI to N={target} --")
        sel, history = dx.cipsi_from_scratch(mol, target, space=space)
        e_final = history[-1]["energy"]
        err = 1e3 * (e_final - e_ref)
        oracle = ORACLE_MHA.get(system, {}).get(target)

        row = {
            "system": system,
            "n_det": int(sel.size),
            "energy": e_final,
            "err_cipsi_mha": err,
            "err_oracle_mha": oracle,
            "gap_to_oracle_mha": (err - oracle) if oracle is not None else None,
            "iterations": len(history),
        }
        rows.append(row)

        print(f"   CIPSI  N={sel.size:>7d}  err = {err:9.4f} mHa")
        if oracle is not None:
            print(f"   oracle N={target:>7d}  err = {oracle:9.4f} mHa"
                  f"   -> CIPSI is {err - oracle:+.4f} mHa above the best possible")

    gqe = GQE_MHA.get(system)
    print(f"\n{'-'*72}")
    print(f"  {'N':>8}  {'CIPSI':>12}  {'oracle':>12}  {'GQE':>12}")
    for r in rows:
        o = f"{r['err_oracle_mha']:.4f}" if r["err_oracle_mha"] is not None else "--"
        g = f"{gqe:.4f}" if (gqe and r["n_det"] >= 2000) else "--"
        print(f"  {r['n_det']:>8d}  {r['err_cipsi_mha']:>12.4f}  {o:>12}  {g:>12}")
    print(f"  (all in mHa; chemical accuracy = 1.6)")

    out = GOLDEN / system / f"stage1_{system}_cipsi.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", default=None, choices=["LiH", "N2", "ScH"])
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    systems = [args.system] if args.system else ["LiH", "ScH"]
    allrows = []
    for s in systems:
        allrows += run_system(s, threads=args.threads)

    with open(GOLDEN / "stage1_summary.json", "w") as fh:
        json.dump(allrows, fh, indent=2)


if __name__ == "__main__":
    main()