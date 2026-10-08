#!/usr/bin/env python
"""
GQE against the oracle bound and CIPSI, at GQE's own determinant budget.

    python tools/gqe_vs_cipsi.py <project dir> [<project dir> ...]

Each project dir must hold results/step2_hamiltonian.pkl (from step 2 or from
tools/export_cas_hamiltonian.py) and results/gqe_train.log from a finished GQE
run. For each dir this prints, in mHa against exact diagonalisation of the
same Hamiltonian:

    GQE      final Global-refined(best_so_far) error from the training log
    oracle   best energy from any N determinants (top-|c| of the exact vector)
    CIPSI    classical perturbative selection grown to the same N

with N = GQE's final QSCI subspace dimension, and then mean +- std over all
dirs. Read-only.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

KEY_E = "Global-refined(best_so_far)/energy - R-CASCI"
KEY_N = "Global-refined(best_so_far)/subspace_dim"


def one(d):
    from quenais.quantum.det_analysis import (casci_vector, projected_energy,
                                              weight_curve)
    from quenais.quantum.det_expansion import cipsi_from_scratch
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle
    from quenais.visualization.plots import parse_gqe_log

    d = Path(d)
    rows = parse_gqe_log(str(d / "results" / "gqe_train.log"))
    errs = [r[KEY_E] for r in rows if r.get(KEY_E) is not None]
    dims = [r[KEY_N] for r in rows if r.get(KEY_N) is not None]
    if not errs or not dims:
        return None
    n = int(dims[-1])
    mol = load_from_dmet_pickle(str(d / "results" / "step2_hamiltonian.pkl"))
    e_exact, flat, space = casci_vector(mol)
    order, cum = weight_curve(flat)
    e_or = projected_energy(mol, order[:n], space=space)
    sel, _ = cipsi_from_scratch(mol, n, space=space, verbose=False)
    e_ci = projected_energy(mol, sel, space=space)
    return {"dir": d.name, "N": n, "ndet": space.ndet, "w1": float(cum[0]),
            "epochs": len(rows), "gqe": 1e3 * errs[-1],
            "oracle": 1e3 * (e_or - e_exact), "cipsi": 1e3 * (e_ci - e_exact)}


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    res = []
    print(f"{'run':<22} {'N':>6} {'of':>7} {'w1':>6} {'epochs':>6} "
          f"{'GQE mHa':>10} {'oracle':>9} {'CIPSI':>9}")
    for d in argv:
        r = one(d)
        if r is None:
            print(f"{Path(d).name:<22}  no finished GQE log")
            continue
        res.append(r)
        print(f"{r['dir']:<22} {r['N']:>6} {r['ndet']:>7} {r['w1']:>6.3f} "
              f"{r['epochs']:>6} {r['gqe']:>10.3f} {r['oracle']:>9.3f} "
              f"{r['cipsi']:>9.3f}")
    if len(res) > 1:
        def ms(k):
            v = [r[k] for r in res]
            return f"{statistics.fmean(v):.3f} +- {statistics.stdev(v):.3f}"
        print(f"\nmean over {len(res)} runs (mHa):  GQE {ms('gqe')}   "
              f"oracle {ms('oracle')}   CIPSI {ms('cipsi')}")
        wins = sum(r["gqe"] < r["cipsi"] for r in res)
        print(f"GQE beat CIPSI in {wins} of {len(res)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
