#!/usr/bin/env python
"""
Merge the determinant sets of several GQE runs and re-diagonalise.

    python tools/gqe_ensemble.py $B/n2_curve --where tag_R=2.1 gqe_ngates=80

WHY
---
GQE training is unreliable from seed to seed (plain-CAS N2 2.1 A: one run
in five beat CIPSI). Keeping only the best seed wastes the others. QSCI is
variational, so the union of several runs' determinant sets can only be at
least as good as the best of them: this measures how much a cheap ensemble
gains, at what subspace size, and compares that size against CIPSI and the
oracle at the SAME budget (the thesis yardsticks).

INPUT
-----
Every matching ok GQE row of <out>/results.csv whose run folder has
results/gqe_determinants.pkl (written by the patched train_pipeline.py;
runs before the patch have none and are listed as skipped). Rows are
grouped by system_id, so --where may select several geometries at once.

OUTPUT, per system
------------------
  each seed     its own set re-diagonalised (checks the dump against the
                logged energy) -- dim, error vs embedding FCI
  union         all seeds' alpha strings x all seeds' beta strings
  CIPSI/oracle  at the union's dimension (skipped above 2 M determinants)
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HARTREE_TO_KCAL = 627.5094740631
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def merge_sets(sets):
    """Union of alpha strings and of beta strings over (sa, sb) pairs."""
    sa = np.unique(np.concatenate([np.asarray(a, np.int64) for a, _ in sets]))
    sb = np.unique(np.concatenate([np.asarray(b, np.int64) for _, b in sets]))
    return sa, sb


def diagonalise(step2, sa, sb):
    """Lowest energy (incl. ecore) in span{|a>|b>: a in sa, b in sb}."""
    from pyscf.fci import selected_ci

    n = int(step2["n_emb"])
    nelec = (int(step2["n_alpha"]), int(step2["n_beta"]))
    myci = selected_ci.SelectedCI()
    myci.conv_tol = 1e-10
    e, _ = selected_ci.kernel_fixed_space(
        myci, np.asarray(step2["h1e"]), np.asarray(step2["h2e"]), n, nelec,
        (np.asarray(sa, np.int64), np.asarray(sb, np.int64)), nroots=1,
        tol=1e-10, max_cycle=200)
    return float(e) + float(step2["ecore"])


def references(step2_path, budget, max_det=2_000_000):
    """(E_exact, E_oracle, E_cipsi) at `budget` determinants, or Nones."""
    from quenais.quantum.det_analysis import (casci_vector, projected_energy,
                                              weight_curve)
    from quenais.quantum.det_expansion import cipsi_from_scratch
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle

    mol = load_from_dmet_pickle(str(step2_path))
    e_exact, flat, space = casci_vector(mol)
    if space.ndet > max_det:
        return e_exact, None, None
    n = int(min(budget, space.ndet))
    if n >= space.ndet:
        return e_exact, e_exact, e_exact
    order, _ = weight_curve(flat)
    e_or = projected_energy(mol, order[:n], space=space)
    sel, _ = cipsi_from_scratch(mol, n, space=space, verbose=False)
    return e_exact, e_or, projected_energy(mol, sel, space=space)


def main(argv=None):
    from show_rows import fmt, value

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--where", nargs="*", default=[])
    ap.add_argument("--no-cipsi", action="store_true",
                    help="skip the CIPSI/oracle comparison (slow on big spaces)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    rows = list(csv.DictReader(open(out / "results.csv")))
    conds = [w.split("=", 1) for w in ["solver=gqe", *a.where]]
    rows = [r for r in rows if r.get("status") == "ok"
            and all(fmt(value(r, k)) == fmt(v) for k, v in conds)]
    groups = defaultdict(list)
    for r in rows:
        groups[r["system_id"]].append(r)
    if not groups:
        print("no matching ok GQE rows")
        return 1

    for sid, rs in groups.items():
        rdir0 = Path(rs[0]["run_dir"])
        with open(rdir0 / "results" / "step2_hamiltonian.pkl", "rb") as fh:
            step2 = pickle.load(fh)
        e_ref = float(rs[0]["solver_ref_energy_Ha"])
        full = int(float(rs[0]["full_dim"]))
        print(f"\n=== system {sid}  R={rs[0].get('tag_R', '-')}  "
              f"{int(step2['n_emb'])} orbitals, {full} determinants ===")
        sets, skipped = [], []
        print(f"{'seed':>4} {'gates':>5} {'epochs':>6} {'dim':>8} "
              f"{'err logged':>11} {'err re-diag':>12}")
        for r in sorted(rs, key=lambda r: (fmt(value(r, "gqe_ngates")),
                                           fmt(value(r, "seed")))):
            dump = Path(r["run_dir"]) / "results" / "gqe_determinants.pkl"
            if not dump.exists():
                skipped.append(r["run_id"])
                continue
            with open(dump, "rb") as fh:
                d = pickle.load(fh)
            if d.get("strs_a") is None:
                skipped.append(r["run_id"] + " (no strings in dump)")
                continue
            sa, sb = d["strs_a"], d["strs_b"]
            e = diagonalise(step2, sa, sb)
            sets.append((sa, sb))
            print(f"{fmt(value(r, 'seed')):>4} {fmt(value(r, 'gqe_ngates')):>5} "
                  f"{fmt(value(r, 'gqe_max_iters')):>6} {len(sa) * len(sb):>8} "
                  f"{float(r['err_solver_kcal']):>11.4f} "
                  f"{(e - e_ref) * HARTREE_TO_KCAL:>12.4f}")
        if skipped:
            print(f"  skipped (no dump -- run before the patch?): {len(skipped)}")
        if len(sets) < 2:
            print("  fewer than 2 dumps: nothing to merge")
            continue
        sa, sb = merge_sets(sets)
        dim = len(sa) * len(sb)
        e_u = diagonalise(step2, sa, sb)
        print(f"union of {len(sets)}: dim {dim} ({100 * dim / full:.1f}%), "
              f"err {(e_u - e_ref) * HARTREE_TO_KCAL:.4f} kcal/mol")
        if not a.no_cipsi:
            try:
                _ex, e_or, e_ci = references(rdir0 / "results" / "step2_hamiltonian.pkl", dim)
                if e_ci is None:
                    print("  CIPSI/oracle skipped: space > 2 M determinants")
                else:
                    print(f"  same budget: oracle {(e_or - e_ref) * HARTREE_TO_KCAL:.4f}"
                          f", CIPSI {(e_ci - e_ref) * HARTREE_TO_KCAL:.4f} kcal/mol")
            except Exception as exc:
                print(f"  CIPSI/oracle failed: {exc!r}"[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
