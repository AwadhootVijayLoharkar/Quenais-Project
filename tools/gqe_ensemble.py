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
  each seed     its own determinant list re-diagonalised (checks the dump
                against the logged energy) -- dim, error vs embedding FCI
  union         all seeds' determinants together (a list union, not a
                product of string sets: GQE's QSCI space is a LIST of
                (alpha, beta) determinants)
  CIPSI/oracle  at the union's dimension (skipped above 2 M determinants)
  coverage      weight of the EXACT wavefunction inside the best single
                set and inside the union, split by excitation level
                relative to the dominant determinant -- shows WHICH kind of
                determinants GQE never reaches (e.g. all quadruples)
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


def dets_from_dump(d):
    """(N, 2) int array of (alpha, beta) occupation strings, or None.
    Dumps written before the fix hold only strs_a/strs_b (the first two
    determinants) and are unusable."""
    dets = d.get("dets")
    if dets is None:
        return None
    dets = np.asarray(dets, dtype=np.int64)
    if dets.ndim == 2 and dets.shape[1] != 2 and dets.shape[0] == 2:
        dets = dets.T
    if dets.ndim != 2 or dets.shape[1] != 2:
        return None
    return dets


def dets_to_indices(dets, astrs, bstrs):
    """(alpha, beta) occupation pairs -> full indices addr_a * nb + addr_b."""
    amap = {int(v): i for i, v in enumerate(astrs)}
    bmap = {int(v): i for i, v in enumerate(bstrs)}
    nb = len(bstrs)
    return np.unique(np.array([amap[int(a)] * nb + bmap[int(b)]
                               for a, b in dets], dtype=np.int64))


def merge_indices(index_sets):
    """Union of several determinant index sets."""
    return np.unique(np.concatenate([np.asarray(x, np.int64) for x in index_sets]))


def _popcount(x):
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros(x.shape, dtype=np.int64)
    while np.any(x):
        out += x & 1
        x = x >> 1
    return out


def coverage_by_level(flat, astrs, bstrs, indices):
    """
    Exact-wavefunction weight per excitation level (relative to the
    largest-|c| determinant), total and inside the determinant set.

    flat: exact CI vector, index = addr_alpha * len(bstrs) + addr_beta.
    Returns {level: (n_dets, weight_total, weight_in_set, n_in_set)} and
    the total weight captured.
    """
    na, nb = len(astrs), len(bstrs)
    c2 = np.abs(np.asarray(flat).reshape(na, nb)) ** 2
    ia0, ib0 = np.unravel_index(int(np.argmax(c2)), c2.shape)
    la = _popcount(np.asarray(astrs) ^ astrs[ia0])
    lb = _popcount(np.asarray(bstrs) ^ bstrs[ib0])
    level = (la[:, None] + lb[None, :]) // 2
    inside = np.zeros(na * nb, dtype=bool)
    inside[np.asarray(indices, np.int64)] = True
    inside = inside.reshape(na, nb)
    table = {}
    for lv in np.unique(level):
        m = level == lv
        table[int(lv)] = (int(m.sum()), float(c2[m].sum()),
                          float(c2[m & inside].sum()), int((m & inside).sum()))
    return table, float(c2[inside].sum())


def print_coverage(name, table, captured):
    print(f"  {name}: captures {100 * captured:.2f}% of the exact wavefunction")
    print(f"    {'level':>5} {'exact weight':>13} {'captured':>9} {'dets in set':>18}")
    for lv, (n, w, wi, ni) in table.items():
        if w < 1e-6 and ni == 0:
            continue
        print(f"    {lv:>5} {100 * w:>12.3f}% {100 * wi:>8.3f}% {ni:>8} / {n:<8}")


def exact_vector(step2_path):
    from quenais.quantum.det_analysis import casci_vector
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle

    mol = load_from_dmet_pickle(str(step2_path))
    e_exact, flat, space = casci_vector(mol)
    return mol, e_exact, flat, space


def references(step2_path, budget, max_det=2_000_000, cached=None):
    """(E_exact, E_oracle, E_cipsi) at `budget` determinants, or Nones."""
    from quenais.quantum.det_analysis import projected_energy, weight_curve
    from quenais.quantum.det_expansion import cipsi_from_scratch

    mol, e_exact, flat, space = cached or exact_vector(step2_path)
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

    from quenais.quantum.det_analysis import projected_energy

    for sid, rs in groups.items():
        rdir0 = Path(rs[0]["run_dir"])
        step2_path = rdir0 / "results" / "step2_hamiltonian.pkl"
        full = int(float(rs[0]["full_dim"]))
        print(f"\n=== system {sid}  R={rs[0].get('tag_R', '-')}  {full} determinants ===")
        if full > 2_000_000:
            print("  space > 2 M determinants: skipped")
            continue
        cached = exact_vector(step2_path)
        mol, e_exact, flat, space = cached
        sets, skipped = [], []
        print(f"{'seed':>4} {'gates':>5} {'angles':>6} {'dets':>6} "
              f"{'err logged':>11} {'err re-diag':>12}")
        for r in sorted(rs, key=lambda r: (fmt(value(r, "gqe_ngates")),
                                           fmt(value(r, "seed")))):
            dump = Path(r["run_dir"]) / "results" / "gqe_determinants.pkl"
            if not dump.exists():
                skipped.append(r["run_id"])
                continue
            with open(dump, "rb") as fh:
                dets = dets_from_dump(pickle.load(fh))
            if dets is None:
                skipped.append(r["run_id"])
                continue
            idx = dets_to_indices(dets, space.astrs, space.bstrs)
            e = projected_energy(mol, idx, space=space)
            sets.append((idx, e))
            ang = "yes" if value(r, "gqe_pool_angles") not in ("-", None, "") else "ccsd"
            print(f"{fmt(value(r, 'seed')):>4} {fmt(value(r, 'gqe_ngates')):>5} "
                  f"{ang:>6} {len(idx):>6} "
                  f"{float(r['err_solver_kcal']):>11.4f} "
                  f"{(e - e_exact) * HARTREE_TO_KCAL:>12.4f}")
        if skipped:
            print(f"  skipped (no usable dump -- run before the fix?): {len(skipped)}")
        if not sets:
            continue
        best = min(sets, key=lambda t: t[1])
        print_coverage("best single seed",
                       *coverage_by_level(flat, space.astrs, space.bstrs, best[0]))
        if len(sets) >= 2:
            union = merge_indices([x for x, _ in sets])
            e_u = projected_energy(mol, union, space=space)
            print(f"union of {len(sets)}: {len(union)} dets ({100 * len(union) / full:.1f}%), "
                  f"err {(e_u - e_exact) * HARTREE_TO_KCAL:.4f} kcal/mol")
            print_coverage("union", *coverage_by_level(flat, space.astrs,
                                                       space.bstrs, union))
            budget = len(union)
        else:
            budget = len(best[0])
        if not a.no_cipsi:
            try:
                _ex, e_or, e_ci = references(step2_path, budget, cached=cached)
                print(f"  same budget ({budget} dets): oracle "
                      f"{(e_or - e_exact) * HARTREE_TO_KCAL:.4f}, CIPSI "
                      f"{(e_ci - e_exact) * HARTREE_TO_KCAL:.4f} kcal/mol")
            except Exception as exc:
                print(f"  CIPSI/oracle failed: {exc!r}"[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
