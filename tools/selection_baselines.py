#!/usr/bin/env python
"""
Classical determinant-selection baselines at the SAME size as finished
quantum runs, in both orbital bases. No quantum run is repeated.

For every matching row of <out>/results.csv (default: solver=gqe), on that
run's own step-2 embedding Hamiltonian and at its subspace_dim N:

  oracle   top-N determinants by exact |c|, re-diagonalised (uses the answer)
  CIPSI    the package's growth CIPSI (start from the lowest determinant,
           double by Epstein-Nesbet PT2, never removes)
  HCI      heat-bath CI (Holmes, Tubman, Umrigar 2016), the standard QSCI
           baseline: add a if max_i |H_ai c_i| > eps; eps tuned to the budget,
           topped up to exactly N by heat-bath score
  ASCI     fixed-size adaptive-sampling CI seeded with the CIPSI set:
           re-ranks members and candidates every iteration, can drop
           determinants (Tubman et al. 2016)

each in the raw DMET basis ("emb", what the results.csv columns use) and in
the embedded Hartree-Fock basis ("mo", the basis GQE uses since the fix).
Selection is basis dependent, so a fair classical baseline is the BEST of
the classical columns.

    python tools/selection_baselines.py $B/n2_curve --where solver=gqe gqe_basis=mo
    python tools/selection_baselines.py $B/n2_curve --where solver=sqd sqd_subspace=list

Writes <out>/selection_baselines.csv. Errors in kcal/mol vs embedding FCI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

HARTREE_TO_KCAL = 627.509474


def _value(row, key):
    v = row.get(key)
    if v not in (None, ""):
        return str(v)
    try:
        p = json.loads(row.get("params_json") or "{}")
    except ValueError:
        return ""
    v = p.get(key)
    return "" if v is None else str(v)


def _match(row, where):
    for cond in where:
        k, _, v = cond.partition("=")
        if _value(row, k) != v:
            return False
    return row.get("status") == "ok"


def baselines(step2_path, n, bases, cache):
    """{basis: {"oracle":E, "cipsi":E, "asci":E}} plus 'exact' (Ha)."""
    from quenais.quantum.det_analysis import casci_vector, projected_energy, weight_curve
    from quenais.quantum.det_expansion import (asci_fixed_size, cipsi_from_scratch,
                                               hci_fixed_size)
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle

    digest = hashlib.sha256(Path(step2_path).read_bytes()).hexdigest()[:16]
    out = {}
    for b in bases:
        key = (digest, n, b)
        if key in cache:
            out[b] = cache[key]
            continue
        mol = load_from_dmet_pickle(str(step2_path), orbital_basis=b,
                                    cache_dir=Path("/tmp") / "qn_baselines" / b)
        e_exact, flat, space = casci_vector(mol)
        order, _ = weight_curve(flat)
        e_or = projected_energy(mol, order[:n], space=space)
        sel, _ = cipsi_from_scratch(mol, n, space=space, verbose=False)
        e_ci = projected_energy(mol, sel, space=space)
        _, e_as = asci_fixed_size(mol, n, space=space, seed=sel)
        _, e_hci, _eps = hci_fixed_size(mol, n, space=space)
        cache[key] = {"exact": e_exact, "oracle": e_or, "cipsi": e_ci,
                      "asci": e_as, "hci": e_hci, "ndet": space.ndet}
        out[b] = cache[key]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--where", nargs="*", default=["solver=gqe"])
    ap.add_argument("--basis", choices=["both", "emb", "mo"], default="both")
    a = ap.parse_args(argv)

    out = Path(a.out)
    rows = list(csv.DictReader(open(out / "results.csv")))
    rows = [r for r in rows if _match(r, a.where)]
    if not rows:
        sys.exit("no matching rows (run --collect-only first?)")
    bases = {"both": ["embedding", "mo"], "emb": ["embedding"], "mo": ["mo"]}[a.basis]
    max_det = int(os.environ.get("QUENAIS_SELECTION_MAX_DET", 2_000_000))
    rows.sort(key=lambda r: (float(r.get("tag_R") or 0), _value(r, "gqe_qsci_max_dim"),
                             _value(r, "gqe_pool_angles"), _value(r, "seed")))

    cache, table = {}, []
    hdr = ["R", "N", "seed", "solver", "HCI_emb", "HCI_mo", "CIPSI_emb", "CIPSI_mo",
           "ASCI_emb", "ASCI_mo", "oracle_mo", "best_classical", "best/solver"]
    print("  ".join(f"{h:>11s}" for h in hdr))
    for r in rows:
        try:
            n = int(float(r["subspace_dim"]))
        except (TypeError, ValueError, KeyError):
            continue
        p = out / "runs" / r["run_id"] / "results" / "step2_hamiltonian.pkl"
        if not p.exists():
            print(f"  {r['run_id']}: no step-2 pickle, skipped")
            continue
        try:
            res = baselines(p, n, bases, cache)
        except Exception as exc:          # one bad row must not stop the table
            print(f"  {r['run_id']}: {exc!r}"[:200])
            continue
        any_b = next(iter(res.values()))
        if any_b["ndet"] > max_det or n >= any_b["ndet"]:
            continue
        ex = any_b["exact"]
        k = lambda e: (e - ex) * HARTREE_TO_KCAL  # noqa: E731
        vals = {f"{m}_{'emb' if b == 'embedding' else 'mo'}": k(res[b][m])
                for b in res for m in ("hci", "cipsi", "asci", "oracle")}
        classical = [v for kk, v in vals.items() if not kk.startswith("oracle")]
        best = min(classical)
        sol = float(r["err_solver_kcal"])
        line = {"R": r.get("tag_R"), "N": n, "seed": _value(r, "seed"),
                "solver": sol,
                "HCI_emb": vals.get("hci_emb"), "HCI_mo": vals.get("hci_mo"),
                "CIPSI_emb": vals.get("cipsi_emb"), "CIPSI_mo": vals.get("cipsi_mo"),
                "ASCI_emb": vals.get("asci_emb"), "ASCI_mo": vals.get("asci_mo"),
                "oracle_emb": vals.get("oracle_emb"), "oracle_mo": vals.get("oracle_mo"),
                "best_classical": best,
                "best/solver": best / sol if sol > 1e-9 else float("inf"),
                "run_id": r["run_id"], "budget": _value(r, "gqe_qsci_max_dim"),
                "angles": _value(r, "gqe_pool_angles")}
        table.append(line)
        fmt = lambda v: "-" if v is None else (f"{v:11.4f}" if isinstance(v, float) else f"{v!s:>11}")  # noqa: E731
        print("  ".join(fmt(line[h]) for h in hdr))
    with open(out / "selection_baselines.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]) if table else hdr)
        w.writeheader()
        w.writerows(table)
    print(f"\n{len(table)} rows -> {out / 'selection_baselines.csv'}")
    print("best/solver > 1 means the quantum solver beat the best classical "
          "selection at the same size.")


if __name__ == "__main__":
    main()
