#!/usr/bin/env python
"""
Regime diagnostics for finished runs, next to how GENIE actually did.

For every matching row of <out>/results.csv: load that run's step-2
embedding, compute cheap correlation indicators (quenais/quantum/regime.py:
CCSD-estimated HF weight, MP2 natural-orbital occupations, optionally the
exact leading-determinant weight), print the router's recommendation, and,
if <out>/selection_baselines.csv exists, the measured ratio
GENIE error / best classical error at the same subspace size.

    python tools/regime_diagnostics.py $B/h_chain --where solver=gqe gqe_basis=mo --exact
    python tools/regime_diagnostics.py $B/n2_curve --where solver=gqe gqe_basis=mo --exact

Writes <out>/regime_diagnostics.csv. Use it to check the thresholds: rows
the router calls "strong" should be the ones where the ratio is large.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


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


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--where", nargs="*", default=["solver=gqe"])
    ap.add_argument("--exact", action="store_true",
                    help="also the exact leading-determinant weight (validation only)")
    ap.add_argument("--weak-w1", type=float, default=None)
    ap.add_argument("--strong-w1", type=float, default=None)
    ap.add_argument("--strong-noon", type=float, default=None)
    ap.add_argument("--budget", type=int, default=None,
                    help="determinant budget for the compactness test "
                         "(default: each run's own subspace size N)")
    a = ap.parse_args(argv)

    from quenais.quantum.gqe_adapter import load_from_dmet_pickle
    from quenais.quantum.regime import Thresholds, compute_diagnostics, recommend

    thr = Thresholds(**{k: v for k, v in (("weak_w1", a.weak_w1),
                                          ("strong_w1", a.strong_w1),
                                          ("strong_noon", a.strong_noon))
                        if v is not None})
    out = Path(a.out)
    rows = [r for r in csv.DictReader(open(out / "results.csv")) if _match(r, a.where)]
    if not rows:
        sys.exit("no matching rows (run --collect-only first?)")
    base = {}
    bpath = out / "selection_baselines.csv"
    if bpath.exists():
        base = {r["run_id"]: r for r in csv.DictReader(open(bpath))}
    max_det = int(os.environ.get("QUENAIS_SELECTION_MAX_DET", 2_000_000))
    rows.sort(key=lambda r: (r.get("molecule", ""), _float(r.get("tag_R")) or 0,
                             _value(r, "gqe_qsci_max_dim"), _value(r, "seed")))

    cache, table = {}, []
    hdr = ["molecule", "R", "N", "seed", "w1_exact", "ccsd_ok", "n_est_mp2",
           "n_est_cc", "n_est", "n99_exact", "regime", "route", "genie",
           "best_cl", "genie/best"]
    print("  ".join(f"{h:>10s}" for h in hdr))
    for r in rows:
        p = out / "runs" / r["run_id"] / "results" / "step2_hamiltonian.pkl"
        if not p.exists():
            print(f"  {r['run_id']}: no step-2 pickle, skipped")
            continue
        key = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        if key not in cache:
            mol = load_from_dmet_pickle(str(p), orbital_basis="embedding",
                                        cache_dir=Path("/tmp") / "qn_regime")
            cache[key] = compute_diagnostics(mol, exact=a.exact, max_det=max_det)
        d = cache[key]
        n_sub = _float(r.get("subspace_dim"))
        budget = a.budget if a.budget is not None else (int(n_sub) if n_sub else None)
        regime, route, why = recommend(d, thr, budget=budget)
        short_route = ("GENIE-deep" if "deeper circuit" in route
                       else "GENIE?" if "GENIE (or" in route
                       else "GENIE" if route.startswith("GENIE (") or route == "GENIE"
                       else "not GENIE")
        b = base.get(r["run_id"], {})
        sol = _float(r.get("err_solver_kcal"))
        best = _float(b.get("best_classical"))
        ratio = (sol / best) if (sol is not None and best and best > 1e-6) else None
        line = {"run_id": r["run_id"], "molecule": r.get("molecule"),
                "R": r.get("tag_R"), "N": r.get("subspace_dim"),
                "seed": _value(r, "seed"), **d, "regime": regime, "route": route,
                "reason": why, "budget": budget, "route_short": short_route,
                "genie": sol, "best_cl": best, "genie/best": ratio}
        table.append(line)

        def fmt(v):
            if v is None or v == "":
                return f"{'-':>10}"
            if isinstance(v, bool):
                return f"{str(v):>10}"
            if isinstance(v, float):
                return f"{v:10.4f}"
            return f"{str(v):>10}"
        shown = dict(line, route=short_route)
        print("  ".join(fmt(shown.get(h)) for h in hdr))

    keys = sorted({k for t in table for k in t})
    with open(out / "regime_diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    print(f"\n{len(table)} rows -> {out / 'regime_diagnostics.csv'}")
    print(f"thresholds: {thr}")
    print("genie/best = GENIE error / best classical error at the same size "
          "(1 = tie, >1 classical better).")


if __name__ == "__main__":
    main()
