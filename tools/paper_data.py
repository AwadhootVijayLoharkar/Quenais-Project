#!/usr/bin/env python
"""
Export the small data files behind the paper figures. Read-only on runs.

    python tools/paper_data.py $B/n2_curve   --where solver=gqe gqe_basis=mo
    python tools/paper_data.py $B/h_chain    --where solver=gqe gqe_basis=mo
    python tools/paper_data.py $B/h14_depth  --where solver=gqe

Writes <out>/paper_data/:

  train/<run_id>.csv     per-epoch metrics parsed from the GQE training log
                         (all numeric keys: raw circuit, local/global refined,
                         best-so-far energies minus reference, subspace dims)
  weights/<system>_<basis>.csv
                         cumulative weight of the exact embedded state for the
                         top-k determinants on a log grid of k (compactness curve)
  selection/<run_id>.json
                         what GENIE actually selected, against the exact state
                         and against CIPSI at the same N (same basis):
                           frac_topk  fraction of the exact top-k determinants
                                      inside the set, for k on a log grid
                           captured   exact weight inside the set
                           by_level   exact weight per excitation level, total
                                      and inside the set
  rows.csv               run_id + the parameters needed to label the plots

Spaces larger than --max-det (default 2e6) get train/ only.
Zip <out>/paper_data and send it; it is a few MB at most.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

LABEL_KEYS = ("run_id", "system_id", "molecule", "basis", "tag_R", "solver",
              "seed", "gqe_ngates", "gqe_qsci_max_dim", "gqe_pool_angles",
              "gqe_basis", "subspace_dim", "full_dim", "err_solver_kcal", "n_epochs")


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


def _grid(n):
    """Integers 1..n, log-spaced, always including n."""
    g = np.unique(np.round(np.logspace(0, np.log10(max(n, 1)), 60)).astype(np.int64))
    return np.unique(np.append(g[g <= n], n))


def _find_log(rdir):
    for p in sorted((rdir / "results").glob("*.log")):
        try:
            if "[epoch " in p.read_text(errors="ignore"):
                return p
        except OSError:
            pass
    return None


def export_train(rdir, dest):
    from quenais.visualization.plots import parse_gqe_log

    log = _find_log(rdir)
    if log is None:
        return False
    rows = parse_gqe_log(str(log))
    if not rows:
        return False
    keys = sorted({k for r in rows for k, v in r.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    keys = ["epoch"] + [k for k in keys if k != "epoch"]
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    return True


def set_profile(idx, order, rank, flat, space, grid):
    from gqe_ensemble import coverage_by_level

    idx = np.asarray(idx, np.int64)
    inside = np.zeros(space.ndet, dtype=bool)
    inside[idx] = True
    hit = np.cumsum(inside[order])            # hits among the exact top-k
    table, captured = coverage_by_level(flat, space.astrs, space.bstrs, idx)
    r = np.sort(rank[idx])
    return {
        "N": int(idx.size),
        "captured": float(captured),
        "overlap_topN": int(hit[idx.size - 1]),
        "frac_topk": {int(k): float(hit[k - 1] / k) for k in grid},
        "rank_quantiles": {q: int(r[min(len(r) - 1, int(q * len(r)))])
                           for q in (0.5, 0.9, 0.99)},
        "by_level": {int(lv): {"n": n, "w": w, "w_in": wi, "n_in": ni}
                     for lv, (n, w, wi, ni) in table.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--where", nargs="*", default=["solver=gqe"])
    ap.add_argument("--max-det", type=float, default=2e6)
    ap.add_argument("--no-cipsi", action="store_true",
                    help="skip the CIPSI set profile (slow above ~1e5 determinants)")
    a = ap.parse_args(argv)

    from gqe_ensemble import dets_from_dump, dets_to_indices
    from quenais.quantum.det_analysis import casci_vector, weight_curve
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle

    out = Path(a.out)
    rows = [r for r in csv.DictReader(open(out / "results.csv")) if _match(r, a.where)]
    if not rows:
        sys.exit("no matching rows")
    dst = out / "paper_data"
    for sub in ("train", "weights", "selection"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    labels, cache, cipsi_cache = [], {}, {}
    for r in rows:
        rid = r["run_id"]
        rdir = out / "runs" / rid
        labels.append({k: _value(r, k) for k in LABEL_KEYS})
        ok_t = export_train(rdir, dst / "train" / f"{rid}.csv")

        p = rdir / "results" / "step2_hamiltonian.pkl"
        full = float(r.get("full_dim") or 0)
        msg = f"{rid}: train {'ok' if ok_t else '-'}"
        if not p.exists() or full == 0 or full > a.max_det:
            print(msg + ", exact skipped")
            continue
        basis = "mo" if _value(r, "gqe_basis") == "mo" else "embedding"
        key = (hashlib.sha256(p.read_bytes()).hexdigest()[:16], basis)
        if key not in cache:
            mol = load_from_dmet_pickle(str(p), orbital_basis=basis,
                                        cache_dir=Path("/tmp") / "qn_paper" / basis)
            e_exact, flat, space = casci_vector(mol)
            order, cum = weight_curve(flat)
            rank = np.empty_like(order)
            rank[order] = np.arange(order.size)
            grid = _grid(space.ndet)
            n99 = int(np.searchsorted(cum, 0.99) + 1)
            with open(dst / "weights" / f"{r['system_id']}_{basis}.csv", "w",
                      newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["k", "cum_weight", "w_k"])
                w2 = np.asarray(flat) ** 2
                for k in grid:
                    w.writerow([int(k), float(cum[k - 1]), float(w2[order[k - 1]])])
            cache[key] = (mol, e_exact, flat, space, order, rank, grid, n99)
        mol, e_exact, flat, space, order, rank, grid, n99 = cache[key]

        dump = rdir / "results" / "gqe_determinants.pkl"
        if not dump.exists():
            print(msg + ", weights ok, no determinant dump")
            continue
        with open(dump, "rb") as fh:
            dets = dets_from_dump(pickle.load(fh))
        if dets is None:
            print(msg + ", weights ok, dump unusable")
            continue
        idx = dets_to_indices(dets, space.astrs, space.bstrs)
        res = {"run_id": rid, "system_id": r["system_id"], "basis": basis,
               "ndet": int(space.ndet), "n99": n99,
               "genie": set_profile(idx, order, rank, flat, space, grid)}
        if not a.no_cipsi:
            ck = (key, int(idx.size))
            if ck not in cipsi_cache:
                from quenais.quantum.det_expansion import cipsi_from_scratch
                sel, _ = cipsi_from_scratch(mol, int(idx.size), space=space, verbose=False)
                cipsi_cache[ck] = set_profile(sel, order, rank, flat, space, grid)
            res["cipsi"] = cipsi_cache[ck]
        with open(dst / "selection" / f"{rid}.json", "w") as fh:
            json.dump(res, fh, indent=1)
        print(msg + f", weights ok, selection ok (N={idx.size}, "
                    f"captured {res['genie']['captured']:.4f})")

    with open(dst / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LABEL_KEYS))
        w.writeheader()
        w.writerows(labels)
    print(f"\n{len(labels)} rows -> {dst}")


if __name__ == "__main__":
    main()
