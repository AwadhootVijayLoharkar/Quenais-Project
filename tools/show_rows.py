#!/usr/bin/env python
"""
Print selected columns of <out>/results.csv as an aligned table.

    python tools/show_rows.py $B/h8_smoke                       # lucj_opt SQD rows
    python tools/show_rows.py $B/n2_curve --where solver=gqe tag_R=2.1 \
        --cols tag_R gqe_ngates gqe_max_iters seed err_solver_kcal t_step3_s

--where KEY=VALUE filters on the CSV text, or on params_json for flags that
are not columns (gqe_ngates, gqe_max_iters, ...). Missing = "-".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_WHERE = ["solver=sqd", "ansatz=lucj_opt"]
DEFAULT_COLS = ["tag_R", "shots_per_circuit", "seed", "err_solver_kcal",
                "err_lucj_kcal", "subspace_fraction", "valid_shot_fraction",
                "n_qubits", "circuit_depth_max", "t_step3_s"]


def value(row, key):
    v = row.get(key)
    if v not in (None, ""):
        return v
    try:
        p = json.loads(row.get("params_json") or "{}")
    except ValueError:
        return "-"
    return p.get(key, "-")


def fmt(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f.is_integer() and abs(f) < 1e9:
        return str(int(f))
    return f"{f:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--where", nargs="*", default=None)
    ap.add_argument("--cols", nargs="*", default=None)
    a = ap.parse_args(argv)
    where = DEFAULT_WHERE if a.where is None else a.where
    cols = a.cols or DEFAULT_COLS
    rows = list(csv.DictReader(open(Path(a.out) / "results.csv")))
    conds = [w.split("=", 1) for w in where]
    sel = [r for r in rows
           if r.get("status") == "ok"
           and all(fmt(value(r, k)) == fmt(v) for k, v in conds)]
    sel.sort(key=lambda r: tuple(fmt(value(r, c)).zfill(12) for c in cols[:3]))
    table = [cols] + [[fmt(value(r, c)) for c in cols] for r in sel]
    w = [max(len(t[i]) for t in table) for i in range(len(cols))]
    for t in table:
        print("  ".join(x.rjust(w[i]) for i, x in enumerate(t)))
    failed = [r for r in rows if r.get("status") != "ok"
              and all(fmt(value(r, k)) == fmt(v) for k, v in conds)]
    print(f"\n{len(sel)} ok rows" + (f", {len(failed)} FAILED" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
