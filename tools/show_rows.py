#!/usr/bin/env python
"""
Print selected columns of <out>/results.csv as an aligned table.

    python tools/show_rows.py $B/h8_smoke                       # lucj_opt SQD rows
    python tools/show_rows.py $B/n2_curve --where solver=gqe tag_R=2.1 \
        --cols tag_R gqe_ngates gqe_max_iters seed err_solver_kcal t_step3_s

--where KEY=VALUE filters on the CSV text, or on params_json for flags that
are not columns (gqe_ngates, gqe_max_iters, ...). Missing = "-".

--agg collapses seed repeats: one line per configuration (same config_id =
same parameters except the seed), numeric columns that differ between seeds
as mean+-std (min..max with --range), plus n = number of seeds.
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


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aggregate(rows, cols, use_range=False):
    """Group by config_id; constant columns once, varying numeric ones as
    mean+-std (or min..max). The seed column is replaced by n."""
    import statistics

    groups = {}
    for r in rows:
        groups.setdefault(r.get("config_id") or r.get("run_id") or id(r), []).append(r)
    out_cols = [c for c in cols if c != "seed"] + ["n"]
    body = []
    for rs in groups.values():
        line = []
        for c in out_cols[:-1]:
            vals = [value(r, c) for r in rs]
            shown = {fmt(v) for v in vals}
            nums = [_float(v) for v in vals]
            if len(shown) == 1:
                line.append(shown.pop())
            elif all(x is not None for x in nums):
                if use_range:
                    line.append(f"{min(nums):.4g}..{max(nums):.4g}")
                else:
                    sd = statistics.stdev(nums) if len(nums) > 1 else 0.0
                    line.append(f"{statistics.fmean(nums):.4g}+-{sd:.2g}")
            else:
                line.append("/".join(sorted(shown)))
        line.append(str(len(rs)))
        body.append(line)
    return out_cols, body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--where", nargs="*", default=None)
    ap.add_argument("--cols", nargs="*", default=None)
    ap.add_argument("--agg", action="store_true",
                    help="one line per configuration, mean+-std over seeds")
    ap.add_argument("--range", action="store_true",
                    help="with --agg: show min..max instead of mean+-std")
    a = ap.parse_args(argv)
    where = DEFAULT_WHERE if a.where is None else a.where
    cols = a.cols or DEFAULT_COLS
    rows = list(csv.DictReader(open(Path(a.out) / "results.csv")))
    conds = [w.split("=", 1) for w in where]
    sel = [r for r in rows
           if r.get("status") == "ok"
           and all(fmt(value(r, k)) == fmt(v) for k, v in conds)]
    sel.sort(key=lambda r: tuple(fmt(value(r, c)).zfill(12) for c in cols[:3]))
    if a.agg:
        cols, body = aggregate(sel, cols, a.range)
    else:
        body = [[fmt(value(r, c)) for c in cols] for r in sel]
    table = [cols] + body
    w = [max(len(t[i]) for t in table) for i in range(len(cols))]
    for t in table:
        print("  ".join(x.rjust(w[i]) for i, x in enumerate(t)))
    failed = [r for r in rows if r.get("status") != "ok"
              and all(fmt(value(r, k)) == fmt(v) for k, v in conds)]
    print(f"\n{len(sel)} ok rows" + (f", {len(failed)} FAILED" if failed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
