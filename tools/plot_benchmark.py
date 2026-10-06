#!/usr/bin/env python
"""
Thesis figures and tables from a run_benchmark.py output directory.

    python tools/plot_benchmark.py <out dir> [<out dir> ...] [--to <dir>]

Reads results.csv from each directory (several are merged), writes PNG + PDF
figures and Markdown/CSV tables into <first out dir>/figures/ (or --to).
Each figure is produced only if the data for it exists, so this runs on any
benchmark directory:

  error_vs_shots      |error| vs shots per circuit, log-log, mean +- std over seeds
  subspace_vs_error   every run: subspace fraction vs error, one point per run
  curve               energies vs bond length (rows with tag_R) + error panel
  las_convergence     LASSQD energy per macro cycle, per seed (carryover figure)
  summary.md/.csv     one row per configuration: n, error mean +- std, subspace,
                      wall time; plus non-parallelity error per curve method

"Error" is err_solver_kcal (vs the solver's own exact limit: embedding FCI for
DMET solvers, LASSCF for LASSQD) unless --error says otherwise.

Needs only numpy and matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# ── visual system ───────────────────────────────────────────────────────
# Colour follows the solver (entity), fixed across every figure. Variants of
# one solver share its colour and differ by marker / line style.
SOLVER_COLOUR = {
    "sqd": "#2a78d6",      # blue
    "skqd": "#eb6834",     # orange
    "sqdrift": "#1baf7a",  # aqua
    "gqe": "#eda100",      # yellow
    "lassqd": "#e87ba4",   # magenta
    "lasscf": "#008300",   # green
}
SOLVER_ORDER = ["sqd", "skqd", "sqdrift", "gqe", "lasscf", "lassqd"]
SOLVER_LABEL = {"sqd": "DMET+SQD", "skqd": "DMET+SKQD", "sqdrift": "DMET+SqDRIFT",
                "gqe": "DMET+GQE", "lasscf": "LASSCF", "lassqd": "LASSQD"}
VARIANT_MARKERS = ["o", "s", "^", "D", "v", "P"]
VARIANT_STYLES = ["-", "--", ":", "-."]
REFERENCE_STYLE = {   # classical references: neutral ink, never series colours
    "FCI": dict(color="#0b0b0b", ls="-", lw=1.6),
    "CASCI": dict(color="#52514e", ls="-", lw=1.2),
    "CASSCF": dict(color="#52514e", ls=":", lw=1.4),
    "CCSD_T": dict(color="#52514e", ls="--", lw=1.4),
    "HF": dict(color="#9a9893", ls="-.", lw=1.2),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CHEM_ACC = 1.0          # kcal/mol
LOG_FLOOR = 1e-4        # kcal/mol: exact-to-print errors are drawn here

#: the axes figures sweep over; never part of a variant's identity
AXES = {"seed", "lassqd_seed", "gqe_seed", "shots", "lassqd_shots",
        "geometry", "xyz"}
#: params that never distinguish a "variant" (they are axes or identity)
NOT_VARIANT = {"seed", "lassqd_seed", "gqe_seed", "shots", "lassqd_shots",
               "geometry", "solver", "molecule", "basis"}
SHORT = {"lassqd_lucj_optimize": "opt", "lassqd_carryover_eps": "eps",
         "las_fragments": "frag", "lassqd_lucj_pairs": "pairs",
         "dmet_reference": "ref", "las_no_orbital_opt": "LASCI"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.markersize": 5.5,
    })
    return plt


# ── data ────────────────────────────────────────────────────────────────

def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(dirs):
    rows = []
    for d in dirs:
        path = Path(d).expanduser() / "results.csv"
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok":
                    r["_params"] = json.loads(r.get("params_json") or "{}")
                    # A "variant" is a method configuration independent of the
                    # axes the figures sweep (seed, shots, geometry). The
                    # driver's config_id still contains shots and geometry,
                    # so it cannot be used to draw a line across them.
                    ident = {k: v for k, v in r["_params"].items()
                             if k not in AXES}
                    r["_variant"] = json.dumps(ident, sort_keys=True)
                    rows.append(r)
    return rows


def variant_labels(rows):
    """config_id -> human label: solver plus whatever distinguishes it from
    the solver's other configurations."""
    by_solver = defaultdict(dict)
    for r in rows:
        by_solver[r["solver"]][r["_variant"]] = r["_params"]
    out = {}
    for solver, cfgs in by_solver.items():
        keys = set().union(*[p.keys() for p in cfgs.values()]) - NOT_VARIANT
        varying = sorted(k for k in keys
                         if len({json.dumps(p.get(k), sort_keys=True)
                                 for p in cfgs.values()}) > 1)
        flags = {k for k in varying if any(p.get(k) is True for p in cfgs.values())}
        for cid, p in cfgs.items():
            bits = []
            for k in varying:
                v = p.get(k)
                name = SHORT.get(k, k)
                if k == "las_fragments" and isinstance(v, list):
                    # label by fragment spin: 2S of each fragment
                    v = "2S=" + ",".join(s.split(":")[3] if s.count(":") == 3
                                         else "?" for s in v)
                    bits.append(v)
                elif v is True:
                    bits.append(name)
                elif v is None:
                    if k in flags:
                        bits.append(f"no {name}")
                else:
                    bits.append(f"{name}={v}")
            out[cid] = SOLVER_LABEL.get(solver, solver) + (
                f" ({', '.join(bits)})" if bits else "")
    return out


def _variant_index(rows):
    idx, seen = {}, defaultdict(list)
    for r in rows:
        lst = seen[r["solver"]]
        if r["_variant"] not in lst:
            lst.append(r["_variant"])
        idx[r["_variant"]] = lst.index(r["_variant"])
    return idx


def _err(r, key):
    v = _num(r.get(key))
    return None if v is None else abs(v)


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    print(f"  wrote {out / name}.png/.pdf")


# ── figures ─────────────────────────────────────────────────────────────

def fig_error_vs_shots(plt, rows, labels, vidx, out, key, mol):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s, e = _num(r.get("shots_per_circuit")), _err(r, key)
        if s and e is not None and r["solver"] != "lasscf":
            groups[r["_variant"]][s].append(max(e, LOG_FLOOR))
    groups = {c: g for c, g in groups.items() if len(g) >= 2}
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    solver_of = {r["_variant"]: r["solver"] for r in rows}
    for cid in sorted(groups, key=lambda c: (SOLVER_ORDER.index(solver_of[c])
                                             if solver_of[c] in SOLVER_ORDER else 9,
                                             vidx[c])):
        g, solver, v = groups[cid], solver_of[cid], vidx[cid]
        xs = sorted(g)
        mean = [statistics.fmean(g[x]) for x in xs]
        std = [statistics.stdev(g[x]) if len(g[x]) > 1 else 0 for x in xs]
        lo = [max(m - s, LOG_FLOOR) for m, s in zip(mean, std)]
        ax.errorbar(xs, mean, yerr=[[m - l for m, l in zip(mean, lo)], std],
                    color=SOLVER_COLOUR.get(solver, INK2),
                    marker=VARIANT_MARKERS[v % 6], ls=VARIANT_STYLES[v % 4],
                    capsize=2.5, elinewidth=1, label=labels[cid])
    ax.axhline(CHEM_ACC, color=INK2, lw=0.9, ls="--")
    ax.text(min(min(g) for g in groups.values()),
            CHEM_ACC * 1.15, "chemical accuracy", color=INK2, fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("shots per circuit")
    ax.set_ylabel(f"|error| (kcal/mol)")
    ax.set_title(f"Error vs shots — {mol}")
    ax.legend(loc="best")
    _save(fig, out, "error_vs_shots")
    plt.close(fig)


def fig_subspace_vs_error(plt, rows, labels, vidx, out, key, mol):
    pts = [(r, _num(r.get("subspace_fraction")), _err(r, key)) for r in rows]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None]
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    done = set()
    for r, x, y in sorted(pts, key=lambda t: SOLVER_ORDER.index(t[0]["solver"])
                          if t[0]["solver"] in SOLVER_ORDER else 9):
        cid = r["_variant"]
        lab = labels[cid] if cid not in done else None
        done.add(cid)
        ax.scatter(x, max(y, LOG_FLOOR), s=28, marker=VARIANT_MARKERS[vidx[cid] % 6],
                   color=SOLVER_COLOUR.get(r["solver"], INK2),
                   edgecolor="white", linewidth=0.8, label=lab, zorder=3)
    ax.axhline(CHEM_ACC, color=INK2, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("subspace fraction (sampled dim / full dim)")
    ax.set_ylabel("|error| (kcal/mol)")
    ax.set_title(f"Subspace coverage vs error — {mol}")
    ax.legend(loc="best", fontsize=7)
    _save(fig, out, "subspace_vs_error")
    plt.close(fig)


def fig_curve(plt, rows, labels, vidx, out, mol):
    cr = [r for r in rows if _num(r.get("tag_R")) is not None]
    if len({r["tag_R"] for r in cr}) < 3:
        return None
    # references: one value per R
    ref = defaultdict(dict)
    for r in cr:
        R = _num(r["tag_R"])
        for m in REFERENCE_STYLE:
            e = _num(r.get(f"E_{m}_Ha"))
            if e is not None:
                ref[m][R] = e
    exact_name = "FCI" if ref.get("FCI") else "CASCI"
    exact = ref.get(exact_name, {})
    meth = defaultdict(lambda: defaultdict(list))
    for r in cr:
        e = _num(r.get("energy_Ha"))
        if e is not None:
            meth[r["_variant"]][_num(r["tag_R"])].append(e)
    solver_of = {r["_variant"]: r["solver"] for r in cr}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for m, st in REFERENCE_STYLE.items():
        if ref.get(m):
            xs = sorted(ref[m])
            a1.plot(xs, [ref[m][x] for x in xs], label=m.replace("_T", "(T)"), **st)
            if m != exact_name and exact:
                xe = [x for x in xs if x in exact]
                a2.plot(xe, [(ref[m][x] - exact[x]) * 627.5094740631 for x in xe],
                        **st)
    npe = {}
    for cid in sorted(meth, key=lambda c: (SOLVER_ORDER.index(solver_of[c])
                                           if solver_of[c] in SOLVER_ORDER else 9,
                                           vidx[c])):
        d, solver, v = meth[cid], solver_of[cid], vidx[cid]
        xs = sorted(d)
        ys = [statistics.fmean(d[x]) for x in xs]
        kw = dict(color=SOLVER_COLOUR.get(solver, INK2),
                  marker=VARIANT_MARKERS[v % 6], ls=VARIANT_STYLES[v % 4], ms=4.5)
        a1.plot(xs, ys, label=labels[cid], **kw)
        xe = [x for x in xs if x in exact]
        errs = [(statistics.fmean(d[x]) - exact[x]) * 627.5094740631 for x in xe]
        sd = [statistics.stdev(d[x]) * 627.5094740631 if len(d[x]) > 1 else 0
              for x in xe]
        a2.errorbar(xe, errs, yerr=sd, capsize=2, elinewidth=0.9, **kw)
        if len(errs) >= 2:
            npe[labels[cid]] = (max(errs) - min(errs), min(errs), max(errs), len(errs))
    for m in REFERENCE_STYLE:
        if m != exact_name and ref.get(m) and exact:
            e = [(ref[m][x] - exact[x]) * 627.5094740631 for x in ref[m] if x in exact]
            if len(e) >= 2:
                npe[m.replace("_T", "(T)")] = (max(e) - min(e), min(e), max(e), len(e))
    a2.axhspan(-CHEM_ACC, CHEM_ACC, color=GRID, zorder=0)
    a2.axhline(0, color=INK, lw=0.8)
    a1.set_ylabel("energy (Ha)")
    a1.set_title(f"Dissociation curve — {mol}")
    a1.legend(loc="best", fontsize=7)
    a2.set_ylabel(f"E − E({exact_name}) (kcal/mol)")
    a2.set_xlabel("R (Å)")
    a2.set_yscale("symlog", linthresh=1.0)
    fig.tight_layout()
    _save(fig, out, "curve")
    plt.close(fig)
    return npe


def fig_las_convergence(plt, rows, labels, vidx, out, mol):
    las = [r for r in rows if r["solver"] == "lassqd" and r.get("energy_trace_json")]
    if not las:
        return
    ref = {}
    for r in rows:
        if r["solver"] == "lasscf":
            ref[r["system_id"]] = _num(r["energy_Ha"])
    # one panel per variant, one line per seed; keep it readable
    from matplotlib.ticker import MaxNLocator

    # one panel per (variant, shots); one line per seed
    cfgs = sorted({(r["_variant"], _num(r.get("shots_per_circuit")) or 0)
                   for r in las}, key=lambda c: (vidx[c[0]], c[1]))[:8]
    ncol = min(4, len(cfgs))
    nrow = math.ceil(len(cfgs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.6 * nrow),
                             sharey=True, squeeze=False)
    for ax in axes.flat[len(cfgs):]:
        ax.set_visible(False)
    for ax, (cid, shots) in zip(axes.flat, cfgs):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for r in [r for r in las if r["_variant"] == cid
                  and (_num(r.get("shots_per_circuit")) or 0) == shots]:
            tr = [e for e in json.loads(r["energy_trace_json"]) if e is not None]
            e0 = ref.get(r["system_id"])
            if e0 is None or not tr:
                continue
            ys = [max(abs(e - e0) * 627.5094740631, LOG_FLOOR) for e in tr]
            ax.plot(range(len(ys)), ys, marker="o", ms=3,
                    color=SOLVER_COLOUR["lassqd"], alpha=0.85,
                    label=f"seed {r.get('seed')}")
        ax.axhline(CHEM_ACC, color=INK2, lw=0.9, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"{labels[cid].replace('LASSQD ', '')} {int(shots)} shots",
                     fontsize=8)
        ax.set_xlabel("macro cycle")
    for row in axes:
        row[0].set_ylabel("|E − E(LASSCF)| (kcal/mol)")
    fig.suptitle(f"LASSQD convergence — {mol}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "las_convergence")
    plt.close(fig)


# ── tables ──────────────────────────────────────────────────────────────

def tables(rows, labels, out, key, npe):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["_variant"], r.get("shots_per_circuit") or "", r.get("tag_R") or "")
               ].append(r)
    lines = []
    for (cid, shots, R), rs in groups.items():
        def ms(field, scale=1.0):
            v = [_num(r.get(field)) for r in rs]
            v = [x * scale for x in v if x is not None]
            if not v:
                return None, None
            return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
        e, es = ms(key)
        sf, _ = ms("subspace_fraction")
        t, _ = ms("t_step3_s")
        lines.append({
            "method": labels[cid], "molecule": rs[0]["molecule"], "R": R,
            "shots": shots, "n": len(rs),
            "error_mean_kcal": e, "error_std_kcal": es,
            "subspace_fraction": sf, "qubits": rs[0].get("n_qubits") or "",
            "wall_s": t, "solver": rs[0]["solver"],
        })
    lines.sort(key=lambda d: (d["molecule"], _num(d["R"]) or 0,
                              SOLVER_ORDER.index(d["solver"])
                              if d["solver"] in SOLVER_ORDER else 9,
                              d["method"], _num(d["shots"]) or 0))
    out.mkdir(parents=True, exist_ok=True)
    keys = ["molecule", "R", "method", "shots", "n", "error_mean_kcal",
            "error_std_kcal", "subspace_fraction", "qubits", "wall_s"]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(lines)

    def f(x, p=3):
        return "–" if x is None else f"{x:.{p}f}"
    md = [f"Error = `{key}` (kcal/mol), mean ± std over seeds.\n",
          "| molecule | R | method | shots | n | error | subspace | qubits | wall (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in lines:
        err = ("–" if d["error_mean_kcal"] is None else
               f"{d['error_mean_kcal']:+.3f} ± {d['error_std_kcal']:.3f}")
        md.append(f"| {d['molecule']} | {d['R'] or '–'} | {d['method']} | "
                  f"{d['shots'] or '–'} | {d['n']} | {err} | "
                  f"{f(d['subspace_fraction'])} | {d['qubits'] or '–'} | "
                  f"{f(d['wall_s'], 1)} |")
    if npe:
        md += ["", "Non-parallelity error along the curve (kcal/mol, vs exact):", "",
               "| method | NPE | min error | max error | points |", "|---|---|---|---|---|"]
        for name, (n, lo, hi, k) in sorted(npe.items(), key=lambda kv: kv[1][0]):
            md.append(f"| {name} | {n:.2f} | {lo:+.2f} | {hi:+.2f} | {k} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {out / 'summary.md'} and summary.csv ({len(lines)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", help="run_benchmark.py output dirs")
    ap.add_argument("--to", default=None, help="figure dir (default <first>/figures)")
    ap.add_argument("--error", default="err_solver_kcal",
                    choices=["err_solver_kcal", "err_vs_best_ref_kcal",
                             "err_vs_casci_kcal", "err_vs_fci_kcal"])
    a = ap.parse_args(argv)

    rows = load(a.dirs)
    if not rows:
        raise SystemExit("no ok rows found")
    out = Path(a.to).expanduser() if a.to else Path(a.dirs[0]).expanduser() / "figures"
    mol = "/".join(sorted({f"{r['molecule']} ({r['basis']})" for r in rows}))
    labels, vidx = variant_labels(rows), _variant_index(rows)
    plt = _style()
    print(f"[plot] {len(rows)} runs, {len(labels)} configurations -> {out}")
    fig_error_vs_shots(plt, rows, labels, vidx, out, a.error, mol)
    fig_subspace_vs_error(plt, rows, labels, vidx, out, a.error, mol)
    npe = fig_curve(plt, rows, labels, vidx, out, mol)
    fig_las_convergence(plt, rows, labels, vidx, out, mol)
    tables(rows, labels, out, a.error, npe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())