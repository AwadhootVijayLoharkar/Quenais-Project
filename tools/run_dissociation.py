#!/usr/bin/env python
"""
Dissociation curve: all classical methods against exact, across bond length.

Goes in: tools/run_dissociation.py

Produces the standard strong-correlation figure. As the bond breaks, single
reference methods fail in a characteristic way -- CCSD in particular does not
merely lose accuracy but crosses through the exact answer, ending up BELOW it,
because the coupled-cluster expansion is non-variational and its amplitudes
diverge. On N2/STO-3G we measure max|t2| = 0.65 at 1.8 A and 0.84 at 2.1 A,
where a healthy value is below 0.1.

That crossing is the cleanest available demonstration that a system is strongly
correlated, and it motivates everything downstream: it is precisely where a
method that treats many configurations on an equal footing is required.

Two outputs:
  dissociation_<mol>.csv   -- energies at every geometry
  dissociation_<mol>.pdf   -- two panels, absolute energy and error vs exact

CAVEAT ON COMPARABILITY. HF/MP2/CCSD/CCSD(T) are all-electron; CASCI is within
the active space with a frozen core. For N2/STO-3G with (10e,8o) the frozen 1s
pair contributes almost no correlation, so the comparison is meaningful, but the
curves are not identical theory levels and the figure caption should say so.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HARTREE_TO_KCAL = 627.5094740631


def compute_point(atoms, r, basis, ncas, nelecas):
    """All classical methods plus exact CASCI at one geometry."""
    from pyscf import ao2mo, cc, fci, gto, mcscf, mp, scf

    mol = gto.M(atom=f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {r:.6f}",
                basis=basis, symmetry=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    mf.kernel()

    out = {"r": r, "HF": float(mf.e_tot), "converged_hf": bool(mf.converged)}

    try:
        out["MP2"] = float(mp.MP2(mf).run(verbose=0).e_tot)
    except Exception:
        out["MP2"] = float("nan")

    # CCSD frequently fails to converge past the point of bond breaking. That
    # failure is itself a result -- record it rather than letting it abort.
    try:
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.max_cycle = 200
        mycc.kernel()
        out["CCSD"] = float(mycc.e_tot)
        out["ccsd_converged"] = bool(mycc.converged)
        out["max_t2"] = float(np.abs(mycc.t2).max())
        try:
            out["CCSD_T"] = float(mycc.e_tot + mycc.ccsd_t())
        except Exception:
            out["CCSD_T"] = float("nan")
    except Exception:
        out.update({"CCSD": float("nan"), "ccsd_converged": False,
                    "max_t2": float("nan"), "CCSD_T": float("nan")})

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, shift=0.5, ss=0)
    mc.fcisolver.conv_tol = 1e-14
    out["CASCI"] = float(mc.kernel()[0])

    # Correlation coordinate, for cross-referencing with the selection studies.
    civec = np.asarray(mc.ci).reshape(-1)
    out["w1"] = float((civec ** 2).max())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", default="N2")
    p.add_argument("--atoms", nargs=2, default=("N", "N"))
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--ncas", type=int, default=8)
    p.add_argument("--nelecas", type=int, default=10)
    p.add_argument("--distances", type=float, nargs="+",
                   default=[0.9, 1.0, 1.0977, 1.2, 1.3, 1.4, 1.5, 1.65, 1.8,
                            1.95, 2.1, 2.25, 2.4, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0])
    p.add_argument("--out", default=".")
    args = p.parse_args()

    rows = []
    print(f"  {'r':>6} {'HF':>13} {'CCSD':>13} {'CASCI':>13} "
          f"{'w1':>7} {'max|t2|':>9}")
    for r in args.distances:
        try:
            row = compute_point(tuple(args.atoms), r, args.basis,
                                args.ncas, args.nelecas)
        except Exception as exc:
            print(f"  {r:>6.3f}  failed: {exc}")
            continue
        rows.append(row)
        flag = "" if row.get("ccsd_converged") else "  <- CCSD not converged"
        print(f"  {r:>6.3f} {row['HF']:>13.6f} {row['CCSD']:>13.6f} "
              f"{row['CASCI']:>13.6f} {row['w1']:>7.4f} "
              f"{row['max_t2']:>9.3f}{flag}")

    if not rows:
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["r", "HF", "MP2", "CCSD", "CCSD_T", "CASCI", "w1", "max_t2",
            "ccsd_converged", "converged_hf"]
    csv_path = out_dir / f"dissociation_{args.molecule}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Dissociation energy from the computed curve.
    e = np.array([x["CASCI"] for x in rows])
    rr = np.array([x["r"] for x in rows])
    de = (e.max() - e.min()) * HARTREE_TO_KCAL
    print(f"\n  minimum at r = {rr[e.argmin()]:.3f} A")
    print(f"  D_e (CASCI, curve endpoints) = {de:.1f} kcal/mol")
    print(f"  wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "font.family": "serif",
                             "legend.frameon": False})
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.0, 6.2), sharex=True)

        series = [("HF", "#888780", "-"), ("MP2", "#eda100", "-."),
                  ("CCSD", "#e34948", "--"), ("CCSD_T", "#2a78d6", ":"),
                  ("CASCI", "#0b0b0b", "-")]
        for k, c, ls in series:
            y = np.array([x.get(k, np.nan) for x in rows], dtype=float)
            a1.plot(rr, y, ls, color=c, lw=1.4,
                    label="exact (CASCI)" if k == "CASCI" else k)
        a1.set_ylabel("total energy (Ha)")
        a1.legend(fontsize=8, ncol=2)

        for k, c, ls in series[:-1]:
            y = np.array([x.get(k, np.nan) for x in rows], dtype=float)
            a2.plot(rr, 1e3 * (y - e), ls, color=c, lw=1.4, label=k)
        a2.axhline(0, color="0.2", lw=1.0)
        a2.axhline(1.6, color="0.6", lw=0.7, ls=":")
        a2.axhline(-1.6, color="0.6", lw=0.7, ls=":")
        a2.set_ylabel("error vs exact (mHa)")
        a2.set_xlabel(r"bond length ($\AA$)")
        a2.set_yscale("symlog", linthresh=1.0)
        a2.legend(fontsize=8, ncol=2)

        fig.tight_layout()
        pdf = out_dir / f"dissociation_{args.molecule}.pdf"
        fig.savefig(pdf)
        print(f"  wrote {pdf}")
    except Exception as exc:
        print(f"  plotting skipped: {exc}")


if __name__ == "__main__":
    main()