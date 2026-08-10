#!/usr/bin/env python
"""
Can a classical expansion rescue GQE's determinant set?

Goes in: tools/expand_gqe_seed.py

THE QUESTION
------------
At r=1.8 A the quantum sampler picks badly: its 200 determinants capture 76.9%
of the wavefunction where the best 200 capture 99.98%, giving 78.9 mHa against
0.40. The diagonalisation is fine -- re-solving GQE's own set independently
reproduces 78.9371 mHa exactly. The selection is the problem.

So: does GQE's set at least point at the right REGION? Take its determinants,
add everything the Hamiltonian couples to them, keep the best by perturbative
score, and stop at the same budget. This is the QiankunNet paper's H-Couple
step, and it is the cheapest possible fix.

  recovers to ~0.4 mHa  -> a classical post-step repairs the quantum sampling.
                           Cheap, no model needed, and it makes the quantum
                           stage useful immediately.
  stays bad             -> GQE's seed is in the wrong region entirely and local
                           expansion cannot reach the important configurations.
                           Only a better proposal distribution will do it, and
                           that is the neural sampler's justification.

FOUR NUMBERS, ALL AT THE SAME BUDGET
------------------------------------
    oracle        best possible
    CIPSI         classical selection from scratch, no quantum input
    GQE           what the quantum sampler chose
    GQE+expand    the quantum seed, classically expanded

The gap between CIPSI and GQE+expand is the honest measure of what the quantum
stage contributes: both end with the same classical machinery, and the only
difference is where they started.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--dets", required=True, help="npz from the dump patch")
    p.add_argument("--atoms", nargs=2, default=("N", "N"))
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--ncas", type=int, default=8)
    p.add_argument("--nelecas", type=int, default=10)
    p.add_argument("--budget", type=int, default=200)
    args = p.parse_args()

    sys.path.insert(0, str(REPO / "tools"))
    from run_correlation_scan import build_cas_molecule
    from compare_gqe_determinants import strings_to_indices
    try:
        from quenais.quantum import det_analysis as da
        from quenais.quantum import det_expansion as dx
    except ImportError:
        import det_analysis as da
        import det_expansion as dx

    mol, _, _, _ = build_cas_molecule(
        tuple(args.atoms), args.r, args.basis, args.ncas, args.nelecas)
    e_exact, flat, space = da.casci_vector(mol)
    order, _ = da.weight_curve(flat)
    w = flat ** 2

    d = np.load(args.dets)
    gqe_idx = strings_to_indices(d["strs"], int(d["norb"]),
                                 tuple(int(x) for x in d["nelec"]), space)
    n = args.budget

    def report(name, idx, energy=None):
        idx = np.asarray(idx)
        e = energy if energy is not None else da.projected_energy(
            mol, idx, space=space)
        print(f"  {name:<14} {len(idx):>5} dets   "
              f"{1e3*(e - e_exact):9.4f} mHa   "
              f"weight {float(w[idx].sum()):.6f}   "
              f"overlap with oracle {len(set(idx.tolist()) & set(order[:n].tolist())):>3}/{n}")
        return e

    print(f"\n  N2  r = {args.r:.4f} A   ndet={space.ndet}   budget={n}")
    print(f"  exact CASCI = {e_exact:.12f} Ha\n")

    report("oracle", order[:n])

    sel_cipsi, hist_cipsi = dx.cipsi_from_scratch(mol, n, space=space,
                                                  verbose=False)
    report("CIPSI", sel_cipsi, hist_cipsi[-1]["energy"])

    report("GQE", gqe_idx)

    # The expansion. Seeded with GQE's determinants, grown by the same
    # perturbative criterion CIPSI uses, stopped at the same budget. Since the
    # seed is already at the budget, this replaces its worst members rather
    # than adding to it -- which is the point: the question is whether the
    # neighbourhood of GQE's picks contains the determinants it missed.
    sel_exp, hist_exp = dx.expand_from_seed(mol, gqe_idx, n + n // 2,
                                            space=space, verbose=False)
    e_exp = report("GQE+expand", sel_exp, hist_exp[-1]["energy"])

    print(f"\n  VERDICT")
    err_exp = 1e3 * (e_exp - e_exact)
    e_cipsi = hist_cipsi[-1]["energy"]
    err_cipsi = 1e3 * (e_cipsi - e_exact)
    if err_exp <= 2.0:
        print(f"    Expansion RESCUES it: {err_exp:.4f} mHa. GQE's picks were in")
        print(f"    the right region after all -- a classical expansion step")
        print(f"    recovers the accuracy with no model at all. Add this to the")
        print(f"    pipeline before considering a neural sampler.")
    else:
        print(f"    Expansion does NOT rescue it: {err_exp:.4f} mHa. GQE's seed")
        print(f"    is far enough from the important configurations that local")
        print(f"    expansion cannot reach them. A better proposal distribution")
        print(f"    is the only route -- which is the neural sampler's case,")
        print(f"    now measured rather than assumed.")
    print(f"\n    vs CIPSI from scratch ({err_cipsi:.4f} mHa): the quantum seed is "
          f"{'HELPING' if err_exp < err_cipsi else 'NOT helping'} here.")


if __name__ == "__main__":
    main()