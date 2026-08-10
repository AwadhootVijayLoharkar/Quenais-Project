#!/usr/bin/env python
"""
Compare GQE's selected determinants against the oracle's.

Goes in: tools/compare_gqe_determinants.py

Splits the r=1.8 A failure into its two possible causes:

  * GQE picked bad determinants          -> a sampling problem, which is what a
                                            learned proposal distribution is for
  * GQE picked fine determinants but got  -> a bug in its diagonalisation or
    a bad energy from them                   refinement, worth fixing first

It does this by re-diagonalising GQE's OWN determinant set with the same code
that produced the oracle number. Same Hamiltonian, same solver, only the
determinant list differs -- so any energy difference is attributable to the
selection alone.

Requires the dump produced by the patch in GQE_DETERMINANT_DUMP.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def strings_to_indices(strs, norb, nelec, space):
    """
    GQE determinant (alpha_bitmask, beta_bitmask) -> full determinant index.

    The bitmasks are PySCF cistring occupation integers, so str2addr inverts
    them directly. The full index convention matches
    det_analysis.DeterminantSpace and gqe_qsci's from_fullci_index:

        index = addr_alpha * n_beta_strings + addr_beta
    """
    from pyscf.fci import cistring

    out = np.empty(len(strs), dtype=np.int64)
    for k, (a, b) in enumerate(strs):
        addr_a = cistring.str2addr(norb, nelec[0], int(a))
        addr_b = cistring.str2addr(norb, nelec[1], int(b))
        out[k] = addr_a * space.nb + addr_b
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--dets", required=True, help="npz from the dump patch")
    p.add_argument("--molecule", default="N2")
    p.add_argument("--atoms", nargs=2, default=("N", "N"))
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--ncas", type=int, default=8)
    p.add_argument("--nelecas", type=int, default=10)
    args = p.parse_args()

    sys.path.insert(0, str(REPO / "tools"))
    from run_correlation_scan import build_cas_molecule
    try:
        from quenais.quantum import det_analysis as da
    except ImportError:
        import det_analysis as da

    mol, e_casci, e_rhf, root_gap = build_cas_molecule(
        tuple(args.atoms), args.r, args.basis, args.ncas, args.nelecas)
    e_exact, flat, space = da.casci_vector(mol)
    order, _ = da.weight_curve(flat)

    d = np.load(args.dets)
    strs = d["strs"]
    norb = int(d["norb"])
    nelec = tuple(int(x) for x in d["nelec"])

    if norb != space.norb or nelec != space.nelec:
        raise SystemExit(
            f"dump is for norb={norb} nelec={nelec}, but this geometry gives "
            f"norb={space.norb} nelec={space.nelec}. Wrong dump file, or the "
            f"Hamiltonian exported into the run directory was not this one."
        )

    gqe_idx = strings_to_indices(strs, norb, nelec, space)
    n = len(gqe_idx)
    oracle_idx = order[:n]

    overlap = len(set(gqe_idx.tolist()) & set(oracle_idx.tolist()))

    e_gqe_set = da.projected_energy(mol, gqe_idx, space=space)
    e_oracle = da.projected_energy(mol, oracle_idx, space=space)

    # How much of the exact wavefunction did GQE's set actually capture?
    w = flat ** 2
    w_gqe = float(w[gqe_idx].sum())
    w_oracle = float(w[oracle_idx].sum())

    print(f"\n  {args.molecule}  r = {args.r:.4f} A   ncas={space.norb} "
          f"nelec={space.nelec}  ndet={space.ndet}")
    print(f"  exact CASCI = {e_exact:.12f} Ha\n")
    print(f"  determinants in GQE's final set : {n}")
    print(f"  shared with the oracle's top {n} : {overlap}"
          f"  ({100.0*overlap/n:.1f}%)\n")
    print(f"  weight captured, GQE's set    : {w_gqe:.8f}")
    print(f"  weight captured, oracle's set : {w_oracle:.8f}\n")
    print(f"  re-diagonalised in GQE's set    : {1e3*(e_gqe_set - e_exact):9.4f} mHa")
    print(f"  re-diagonalised in oracle's set : {1e3*(e_oracle - e_exact):9.4f} mHa")

    print(f"\n  VERDICT")
    err_gqe = 1e3 * (e_gqe_set - e_exact)
    err_orc = 1e3 * (e_oracle - e_exact)
    if err_gqe < 5 * max(err_orc, 1e-6):
        print(f"    GQE's determinants are FINE -- re-diagonalising them here")
        print(f"    gives {err_gqe:.4f} mHa. If the run reported far worse, the")
        print(f"    fault is in GQE's own diagonalisation or refinement, not in")
        print(f"    its sampling. Fix that before building any model.")
    else:
        print(f"    GQE's determinants are genuinely POOR -- {err_gqe:.4f} mHa")
        print(f"    even with a correct diagonalisation, against {err_orc:.4f}")
        print(f"    for the best {n}. The sampler is not finding the important")
        print(f"    configurations here. That is the problem a learned proposal")
        print(f"    distribution is meant to solve.")


if __name__ == "__main__":
    main()