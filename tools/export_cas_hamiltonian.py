#!/usr/bin/env python
"""
Export a CAS Hamiltonian in step-2 pickle format.

Goes in: tools/export_cas_hamiltonian.py

WHY THIS EXISTS
---------------
run_correlation_scan.py builds the active-space Hamiltonian directly with
PySCF, bypassing DMET -- N2/STO-3G has too few orbitals to leave DMET an
environment, and the correlation-crossover question does not need an embedding
anyway.

But the quantum solvers read a step-2 pickle. Without a bridge you would have
to re-derive the Hamiltonian through a different route to run them, and then
the quantum numbers would not be comparable to the oracle and CIPSI numbers
that motivated the run. Comparing selection methods on two different
Hamiltonians measures nothing.

This writes the identical integrals the scan measured into the format
load_from_dmet_pickle() expects, so:

    python tools/export_cas_hamiltonian.py --r 2.1 --project-dir runs/N2_r2.1
    quenais-run --molecule N2 --basis sto-3g --solver gqe --steps 3 4 \
                --project-dir runs/N2_r2.1 --gqe-qsci-max-dim 200

runs the quantum solver on exactly the Hamiltonian the scan characterised.

HONESTY ABOUT THE FIELDS
------------------------
The DMET-specific entries (mu, n_imp, n_bath, sv, sv2_cov) have no meaning for
a plain CAS. They are filled with values that are true for this construction --
the "impurity" is the whole active space, there is no bath, and there are no
Schmidt singular values -- rather than with plausible-looking numbers. Anything
reading them will see an embedding that is trivially the active space, which is
exactly what it is.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def build(molecule, atoms, r, basis, ncas, nelecas):
    from pyscf import ao2mo, fci, gto, mcscf, scf

    # symmetry=True for the same reason as the scan: without it the degenerate
    # pi orbitals are fixed only up to a rotation, so the integrals -- and
    # therefore the determinant basis the quantum sampler works in -- change
    # from run to run.
    mol = gto.M(atom=f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {r:.6f}",
                basis=basis, symmetry=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at r={r:.4f}")

    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver = fci.addons.fix_spin_(mc.fcisolver, shift=0.5, ss=0)
    mc.fcisolver.conv_tol = 1e-14
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)
    e_casci = float(mc.kernel()[0])

    n_a = (nelecas + mol.spin) // 2
    n_b = nelecas - n_a

    step2 = {
        "h1e": np.asarray(h1),
        "h2e": np.asarray(h2),
        "ecore": float(ecore),
        "n_alpha": int(n_a),
        "n_beta": int(n_b),
        "n_emb": int(ncas),
        # The active space IS the impurity; there is no bath and no Schmidt
        # decomposition in this construction.
        "n_imp": int(ncas),
        "n_bath": 0,
        "mu": 0.0,
        "sv": np.zeros(0),
        "sv_all": np.zeros(0),
        "sv_gap": 0.0,
        "sv2_cov": 0.0,
        "uhf_energy": float(mf.e_tot),
        "reference_density_info": {
            "method": "casci-direct",
            "e_cas": e_casci,
            "n_active": int(ncas),
            "nel_active": int(nelecas),
        },
        "ref_occ_alpha": np.zeros(ncas),
        "ref_occ_beta": np.zeros(ncas),
        "mol_info": {
            "molecule": molecule,
            "basis": basis,
            "n_atoms": 2,
            "atom_syms": list(atoms),
            "n_electrons": int(mol.nelectron),
            "n_ao": int(mol.nao_nr()),
            # Provenance -- so a stale pickle cannot be mistaken for another
            # geometry. Config.cached_result_is_current() validates on
            # mol_info["molecule"], which is not enough when the only
            # difference between two runs is the bond length.
            "bond_length": float(r),
            "source": "export_cas_hamiltonian.py (direct CAS, no DMET)",
        },
    }
    return step2, e_casci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", default="N2")
    p.add_argument("--atoms", nargs=2, default=("N", "N"))
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--ncas", type=int, default=8)
    p.add_argument("--nelecas", type=int, default=10)
    p.add_argument("--project-dir", required=True)
    args = p.parse_args()

    step2, e_casci = build(args.molecule, tuple(args.atoms), args.r,
                           args.basis, args.ncas, args.nelecas)

    out_dir = Path(args.project_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "step2_hamiltonian.pkl"
    with open(out, "wb") as fh:
        pickle.dump(step2, fh)

    from math import comb
    ndet = comb(args.ncas, step2["n_alpha"]) * comb(args.ncas, step2["n_beta"])

    print(f"  molecule    : {args.molecule}/{args.basis}  r = {args.r:.4f} A")
    print(f"  active space: ({args.nelecas}e,{args.ncas}o)  "
          f"nelec = ({step2['n_alpha']}, {step2['n_beta']})")
    print(f"  determinants: {ndet}")
    print(f"  RHF         : {step2['uhf_energy']:.12f} Ha")
    print(f"  exact CASCI : {e_casci:.12f} Ha   <- the target")
    print(f"  ecore       : {step2['ecore']:.12f} Ha")
    print(f"\n  wrote {out}")
    print(f"\n  next:")
    print(f"    quenais-run --molecule {args.molecule} --basis {args.basis} "
          f"--solver gqe --steps 3 4 \\")
    print(f"                --project-dir {args.project_dir}")


if __name__ == "__main__":
    main()