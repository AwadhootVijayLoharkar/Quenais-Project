"""
Is GQE's starting state on the DMET embedding really Hartree-Fock?

DMETEmbeddingMolecule hands GQE the RAW embedding integrals
(cas_hamiltonian = h1e_emb, h2e_emb: impurity + bath, a Schmidt basis),
while the CCSD amplitudes that build the operator pool come from an SCF
inside that space, i.e. in the SCF MO basis. A circuit that starts from
"the lowest n orbitals occupied" is then HF only if the raw embedding
basis already coincides with the SCF occupied space.

This script measures that, for every step-2 pickle it finds:

  overlap  singular values of C_mo[:n_occ, :n_occ] (occupied SCF MOs
           projected on the first n_occ raw embedding orbitals);
           all ~1 -> same space, no problem; small -> the reference
           determinant GQE starts from is NOT HF
  E(ref)   energy of the determinant |lowest orbitals occupied> in the
           raw basis, vs the SCF energy

    python tools/check_gqe_basis.py $B/n2_curve
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def check(path):
    from quenais.quantum.det_analysis import DeterminantSpace, _sigma_operator
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle

    mol = load_from_dmet_pickle(path, cache_dir=Path("/tmp") / "qn_basis_check")
    na, nb = mol.nelec
    c = np.asarray(mol.hf.mo_coeff)
    if c.ndim == 3:                      # UHF
        c = c[0]
    sv = np.linalg.svd(c[:na, :na], compute_uv=False)
    space = DeterminantSpace(mol.norb, mol.nelec)
    sigma = _sigma_operator(mol, space)
    v = np.zeros((space.na, space.nb))
    v[0, 0] = 1.0                        # lowest orbitals occupied (raw basis)
    e_ref = float(np.sum(v * sigma(v))) + mol.cas_hamiltonian.e_core
    e_hf = float(mol.hf.e_tot)
    print(f"{path}")
    print(f"   norb={mol.norb} nelec={mol.nelec}  overlap min/mean "
          f"{sv.min():.3f}/{sv.mean():.3f}   E(ref det) - E(HF) = "
          f"{(e_ref - e_hf) * 627.5095:9.2f} kcal/mol")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    paths = sorted(p for p in root.rglob("*.pkl") if "step2" in p.name)
    if not paths:
        sys.exit(f"no step2*.pkl under {root}")
    seen = set()
    for p in paths:
        key = p.stat().st_size
        if key in seen:
            continue
        seen.add(key)
        try:
            check(p)
        except Exception as exc:         # keep going over other systems
            print(f"{p}: {exc}")
        if len(seen) >= 8:
            break


if __name__ == "__main__":
    main()
