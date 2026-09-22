"""
Our LASSCF against mrh numbers recorded in
tests/regression/golden/las_mrh_reference.json (produced locally; see its
_readme). Skips every system whose reference is still null.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytestmark = pytest.mark.needs_pyscf

REF = json.loads((Path(__file__).parent / "regression" / "golden"
                  / "las_mrh_reference.json").read_text())
SYSTEMS = [k for k in REF if not k.startswith("_")]


@pytest.mark.parametrize("name", SYSTEMS)
def test_lasscf_matches_mrh(name):
    entry = REF[name]
    if entry.get("e_lasscf") is None:
        pytest.skip(f"{name}: no mrh reference recorded yet")
    from pyscf import gto, scf

    from quenais.las.driver import LasOptions, run_las
    from quenais.las.energy import LasSpace
    from quenais.las.fci_solver import FCIFragmentSolver
    from quenais.las.fragments import (
        fragment_ao_indices,
        localize_active,
        parse_fragment_spec,
        resolve_atoms,
    )
    from quenais.las.integrals import PyscfIntegrals

    mol = gto.M(atom=entry["atom"], basis=entry["basis"], verbose=0)
    mf = scf.RHF(mol).run()
    specs = [parse_fragment_spec(s) for s in entry["fragments"]]
    syms = [mol.atom_symbol(i) for i in range(mol.natm)]
    aos = [fragment_ao_indices(mol, resolve_atoms(s, syms)) for s in specs]
    blocks, _ = localize_active(mol, mf.mo_coeff, aos, [s.norb for s in specs])
    C = np.hstack(blocks)
    space = LasSpace(ncore=0, ncas_sub=tuple(s.norb for s in specs), nmo=C.shape[1])
    res = run_las(PyscfIntegrals(mol), C, space, [s.nelec for s in specs],
                  [s.spin2s for s in specs], FCIFragmentSolver(),
                  LasOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-6),
                  log=lambda *_: None)
    assert res.converged
    assert abs(res.energy - entry["e_lasscf"]) < 1e-6
