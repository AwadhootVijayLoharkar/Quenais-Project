"""
GQE orbital basis on DMET embeddings (gqe_adapter orbital_basis).

The physics tests (needs_pyscf) build an "embedding" Hamiltonian by
rotating H4/STO-3G MO integrals with a random orthogonal matrix -- the
same situation as a Schmidt basis: correct Hamiltonian, but the lowest
orbitals are not the HF ones. Then:

  * "embedding": determinant 0 is NOT HF (the bug)
  * "mo":        determinant 0 IS HF, CASCI energy unchanged,
                 occupations (rotated back) unchanged
"""

from __future__ import annotations

import numpy as np
import pytest

from quenais.settings import GqeSettings


def test_settings_and_cli_wiring(tmp_path):
    from quenais.cli import build_gqe_settings, build_parser

    assert GqeSettings().dmet_basis == "embedding"
    with pytest.raises(ValueError):
        GqeSettings(dmet_basis="canonical").validate()
    over = GqeSettings(dmet_basis="mo").hydra_overrides(step2_pickle_path="x.pkl")
    assert "molecule.orbital_basis=mo" in over
    args = build_parser().parse_args(["--molecule", "N2", "--solver", "gqe",
                                      "--gqe-basis", "mo"])
    assert build_gqe_settings(args).dmet_basis == "mo"

    import run_benchmark as d
    a, _ = d.normalise_params({"solver": "gqe", "gqe_basis": "mo"})
    assert a["gqe_basis"] == "mo"
    b, _ = d.normalise_params({"solver": "gqe"})
    assert "gqe_basis" not in b          # old runs keep their identity


def _rotated_h4():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6",
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    n = c.shape[1]
    h1 = c.T @ mf.get_hcore() @ c
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, c), n)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(n, n)))
    h1r = q.T @ h1 @ q
    h2r = np.einsum("pqrs,pi,qj,rk,sl->ijkl", h2, q, q, q, q)
    return h1r, h2r, float(mol.energy_nuc())


def _det0_energy(m):
    from quenais.quantum.det_analysis import DeterminantSpace, _sigma_operator

    sp = DeterminantSpace(m.norb, m.nelec)
    v = np.zeros((sp.na, sp.nb))
    v[0, 0] = 1.0
    return float(np.sum(v * _sigma_operator(m, sp)(v))) + m.cas_hamiltonian.e_core


@pytest.mark.needs_pyscf
def test_mo_basis_makes_reference_hf(tmp_path):
    pytest.importorskip("pyscf")
    from quenais.quantum.gqe_adapter import DMETEmbeddingMolecule

    h1, h2, ec = _rotated_h4()
    emb = DMETEmbeddingMolecule(h1, h2, ec, 2, 2, cache_dir=tmp_path / "a")
    mo = DMETEmbeddingMolecule(h1, h2, ec, 2, 2, cache_dir=tmp_path / "b",
                               orbital_basis="mo")
    e_hf = float(mo.hf.e_tot)
    assert abs(_det0_energy(mo) - e_hf) < 1e-8
    assert abs(_det0_energy(emb) - e_hf) > 1e-3          # the bug, reproduced
    assert abs(emb.compute_casci() - mo.compute_casci()) < 1e-9
    oa, ob = emb.casci_avg_occs()
    ma, mb = mo.casci_avg_occs()
    assert np.allclose(oa, ma, atol=1e-7) and np.allclose(ob, mb, atol=1e-7)
    # CCSD amplitudes live in the same (MO) basis as the rotated Hamiltonian
    assert np.allclose(np.asarray(mo.ccsd_amplitude["t2"]),
                       np.asarray(emb.ccsd_amplitude["t2"]))


@pytest.mark.needs_pyscf
def test_mo_basis_cache_is_separate(tmp_path):
    pytest.importorskip("pyscf")
    from quenais.quantum.gqe_adapter import DMETEmbeddingMolecule

    h1, h2, ec = _rotated_h4()
    a = DMETEmbeddingMolecule(h1, h2, ec, 2, 2, cache_dir=tmp_path)
    b = DMETEmbeddingMolecule(h1, h2, ec, 2, 2, cache_dir=tmp_path,
                              orbital_basis="mo")
    assert a._cache_key != b._cache_key
