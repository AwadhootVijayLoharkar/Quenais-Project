"""
Exact reference methods against PySCF.

Every assertion here is against a number PySCF produces independently, so
the module pins the integral handoff and the active-space bookkeeping
rather than re-deriving the physics.

Nothing here requires block2: the DMRG test asserts that a missing block2
is reported as a skip rather than taking down a step-0 run that has
already paid for CCSD, and checks the energy only if block2 is present.
"""

import pytest

pyscf = pytest.importorskip("pyscf")
pytestmark = pytest.mark.needs_pyscf

from pyscf import fci, gto, mcscf, scf  # noqa: E402

from quenais.classical.references import (  # noqa: E402
    restricted_reference,
    run_casci,
    run_dmrg,
    run_fci,
)
from quenais.classical.runner import best_reference  # noqa: E402
from quenais.config import Config  # noqa: E402
from quenais.settings import ReferenceSettings  # noqa: E402

H4 = "H 0 0 0; H 0 0 1.0; H 0 0 2.6; H 0 0 3.6"
H3 = "H 0 0 0; H 0 0 1.0; H 0 0 2.6"


def _h4():
    mol = gto.M(atom=H4, basis="sto-3g", verbose=0)
    return mol, scf.RHF(mol).run()


def _h3_doublet():
    mol = gto.M(atom=H3, basis="sto-3g", spin=1, verbose=0)
    return mol, scf.UHF(mol).run()


# ═════════════════════════════════════════════════════════════════════════
# FCI
# ═════════════════════════════════════════════════════════════════════════

def test_full_space_fci_equals_pyscf_fci():
    mol, mf = _h4()
    e, info = run_fci(mol, mf, ReferenceSettings())
    assert e is not None
    assert abs(e - fci.FCI(mf).kernel()[0]) < 1e-9
    assert info["n_determinants"] == 36
    assert info["in_active_space"] is False
    assert info["s_squared"] == pytest.approx(0.0, abs=1e-6)


def test_fci_refuses_an_oversized_space_before_doing_work():
    """
    The guard must fire on the size alone. FCI has no graceful
    degradation: past the memory limit the job is killed having produced
    nothing, so a late check is no check at all.
    """
    mol, mf = _h4()
    with pytest.warns(RuntimeWarning, match="over the"):
        e, info = run_fci(mol, mf, ReferenceSettings(fci_max_dets=10))
    assert e is None
    assert info["skipped"] == "too large"
    assert info["n_determinants"] == 36


def test_fci_in_active_space_matches_casci():
    mol, mf = _h4()
    e_fci, info = run_fci(mol, mf, ReferenceSettings(fci_in_active_space=True),
                          active=(2, 2, None))
    e_cas, _ = run_casci(mf, (2, 2, None), ReferenceSettings())
    assert abs(e_fci - e_cas) < 1e-9
    assert info["in_active_space"] is True


def test_fci_in_active_space_skipped_without_step1():
    mol, mf = _h4()
    e, info = run_fci(mol, mf, ReferenceSettings(fci_in_active_space=True),
                      active=None)
    assert e is None and "skipped" in info


# ═════════════════════════════════════════════════════════════════════════
# CASCI
# ═════════════════════════════════════════════════════════════════════════

def test_casci_equals_pyscf_casci_in_the_same_space():
    _mol, mf = _h4()
    e, info = run_casci(mf, (2, 2, None), ReferenceSettings())
    assert e is not None
    assert abs(e - mcscf.CASCI(mf, 2, 2).kernel()[0]) < 1e-9
    assert info["n_determinants"] == 4        # C(2,1) * C(2,1)
    assert info["norb"] == 2


def test_casci_in_the_full_active_space_equals_fci():
    """(4e,4o) is the whole basis for H4/STO-3G, so CASCI must be FCI."""
    mol, mf = _h4()
    e_cas, _ = run_casci(mf, (4, 4, None), ReferenceSettings())
    e_fci, _ = run_fci(mol, mf, ReferenceSettings())
    assert abs(e_cas - e_fci) < 1e-9


def test_casci_is_bounded_by_hf_and_fci():
    mol, mf = _h4()
    e_cas, _ = run_casci(mf, (2, 2, None), ReferenceSettings())
    e_fci, _ = run_fci(mol, mf, ReferenceSettings())
    assert e_fci - 1e-9 <= e_cas <= mf.e_tot + 1e-9


def test_casci_reports_excited_roots():
    _mol, mf = _h4()
    e, info = run_casci(mf, (4, 4, None), ReferenceSettings(casci_nroots=3))
    assert len(info["roots"]) == 3
    assert info["roots"] == sorted(info["roots"])
    assert e == pytest.approx(info["roots"][0])


def test_casci_skipped_without_an_active_space():
    _mol, mf = _h4()
    e, info = run_casci(mf, None, ReferenceSettings())
    assert e is None and "skipped" in info


# ═════════════════════════════════════════════════════════════════════════
# The spin-free reference
# ═════════════════════════════════════════════════════════════════════════

def test_closed_shell_reuses_the_callers_scf():
    """No pointless second SCF when the reference is already spin free."""
    mol, mf = _h4()
    ref, rebuilt = restricted_reference(mol, mf)
    assert ref is mf and rebuilt is False


def test_open_shell_gets_its_own_rohf():
    """
    UHF orbitals are spin dependent, so a CASCI core potential built from
    them is not spin free. The module must converge an ROHF instead, and
    record that it did.
    """
    mol, mf = _h3_doublet()
    ref, rebuilt = restricted_reference(mol, mf)
    assert rebuilt is True
    assert isinstance(ref, scf.rohf.ROHF)
    assert ref.converged
    # ROHF is variationally above UHF in the same basis.
    assert ref.e_tot >= mf.e_tot - 1e-8


def test_open_shell_casci_runs_on_the_rohf_reference():
    mol, mf = _h3_doublet()
    ref, _ = restricted_reference(mol, mf)
    e, info = run_casci(ref, (3, 3, None), ReferenceSettings())
    assert e is not None
    assert e <= ref.e_tot + 1e-9
    # Three electrons: the ground state is a doublet, S(S+1) = 0.75.
    assert info["s_squared"] == pytest.approx(0.75, abs=1e-3)


# ═════════════════════════════════════════════════════════════════════════
# DMRG
# ═════════════════════════════════════════════════════════════════════════

def _small_dmrg_settings():
    return ReferenceSettings(dmrg_bond_dims=(50, 100, 200),
                             dmrg_noises=(1e-4, 1e-5, 0.0),
                             dmrg_thrds=(1e-8, 1e-9, 1e-10),
                             dmrg_sweeps_per_stage=4)


def test_dmrg_reports_a_clean_skip_without_block2(tmp_path, monkeypatch):
    """
    A missing block2 must be reported, not raised.

    The skip path is forced by poisoning the import rather than by
    checking whether block2 happens to be installed. An environment-
    dependent branch means the assertion that actually runs differs
    between machines, so the case you care about is the one that never
    gets tested -- and on a machine WITH block2 this test would quietly
    become a live DMRG run inside the unit suite.
    """
    import sys

    monkeypatch.setitem(sys.modules, "pyblock2.driver.core", None)

    mol, mf = _h4()
    with pytest.warns(RuntimeWarning, match="block2 is not importable"):
        e, info = run_dmrg(mol, mf, (4, 4, None), _small_dmrg_settings(),
                           str(tmp_path))
    assert e is None
    assert info["skipped"] == "block2 not installed"


@pytest.mark.slow
@pytest.mark.needs_block2
def test_dmrg_matches_fci_on_a_full_active_space(tmp_path):
    """
    (4e,4o) is the whole basis for H4/STO-3G, so DMRG at any usable bond
    dimension is exact. This pins the PySCF -> block2 integral handoff,
    not the DMRG algorithm.

    Marked slow AND needs_block2: block2 is a C++ extension that can take
    the interpreter down with it, and a crash there must not be able to
    kill an otherwise healthy `pytest -m "not slow"` run.
    """
    pytest.importorskip("pyblock2")

    mol, mf = _h4()
    e, info = run_dmrg(mol, mf, (4, 4, None), _small_dmrg_settings(),
                       str(tmp_path))
    assert e is not None, info
    e_fci, _ = run_fci(mol, mf, ReferenceSettings())
    assert abs(e - e_fci) < 1e-6, (e, e_fci)
    assert [s["bond_dim"] for s in info["stages"]] == [50, 100, 200]
    # Variational: every stage sits above the exact answer.
    assert all(s["energy"] >= e_fci - 1e-9 for s in info["stages"])


def test_dmrg_skipped_without_an_active_space(tmp_path):
    mol, mf = _h4()
    settings = _small_dmrg_settings()          # dmrg_in_active_space=True
    e, info = run_dmrg(mol, mf, None, settings, str(tmp_path))
    assert e is None
    assert "skipped" in info


# ═════════════════════════════════════════════════════════════════════════
# End to end
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_step0_table_reports_errors_against_the_exact_reference(tmp_path, capsys):
    """The error column against an exact reference is the point of this work."""
    from quenais.classical import runner

    cfg = Config(molecule="H4", basis="sto-3g", project_dir=str(tmp_path),
                 geometry=H4, classical_methods=["HF", "MP2", "CCSD", "FCI"])
    cfg.validate().make_dirs().load_geometry()
    results = runner.main(cfg)

    assert results["methods"]["FCI"]["energy"] is not None
    name, e_ref = best_reference(results["methods"])
    assert name == "FCI"
    # Every approximate method sits above the exact answer.
    for method in ("HF", "MP2", "CCSD"):
        assert results["methods"][method]["energy"] >= e_ref - 1e-9
    assert "vs FCI" in capsys.readouterr().out
