"""
LAS against PySCF, and LASSQD against LASSCF.

Independent checks that need no mrh:
  * a determinant through the LAS energy reproduces RHF
  * one fragment spanning the active space: LASCI = CASCI, LASSCF = CASSCF
  * two fragments: E(FCI) <= E(LASSCF) <= E(LASCI) and a vanishing gradient
  * LASSQD with fragments small enough for SQD to see the whole Hilbert
    space reproduces LASSCF (the paper's first validation, Table 1)
"""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytestmark = pytest.mark.needs_pyscf

from pyscf import fci, gto, mcscf, scf  # noqa: E402

from quenais.las.driver import LasOptions, run_las  # noqa: E402
from quenais.las.energy import LasSpace, las_energy, mean_field_2rdm  # noqa: E402
from quenais.las.fci_solver import FCIFragmentSolver  # noqa: E402
from quenais.las.fragments import fragment_ao_indices, localize_active  # noqa: E402
from quenais.las.integrals import PyscfIntegrals  # noqa: E402

QUIET = dict(log=lambda *_: None)


def _mol(atom):
    return gto.M(atom=atom, basis="sto-3g", verbose=0)


H4 = "H 0 0 0; H 0 0 1.0; H 0 0 2.6; H 0 0 3.6"
H6 = "H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7; H 0 0 3.6; H 0 0 4.5"


def test_determinant_energy_is_rhf():
    mol = _mol(H6)
    mf = scf.RHF(mol).run()
    ints = PyscfIntegrals(mol)
    space = LasSpace(ncore=2, ncas_sub=(2,), nmo=6)
    d = np.zeros((2, 2, 2))
    d[:, 0, 0] = 1.0
    e = las_energy(ints, mf.mo_coeff, space, [d], [mean_field_2rdm(d)])
    assert abs(e - mf.e_tot) < 1e-10


def test_single_fragment_lasci_and_lasscf_equal_casci_and_casscf():
    mol = _mol(H6)
    mf = scf.RHF(mol).run()
    ints = PyscfIntegrals(mol)
    space = LasSpace(ncore=2, ncas_sub=(2,), nmo=6)
    solver = FCIFragmentSolver()
    lasci = run_las(ints, mf.mo_coeff, space, [(1, 1)], [0], solver,
                    LasOptions(orbital_optimization=False), **QUIET)
    e_casci = mcscf.CASCI(mf, 2, 2).kernel()[0]
    assert abs(lasci.energy - e_casci) < 1e-9

    lasscf = run_las(ints, mf.mo_coeff, space, [(1, 1)], [0], FCIFragmentSolver(),
                     LasOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-6), **QUIET)
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.conv_tol = 1e-11
    e_casscf = mc.kernel()[0]
    assert lasscf.converged
    assert abs(lasscf.energy - e_casscf) < 1e-7


def _h4_problem():
    mol = _mol(H4)
    mf = scf.RHF(mol).run()
    frag_aos = [fragment_ao_indices(mol, [0, 1]), fragment_ao_indices(mol, [2, 3])]
    blocks, sv = localize_active(mol, mf.mo_coeff, frag_aos, [2, 2])
    C = np.hstack(blocks)
    return mol, mf, C, LasSpace(ncore=0, ncas_sub=(2, 2), nmo=4), sv


def test_two_fragment_lasscf_bounds_and_stationarity():
    mol, mf, C, space, sv = _h4_problem()
    assert min(v.min() for v in sv) > 0.9           # H2 units localise cleanly
    ints = PyscfIntegrals(mol)
    nel = [(1, 1), (1, 1)]
    lasci = run_las(ints, C, space, nel, [0, 0], FCIFragmentSolver(),
                    LasOptions(orbital_optimization=False), **QUIET)
    lasscf = run_las(ints, C, space, nel, [0, 0], FCIFragmentSolver(),
                     LasOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-6), **QUIET)
    e_fci = fci.FCI(mf).kernel()[0]
    assert lasscf.converged and lasscf.grad_norm < 1e-6
    assert e_fci - 1e-9 <= lasscf.energy <= lasci.energy + 1e-9
    assert lasscf.energy < mf.e_tot


def test_density_fitting_is_close():
    mol, _mf, C, space, _sv = _h4_problem()
    nel = [(1, 1), (1, 1)]
    opts = LasOptions(orbital_optimization=False)
    e = run_las(PyscfIntegrals(mol), C, space, nel, [0, 0], FCIFragmentSolver(),
                opts, **QUIET).energy
    e_df = run_las(PyscfIntegrals(mol, density_fit=True), C, space, nel, [0, 0],
                   FCIFragmentSolver(), opts, **QUIET).energy
    assert abs(e - e_df) < 1e-3


def test_spin_penalty_selects_requested_spin():
    """
    Two electrons in two orbitals, M_S = 0: singlet and triplet both live in
    this sector. Whichever the bare Hamiltonian prefers, the penalised
    solver must return the requested S^2 and that state's exact energy.
    """
    from pyscf.fci import direct_uhf, spin_op

    from quenais.las.fci_solver import spin_penalized_uhf_solver

    n, nelec = 2, (1, 1)
    h = np.array([[0.0, 0.05], [0.05, 0.02]])
    eri = np.zeros((n,) * 4)
    eri[0, 0, 0, 0] = eri[1, 1, 1, 1] = 1.0
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.6
    eri[0, 1, 0, 1] = eri[1, 0, 1, 0] = eri[0, 1, 1, 0] = eri[1, 0, 0, 1] = 0.3
    ham = ((h, h), (eri, eri, eri))
    bare = direct_uhf.FCISolver()
    bare.conv_tol = 1e-12
    w = np.asarray(bare.kernel(*ham, n, nelec, nroots=4)[0])   # all 4 states
    for target_ss in (0.0, 2.0):
        solver = spin_penalized_uhf_solver(1.0, target_ss)
        solver.conv_tol = 1e-12
        e, ci = solver.kernel(*ham, n, nelec, nroots=1)
        ss, _ = spin_op.spin_square0(ci, n, nelec)
        assert abs(ss - target_ss) < 1e-6
        assert np.min(np.abs(w - e)) < 1e-8      # an eigenvalue of the bare H


def test_selected_ci_with_spin_dependent_h_equals_uhf_fci():
    """
    Guards the one PySCF internal LASSQD relies on: kernel_fixed_space must
    route through our contract_2e/make_hdiag overrides. In the full string
    space the result must equal direct_uhf FCI with h_alpha != h_beta.
    """
    from pyscf.fci import cistring

    from quenais.las.fci_solver import spin_penalized_uhf_solver
    from quenais.las.sqd_solver import _sci_solve
    from quenais.settings import LasSettings

    n, nelec = 4, (2, 1)
    rng = np.random.default_rng(3)
    h = rng.normal(size=(2, n, n))
    h1s = 0.5 * (h + h.transpose(0, 2, 1))
    b = rng.normal(scale=0.3, size=(6, n, n))
    b = 0.5 * (b + b.transpose(0, 2, 1))
    eri = np.einsum("Lpq,Lrs->pqrs", b, b)
    s = LasSettings(fragments=["0:4:2,1"], sqd_nroots=1, sqd_davidson_tol=1e-12)
    sa = cistring.make_strings(range(n), nelec[0])
    sb = cistring.make_strings(range(n), nelec[1])
    e_sci = _sci_solve(h1s, eri, n, nelec, sa, sb, 1, s)[0]
    solver = spin_penalized_uhf_solver(s.sqd_spin_shift, 0.75)
    solver.conv_tol = 1e-12
    e_fci = solver.kernel((h1s[0], h1s[1]), (eri, eri, eri), n, nelec)[0]
    assert abs(e_sci - e_fci) < 1e-8, (e_sci, e_fci)


@pytest.mark.slow
@pytest.mark.needs_qiskit
@pytest.mark.parametrize("eps", [0.0, 1e-5])
def test_lassqd_reproduces_lasscf_when_sqd_spans_fragment_space(eps):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit_addon_sqd")
    from quenais.las.sqd_solver import LassqdFragmentSolver
    from quenais.settings import LasSettings

    mol, _mf, C, space, _sv = _h4_problem()
    ints = PyscfIntegrals(mol)
    nel = [(1, 1), (1, 1)]
    ref = run_las(ints, C, space, nel, [0, 0], FCIFragmentSolver(),
                  LasOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-6), **QUIET)
    s = LasSettings(fragments=["0+1:2:1,1", "2+3:2:1,1"], backend="aer_statevector",
                    shots=4000, sqd_batches=2, sqd_samples_per_batch=16,
                    sqd_iterations=2, carryover_eps=eps, seed=7)
    res = run_las(ints, C, space, nel, [0, 0], LassqdFragmentSolver(s, log=lambda *_: None),
                  LasOptions(max_macro=40, conv_tol_energy=1e-8, n_consecutive=2),
                  **QUIET)
    assert all(f["subspace_fraction"] == 1.0 for f in res.history[-1]["fragments"])
    assert abs(res.energy - ref.energy) < 1e-6, (res.energy, ref.energy)