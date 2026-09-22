"""
LASSQD bookkeeping without Qiskit, ffsim or PySCF.

The three external calls of LassqdFragmentSolver are replaced by exact
stand-ins: circuits by the exact fragment ground state, the sampler by
drawing bitstrings from |psi|^2 in Qiskit's printed bit order, and the
selected-CI solve by brute-force diagonalisation in the alpha x beta
product space. Everything else -- post-selection, batching, string
conventions, carryover and its transport between cycles, rotating RDMs
back to the LAS basis, the macro loop -- is the production code.
"""

import sys
import types

import numpy as np

from quenais.las import sqd_bits as bits
from quenais.las import sqd_solver
from quenais.las.driver import LasOptions, run_las
from quenais.las.energy import LasSpace
from quenais.settings import LasSettings
from tests._las_toy import (
    ToyFCISolver,
    ToyIntegrals,
    exact_rdms,
    hamiltonian_matrix,
    local_qubits,
    local_state,
    random_orthogonal,
    sector_states,
)

SPACE = LasSpace(ncore=1, ncas_sub=(2, 3), nmo=6)
CLOSED = ([(1, 1), (1, 1)], [0, 0], ["0:2:1,1", "1:3:1,1"])
OPEN = ([(1, 1), (2, 1)], [0, 1], ["0:2:1,1", "1:3:2,1"])


class _ExactCircuit:
    def __init__(self, psi, n):
        self.psi, self.num_qubits = psi, 2 * n


def _fake_prepare(self, ham):
    n, (na, nb) = ham.norb, ham.nelec
    h_avg = 0.5 * (ham.h1s[0] + ham.h1s[1])
    u = np.linalg.eigh(h_avg)[1]
    h1_mo = np.einsum("pi,spq,qj->sij", u, ham.h1s, u)
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", ham.eri, u, u, u, u)
    st = sector_states(n, na, nb)
    w, v = np.linalg.eigh(hamiltonian_matrix(h1_mo, eri_mo, n, st))
    psi = local_state(n, na, nb, v[:, 0], st)
    return {"u": u, "h1_mo": h1_mo, "eri_mo": eri_mo, "circuit": _ExactCircuit(psi, n)}


def _fake_sample(circuits, settings):
    rng = np.random.default_rng(settings.seed)
    out = []
    for c in circuits:
        p = np.abs(c.psi) ** 2
        draws = rng.choice(p.size, size=settings.shots, p=p / p.sum())
        idx, cnt = np.unique(draws, return_counts=True)
        # JW index bit j = qubit j; Qiskit prints the highest qubit first.
        out.append({format(int(i), f"0{c.num_qubits}b"): int(k)
                    for i, k in zip(idx, cnt)})
    return out


def _fake_sci(h1s, eri, n, nelec, sa, sb, spin2s, settings):
    """
    Mirrors production: spin-averaged Hamiltonian (here brute force instead
    of PySCF) plus the production spin-dependent string operators.
    """
    sa = np.unique(np.asarray(sa, dtype=np.int64))
    sb = np.unique(np.asarray(sb, dtype=np.int64))
    states = np.array([a | (b << n) for a in sa for b in sb])
    h_avg, dh = 0.5 * (h1s[0] + h1s[1]), 0.5 * (h1s[0] - h1s[1])
    hmat = hamiltonian_matrix([h_avg, h_avg], eri, n, states)
    a_op = bits.one_body_string_operator(sa, n, dh)
    b_op = bits.one_body_string_operator(sb, n, dh)
    hmat = hmat + np.kron(a_op, np.eye(sb.size)) - np.kron(np.eye(sa.size), b_op)
    w, v = np.linalg.eigh(hmat)
    psi = local_state(n, *nelec, v[:, 0], states)
    dm1s, dm2 = exact_rdms(psi, n, local_qubits(n))
    return float(w[0]), v[:, 0].reshape(sa.size, sb.size), sa, sb, dm1s, dm2


def _fake_recover(mat, probs, occs, num_elec_a, num_elec_b, rand_seed=None):
    return mat, probs


def _patched(fn):
    fake_mod = types.ModuleType("qiskit_addon_sqd.configuration_recovery")
    fake_mod.recover_configurations = _fake_recover
    orig = (sqd_solver.LassqdFragmentSolver._prepare,
            sqd_solver.sample_fragment_circuits, sqd_solver._sci_solve)
    mods = {k: sys.modules.get(k) for k in
            ("qiskit_addon_sqd", "qiskit_addon_sqd.configuration_recovery")}
    sqd_solver.LassqdFragmentSolver._prepare = _fake_prepare
    sqd_solver.sample_fragment_circuits = _fake_sample
    sqd_solver._sci_solve = _fake_sci
    sys.modules["qiskit_addon_sqd"] = types.ModuleType("qiskit_addon_sqd")
    sys.modules["qiskit_addon_sqd.configuration_recovery"] = fake_mod
    try:
        return fn()
    finally:
        (sqd_solver.LassqdFragmentSolver._prepare,
         sqd_solver.sample_fragment_circuits, sqd_solver._sci_solve) = orig
        for k, m in mods.items():
            if m is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = m


def _run(eps, samples_per_batch, shots=20000, system=OPEN):
    nelec, spin2s, frags = system
    ints = ToyIntegrals(6, seed=11)
    C = random_orthogonal(6, 211)
    s = LasSettings(fragments=frags, shots=shots, sqd_batches=3,
                    sqd_samples_per_batch=samples_per_batch, sqd_iterations=2,
                    carryover_eps=eps, seed=3)
    solver = sqd_solver.LassqdFragmentSolver(s, log=lambda *_: None)
    opts = LasOptions(max_macro=60, conv_tol_energy=1e-10, n_consecutive=2)
    res = run_las(ints, C, SPACE, nelec, spin2s, solver, opts, log=lambda *_: None)
    ref = run_las(ints, C, SPACE, nelec, spin2s, ToyFCISolver(),
                  LasOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-6),
                  log=lambda *_: None)
    return res, ref


def test_full_subspace_sqd_reproduces_lasscf_closed_shell():
    """Samples span each fragment's space: LASSQD must equal LASSCF."""
    res, ref = _patched(lambda: _run(eps=1e-5, samples_per_batch=50, system=CLOSED))
    fr = res.history[-1]["fragments"]
    assert all(f["subspace_fraction"] == 1.0 for f in fr), fr
    assert res.converged
    assert abs(res.energy - ref.energy) < 1e-7, (res.energy, ref.energy)


def test_full_subspace_sqd_open_shell_environment_matches_lasscf():
    """
    Fragment 0 sees fragment 1's (2a,1b) density, so h_alpha != h_beta.
    With the spin-dependent term handled, LASSQD sits on an exact LASSCF
    stationary point. (Random toy Hamiltonians have several; compare with
    LASSCF restarted from the LASSQD solution, i.e. in the same basin.)
    """
    res, _ = _patched(lambda: _run(eps=1e-5, samples_per_batch=50, system=OPEN))
    assert res.converged
    ints = ToyIntegrals(6, seed=11)
    ref = run_las(ints, res.C, SPACE, OPEN[0], OPEN[1], ToyFCISolver(),
                  LasOptions(conv_tol_energy=1e-10, conv_tol_grad=1e-6),
                  dm1s_guess=res.dm1s, log=lambda *_: None)
    assert abs(res.energy - ref.energy) < 1e-6, (res.energy, ref.energy)


def test_spin_dependent_string_operator_matches_brute_force():
    """A (x) 1 - 1 (x) B == sum dh (E^a - E^b) in a partial product space."""
    n = 3
    rng = np.random.default_rng(4)
    dh = rng.normal(size=(n, n))
    dh = dh + dh.T
    sa = np.array([0b011, 0b101, 0b110])
    sb = np.array([0b001, 0b100])                 # deliberately incomplete
    states = np.array([a | (b << n) for a in sa for b in sb])
    ref = hamiltonian_matrix([dh, -dh], np.zeros((n,) * 4), n, states)
    a_op = bits.one_body_string_operator(sa, n, dh)
    b_op = bits.one_body_string_operator(sb, n, dh)
    got = np.kron(a_op, np.eye(sb.size)) - np.kron(np.eye(sa.size), b_op)
    assert np.allclose(got, ref, atol=1e-12)


def test_carryover_is_carried_and_transported():
    res, _ref = _patched(lambda: _run(eps=1e-5, samples_per_batch=2, shots=400))
    its = [f["iterations"] for h in res.history for f in h["fragments"]]
    assert any(it[-1]["n_carry"][0] > 0 for it in its)
    # With carryover the subspace cannot shrink below what was carried.
    dims = [f["subspace_dim"] for h in res.history for f in h["fragments"][1:2]]
    assert max(dims) >= dims[0]


def test_no_carryover_is_plain_sqd():
    res, _ref = _patched(lambda: _run(eps=0.0, samples_per_batch=2, shots=400))
    its = [f["iterations"] for h in res.history for f in h["fragments"]]
    assert all(it[-1]["n_carry"] == (0, 0) for it in its)
