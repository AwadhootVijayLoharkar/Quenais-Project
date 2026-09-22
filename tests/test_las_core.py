"""
LAS core math against brute force. NumPy/SciPy only -- runs everywhere.

Toy system: 6 orthonormal orbitals, rotated by a random orthogonal C,
partitioned [1 core | frag A (2 orb) | frag B (2 orb) | 1 virtual].
"""

import numpy as np
import pytest

from quenais.las.driver import LasOptions, run_las
from quenais.las.energy import (
    LasSpace,
    fragment_hamiltonians,
    las_energy,
    las_energy_and_gradient,
    nonredundant_mask,
)
from quenais.las.optimizer import rotate
from tests._las_toy import (
    ToyFCISolver,
    ToyIntegrals,
    closed_block,
    empty_block,
    exact_rdms,
    full_fci_energy,
    local_qubits,
    product_state,
    random_fragment_state,
    random_orthogonal,
)

NMO = 6
SPACE = LasSpace(ncore=1, ncas_sub=(2, 2), nmo=NMO)


def _setup(seed=3, nelec_sub=((1, 1), (1, 1))):
    ints = ToyIntegrals(NMO, seed=seed)
    C = random_orthogonal(NMO, seed + 100)
    states = [random_fragment_state(2, na, nb, seed + 10 + k)
              for k, (na, nb) in enumerate(nelec_sub)]
    rdms = [exact_rdms(psi, 2, local_qubits(2)) for psi in states]
    dm1s = [r[0] for r in rdms]
    dm2 = [r[1] for r in rdms]
    return ints, C, states, dm1s, dm2


def _global_qubits(space, nvirt):
    """JW map of the product state: blocks [core, A, B, virt], each local."""
    sizes = [space.ncore, *space.ncas_sub, nvirt]
    qa, qb, off = [], [], 0
    for n in sizes:
        qa += [off + p for p in range(n)]
        qb += [off + n + p for p in range(n)]
        off += 2 * n
    return [qa, qb]


def _exact_energy(ints, C, space, states):
    nvirt = space.nmo - space.nocc
    psi = product_state([closed_block(space.ncore), *states, empty_block(nvirt)])
    dm1s, g = exact_rdms(psi, space.nmo, _global_qubits(space, nvirt))
    h = C.T @ ints.hcore @ C
    eri = ints.ao2mo(C, C, C, C)
    return (ints.energy_nuc + np.sum(h * (dm1s[0] + dm1s[1]))
            + 0.5 * np.einsum("pqrs,pqrs->", eri, g))


@pytest.mark.parametrize("nelec_sub", [((1, 1), (1, 1)), ((2, 1), (0, 1)), ((1, 0), (2, 2))])
def test_energy_matches_explicit_product_state(nelec_sub):
    ints, C, states, dm1s, dm2 = _setup(nelec_sub=nelec_sub)
    e_las = las_energy(ints, C, SPACE, dm1s, dm2)
    e_ref = _exact_energy(ints, C, SPACE, states)
    assert abs(e_las - e_ref) < 1e-10, (e_las, e_ref)


def test_fragment_hamiltonian_reproduces_total_energy():
    """const_K + <H_K> must equal the total energy for ANY state of fragment K."""
    nelec_sub = ((1, 1), (2, 1))
    ints, C, states, dm1s, dm2 = _setup(nelec_sub=nelec_sub)
    hams = fragment_hamiltonians(ints, C, SPACE, dm1s, nelec_sub, (0, 1), dm2=dm2)
    for k, ham in enumerate(hams):
        for seed in range(3):
            na, nb = nelec_sub[k]
            psi = random_fragment_state(2, na, nb, 500 + seed)
            d1, d2 = exact_rdms(psi, 2, local_qubits(2))
            new1, new2 = list(dm1s), list(dm2)
            new1[k], new2[k] = d1, d2
            # the environment density is held fixed: only fragment k changes
            e_frag = (np.sum(ham.h1s[0] * d1[0]) + np.sum(ham.h1s[1] * d1[1])
                      + 0.5 * np.einsum("pqrs,pqrs->", d2, ham.eri))
            e_tot = las_energy(ints, C, SPACE, new1, new2)
            assert abs(ham.const + e_frag - e_tot) < 1e-10


@pytest.mark.parametrize("nelec_sub", [((1, 1), (1, 1)), ((2, 1), (0, 1))])
def test_orbital_gradient_finite_difference(nelec_sub):
    ints, C, _states, dm1s, dm2 = _setup(nelec_sub=nelec_sub)
    mask = nonredundant_mask(SPACE)
    _e, g = las_energy_and_gradient(ints, C, SPACE, dm1s, dm2)
    rng = np.random.default_rng(7)
    for _ in range(4):
        x = rng.normal(size=mask.sum())
        t = 1e-5
        ep = las_energy(ints, rotate(C, mask, t * x), SPACE, dm1s, dm2)
        em = las_energy(ints, rotate(C, mask, -t * x), SPACE, dm1s, dm2)
        fd = (ep - em) / (2 * t)
        an = g[mask] @ x
        assert abs(fd - an) < 1e-7 * max(1.0, abs(an)), (fd, an)


def test_redundant_rotations_excluded():
    m = nonredundant_mask(SPACE)
    assert not m[2, 1]          # within fragment A
    assert not m[4, 3]          # within fragment B
    assert m[3, 1]              # A-B
    assert m[1, 0] and m[5, 0]  # core-active, core-virtual
    assert not np.triu(m).any()


def test_lasscf_converges_to_stationary_point_above_fci():
    nelec_sub = ((1, 1), (1, 1))
    ints = ToyIntegrals(NMO, seed=11)
    C0 = random_orthogonal(NMO, 211)
    opts = LasOptions(max_macro=60, conv_tol_energy=1e-10, conv_tol_grad=1e-6)
    lasci = run_las(ints, C0, SPACE, nelec_sub, (0, 0), ToyFCISolver(),
                    LasOptions(orbital_optimization=False), log=lambda *_: None)
    res = run_las(ints, C0, SPACE, nelec_sub, (0, 0), ToyFCISolver(), opts,
                  log=lambda *_: None)
    assert res.converged, res.history[-1]
    assert res.grad_norm < 1e-6
    assert res.energy <= lasci.energy + 1e-10
    e_fci = full_fci_energy(ints, C0, (3, 3))
    assert res.energy >= e_fci - 1e-10
    # Fragment states are eigenstates of their final Hamiltonians.
    hams = fragment_hamiltonians(ints, res.C, SPACE, res.dm1s, nelec_sub, (0, 0),
                                 dm2=res.dm2)
    again = ToyFCISolver().solve(hams)
    e_again = las_energy(ints, res.C, SPACE, [r.dm1s for r in again],
                         [r.dm2 for r in again])
    assert abs(e_again - res.energy) < 1e-8


def test_lasci_single_fragment_equals_casci():
    """One fragment spanning the whole active space is CASCI."""
    space = LasSpace(ncore=1, ncas_sub=(4,), nmo=NMO)
    ints = ToyIntegrals(NMO, seed=5)
    C = random_orthogonal(NMO, 77)
    res = run_las(ints, C, space, [(2, 2)], [0], ToyFCISolver(),
                  LasOptions(orbital_optimization=False), log=lambda *_: None)
    # CASCI by brute force: core frozen, 4 active orbitals, full diagonalisation
    from tests._las_toy import hamiltonian_matrix, sector_states
    h = C.T @ ints.hcore @ C
    eri = ints.ao2mo(C, C, C, C)
    core = [0]
    act = slice(1, 5)
    hc = h + 2 * np.einsum("pqii->pq", eri[:, :, core][:, :, :, core]) \
        - np.einsum("piiq->pq", eri[:, core][:, :, core])
    ecore = ints.energy_nuc + 2 * h[0, 0] + 2 * eri[0, 0, 0, 0] - eri[0, 0, 0, 0]
    st = sector_states(4, 2, 2)
    hm = hamiltonian_matrix([hc[act, act]] * 2, eri[act, act, act, act], 4, st)
    e_casci = np.linalg.eigvalsh(hm)[0] + ecore
    assert abs(res.energy - e_casci) < 1e-10
