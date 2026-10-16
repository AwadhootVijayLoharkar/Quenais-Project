"""
Fixed-size ASCI (det_expansion._asci_core) on a brute-force toy Hamiltonian,
no PySCF: full size gives FCI, any size stays variational, and ASCI seeded
with the growth-CIPSI set is never worse than that set.
"""

from __future__ import annotations

import types

import numpy as np

from quenais.quantum import det_expansion as dx
from quenais.quantum import sqd_list
from tests._las_toy import ToyIntegrals, hamiltonian_matrix


def _toy(n=5, na=2, nb=2, seed=3):
    ints = ToyIntegrals(n, seed=seed)
    c = np.eye(n)
    eri = ints.ao2mo(c, c, c, c).reshape(n, n, n, n)
    a, b = sqd_list.ci_strings(n, na), sqd_list.ci_strings(n, nb)
    states = np.array([x | (y << n) for x in a for y in b])
    H = hamiltonian_matrix([ints.hcore, ints.hcore], eri, n, states)
    space = types.SimpleNamespace(ndet=H.shape[0], na=a.size, nb=b.size)
    sigma = lambda v: (H @ np.asarray(v).reshape(-1)).reshape(np.shape(v))  # noqa: E731
    mol = types.SimpleNamespace(cas_hamiltonian=types.SimpleNamespace(e_core=0.0))
    return H, space, sigma, np.diag(H).copy(), mol


def test_full_size_is_exact():
    H, space, sigma, hd, _ = _toy()
    sel, e, _ = dx._asci_core([int(np.argmin(hd))], space.ndet, space, sigma, hd)
    assert sel.size == space.ndet
    assert abs(e - np.linalg.eigvalsh(H)[0]) < 1e-9


def test_variational_and_not_worse_than_cipsi():
    H, space, sigma, hd, mol = _toy(n=6, na=3, nb=3, seed=7)
    e_fci = np.linalg.eigvalsh(H)[0]
    for n in (5, 12, 30, 60):
        grown, _ = dx._grow(mol, [int(np.argmin(hd))], n, space, sigma, hd,
                            verbose=False)
        e_ci, _ = dx._diagonalise_subspace(grown, space, sigma)
        sel, e, _ = dx._asci_core(grown, n, space, sigma, hd)
        assert e >= e_fci - 1e-10
        assert e <= e_ci + 1e-10
        assert sel.size <= n


def test_hci_exact_size_variational_and_full_space_exact():
    H, space, sigma, hd, _ = _toy(n=6, na=3, nb=3, seed=7)
    e_fci = np.linalg.eigvalsh(H)[0]
    for n in (7, 20, 50, 120):
        sel, e, eps = dx._hci_core(n, space, sigma, hd)
        assert sel.size == n, (n, sel.size)
        assert e >= e_fci - 1e-10
    sel, e, _ = dx._hci_core(space.ndet, space, sigma, hd)
    assert abs(e - e_fci) < 1e-8


def test_hci_heat_bath_criterion():
    """Threshold stage: every determinant left out scores <= eps."""
    H, space, sigma, hd, _ = _toy(n=5, na=2, nb=2, seed=4)
    cols = dx._Columns(space, sigma)
    sel, e, c, score = dx._hci_threshold([int(np.argmin(hd))], 0.05, space,
                                         sigma, cols)
    assert np.all(score[np.isfinite(score)] <= 0.05 + 1e-12)
    # brute force: max_i |H_ai c_i| for a not in sel
    mask = np.ones(space.ndet, bool); mask[sel] = False
    brute = np.max(np.abs(H[np.ix_(mask, sel)] * c[None, :]), axis=1)
    assert np.allclose(brute, score[mask])
