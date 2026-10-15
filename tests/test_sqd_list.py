"""
Determinant-list SQD (--sqd-subspace list, quenais/quantum/sqd_list.py).

The first group runs without Qiskit, ffsim or PySCF: circuit = exact
ground state, sampler = |psi|^2 (the stand-ins of test_las_sqd_logic), and
the full-space sigma is a brute-force Jordan-Wigner matrix in the list
module's own determinant order. Everything else is production code.

The needs_pyscf group repeats the two physics checks on real H4 / STO-3G
integrals with PySCF's sigma (the operator used in production and by the
CIPSI column).
"""

from __future__ import annotations

import numpy as np
import pytest

from quenais.quantum import sqd_list
from quenais.settings import QiskitSolverSettings
from tests._las_toy import hamiltonian_matrix
from tests.test_las_sqd_logic import _patched
from tests.test_sqd_opt import _exact, _toy_step2


def _toy_sigma(h1, eri, norb, nelec):
    """Dense H in the list module's order: full index = ia * n_b + ib."""
    a = sqd_list.ci_strings(norb, nelec[0])
    b = sqd_list.ci_strings(norb, nelec[1])
    states = np.array([x | (y << norb) for x in a for y in b])
    H = hamiltonian_matrix([h1, h1], eri, norb, states)

    def sigma(c):
        return (H @ np.asarray(c).reshape(-1)).reshape(np.shape(c))

    return sigma


def _run_list(step2, shots=20000, **kw):
    from quenais.quantum.sqd_opt import run_lucj_opt_sqd

    q = QiskitSolverSettings(ansatz="lucj_opt", n_shots=shots, seed=3,
                             sqd_batches=3, sqd_opt_iterations=2,
                             sqd_subspace="list", **kw)
    orig = sqd_list.make_sigma
    sqd_list.make_sigma = _toy_sigma
    try:
        return _patched(lambda: run_lucj_opt_sqd(step2, q, log=lambda *_: None))
    finally:
        sqd_list.make_sigma = orig


# ── bookkeeping ───────────────────────────────────────────────────────────

def test_strings_ascending_and_indices_roundtrip():
    s = sqd_list.ci_strings(5, 2)
    assert s.size == 10 and np.all(np.diff(s) > 0)
    assert all(bin(int(x)).count("1") == 2 for x in s)
    ia, ib = np.array([0, 3, 9]), np.array([9, 0, 4])
    idx = sqd_list.full_indices(s, s, s[ia], s[ib])
    assert np.array_equal(idx, ia * 10 + ib)
    assert np.array_equal(sqd_list.spin_flip(idx, 10), ib * 10 + ia)
    with pytest.raises(ValueError):
        sqd_list.full_indices(s, s, [0b111], [0b11])


def test_occupancies_of_single_determinant():
    s = sqd_list.ci_strings(4, 2)
    idx = sqd_list.full_indices(s, s, [0b0101], [0b0011])
    oa, ob = sqd_list.list_occupancies(idx, [1.0], s, s, 4)
    assert np.allclose(oa, [1, 0, 1, 0]) and np.allclose(ob, [1, 1, 0, 0])


# ── physics, stand-ins ────────────────────────────────────────────────────

def test_list_full_sampling_equals_fci():
    """Every determinant with weight is sampled: list SQD must be exact,
    with ecore added once."""
    step2 = _toy_step2()
    r = _run_list(step2, shots=200000)
    assert abs(r["energy"] - _exact(step2)) < 1e-8, (r["energy"], _exact(step2))
    assert r["lucj_opt"]["sqd_subspace"] == "list"
    assert r["stats"]["n_diagonalisations"] == 2          # iterations, no batches
    assert r["iterations"][-1]["subspace_dim"] <= 36


def test_list_open_shell_exact():
    """Open shell (no spin completion). Here one of the 24 determinants
    has too little weight to be drawn even in 200k shots, so the list is
    23 determinants: variational and within a few micro-Hartree."""
    step2 = _toy_step2(n=4, na=2, nb=1, seed=9)
    r = _run_list(step2, shots=200000)
    err = r["energy"] - _exact(step2)
    assert -1e-10 <= err < 1e-5, err
    if r["iterations"][-1]["subspace_dim"] == 24:
        assert err < 1e-8


def test_list_few_samples_variational_and_smaller_than_product():
    from quenais.quantum.sqd_opt import run_lucj_opt_sqd

    step2 = _toy_step2(n=6, na=3, nb=3, seed=2)
    e_fci = _exact(step2)
    r = _run_list(step2, shots=25, sqd_carryover_eps=0.0,
                  sqd_samples_per_batch=1000)
    dim = r["iterations"][-1]["subspace_dim"]
    assert r["energy"] >= e_fci - 1e-9
    assert dim < 400                                     # C(6,3)^2 = 400
    assert r["lucj_opt"]["list_n_sampled_dets"] <= dim
    # same samples in the product space: product contains the list
    q = QiskitSolverSettings(ansatz="lucj_opt", n_shots=25, seed=3,
                             sqd_batches=1, sqd_samples_per_batch=1000,
                             sqd_opt_iterations=2, sqd_carryover_eps=0.0)
    p = _patched(lambda: run_lucj_opt_sqd(step2, q, log=lambda *_: None))
    assert p["iterations"][0]["subspace_dim"] >= r["iterations"][0]["subspace_dim"]
    assert p["energy"] <= r["iterations"][0]["energy"] + 1e-9


def test_list_closed_shell_spin_completed():
    """One sampled (a, b) with a != b: the list must also hold (b, a) and
    the HF determinant, and nothing else."""
    from quenais.las import sqd_bits as bits
    from quenais.settings import LasSettings

    step2 = _toy_step2(n=4, na=2, nb=2, seed=4)
    n = 4
    h1 = np.asarray(step2["h1e"])
    a, b = 0b0101, 0b1001
    row = bits.strings_to_matrix([a], [b], n)[0]
    counts = {"".join("1" if x else "0" for x in row): 7}
    s = LasSettings(sqd_iterations=1, carryover_eps=0.0)
    r = sqd_list.run_list_sqd(np.stack([h1, h1]), step2["h2e"], n, (2, 2),
                              counts, s, np.random.default_rng(0),
                              log=lambda *_: None,
                              sigma=_toy_sigma(h1, step2["h2e"], n, (2, 2)))
    strs = sqd_list.ci_strings(n, 2)
    want = sqd_list.full_indices(strs, strs, [a, b, 0b11], [b, a, 0b11])
    assert r["subspace_dim"] == 3 and r["n_sampled_dets"] == 1
    assert r["unique_valid_bitstrings"] == 1
    # the list energy equals a direct 3x3 diagonalisation
    H = hamiltonian_matrix([h1, h1], step2["h2e"], n,
                           np.array([x | (y << n) for x, y in
                                     ((a, b), (b, a), (0b11, 0b11))]))
    assert abs(r["energy"] - np.linalg.eigvalsh(H)[0]) < 1e-10
    assert np.unique(want).size == 3


def test_settings_cli_and_driver():
    from quenais.cli import build_parser, build_qiskit_settings

    with pytest.raises(ValueError):
        QiskitSolverSettings(sqd_subspace="dense").validate()
    args = build_parser().parse_args([
        "--molecule", "LiH", "--solver", "sqd", "--ansatz", "lucj_opt",
        "--sqd-subspace", "list"])
    assert build_qiskit_settings(args).validate().sqd_subspace == "list"
    assert QiskitSolverSettings().sqd_subspace == "product"

    import run_benchmark as d
    p, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj_opt",
                               "sqd_subspace": "list"})
    assert p["sqd_subspace"] == "list"
    p, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj",
                               "sqd_subspace": "list"})
    assert "sqd_subspace" not in p
    # null = product = the identity of runs that never mentioned the flag
    a, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj_opt",
                               "sqd_subspace": None})
    b, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj_opt"})
    assert d.short_hash(a) == d.short_hash(b)


# ── physics, real PySCF integrals (H4 linear, STO-3G) ─────────────────────

def _h4_step2():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0",
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    n = c.shape[1]
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), n)
    return {"h1e": h1, "h2e": eri, "n_emb": n, "n_alpha": 2, "n_beta": 2,
            "ecore": float(mol.energy_nuc())}


def _pyscf_fci(step2):
    from pyscf import fci

    n = step2["n_emb"]
    e, _ = fci.direct_spin1.kernel(step2["h1e"], step2["h2e"], n,
                                   (step2["n_alpha"], step2["n_beta"]),
                                   ecore=step2["ecore"], conv_tol=1e-12)
    return float(e)


def _counts_for(dets_a, dets_b, n):
    from quenais.las import sqd_bits as bits

    mat = bits.strings_to_matrix(dets_a, dets_b, n)
    return {"".join("1" if x else "0" for x in row): 1 for row in mat}


def _list_pyscf(step2, dets_a, dets_b):
    """Production list SQD with PySCF's sigma, one iteration (no recovery)."""
    from quenais.settings import LasSettings

    n = step2["n_emb"]
    h1 = np.asarray(step2["h1e"])
    s = LasSettings(sqd_iterations=1, carryover_eps=0.0)
    return sqd_list.run_list_sqd(np.stack([h1, h1]), step2["h2e"], n,
                                 (step2["n_alpha"], step2["n_beta"]),
                                 _counts_for(dets_a, dets_b, n), s,
                                 np.random.default_rng(0), log=lambda *_: None)


@pytest.mark.needs_pyscf
def test_strings_match_pyscf():
    cistring = pytest.importorskip("pyscf.fci.cistring")
    for n, k in ((4, 2), (6, 3), (8, 5)):
        ref = np.asarray(cistring.make_strings(range(n), k), dtype=np.int64)
        assert np.array_equal(sqd_list.ci_strings(n, k), ref)


@pytest.mark.needs_pyscf
def test_h4_all_determinants_equals_fci():
    """All 36 determinants 'sampled': list SQD = FCI, singlet."""
    pytest.importorskip("pyscf")
    step2 = _h4_step2()
    strs = sqd_list.ci_strings(4, 2)
    a = np.repeat(strs, strs.size)
    b = np.tile(strs, strs.size)
    r = _list_pyscf(step2, a, b)
    assert r["subspace_dim"] == 36 and r["subspace_fraction"] == 1.0
    assert abs((r["energy"] + step2["ecore"]) - _pyscf_fci(step2)) < 1e-9
    assert r["spin_sq"] is not None and abs(r["spin_sq"]) < 1e-6


@pytest.mark.needs_pyscf
def test_h4_few_determinants_variational_and_exact_in_list():
    """A few determinants: E >= FCI, and E equals a brute-force
    diagonalisation of exactly that list (checks PySCF's string order and
    the index mapping, independent of phase conventions)."""
    pytest.importorskip("pyscf")
    step2 = _h4_step2()
    e_fci = _pyscf_fci(step2)
    strs = sqd_list.ci_strings(4, 2)
    rng = np.random.default_rng(7)
    for k in (2, 5, 9):
        ia = rng.integers(0, strs.size, size=k)
        ib = rng.integers(0, strs.size, size=k)
        r = _list_pyscf(step2, strs[ia], strs[ib])
        e = r["energy"] + step2["ecore"]
        assert e >= e_fci - 1e-10
        idx = sqd_list.full_indices(strs, strs, strs[ia], strs[ib])
        idx = np.union1d(idx, sqd_list.full_indices(strs, strs, [3], [3]))
        idx = np.union1d(idx, sqd_list.spin_flip(idx, strs.size))
        assert r["subspace_dim"] == idx.size < 36
        states = np.array([int(strs[i // 6]) | (int(strs[i % 6]) << 4) for i in idx])
        h1 = np.asarray(step2["h1e"])
        H = hamiltonian_matrix([h1, h1], step2["h2e"], 4, states)
        assert abs(r["energy"] - np.linalg.eigvalsh(H)[0]) < 1e-9
