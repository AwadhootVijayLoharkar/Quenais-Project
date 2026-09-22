"""SQD bitstring handling and carryover bookkeeping. NumPy only."""

import numpy as np
import pytest

from quenais.las import sqd_bits as b


def test_qiskit_bit_order_to_ci_strings():
    # n = 3 orbitals: qubits 0-2 alpha, 3-5 beta; string printed high qubit first.
    # alpha occupies orbitals 0 and 2 (qubits 0, 2); beta orbital 1 (qubit 4).
    key = "010101"
    mat, p = b.counts_to_matrix({key: 7}, 6)
    a, bb = b.matrix_to_strings(mat, 3)
    assert a.tolist() == [0b101] and bb.tolist() == [0b010]
    assert p.tolist() == [1.0]
    back = b.strings_to_matrix(a, bb, 3)
    assert "".join("1" if x else "0" for x in back[0]) == key


def test_register_spaces_are_ignored():
    mat, _ = b.counts_to_matrix({"01 10": 1}, 4)
    assert mat.shape == (1, 4)


def test_postselect_keeps_only_right_electron_numbers():
    rows = b.strings_to_matrix([0b011, 0b001, 0b110], [0b001, 0b001, 0b100], 3)
    p = np.array([0.5, 0.3, 0.2])
    mat, q = b.postselect(rows, p, 3, na=2, nb=1)
    a, bb = b.matrix_to_strings(mat, 3)
    assert sorted(a.tolist()) == [0b011, 0b110]
    assert np.isclose(q.sum(), 1.0)


def test_subsample_draws_distinct_rows():
    rows = b.strings_to_matrix(np.arange(1, 9), np.ones(8, int), 4)
    p = np.full(8, 1 / 8)
    batches = b.subsample(rows, p, 5, 3, np.random.default_rng(0))
    assert len(batches) == 3
    for x in batches:
        assert x.shape == (5, 8)
        assert len({tuple(r) for r in x}) == 5
    assert b.subsample(rows, p, 50, 1, np.random.default_rng(0))[0].shape[0] == 8


def test_hf_strings():
    assert b.hf_strings(3, 1) == (0b111, 0b1)


def test_carryover_selects_large_amplitudes():
    amps = np.array([[0.9, 1e-7], [0.2, 0.0]])
    ca, cb = b.select_carryover(amps, [3, 5], [1, 2], 1e-3)
    assert ca.tolist() == [3, 5] and cb.tolist() == [1]


def test_overlap_permutation_and_remap():
    u_old = np.eye(3)
    u_new = np.eye(3)[:, [2, 0, 1]]          # new orbital 0 = old 2, etc.
    perm = b.overlap_permutation(u_old, u_new)
    assert perm.tolist() == [1, 2, 0]        # old q -> new perm[q]
    # old string with orbitals 0 and 2 occupied -> new orbitals 1 and 0
    assert b.remap_strings([0b101], perm).tolist() == [0b011]
    assert b.popcount(b.remap_strings([0b101, 0b110], perm)).tolist() == [2, 2]


def test_overlap_permutation_refuses_ambiguous_maps():
    c = np.cos(np.pi / 4)
    u_new = np.array([[c, -c], [c, c]])
    assert b.overlap_permutation(np.eye(2), u_new, min_overlap=0.8) is None
    assert b.overlap_permutation(np.eye(2), np.ones((2, 2)) / 2) is None


def test_counts_length_mismatch_raises():
    with pytest.raises(ValueError):
        b.counts_to_matrix({"101": 1}, 4)
