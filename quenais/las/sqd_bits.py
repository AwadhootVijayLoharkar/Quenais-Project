"""
Bitstring bookkeeping for SQD. Pure NumPy.

BIT CONVENTION (Qiskit + ffsim Jordan-Wigner)
---------------------------------------------
A fragment with n orbitals uses 2n qubits: qubit p = alpha orbital p,
qubit n+p = beta orbital p. A measured bitstring is printed with the
HIGHEST clbit first, so as a row of characters

    column j  <->  qubit 2n-1-j
    columns [0, n)    : beta orbitals  n-1 ... 0
    columns [n, 2n)   : alpha orbitals n-1 ... 0

A CI string (PySCF selected_ci) is an int with bit p = orbital p occupied.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "counts_to_matrix",
    "matrix_to_strings",
    "strings_to_matrix",
    "postselect",
    "subsample",
    "hf_strings",
    "popcount",
    "select_carryover",
    "overlap_permutation",
    "remap_strings",
    "one_body_string_operator",
]


def popcount(x):
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros(x.shape, dtype=np.int64)
    y = x.copy()
    while np.any(y):
        out += y & 1
        y >>= 1
    return out


def counts_to_matrix(counts, nbits):
    """{bitstring: count} -> (bool matrix (m, nbits), probabilities (m,))."""
    keys = [k.replace(" ", "") for k in counts]
    if any(len(k) != nbits for k in keys):
        raise ValueError(f"expected {nbits}-bit strings, got lengths "
                         f"{sorted({len(k) for k in keys})}")
    mat = np.array([[c == "1" for c in k] for k in keys], dtype=bool).reshape(-1, nbits)
    w = np.array([counts[k] for k in counts], dtype=float)
    return mat, w / w.sum()


def matrix_to_strings(mat, norb):
    """Rows -> (alpha ints, beta ints)."""
    mat = np.asarray(mat, dtype=np.int64)
    weights = 1 << np.arange(norb, dtype=np.int64)          # orbital p -> 2^p
    alpha = mat[:, norb:][:, ::-1] @ weights
    beta = mat[:, :norb][:, ::-1] @ weights
    return alpha, beta


def strings_to_matrix(alpha, beta, norb):
    alpha = np.asarray(alpha, dtype=np.int64)[:, None]
    beta = np.asarray(beta, dtype=np.int64)[:, None]
    bits = np.arange(norb, dtype=np.int64)[::-1]            # column order n-1..0
    return np.hstack([(beta >> bits) & 1, (alpha >> bits) & 1]).astype(bool)


def postselect(mat, probs, norb, na, nb):
    """Keep rows with the right number of alpha and beta electrons."""
    ok = (mat[:, norb:].sum(axis=1) == na) & (mat[:, :norb].sum(axis=1) == nb)
    p = probs[ok]
    return mat[ok], (p / p.sum() if p.size else p)


def subsample(mat, probs, samples_per_batch, n_batches, rng):
    """n_batches weighted draws of distinct rows, without replacement."""
    m = mat.shape[0]
    if m == 0:
        return [mat[:0] for _ in range(n_batches)]
    k = min(samples_per_batch, m)
    batches = []
    for _ in range(n_batches):
        idx = rng.choice(m, size=k, replace=False, p=probs)
        batches.append(mat[idx])
    return batches


def hf_strings(na, nb):
    """Aufbau determinant in the fragment's own canonical orbitals."""
    return (1 << na) - 1, (1 << nb) - 1


def select_carryover(amplitudes, strs_a, strs_b, eps):
    """
    Alpha and beta strings of every determinant with |c| > eps
    (Algorithm 2 of Wang et al., PNAS 2026).
    """
    amp = np.abs(np.asarray(amplitudes))
    ia, ib = np.nonzero(amp > eps)
    return (np.unique(np.asarray(strs_a)[ia]).astype(np.int64),
            np.unique(np.asarray(strs_b)[ib]).astype(np.int64))


def overlap_permutation(u_old, u_new, min_overlap=0.5):
    """
    perm[q] = p mapping old orbital q to the new orbital it overlaps most,
    or None when that is not a clean one-to-one map (then carryover strings
    cannot be transported and are dropped for this cycle).
    """
    o = np.abs(np.asarray(u_old).T @ np.asarray(u_new))
    perm = np.argmax(o, axis=1)
    if len(set(perm.tolist())) != len(perm):
        return None
    if np.min(o[np.arange(len(perm)), perm]) < min_overlap:
        return None
    return perm


def remap_strings(strs, perm):
    """Move bit q to bit perm[q] in every string."""
    strs = np.asarray(strs, dtype=np.int64)
    out = np.zeros_like(strs)
    for q, p in enumerate(perm):
        out |= ((strs >> q) & 1) << int(p)
    return out


def one_body_string_operator(strs, norb, h):
    """
    Matrix of sum_pq h_pq a+_p a_q (one spin) between the given CI strings,
    projected onto that string set. Sign = parity of occupied orbitals
    strictly between p and q, which is convention independent.
    """
    strs = np.asarray(strs, dtype=np.int64)
    index = {int(x): i for i, x in enumerate(strs)}
    m = np.zeros((strs.size, strs.size))
    for j, s in enumerate(strs.tolist()):
        occ = [q for q in range(norb) if (s >> q) & 1]
        for q in occ:
            s1 = s ^ (1 << q)
            sq = bin(s & ((1 << q) - 1)).count("1")
            for p in range(norb):
                if p != q and (s1 >> p) & 1:
                    continue
                if h[p, q] == 0.0:
                    continue
                i = index.get(s1 | (1 << p))
                if i is None:
                    continue
                sp = bin(s1 & ((1 << p) - 1)).count("1")
                m[i, j] += h[p, q] * (-1.0) ** (sq + sp)
    return m
