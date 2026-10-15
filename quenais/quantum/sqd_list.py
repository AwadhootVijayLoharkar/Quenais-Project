"""
Determinant-list SQD (--sqd-subspace list, step 4).

WHY
---
Standard SQD diagonalises in the PRODUCT space {sampled alpha strings} x
{sampled beta strings}. A few hundred distinct strings already give a
product that covers the whole space of a small system, so SQD becomes
exact diagonalisation and its error says nothing about how well the
circuit SELECTS determinants ("filling, not selection", results_log).

The list mode keeps only the determinants that were actually sampled --
the (alpha, beta) PAIRS of the recovered bitstrings -- and diagonalises
H projected onto exactly that list. Its subspace size is therefore
directly comparable with CIPSI / the oracle at the same number of
determinants (the err_cipsi_kcal / err_oracle_kcal columns).

WHAT IS IN THE LIST (iteration i)
---------------------------------
  * every distinct valid bitstring after configuration recovery
    (iteration 0: post-selection only, as in the product mode)
  * the HF determinant of the sampling basis
  * carryover: determinants with |c| > eps from iteration i-1
  * closed shell (n_alpha = n_beta) and sqd_symmetrize_spin: the
    spin-flipped partner (b, a) of every (a, b), the list analogue of the
    product mode's alpha/beta string union. Without it the eigenvector is
    spin contaminated. n_sampled_dets records the count BEFORE these
    additions.

There are no batches: every iteration diagonalises ONE list built from
all samples (sqd_batches / sqd_samples_per_batch are ignored). The
reported energy is the last iteration's, like the product mode.

HOW IT IS DIAGONALISED
----------------------
det_analysis.projected_eigh with PySCF's full-space sigma build -- the
same operator that produces the oracle and CIPSI columns. One matvec
costs one full-space sigma (milliseconds for N2 STO-3G, seconds for the
2.9 M determinants of N2 cc-pVDZ). Always variational: E >= E_FCI.

Bitstrings are in the SAMPLING basis (the embedding mean-field orbitals
of the LASSQD fragment solver), so the integrals are prep["h1_mo"],
prep["eri_mo"], never the raw step-2 ones.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from quenais.las import sqd_bits as bits

__all__ = ["ci_strings", "full_indices", "make_sigma", "list_occupancies",
           "spin_flip", "run_list_sqd"]


def ci_strings(norb, nel):
    """All occupation ints with nel bits in norb orbitals, ascending --
    PySCF's cistring.make_strings order (checked in the tests)."""
    s = [sum(1 << p for p in c) for c in combinations(range(norb), nel)]
    return np.array(sorted(s), dtype=np.int64)


def _addresses(strs, x):
    x = np.asarray(x, dtype=np.int64)
    a = np.searchsorted(strs, x)
    if x.size and (np.any(a >= strs.size) or np.any(strs[np.minimum(a, strs.size - 1)] != x)):
        raise ValueError("string with the wrong electron number in the list")
    return a.astype(np.int64)


def full_indices(astrs, bstrs, a, b):
    """(alpha int, beta int) pairs -> full index addr_a * n_b + addr_b
    (C-order flattening of the (n_a, n_b) CI array, as in det_analysis)."""
    return _addresses(astrs, a) * bstrs.size + _addresses(bstrs, b)


def spin_flip(idx, nb):
    """(ia, ib) -> (ib, ia); only meaningful when alpha and beta string
    lists are identical (n_alpha = n_beta)."""
    idx = np.asarray(idx, dtype=np.int64)
    return (idx % nb) * nb + idx // nb


def make_sigma(h1, eri, norb, nelec):
    """Full-space H|c> with PySCF (same construction as det_analysis)."""
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)

    def sigma(c):
        return direct_spin1.contract_2e(h2eff, c, norb, nelec)

    return sigma


def list_occupancies(idx, v, astrs, bstrs, norb):
    """Orbital occupancies (alpha, beta) of sum_I v_I |I>, for recovery."""
    w = np.asarray(v) ** 2
    w = w / w.sum()
    ia, ib = np.asarray(idx) // bstrs.size, np.asarray(idx) % bstrs.size
    sh = np.arange(norb)
    occ_a = (((astrs[ia][:, None] >> sh) & 1) * w[:, None]).sum(axis=0)
    occ_b = (((bstrs[ib][:, None] >> sh) & 1) * w[:, None]).sum(axis=0)
    return occ_a.astype(float), occ_b.astype(float)


def _spin_square(idx, v, shape, norb, nelec):
    try:
        from pyscf.fci import spin_op

        c = np.zeros(shape)
        c.reshape(-1)[idx] = v
        return float(spin_op.spin_square0(c, norb, nelec)[0])
    except Exception:            # diagnostic only
        return None


def run_list_sqd(h1s, eri, norb, nelec, counts, s, rng, log=print, sigma=None):
    """
    Determinant-list SQD on one Hamiltonian in the sampling basis.

    h1s    (2, n, n) one-electron integrals; alpha and beta must agree
           (true for every DMET embedding)
    counts Qiskit counts of the circuit (bit order of sqd_bits)
    s      LasSettings (sqd_iterations, carryover_eps, sqd_symmetrize_spin,
           sqd_davidson_tol)
    rng    numpy Generator (seeds configuration recovery)
    sigma  optional full-space sigma; default PySCF (tests inject their own)

    Returns a dict: energy (WITHOUT the constant), iterations, subspace_dim,
    full_dim, subspace_fraction, shots_valid_fraction,
    unique_valid_bitstrings, n_sampled_dets, spin_sq.
    """
    from quenais.quantum.det_analysis import projected_eigh

    n = int(norb)
    na, nb = int(nelec[0]), int(nelec[1])
    h1s = np.asarray(h1s, dtype=float)
    if np.max(np.abs(h1s[0] - h1s[1])) > 1e-10:
        raise ValueError("list SQD needs h_alpha = h_beta (spin-restricted "
                         "one-electron integrals)")
    h1 = 0.5 * (h1s[0] + h1s[1])
    astrs, bstrs = ci_strings(n, na), ci_strings(n, nb)
    shape = (astrs.size, bstrs.size)
    full_dim = int(astrs.size * bstrs.size)
    if sigma is None:
        sigma = make_sigma(h1, eri, n, (na, nb))

    raw, raw_p = bits.counts_to_matrix(counts, 2 * n)
    ok = (raw[:, n:].sum(axis=1) == na) & (raw[:, :n].sum(axis=1) == nb)
    valid_fraction = float(raw_p[ok].sum())
    n_unique_valid = int(ok.sum())

    hf_a, hf_b = bits.hf_strings(na, nb)
    hf_idx = full_indices(astrs, bstrs, [hf_a], [hf_b])
    closed = na == nb and bool(getattr(s, "sqd_symmetrize_spin", True))
    tol = float(getattr(s, "sqd_davidson_tol", 1e-10))
    eps = float(s.carryover_eps)

    carry = np.zeros(0, np.int64)
    occs, iters = None, []
    e = v = sel = None
    n_sampled = 0
    for it in range(int(s.sqd_iterations)):
        if occs is None:
            mat, _p = bits.postselect(raw, raw_p, n, na, nb)
        else:
            from qiskit_addon_sqd.configuration_recovery import (
                recover_configurations,
            )
            mat, p = recover_configurations(
                raw, raw_p, occs, num_elec_a=na, num_elec_b=nb,
                rand_seed=int(rng.integers(2**31)))
            mat, _p = bits.postselect(np.asarray(mat, dtype=bool),
                                      np.asarray(p, dtype=float), n, na, nb)
        if mat.shape[0]:
            a, b = bits.matrix_to_strings(mat, n)
            sampled = np.unique(full_indices(astrs, bstrs, a, b))
        else:
            sampled = np.zeros(0, np.int64)
        n_sampled = int(sampled.size)
        sel = np.union1d(np.union1d(sampled, hf_idx), carry)
        if closed:
            sel = np.union1d(sel, spin_flip(sel, bstrs.size))
        e, v = projected_eigh(sigma, shape, sel, tol=tol, vectors=True)
        occs = list_occupancies(sel, v, astrs, bstrs, n)
        if eps > 0:
            carry = sel[np.abs(v) > eps]
        iters.append({"energy": float(e), "dim": int(sel.size),
                      "n_sampled": n_sampled, "n_carry": (int(carry.size),)})
        log(f"    list SQD it {it + 1}: {n_sampled} sampled -> {sel.size} "
            f"determinants ({100 * sel.size / full_dim:.2f}%), E = {e:.10f}")

    return {
        "energy": float(e),
        "iterations": iters,
        "subspace_dim": int(sel.size),
        "full_dim": full_dim,
        "subspace_fraction": float(sel.size / full_dim),
        "shots_valid_fraction": valid_fraction,
        "unique_valid_bitstrings": n_unique_valid,
        "n_sampled_dets": n_sampled,
        "spin_sq": _spin_square(sel, v, shape, n, (na, nb)),
    }
