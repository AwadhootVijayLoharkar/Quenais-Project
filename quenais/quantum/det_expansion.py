"""
Determinant-set expansion and selection (Stage 1).

Goes in: quenais/quantum/det_expansion.py

WHAT THIS DOES
--------------
Given a starting set of determinants, repeatedly:

    1. diagonalise H inside the current set          -> energy E0, vector c
    2. find every determinant the Hamiltonian couples to that set
    3. score each candidate by how much it would lower the energy
    4. keep the best ones, and repeat

This is CIPSI (Huron, Malrieu & Rancurel 1973) and it is also what the
QiankunNet-QSCI paper calls "H-Couple": adding determinants directly connected
to the sampled subspace through the electronic Hamiltonian.

Two entry points, and the difference between them is the whole point:

    cipsi_from_scratch()  -- starts from the single lowest-energy determinant.
                             Uses no quantum information at all.
    expand_from_seed()    -- starts from a determinant set produced by the
                             quantum sampler.

Run both to the same final size. The gap between them is the measured
contribution of the quantum step. Without that comparison there is no evidence
the quantum sampler did anything a 1973 algorithm could not.

SCOPE -- READ THIS BEFORE REUSING
---------------------------------
The candidate search here is exact and cheap because it never enumerates
excitations. It applies H to the current vector across the FULL determinant
space and reads the couplings straight off the result: <D|H|psi> is just
element D of H|psi>. Every determinant with a non-zero entry is, by definition,
connected to the current set.

That trick requires holding a full-space vector, which is fine at ScH's 108,900
determinants and impossible at the sizes this project eventually targets. This
module is therefore a MEASUREMENT tool for establishing baselines on systems
where the full space fits, not a scalable selected-CI implementation. A
production version needs explicit excitation generation with screening.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    "cipsi_from_scratch",
    "expand_from_seed",
    "lowest_diagonal_determinant",
]


def _hdiag(mol, space):
    """Diagonal of H over the full determinant space, without e_core."""
    from pyscf.fci import direct_spin1

    h = mol.cas_hamiltonian
    return direct_spin1.make_hdiag(h.h1, h.h2, space.norb, space.nelec)


def lowest_diagonal_determinant(mol, space=None):
    """
    Index of the determinant with the lowest diagonal energy.

    This is the honest classical starting point: it uses only the Hamiltonian,
    never the exact answer. In the DMET embedding basis it is NOT index 0 --
    that basis is not energy-ordered.
    """
    from quenais.quantum.det_analysis import DeterminantSpace

    space = space or DeterminantSpace(mol.norb, mol.nelec)
    return int(np.argmin(_hdiag(mol, space)))


def _diagonalise_subspace(sel, space, sigma, tol=1e-10):
    """
    Lowest eigenpair of P H P. Returns (E_electronic, coefficients).
    Electronic energy only -- e_core is added by the caller.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    sel = np.asarray(sel, dtype=np.int64)
    n = sel.size
    buf = np.zeros((space.na, space.nb))

    def matvec(v):
        buf.fill(0.0)
        buf.reshape(-1)[sel] = np.asarray(v).reshape(-1)
        return sigma(buf).reshape(-1)[sel]

    if n == 1:
        e = float(matvec(np.ones(1))[0])
        return e, np.ones(1)

    if n <= 200:
        H = np.empty((n, n))
        unit = np.zeros(n)
        for i in range(n):
            unit[:] = 0.0
            unit[i] = 1.0
            H[:, i] = matvec(unit)
        H = 0.5 * (H + H.T)
        vals, vecs = np.linalg.eigh(H)
        return float(vals[0]), vecs[:, 0]

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    vals, vecs = eigsh(op, k=1, which="SA", tol=tol)
    return float(vals[0]), vecs[:, 0]


def _grow(mol, seed, target_size, space, sigma, hdiag,
          growth=2.0, max_iter=200, verbose=True, label=""):
    """
    The selection loop shared by both entry points.

    growth: multiplier on the set size per iteration. 2.0 doubles each round,
    which reaches any target in log2 steps and keeps the number of expensive
    diagonalisations small.
    """
    e_core = mol.cas_hamiltonian.e_core
    sel = np.unique(np.asarray(seed, dtype=np.int64))
    in_set = np.zeros(space.ndet, dtype=bool)
    in_set[sel] = True

    history = []
    full = np.zeros(space.ndet)

    for it in range(max_iter):
        t0 = time.time()
        e_elec, c = _diagonalise_subspace(sel, space, sigma)
        e_tot = e_elec + e_core

        history.append({
            "iteration": it,
            "n_det": int(sel.size),
            "energy": e_tot,
            "seconds": time.time() - t0,
        })
        if verbose:
            print(f"    {label}iter {it:>3d}  N={sel.size:>7d}  "
                  f"E={e_tot:.10f}  [{time.time()-t0:.1f}s]")

        if sel.size >= target_size:
            break

        # H|psi> over the whole space. Element D is <D|H|psi>, so every
        # determinant coupled to the current set shows up here directly --
        # no excitation enumeration needed.
        full.fill(0.0)
        full[sel] = c
        hpsi = sigma(full.reshape(space.na, space.nb)).reshape(-1)

        # Epstein-Nesbet second-order estimate of what each candidate is worth:
        #     |<D|H|psi>|^2 / (H_DD - E0)
        #
        # DEGENERACY. The denominator vanishes for candidates degenerate with
        # the current energy, and perturbation theory is simply invalid there.
        # An earlier version floored |denom| at 1e-8, which is exactly wrong:
        # it turned "PT2 says nothing here" into "PT2 says this is worth 1e16",
        # so the loop spent its whole budget on degenerate junk. On stretched
        # N2 that produced a CIPSI energy 123 mHa above exact -- far worse than
        # the single-determinant starting point.
        #
        # Bounding the denominator instead keeps such candidates attractive
        # (they are strongly coupled, so they should be picked) without letting
        # the estimate diverge. DENOM_FLOOR is in Hartree and deliberately
        # coarse: it is a regulariser, not a physical parameter.
        DENOM_FLOOR = 1e-3
        denom = hdiag - e_elec
        # Candidates below the current energy get maximum priority: they lower
        # it outright. Those are also exactly the ones a small denominator
        # would have mis-scored.
        denom = np.where(denom < DENOM_FLOOR, DENOM_FLOOR, denom)
        score = (hpsi ** 2) / denom
        score[in_set] = -np.inf

        n_new = max(1, int(sel.size * (growth - 1.0)))
        n_new = min(n_new, target_size - sel.size)
        viable = int(np.count_nonzero(np.isfinite(score) & (score > 0)))
        if viable == 0:
            if verbose:
                print(f"    {label}no coupled determinants left; "
                      f"set is closed at N={sel.size}")
            break
        n_new = min(n_new, viable)

        picked = np.argpartition(score, -n_new)[-n_new:]
        in_set[picked] = True
        sel = np.flatnonzero(in_set)

    return sel, history


def cipsi_from_scratch(mol, target_size, space=None, verbose=True, **kw):
    """
    Classical selected CI with no quantum input. The control.

    Starts from the single lowest-diagonal determinant and grows to
    target_size by perturbative selection.
    """
    from quenais.quantum.det_analysis import DeterminantSpace, _sigma_operator

    space = space or DeterminantSpace(mol.norb, mol.nelec)
    sigma = _sigma_operator(mol, space)
    hdiag = _hdiag(mol, space)
    seed = [int(np.argmin(hdiag))]

    if verbose:
        print(f"  CIPSI from scratch, seed determinant {seed[0]}, "
              f"target {target_size}")
    return _grow(mol, seed, target_size, space, sigma, hdiag,
                 verbose=verbose, label="", **kw)


def expand_from_seed(mol, seed_determinants, target_size, space=None,
                     verbose=True, **kw):
    """
    Same selection, but seeded with determinants the quantum sampler found.

    seed_determinants: iterable of full determinant indices
        (addr_alpha * n_beta_strings + addr_beta), the same convention as
        det_analysis.DeterminantSpace and gqe_qsci's from_fullci_index.
    """
    from quenais.quantum.det_analysis import DeterminantSpace, _sigma_operator

    space = space or DeterminantSpace(mol.norb, mol.nelec)
    sigma = _sigma_operator(mol, space)
    hdiag = _hdiag(mol, space)
    seed = np.unique(np.asarray(list(seed_determinants), dtype=np.int64))

    if seed.min() < 0 or seed.max() >= space.ndet:
        raise ValueError(
            f"seed determinant indices out of range for {space}. Check they "
            f"are full indices (addr_a * n_b + addr_b), not alpha/beta pairs."
        )
    if verbose:
        print(f"  expansion from {seed.size} quantum-sampled determinants, "
              f"target {target_size}")
    return _grow(mol, seed, target_size, space, sigma, hdiag,
                 verbose=verbose, label="", **kw)