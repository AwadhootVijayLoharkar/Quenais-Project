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
    "asci_fixed_size",
    "hci_fixed_size",
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

    # Fixed v0 -- ARPACK seeds randomly otherwise, which makes the whole
    # selection trajectory irreproducible: a slightly different eigenvector
    # changes the PT2 scores, which changes the next batch of determinants,
    # which compounds over iterations.
    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    v0 = np.full(n, 1.0 / np.sqrt(n))
    vals, vecs = eigsh(op, k=1, which="SA", tol=tol, v0=v0)
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

# ─────────────────────────────────────────────────────────────────────────────
# ASCI: the best determinant set of a FIXED size
# ─────────────────────────────────────────────────────────────────────────────

def _asci_core(sel0, target_size, space, sigma, hdiag, max_iter=40,
               denom_floor=1e-3, verbose=False):
    """
    Adaptive-sampling CI at fixed size (Tubman et al., J. Chem. Phys. 145,
    044112 (2016)), with exact candidate search through the full-space
    sigma vector.

    Every iteration ranks ALL determinants by
        in the set     |c_i|                       (current coefficient)
        outside        |<a|H|psi>| / |E0 - H_aa|   (first-order amplitude)
    and keeps the top `target_size`. Unlike CIPSI's growth loop, a determinant
    picked early can be dropped again. Stops when the set no longer changes;
    returns the lowest-energy set seen (energies are variational, so this is
    always a valid upper bound).

    Returns (selected indices, electronic energy, n_iterations).
    """
    n = int(target_size)
    sel = np.unique(np.asarray(sel0, dtype=np.int64))
    best_e, best_sel = np.inf, sel
    full = np.zeros(space.ndet)
    seen = set()
    it = 0
    for it in range(1, max_iter + 1):
        e, c = _diagonalise_subspace(sel, space, sigma)
        if e < best_e - 1e-12:
            best_e, best_sel = e, sel
        key = sel.tobytes()
        if key in seen:                     # converged or cycling
            break
        seen.add(key)
        full.fill(0.0)
        full[sel] = c
        hpsi = sigma(full.reshape(space.na, space.nb)).reshape(-1)
        denom = np.abs(e - hdiag)
        denom = np.where(denom < denom_floor, denom_floor, denom)
        score = np.abs(hpsi) / denom
        score[sel] = np.abs(c)              # members keep their coefficient
        k = min(n, int(np.count_nonzero(score > 0)))
        if k <= 0:
            break
        new = np.sort(np.argpartition(score, -k)[-k:]).astype(np.int64)
        if verbose:
            print(f"    ASCI it {it:2d}  N={new.size}  E={e:.10f}  "
                  f"changed {np.setdiff1d(new, sel).size}")
        sel = new
    return best_sel, float(best_e), it


def asci_fixed_size(mol, target_size, space=None, seed=None, verbose=False,
                    **kw):
    """
    Classical fixed-size selected CI (ASCI), no quantum input.

    Starts from the CIPSI set of the same size (or from `seed`), then
    iterates the ASCI ranking until the set stops changing.
    Returns (selected indices, total energy incl. e_core).
    """
    from quenais.quantum.det_analysis import DeterminantSpace, _sigma_operator

    space = space or DeterminantSpace(mol.norb, mol.nelec)
    sigma = _sigma_operator(mol, space)
    hdiag = _hdiag(mol, space)
    if seed is None:
        seed, _ = _grow(mol, [int(np.argmin(hdiag))], target_size, space,
                        sigma, hdiag, verbose=False)
    sel, e, _ = _asci_core(seed, target_size, space, sigma, hdiag,
                           verbose=verbose, **kw)
    return sel, e + mol.cas_hamiltonian.e_core


# ─────────────────────────────────────────────────────────────────────────────
# Heat-bath CI (HCI) at a fixed size -- the standard QSCI baseline
# ─────────────────────────────────────────────────────────────────────────────

class _Columns:
    """H e_i for single determinants i (one full-space sigma each), cached."""

    def __init__(self, space, sigma, cache_limit=4000):
        self.space, self.sigma = space, sigma
        self.cache, self.limit = {}, cache_limit
        self._buf = np.zeros(space.ndet)

    def __call__(self, i):
        i = int(i)
        col = self.cache.get(i)
        if col is None:
            self._buf.fill(0.0)
            self._buf[i] = 1.0
            col = self.sigma(self._buf.reshape(self.space.na, self.space.nb)
                             ).reshape(-1).copy()
            if len(self.cache) < self.limit:
                self.cache[i] = col
        return col


def _hb_scores(sel, c, cols, ndet):
    """Heat-bath score of every determinant: max_i |H_ai c_i| over i in sel."""
    score = np.zeros(ndet)
    for i, ci in zip(sel, c):
        np.maximum(score, np.abs(cols(i) * ci), out=score)
    score[sel] = -np.inf
    return score


def _hci_threshold(seed, eps, space, sigma, cols, max_iter=60, max_dim=None):
    """Variational HCI stage at one threshold: add every a with
    max_i |H_ai c_i| > eps, re-diagonalise, until nothing new is added."""
    sel = np.unique(np.asarray(seed, dtype=np.int64))
    for _ in range(max_iter):
        e, c = _diagonalise_subspace(sel, space, sigma)
        score = _hb_scores(sel, c, cols, space.ndet)
        new = np.flatnonzero(score > eps)
        if new.size == 0:
            return sel, e, c, score
        sel = np.union1d(sel, new)
        if max_dim is not None and sel.size > max_dim:
            break
    e, c = _diagonalise_subspace(sel, space, sigma)
    return sel, e, c, _hb_scores(sel, c, cols, space.ndet)


def _hci_core(target_size, space, sigma, hdiag, n_bisect=30, verbose=False):
    """
    HCI (Holmes, Tubman, Umrigar, JCTC 12, 3674 (2016)) brought to exactly
    `target_size` determinants for an equal-budget comparison:

      1. bisect the threshold eps (log scale) for the largest converged HCI
         set with at most N determinants,
      2. top it up to exactly N with the highest remaining heat-bath scores,
         and re-diagonalise.

    Returns (selected indices, electronic energy, eps used).
    """
    n = int(target_size)
    seed = [int(np.argmin(hdiag))]
    cols = _Columns(space, sigma)
    lo, hi = -8.0, 1.0                     # log10 eps
    best = None
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        sel, e, c, score = _hci_threshold(seed, 10.0 ** mid, space, sigma, cols,
                                          max_dim=4 * n)
        if verbose:
            print(f"    HCI eps=1e{mid:+.2f}  N={sel.size}  E={e:.10f}")
        if sel.size <= n:
            best = (sel, e, c, score, 10.0 ** mid)
            hi = mid                       # allow more determinants
        else:
            lo = mid
        if best is not None and best[0].size == n:
            break
    if best is None:                       # even the largest eps overshoots
        sel, e, c, score = _hci_threshold(seed, 10.0 ** hi, space, sigma, cols)
        best = (sel, e, c, score, 10.0 ** hi)
    sel, e, c, score, eps = best
    while sel.size < n:                    # top up by heat-bath score
        k = min(n - sel.size, int(np.count_nonzero(score > 0)))
        if k <= 0:
            break
        sel = np.union1d(sel, np.argpartition(score, -k)[-k:])
        e, c = _diagonalise_subspace(sel, space, sigma)
        score = _hb_scores(sel, c, cols, space.ndet)
    return sel.astype(np.int64), float(e), float(eps)


def hci_fixed_size(mol, target_size, space=None, verbose=False, **kw):
    """
    Heat-bath CI with exactly `target_size` determinants, no quantum input.
    Returns (selected indices, total energy incl. e_core, eps used).
    """
    from quenais.quantum.det_analysis import DeterminantSpace, _sigma_operator

    space = space or DeterminantSpace(mol.norb, mol.nelec)
    sigma = _sigma_operator(mol, space)
    hdiag = _hdiag(mol, space)
    sel, e, eps = _hci_core(target_size, space, sigma, hdiag, verbose=verbose, **kw)
    return sel, e + mol.cas_hamiltonian.e_core, eps
