"""
Determinant-space measurement utilities (Stage 0, Experiments A and B).

Goes in: quenais/quantum/det_analysis.py

No torch. No model. Numpy/scipy/pyscf only, so this stays importable on the
base install. Every heavy import is inside a function, matching the convention
in gqe_adapter.py.

WHY THE ENERGY IS COMPUTED THIS WAY
-----------------------------------
The oracle bound needs the variational energy in the span of an ARBITRARY list
of determinants -- not a product space, not a lowest-diagonal subspace. The
obvious implementation is to build that subspace Hamiltonian from Slater-Condon
rules. This module does not do that.

Instead it projects PySCF's own sigma builder:

    v (length N)  ->  scatter into a zero (n_a, n_b) CI array at the selected
                      positions  ->  contract_2e  ->  gather back to length N

so the operator being diagonalised is P H P with exactly the same integrals,
the same phase conventions and the same contraction that produced the validated
DMET_CASCI reference. A second, independent Hamiltonian implementation would
re-open the Jordan-Wigner phase question that was already tested and closed,
and would do it in code with no reference to check against. Cost is one
full-space sigma build per matvec, which is milliseconds at ScH's 108,900
determinants.

DETERMINANT ORDERING
--------------------
    full index = addr_alpha * n_beta_strings + addr_beta     (C order)

which is the flattening of PySCF's (n_a, n_b) CI array AND is bit-identical to
gqe_qsci.qsci.determinant.Determinant.from_fullci_index. The two codebases
agree, so determinant indices are portable between them.

BITSTRING CONVENTIONS -- THERE ARE TWO IN THIS PROJECT
------------------------------------------------------
    block       [a0 a1 ... a(M-1) | b0 b1 ... b(M-1)]   quenais/quantum/solver.py
    interleaved [a0 b0 a1 b1 ...], little-endian        gqe_qsci Determinant

This module works in `block` and exposes to_interleaved() for the crossing.
Getting this wrong produces valid-looking determinants belonging to a different
state, which is exactly the failure mode this project keeps paying for.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    "DeterminantSpace",
    "weight_curve",
    "projected_energy",
    "oracle_curve",
    "n_for_accuracy",
    "run_gates",
]

CHEMICAL_ACCURACY_HA = 1.6e-3


# ─────────────────────────────────────────────────────────────────────────────
# Determinant bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

class DeterminantSpace:
    """Index <-> determinant map for a fixed (norb, n_alpha, n_beta)."""

    def __init__(self, norb, nelec):
        from pyscf.fci import cistring

        self.norb = int(norb)
        self.nelec = (int(nelec[0]), int(nelec[1]))
        self.astrs = np.asarray(cistring.make_strings(range(self.norb), self.nelec[0]))
        self.bstrs = np.asarray(cistring.make_strings(range(self.norb), self.nelec[1]))
        self.na = len(self.astrs)
        self.nb = len(self.bstrs)
        self.ndet = self.na * self.nb

    def __repr__(self):
        return (f"DeterminantSpace(norb={self.norb}, nelec={self.nelec}, "
                f"na={self.na}, nb={self.nb}, ndet={self.ndet})")

    def split(self, idx):
        """full index -> (addr_alpha, addr_beta)"""
        idx = np.asarray(idx)
        return idx // self.nb, idx % self.nb

    def strings(self, idx):
        """full index -> (alpha_occupation_int, beta_occupation_int)"""
        ia, ib = self.split(idx)
        return self.astrs[ia], self.bstrs[ib]

    def to_block_bitstring(self, idx):
        """
        full index -> bool array of length 2*norb, [alpha block | beta block].
        Matches quenais/quantum/solver.py hf_bitstring().
        """
        a, b = self.strings(idx)
        a = np.atleast_1d(a)
        b = np.atleast_1d(b)
        shifts = np.arange(self.norb)
        bits_a = ((a[:, None] >> shifts) & 1).astype(bool)
        bits_b = ((b[:, None] >> shifts) & 1).astype(bool)
        return np.hstack([bits_a, bits_b])

    def to_interleaved_bitstring(self, idx):
        """
        full index -> str in gqe_qsci convention: [a0 b0 a1 b1 ...], little-endian
        (leftmost character is orbital 0).
        """
        blocks = np.atleast_2d(self.to_block_bitstring(idx))
        out = []
        for row in blocks:
            a, b = row[:self.norb], row[self.norb:]
            out.append("".join(f"{int(x)}{int(y)}" for x, y in zip(a, b)))
        return out

    def hf_index(self):
        """Index of the HF determinant (lowest orbitals filled) -- always 0."""
        return 0


def casci_vector(mol):
    """
    Exact CI vector for the embedding, flattened to full-index order.

    Returns (energy, flat_civec, DeterminantSpace).
    """
    e = mol.compute_casci()
    civec = np.asarray(mol._last_casci_civec)
    space = DeterminantSpace(mol.norb, mol.nelec)
    if civec.shape != (space.na, space.nb):
        raise ValueError(
            f"CI vector shape {civec.shape} does not match the determinant "
            f"space {(space.na, space.nb)}. The molecule and the space "
            f"disagree -- check n_alpha/n_beta came from the pickle, not from "
            f"the active-space count."
        )
    flat = civec.reshape(-1)
    norm = float(flat @ flat)
    if abs(norm - 1.0) > 1e-8:
        raise ValueError(f"CI vector is not normalised: <c|c> = {norm!r}")
    return float(e), flat, space


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A
# ─────────────────────────────────────────────────────────────────────────────

def weight_curve(flat_civec):
    """
    Returns (order, cumulative_weight).

    order[k] is the full index of the k-th most important determinant;
    cumulative_weight[k] is the weight captured by the top k+1 of them.
    """
    w = np.asarray(flat_civec) ** 2
    order = np.argsort(w)[::-1]
    return order, np.cumsum(w[order])


def n_for_weight(cumulative, targets=(0.9, 0.99, 0.999, 0.9999, 0.99999)):
    """Smallest N capturing each target weight."""
    return {t: int(np.searchsorted(cumulative, t) + 1) for t in targets}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B
# ─────────────────────────────────────────────────────────────────────────────

def _sigma_operator(mol, space):
    """Build the closure that applies the full-space Hamiltonian to a CI array."""
    from pyscf.fci import direct_spin1

    h = mol.cas_hamiltonian
    h2eff = direct_spin1.absorb_h1e(h.h1, h.h2, space.norb, space.nelec, 0.5)

    def sigma(civec_2d):
        return direct_spin1.contract_2e(h2eff, civec_2d, space.norb, space.nelec)

    return sigma


def projected_energy(mol, det_indices, space=None, sigma=None, tol=1e-10):
    """
    Lowest eigenvalue of P H P, where P projects onto span(det_indices).

    This is the exact variational energy obtainable from that determinant set.
    Includes e_core.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    space = space or DeterminantSpace(mol.norb, mol.nelec)
    sigma = sigma or _sigma_operator(mol, space)
    sel = np.asarray(det_indices, dtype=np.int64)
    n = sel.size
    if n == 0:
        raise ValueError("empty determinant list")

    buf = np.zeros((space.na, space.nb))

    def matvec(v):
        buf.fill(0.0)
        buf.reshape(-1)[sel] = np.asarray(v).reshape(-1)
        return sigma(buf).reshape(-1)[sel]

    e_core = mol.cas_hamiltonian.e_core

    # eigsh needs k < n. Below the crossover a dense build is both cheaper and
    # more robust than an iterative solve on a tiny space.
    if n <= 200:
        H = np.empty((n, n))
        unit = np.zeros(n)
        for i in range(n):
            unit[:] = 0.0
            unit[i] = 1.0
            H[:, i] = matvec(unit)
        H = 0.5 * (H + H.T)
        return float(np.linalg.eigvalsh(H)[0]) + e_core

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    vals = eigsh(op, k=1, which="SA", tol=tol, return_eigenvectors=False)
    return float(vals[0]) + e_core


def truncated_energy(mol, det_indices, flat_civec, space=None, sigma=None):
    """
    Energy of the exact CI vector truncated to det_indices and renormalised,
    with NO re-diagonalisation.

    The gap between this and projected_energy() is the relaxation energy: what a
    sampler that finds the right support but the wrong amplitudes still recovers
    by re-diagonalising. It answers whether the model needs to learn amplitudes
    or only support.
    """
    space = space or DeterminantSpace(mol.norb, mol.nelec)
    sigma = sigma or _sigma_operator(mol, space)
    sel = np.asarray(det_indices, dtype=np.int64)

    v = np.zeros(space.ndet)
    v[sel] = np.asarray(flat_civec)[sel]
    nrm = np.linalg.norm(v)
    if nrm == 0.0:
        raise ValueError("truncation removed all amplitude")
    v /= nrm
    hv = sigma(v.reshape(space.na, space.nb)).reshape(-1)
    return float(v @ hv) + mol.cas_hamiltonian.e_core


def oracle_curve(mol, n_grid=None, extra_points=(), verbose=True):
    """
    Experiment B.

    For each N: take the top-N determinants by exact |c_I|^2, and record
      - the projected (re-diagonalised) energy  -- the oracle
      - the truncate-and-renormalise energy     -- no relaxation

    NOTE ON WHAT THIS BOUNDS. Top-N by amplitude is not the energy-optimal
    N-determinant set, so E(top-N) >= E(optimal-N). The bound is therefore
    slightly pessimistic, which is the safe direction for a go/no-go: the true
    headroom is at least what is measured here.

    Returns a list of dicts.
    """
    e_exact, flat, space = casci_vector(mol)
    order, cum = weight_curve(flat)
    sigma = _sigma_operator(mol, space)

    if n_grid is None:
        n_grid = sorted(set(
            [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
            + [int(round(10 ** e)) for e in np.arange(3.0, np.log10(space.ndet), 0.15)]
            + [space.ndet]
        ))
    n_grid = sorted({n for n in list(n_grid) + list(extra_points)
                     if 1 <= n <= space.ndet})

    rows = []
    for n in n_grid:
        sel = order[:n]
        t0 = time.time()
        e_proj = projected_energy(mol, sel, space=space, sigma=sigma)
        e_trunc = truncated_energy(mol, sel, flat, space=space, sigma=sigma)
        dt = time.time() - t0
        row = {
            "n_det": int(n),
            "fraction": n / space.ndet,
            "weight_captured": float(cum[n - 1]),
            "e_projected": e_proj,
            "e_truncated": e_trunc,
            "err_projected_mha": 1e3 * (e_proj - e_exact),
            "err_truncated_mha": 1e3 * (e_trunc - e_exact),
            "relaxation_mha": 1e3 * (e_trunc - e_proj),
            "seconds": dt,
        }
        rows.append(row)
        if verbose:
            print(f"  N={n:>8d}  ({100*row['fraction']:6.2f}%)  "
                  f"weight={row['weight_captured']:.8f}  "
                  f"err={row['err_projected_mha']:9.4f} mHa  "
                  f"relax={row['relaxation_mha']:8.4f} mHa  "
                  f"[{dt:.1f}s]")

        # Every point is variational: E >= E_exact. A negative error means the
        # projection or the reference is wrong, and everything after it is noise.
        if row["err_projected_mha"] < -1e-6:
            raise RuntimeError(
                f"Projected energy fell BELOW the exact reference at N={n} "
                f"({row['err_projected_mha']:.6f} mHa). This is not possible "
                f"variationally -- the determinant indexing or the reference "
                f"is wrong. Stop and fix before reading any other number."
            )
    return rows


def n_for_accuracy(rows, threshold=CHEMICAL_ACCURACY_HA):
    """
    Smallest N on the measured grid reaching `threshold` (in Ha).

    This is N_chem -- the quantity the scaling series tracks across system
    sizes, and the number that actually sizes the sampler.
    """
    thr_mha = 1e3 * threshold
    for r in rows:
        if r["err_projected_mha"] <= thr_mha:
            return r["n_det"]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Gates -- run these before believing anything above
# ─────────────────────────────────────────────────────────────────────────────

def run_gates(mol, reference_casci, tol=1e-9, verbose=True):
    """
    1. Full determinant space must reproduce the DETERMINISTIC reference exactly.
    2. The HF determinant alone must reproduce the embedding SCF energy.
    3. The CI vector's largest amplitude must sit at the HF determinant.

    Gate 1 is the one that matters: if P H P over the whole space does not give
    back the validated number, the indexing is wrong and every curve is fiction.
    """
    e_exact, flat, space = casci_vector(mol)
    sigma = _sigma_operator(mol, space)
    results = {}

    e_full = projected_energy(mol, np.arange(space.ndet), space=space, sigma=sigma)
    results["full_space"] = (e_full, reference_casci, abs(e_full - reference_casci))

    e_hf = projected_energy(mol, [space.hf_index()], space=space, sigma=sigma)
    results["hf_determinant"] = (e_hf, float(mol.hf.e_tot), abs(e_hf - mol.hf.e_tot))

    order, _ = weight_curve(flat)
    results["hf_is_dominant"] = (int(order[0]), space.hf_index(),
                                 0.0 if order[0] == space.hf_index() else 1.0)

    if verbose:
        print(f"  {space}")
        print(f"  gate 1  full space      : {e_full:.12f}  vs ref "
              f"{reference_casci:.12f}   diff {results['full_space'][2]:.3e}  "
              f"{'PASS' if results['full_space'][2] < tol else 'FAIL'}")
        print(f"  gate 2  HF determinant  : {e_hf:.12f}  vs mf.e_tot "
              f"{mol.hf.e_tot:.12f}   diff {results['hf_determinant'][2]:.3e}")
        print(f"  gate 3  HF is dominant  : top index {order[0]} "
              f"{'PASS' if order[0] == space.hf_index() else 'FAIL'}")

    if results["full_space"][2] >= tol:
        raise RuntimeError(
            f"GATE 1 FAILED. Full-space projected energy {e_full:.12f} does "
            f"not match the DETERMINISTIC reference {reference_casci:.12f} "
            f"(diff {results['full_space'][2]:.3e} > {tol:.0e}). The "
            f"determinant indexing is wrong. Do not read any other output."
        )
    return results