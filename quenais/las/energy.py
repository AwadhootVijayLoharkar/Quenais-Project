"""
LAS energy, fragment Hamiltonians and orbital gradient. Pure NumPy.

CONVENTIONS
-----------
Orbitals  C has shape (nao, nmo), columns ordered

              [ core | frag 0 | frag 1 | ... | virtual ]

          with C^T S C = 1. The core is doubly occupied.

RDMs      per fragment K (norb_K = n):
            dm1s[K]  shape (2, n, n)     <a+_p,s a_q,s>, s = alpha, beta
            dm2[K]   shape (n, n, n, n)  spin-summed, PySCF order:
                     G_pqrs = sum_st <a+_p,s a+_r,t a_s,t a_q,s>
          so the fragment two-electron energy is 1/2 sum G_pqrs (pq|rs).

Integrals are chemist's notation (pq|rs). All quantities are real.

THE ENERGY
----------
A LAS wave function is an antisymmetrised product of fragment states with
fixed particle numbers. Its 1-RDM is block diagonal, and every
two-electron term that couples two different blocks (core-fragment,
fragment-fragment) factorises exactly into products of 1-RDMs -- that is
what "mean field between fragments" means here, and it is exact for this
ansatz. So

    E = E_mf[D] + sum_K ( E2_K[G_K] - E2mf_K[D_K] )

where E_mf is the Hartree-Fock-like energy functional evaluated with the
full block-diagonal spin density D, E2_K is the exact intra-fragment
two-electron energy from G_K, and E2mf_K is the mean-field approximation
to it that E_mf contains and must be swapped out.

FRAGMENT HAMILTONIAN
--------------------
E is linear in fragment K's RDMs apart from its own two-electron term:

    E = const_K + sum_s tr(h_K^s D_K^s) + E2_K

    h_K^s = C_K^T ( h + J[D_env] - K[D_env^s] ) C_K,   D_env = D - D_K

so minimising <H_K> with one-electron part h_K^s and ERIs (pq|rs)_K is the
exact fragment problem. h_K^s is spin dependent when the environment is
open shell.

ORBITAL GRADIENT
----------------
C -> C exp(k), k antisymmetric. dE = sum_pq k_pq X_pq, and the gradient
with respect to the independent parameters k_pq (p > q) is

    g_pq = X_pq - X_qp.

X has a mean-field part (A = sum_s D^s f^s, X_mf = A^T - A with
f^s = h + J - K^s in the MO basis) and a fragment part
X[p, u in K] += 2 sum_vwx W_uvwx (pv|wx), W = G_K - Gmf_K.
Verified against finite differences in tests/test_las_core.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LasSpace",
    "FragmentHamiltonian",
    "fragment_slices",
    "block_density",
    "mean_field_2rdm",
    "fragment_hamiltonians",
    "las_energy",
    "las_energy_and_gradient",
    "nonredundant_mask",
    "diagonal_hessian",
]


@dataclass(frozen=True)
class LasSpace:
    """Orbital partition. nmo is the total number of MOs in C."""

    ncore: int
    ncas_sub: tuple
    nmo: int

    def __post_init__(self):
        object.__setattr__(self, "ncas_sub", tuple(int(n) for n in self.ncas_sub))
        if self.ncore < 0 or any(n <= 0 for n in self.ncas_sub):
            raise ValueError(f"invalid LAS partition {self}")
        if self.ncore + sum(self.ncas_sub) > self.nmo:
            raise ValueError(
                f"ncore + sum(ncas_sub) = {self.ncore + sum(self.ncas_sub)} "
                f"exceeds nmo = {self.nmo}"
            )

    @property
    def nfrag(self):
        return len(self.ncas_sub)

    @property
    def ncas(self):
        return sum(self.ncas_sub)

    @property
    def nocc(self):
        return self.ncore + self.ncas


@dataclass
class FragmentHamiltonian:
    """Everything a fragment solver needs, in the fragment's LAS orbitals."""

    index: int
    h1s: np.ndarray     # (2, n, n) spin-resolved effective one-electron part
    eri: np.ndarray     # (n, n, n, n) chemist's (pq|rs)
    const: float        # E_total = const + sum_s tr(h1s[s] D^s) + E2
    nelec: tuple        # (n_alpha, n_beta)
    spin2s: int         # 2S of the target state

    @property
    def norb(self):
        return self.h1s.shape[-1]


def fragment_slices(space):
    out, start = [], space.ncore
    for n in space.ncas_sub:
        out.append(slice(start, start + n))
        start += n
    return out


def block_density(space, dm1s):
    """Full (2, nmo, nmo) MO-basis spin density: core + fragment blocks."""
    d = np.zeros((2, space.nmo, space.nmo))
    idx = np.arange(space.ncore)
    d[:, idx, idx] = 1.0
    for sl, dm in zip(fragment_slices(space), dm1s):
        d[:, sl, sl] = dm
    return d


def mean_field_2rdm(dm1s_k):
    """Spin-summed 2-RDM of a single determinant-like density (Coulomb - exchange)."""
    da, db = dm1s_k
    d = da + db
    g = np.einsum("pq,rs->pqrs", d, d)
    g -= np.einsum("ps,rq->pqrs", da, da)
    g -= np.einsum("ps,rq->pqrs", db, db)
    return g


def _jk_mo(eri, dm):
    """J and K of a density confined to the orbitals eri is written in."""
    j = np.einsum("pqrs,rs->pq", eri, dm)
    k = np.einsum("prsq,rs->pq", eri, dm)
    return j, k


def _sym_w(w):
    """Symmetrise a 2-RDM-like tensor under the real-orbital symmetries."""
    return 0.25 * (w + w.transpose(2, 3, 0, 1) + w.transpose(1, 0, 3, 2)
                   + w.transpose(3, 2, 1, 0))


class _Common:
    """Shared intermediates for one set of orbitals and 1-RDMs."""

    def __init__(self, ints, C, space, dm1s):
        if C.shape[1] != space.nmo:
            raise ValueError(f"C has {C.shape[1]} columns, space.nmo = {space.nmo}")
        self.ints, self.C, self.space = ints, C, space
        self.slices = fragment_slices(space)
        self.dmo = block_density(space, dm1s)                      # (2,nmo,nmo)
        dao = np.einsum("pi,sij,qj->spq", C, self.dmo, C)          # (2,nao,nao)
        vj, vk = ints.get_jk(dao)
        vj = vj[0] + vj[1]
        self.h_mo = C.T @ ints.hcore @ C
        self.j_mo = C.T @ vj @ C
        self.k_mo = np.einsum("pi,spq,qj->sij", C, vk, C)
        self.eri_k = [ints.ao2mo(C[:, sl], C[:, sl], C[:, sl], C[:, sl])
                      for sl in self.slices]
        self.dm1s = [np.asarray(d) for d in dm1s]

    def e_mf(self):
        e = self.ints.energy_nuc
        for s in range(2):
            f2 = self.j_mo - self.k_mo[s]
            e += np.sum(self.h_mo * self.dmo[s]) + 0.5 * np.sum(f2 * self.dmo[s])
        return float(e)

    def frag_mf_e2(self, k):
        eri, (da, db) = self.eri_k[k], self.dm1s[k]
        ja, ka = _jk_mo(eri, da)
        jb, kb = _jk_mo(eri, db)
        j = ja + jb
        return 0.5 * (np.sum((j - ka) * da) + np.sum((j - kb) * db))


def fragment_hamiltonians(ints, C, space, dm1s, nelec_sub, spin2s_sub, dm2=None):
    """
    Fragment Hamiltonians in the current orbitals, with every other fragment
    and the core entering through its current 1-RDM.

    ``const`` makes const + <H_K> equal the total LAS energy. It includes the
    other fragments' correlation corrections only when their current 2-RDMs
    are passed as ``dm2``; the solvers never need it, it is for reporting.
    """
    com = _Common(ints, C, space, dm1s)
    e_mf = com.e_mf()
    corr = [0.0] * space.nfrag
    if dm2 is not None:
        corr = [0.5 * np.einsum("pqrs,pqrs->", dm2[k], com.eri_k[k])
                - com.frag_mf_e2(k) for k in range(space.nfrag)]
    out = []
    for k, sl in enumerate(com.slices):
        eri = com.eri_k[k]
        da, db = com.dm1s[k]
        ja, ka = _jk_mo(eri, da)
        jb, kb = _jk_mo(eri, db)
        j_k = ja + jb
        h1s = np.empty((2,) + da.shape)
        for s, k_own in ((0, ka), (1, kb)):
            j_env = com.j_mo[sl, sl] - j_k
            k_env = com.k_mo[s][sl, sl] - k_own
            h1s[s] = com.h_mo[sl, sl] + j_env - k_env
        lin = np.sum(h1s[0] * da) + np.sum(h1s[1] * db)
        const = (e_mf - lin - com.frag_mf_e2(k)
                 + sum(c for j, c in enumerate(corr) if j != k))
        out.append(FragmentHamiltonian(
            index=k, h1s=h1s, eri=eri, const=float(const),
            nelec=tuple(int(x) for x in nelec_sub[k]),
            spin2s=int(spin2s_sub[k]),
        ))
    return out


def las_energy(ints, C, space, dm1s, dm2):
    com = _Common(ints, C, space, dm1s)
    return _energy(com, dm2)


def _energy(com, dm2):
    e = com.e_mf()
    for k in range(com.space.nfrag):
        e += 0.5 * np.einsum("pqrs,pqrs->", dm2[k], com.eri_k[k])
        e -= com.frag_mf_e2(k)
    return float(e)


def las_energy_and_gradient(ints, C, space, dm1s, dm2):
    """(E, g) with g the full antisymmetric (nmo, nmo) gradient matrix."""
    com = _Common(ints, C, space, dm1s)
    e = _energy(com, dm2)

    a = np.zeros((space.nmo, space.nmo))
    for s in range(2):
        f = com.h_mo + com.j_mo - com.k_mo[s]
        a += com.dmo[s] @ f
    x = a.T - a

    for k, sl in enumerate(com.slices):
        ck = com.C[:, sl]
        w = _sym_w(np.asarray(dm2[k]) - mean_field_2rdm(com.dm1s[k]))
        eri_pk = com.ints.ao2mo(com.C, ck, ck, ck)          # (nmo, n, n, n)
        x[:, sl] += 2.0 * np.einsum("uvwx,pvwx->pu", w, eri_pk)

    return e, x - x.T


def nonredundant_mask(space):
    """
    Boolean (nmo, nmo) mask, True for rotations the optimiser should use
    (lower triangle only). Excluded: core-core, virtual-virtual and
    rotations within one fragment (redundant for an exact fragment solver;
    the LASSQD paper also freezes them for approximate solvers).
    """
    label = np.empty(space.nmo, dtype=int)
    label[: space.ncore] = -1
    for k, sl in enumerate(fragment_slices(space)):
        label[sl] = k
    label[space.nocc:] = -2
    m = label[:, None] != label[None, :]
    return np.tril(m, -1)


def diagonal_hessian(ints, C, space, dm1s, floor=0.05):
    """
    Cheap positive diagonal Hessian estimate 2|(n_p - n_q)(e_p - e_q)|,
    floored, used only as the optimiser's preconditioner. For closed-shell
    occupied-virtual pairs it reduces to the familiar 4(e_a - e_i).
    """
    com = _Common(ints, C, space, dm1s)
    n = com.dmo[0].diagonal() + com.dmo[1].diagonal()
    f = com.h_mo + com.j_mo - 0.5 * (com.k_mo[0] + com.k_mo[1])
    e = f.diagonal()
    h = 2.0 * np.abs(np.subtract.outer(n, n) * np.subtract.outer(e, e))
    return np.maximum(h, floor)
