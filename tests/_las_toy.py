"""
Brute-force reference machinery for the LAS tests. NumPy/SciPy only.

Nothing here shares code with quenais.las: energies and RDMs are computed
by applying Jordan-Wigner creation/annihilation operators to explicit
Fock-space vectors, which is slow and obviously correct. That independence
is the point -- it is what the LAS energy, fragment Hamiltonians and
orbital gradient are validated against, without PySCF.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from quenais.las.driver import FragmentResult


# ── integrals ─────────────────────────────────────────────────────────────

class ToyIntegrals:
    """Orthonormal 'AO' basis (S = 1), random but physical-looking integrals."""

    def __init__(self, nao, seed=0, naux=None):
        rng = np.random.default_rng(seed)
        h = rng.normal(size=(nao, nao))
        self.hcore = 0.5 * (h + h.T) - np.diag(np.arange(nao) * -0.6 + 2.0)
        naux = naux or 2 * nao
        b = rng.normal(scale=0.35, size=(naux, nao, nao))
        b = 0.5 * (b + b.transpose(0, 2, 1))
        self.eri = np.einsum("Lpq,Lrs->pqrs", b, b)
        self.ovlp = np.eye(nao)
        self.energy_nuc = 1.25
        self.nao = nao

    def get_jk(self, dms):
        dms = np.asarray(dms)
        vj = np.einsum("pqrs,nrs->npq", self.eri, dms)
        vk = np.einsum("prsq,nrs->npq", self.eri, dms)
        return vj, vk

    def ao2mo(self, c1, c2, c3, c4):
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", self.eri, c1, c2, c3, c4,
                         optimize=True)


def random_orthogonal(n, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q * np.sign(np.diag(r))


# ── Fock space ────────────────────────────────────────────────────────────

def annihilators(nso):
    """Sparse JW annihilation operators; bit i of the index = spin-orbital i."""
    dim = 1 << nso
    states = np.arange(dim)
    ops = []
    for i in range(nso):
        occ = (states >> i) & 1
        src = states[occ == 1]
        dst = src ^ (1 << i)
        below = src & ((1 << i) - 1)
        sign = (-1.0) ** np.array([bin(x).count("1") for x in below])
        ops.append(sp.csr_matrix((sign, (dst, src)), shape=(dim, dim)))
    return ops


def exact_rdms(psi, n, qubit):
    """
    Spin 1-RDM (2, n, n) and spin-summed 2-RDM (n,n,n,n) of a Fock vector.
    qubit[s][p] is the JW index of orbital p with spin s.
    """
    nso = int(np.log2(psi.size))
    a = annihilators(nso)
    # v[s, p, q] = E^s_pq psi = a+_p,s a_q,s psi
    v = np.empty((2, n, n, psi.size))
    for s in range(2):
        for p in range(n):
            for q in range(n):
                v[s, p, q] = a[qubit[s][p]].T @ (a[qubit[s][q]] @ psi)
    dm1s = np.einsum("x,spqx->spq", psi, v)
    # <E_pq E_rs> = <E_qp psi | E_rs psi>
    # g[p,q,r,s] = sum_{a,b spins} <E^a_qp psi | E^b_rs psi>
    vs = v.sum(axis=0)                      # spin-summed excitation vectors
    g = np.einsum("qpx,rsx->pqrs", vs, vs)
    for s in range(2):
        g -= np.einsum("qr,ps->pqrs", np.eye(n), dm1s[s])
    return dm1s, g


def sector_states(n, na, nb):
    """Fragment-local ordering: alpha orbital p -> bit p, beta p -> bit n+p."""
    out = []
    for x in range(1 << (2 * n)):
        a, b = x & ((1 << n) - 1), x >> n
        if bin(a).count("1") == na and bin(b).count("1") == nb:
            out.append(x)
    return np.array(out)


def local_qubits(n):
    return [list(range(n)), list(range(n, 2 * n))]


def hamiltonian_matrix(h1s, eri, n, states):
    """Dense H in the given basis states (fragment-local ordering)."""
    a = annihilators(2 * n)
    q = local_qubits(n)
    e = [[[(a[q[s][p]].T @ a[q[s][r]]).tocsr() for r in range(n)] for p in range(n)]
         for s in range(2)]
    h = sp.csr_matrix((1 << (2 * n), 1 << (2 * n)))
    for s in range(2):
        for p in range(n):
            for r in range(n):
                if h1s[s][p, r]:
                    h = h + h1s[s][p, r] * e[s][p][r]
    for s in range(2):
        for t in range(2):
            for p in range(n):
                for qq in range(n):
                    for r in range(n):
                        for ss in range(n):
                            c = 0.5 * eri[p, qq, r, ss]
                            if abs(c) < 1e-14:
                                continue
                            term = e[s][p][qq] @ e[t][r][ss]
                            if s == t and qq == r:
                                term = term - e[s][p][ss]
                            h = h + c * term
    h = h.tocsr()[states][:, states]
    return h.toarray()


def local_state(n, na, nb, coeffs, states):
    psi = np.zeros(1 << (2 * n))
    psi[states] = coeffs
    return psi / np.linalg.norm(psi)


def random_fragment_state(n, na, nb, seed):
    st = sector_states(n, na, nb)
    c = np.random.default_rng(seed).normal(size=st.size)
    return local_state(n, na, nb, c, st)


def product_state(blocks):
    """
    Kronecker product of fixed-particle-number block states; block 0 occupies
    the lowest JW indices. For contiguous blocks this is exactly the
    antisymmetrised product state.
    """
    psi = np.array([1.0])
    for b in reversed(blocks):
        psi = np.kron(psi, b)
    return psi


def closed_block(ncore):
    """All 2*ncore spin orbitals occupied."""
    v = np.zeros(1 << (2 * ncore))
    v[-1] = 1.0
    return v


def empty_block(nvirt):
    v = np.zeros(1 << (2 * nvirt))
    v[0] = 1.0
    return v


# ── exact fragment solver for the driver ──────────────────────────────────

class ToyFCISolver:
    """Exact fragment ground states by dense diagonalisation."""

    stochastic = False
    name = "toy-fci"

    def solve(self, hams, cycle=0):
        out = []
        for ham in hams:
            n = ham.norb
            na, nb = ham.nelec
            st = sector_states(n, na, nb)
            hmat = hamiltonian_matrix(ham.h1s, ham.eri, n, st)
            w, v = np.linalg.eigh(hmat)
            psi = local_state(n, na, nb, v[:, 0], st)
            dm1s, dm2 = exact_rdms(psi, n, local_qubits(n))
            out.append(FragmentResult(dm1s=dm1s, dm2=dm2, energy=float(w[0]),
                                      info={"dim": int(st.size)}))
        return out


def full_fci_energy(ints, C, nelec):
    """Exact ground-state energy of the whole toy system in the MO basis."""
    n = C.shape[1]
    h = C.T @ ints.hcore @ C
    eri = ints.ao2mo(C, C, C, C)
    st = sector_states(n, *nelec)
    hmat = hamiltonian_matrix([h, h], eri, n, st)
    return float(np.linalg.eigvalsh(hmat)[0] + ints.energy_nuc)
