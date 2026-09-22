"""
LASSQD fragment solver: LUCJ circuits -> one glued sampling job -> SQD with
carryover, per fragment. Implemented from Wang et al., PNAS 2026 (Methods,
Algorithms 1 and 2) using ffsim, Qiskit, qiskit-addon-sqd and PySCF.

PER MACRO CYCLE
---------------
For every fragment K (Hamiltonian from energy.fragment_hamiltonians):

  1. Fragment mean field (RHF/ROHF on the fragment Hamiltonian). Its
     orbitals U_K define the basis the circuit is built and measured in --
     the HF determinant dominates there, which is what makes sampling
     efficient.
  2. LUCJ ansatz from fragment UCCSD amplitudes (optionally refined with
     ffsim's linear method), heavy-hex interaction pairs by default.

All fragment circuits are placed side by side on disjoint qubits, measured
into one classical register each, and run as ONE job.

  3. SQD, iterated (Algorithm 1): post-select or recover configurations,
     draw K batches of d configurations, diagonalise H in each batch's
     alpha x beta product space (PySCF selected CI, spin penalty), keep the
     lowest batch, update orbital occupancies.
  4. Carryover (Algorithm 2): determinants with |c| > eps from the best
     batch are appended to every batch of the next iteration, and to the
     next macro cycle after transporting them into the new fragment basis
     by maximum orbital overlap.
  5. RDMs are rotated back from U_K into the fragment's LAS orbitals.

SPIN
----
With an open-shell environment the fragment Hamiltonian has h_alpha !=
h_beta. PySCF's selected CI is spin restricted, so it is given the average
and the remainder (h_alpha - h_beta)/2 is added exactly inside the
subspace (_spin_dependent_sci). The circuit is built from the average --
it only has to generate the right configurations.
"""

from __future__ import annotations

import numpy as np

from quenais.las import sqd_bits as bits
from quenais.las.driver import FragmentResult
from quenais.las.fci_solver import spin_summed_2rdm

__all__ = ["LassqdFragmentSolver", "sample_fragment_circuits"]


# ═════════════════════════════════════════════════════════════════════════
# Fragment mean field and LUCJ circuit
# ═════════════════════════════════════════════════════════════════════════

def _fragment_mean_field(h1, eri, n, na, nb):
    """RHF/ROHF on a bare Hamiltonian. Requires na >= nb (swap outside)."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(verbose=0)
    mol.nelectron = na + nb
    mol.spin = na - nb
    mol.incore_anyway = True
    try:
        mol.nao = n
    except AttributeError:
        pass
    mf = scf.RHF(mol) if na == nb else scf.ROHF(mol)
    mf.get_hcore = lambda *a, **k: h1
    mf.get_ovlp = lambda *a, **k: np.eye(n)
    mf._eri = ao2mo.restore(8, eri, n)
    mf.init_guess = "1e"
    mf.max_cycle = 200
    mf.kernel()
    return mf


def _uccsd_amplitudes(mf):
    from pyscf import cc, scf

    umf = scf.addons.convert_to_uhf(mf)
    mycc = cc.UCCSD(umf)
    mycc.verbose = 0
    mycc.kernel()
    return mycc.t1, mycc.t2


def _interaction_pairs(n, kind):
    if kind == "all":
        return None
    aa = [(p, p + 1) for p in range(n - 1)]
    ab = [(p, p) for p in range(0, n, 4)]
    return (aa, ab, list(aa))


def _lucj_operator(h1_mo, eri_mo, n, nelec, mf_amplitudes, settings, rng, log):
    """UCJ (spin-unbalanced) operator, as in the LASSQD paper."""
    import ffsim

    pairs = _interaction_pairs(n, settings.lucj_pairs)
    n_reps = settings.lucj_n_reps
    x0, final = None, False
    if mf_amplitudes is not None:
        t1, t2 = mf_amplitudes
        try:
            full = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
                t2, t1=t1, n_reps=n_reps)
            final = full.final_orbital_rotation is not None
            x0 = full.to_parameters(interaction_pairs=pairs)
        except Exception as exc:   # tiny spaces: no doubles, shape edge cases
            log(f"      LUCJ from t-amplitudes failed ({exc}); random start")
    if x0 is None:
        npar = ffsim.UCJOpSpinUnbalanced.n_params(
            n, n_reps, interaction_pairs=pairs, with_final_orbital_rotation=False)
        x0 = rng.uniform(-0.1, 0.1, size=npar)

    def build(x):
        return ffsim.UCJOpSpinUnbalanced.from_parameters(
            x, norb=n, n_reps=n_reps, interaction_pairs=pairs,
            with_final_orbital_rotation=final)

    if settings.lucj_optimize:
        from ffsim.optimize import minimize_linear_method

        ham = ffsim.MolecularHamiltonian(h1_mo, eri_mo, 0.0)
        linop = ffsim.linear_operator(ham, norb=n, nelec=nelec)
        ref = ffsim.hartree_fock_state(n, nelec)

        def params_to_vec(x):
            return ffsim.apply_unitary(ref, build(x), norb=n, nelec=nelec)

        res = minimize_linear_method(params_to_vec, linop, x0=x0,
                                     maxiter=settings.lucj_opt_maxiter)
        x0 = res.x
        log(f"      LUCJ linear method: E = {res.fun:.8f} ({res.nit} it)")
    return build(x0)


def _fragment_circuit(n, nelec, op):
    import ffsim
    from qiskit import QuantumCircuit, QuantumRegister

    q = QuantumRegister(2 * n, "q")
    qc = QuantumCircuit(q)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(n, nelec), q)
    qc.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(op), q)
    return qc


# ═════════════════════════════════════════════════════════════════════════
# Sampling
# ═════════════════════════════════════════════════════════════════════════

def sample_fragment_circuits(circuits, settings):
    """
    Place every fragment circuit on its own qubits, measure each into its
    own register ("f0", "f1", ...), run ONE job, and return one counts dict
    per fragment. Bitstrings follow the convention in sqd_bits.
    """
    import ffsim
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    total = sum(c.num_qubits for c in circuits)
    q = QuantumRegister(total, "q")
    glued = QuantumCircuit(q)
    names, off = [], 0
    for i, c in enumerate(circuits):
        creg = ClassicalRegister(c.num_qubits, f"f{i}")
        glued.add_register(creg)
        span = list(range(off, off + c.num_qubits))
        glued.compose(c, qubits=span, inplace=True)
        glued.measure([q[j] for j in span], creg)
        names.append(f"f{i}")
        off += c.num_qubits

    backend = settings.backend
    if backend in ("aer_mps", "aer_statevector"):
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2

        opts = {"method": "matrix_product_state" if backend == "aer_mps"
                else "statevector"}
        if backend == "aer_mps" and settings.mps_max_bond_dim:
            opts["matrix_product_state_max_bond_dimension"] = settings.mps_max_bond_dim
        sim = AerSimulator(**opts)
        pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
        pm.pre_init = ffsim.qiskit.PRE_INIT
        tqc = pm.run(glued)
        sampler = SamplerV2(seed=settings.seed, options={"backend_options": opts})
        pub = sampler.run([tqc], shots=settings.shots).result()[0]
    elif backend == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        hw = service.backend(settings.ibm_backend_name)
        pm = generate_preset_pass_manager(
            optimization_level=settings.ibm_optimization_level, backend=hw)
        pm.pre_init = ffsim.qiskit.PRE_INIT
        tqc = pm.run(glued)
        pub = SamplerV2(mode=hw).run([tqc], shots=settings.shots).result()[0]
    else:
        raise ValueError(f"unknown LASSQD backend {backend!r}")

    return [getattr(pub.data, name).get_counts() for name in names]


# ═════════════════════════════════════════════════════════════════════════
# Subspace diagonalisation
# ═════════════════════════════════════════════════════════════════════════

def _sci_solve(h1s, eri, n, nelec, sa, sb, spin2s, settings):
    """
    Diagonalise in span{|a>|b> : a in sa, b in sb} with PySCF selected CI.
    h1s is (2, n, n): the spin-dependent part (h_a - h_b)/2 is added exactly
    on top of PySCF's spin-restricted contraction (see _spin_dependent_sci).
    Returns (e, amplitudes, sa, sb, dm1s, dm2).
    """
    from pyscf.fci import addons, selected_ci

    sa = np.unique(np.asarray(sa, dtype=np.int64))
    sb = np.unique(np.asarray(sb, dtype=np.int64))
    dim = sa.size * sb.size
    if dim > settings.sqd_max_dim:
        raise RuntimeError(
            f"SQD subspace {sa.size} x {sb.size} = {dim} exceeds sqd_max_dim="
            f"{settings.sqd_max_dim}. Lower samples_per_batch or carryover_eps."
        )
    h_avg = 0.5 * (h1s[0] + h1s[1])
    dh = 0.5 * (h1s[0] - h1s[1])
    if np.max(np.abs(dh)) > 1e-12:
        myci = _spin_dependent_sci(selected_ci, bits.one_body_string_operator(sa, n, dh),
                                   bits.one_body_string_operator(sb, n, dh))
    else:
        myci = selected_ci.SelectedCI()
    myci.conv_tol = settings.sqd_davidson_tol
    myci.max_cycle = settings.sqd_max_davidson
    ss = 0.25 * spin2s * (spin2s + 2)
    myci = addons.fix_spin_(myci, ss=ss, shift=settings.sqd_spin_shift)
    nroots = max(1, min(settings.sqd_nroots, dim))
    e, civec = selected_ci.kernel_fixed_space(
        myci, h_avg, eri, n, nelec, (sa, sb), nroots=nroots,
        tol=settings.sqd_davidson_tol, max_cycle=settings.sqd_max_davidson)
    if nroots > 1:
        i = int(np.argmin(e))
        e, civec = e[i], civec[i]
    dma, dmb = selected_ci.make_rdm1s(civec, n, nelec)
    dm2 = spin_summed_2rdm(selected_ci.make_rdm2s(civec, n, nelec))
    # Carryover maps amplitudes to strings, so use the strings the vector
    # actually carries rather than assuming PySCF kept our ordering.
    strs = getattr(civec, "_strs", None)
    if strs is not None:
        sa, sb = np.asarray(strs[0], np.int64), np.asarray(strs[1], np.int64)
    return (float(e), np.asarray(civec).reshape(sa.size, sb.size), sa, sb,
            np.stack([dma, dmb]), dm2)


def _spin_dependent_sci(selected_ci, a_mat, b_mat):
    """
    SelectedCI whose Hamiltonian gains  sum_pq dh_pq (E^a_pq - E^b_pq),
    applied as  A C - C B^T  on the (n_alpha_strings, n_beta_strings)
    coefficient matrix. Needed when the fragment's environment is open
    shell, i.e. h_alpha != h_beta.
    """
    da, db = np.diag(a_mat), np.diag(b_mat)

    class _SCI(selected_ci.SelectedCI):
        def contract_2e(self, eri, civec_strs, norb, nelec, link_index=None, **kw):
            out = super().contract_2e(eri, civec_strs, norb, nelec, link_index, **kw)
            c = np.asarray(civec_strs).reshape(a_mat.shape[0], b_mat.shape[0])
            extra = a_mat @ c - c @ b_mat.T
            return out + extra.reshape(np.shape(out))

        def make_hdiag(self, h1e, eri, ci_strs, norb, nelec, *args, **kw):
            hd = super().make_hdiag(h1e, eri, ci_strs, norb, nelec, *args, **kw)
            return hd + (da[:, None] - db[None, :]).reshape(np.shape(hd))

    return _SCI()


# ═════════════════════════════════════════════════════════════════════════
# Solver
# ═════════════════════════════════════════════════════════════════════════

class LassqdFragmentSolver:
    stochastic = True
    name = "lassqd"

    def __init__(self, settings, log=print):
        self.s = settings
        self.log = log
        self.rng = np.random.default_rng(settings.seed)
        self._basis = {}       # fragment -> U of the previous macro cycle
        self._carry = {}       # fragment -> (alpha strings, beta strings)

    # ── per fragment preparation ─────────────────────────────────────────
    def _prepare(self, ham):
        n, (na, nb) = ham.norb, ham.nelec
        h_avg = 0.5 * (ham.h1s[0] + ham.h1s[1])
        swap = na < nb
        mf = _fragment_mean_field(h_avg, ham.eri, n, *((nb, na) if swap else (na, nb)))
        u = np.asarray(mf.mo_coeff)
        if not mf.converged:
            self.log(f"    fragment {ham.index}: mean field not converged; "
                     f"using it anyway (it only defines the sampling basis)")
        h1_mo = np.einsum("pi,spq,qj->sij", u, ham.h1s, u)
        eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", ham.eri, u, u, u, u,
                           optimize=True)
        amps = None
        if na + nb > 0 and min(na, nb) < n:
            try:
                t1, t2 = _uccsd_amplitudes(mf)
                if swap:
                    t1 = (t1[1], t1[0])
                    t2 = (t2[2], t2[1].transpose(1, 0, 3, 2), t2[0])
                amps = (t1, t2)
            except Exception as exc:
                self.log(f"    fragment {ham.index}: UCCSD failed ({exc})")
        op = _lucj_operator(0.5 * (h1_mo[0] + h1_mo[1]), eri_mo, n, (na, nb),
                            amps, self.s, self.rng, self.log)
        return {"u": u, "h1_mo": h1_mo, "eri_mo": eri_mo,
                "circuit": _fragment_circuit(n, (na, nb), op)}

    # ── SQD with carryover for one fragment ──────────────────────────────
    def _sqd(self, ham, prep, counts):
        s = self.s
        n, (na, nb) = ham.norb, ham.nelec
        eri = prep["eri_mo"]
        raw, raw_p = bits.counts_to_matrix(counts, 2 * n)
        n_valid = int(bits.postselect(raw, raw_p, n, na, nb)[0].shape[0])
        hf_a, hf_b = bits.hf_strings(na, nb)

        carry_a, carry_b = self._transported_carryover(ham.index, prep["u"])
        occs, best, iters = None, None, []
        for it in range(s.sqd_iterations):
            if occs is None:
                mat, p = bits.postselect(raw, raw_p, n, na, nb)
            else:
                from qiskit_addon_sqd.configuration_recovery import (
                    recover_configurations,
                )
                mat, p = recover_configurations(
                    raw, raw_p, occs, num_elec_a=na, num_elec_b=nb,
                    rand_seed=int(self.rng.integers(2**31)))
                mat, p = bits.postselect(np.asarray(mat, dtype=bool),
                                         np.asarray(p, dtype=float), n, na, nb)
            batches = bits.subsample(mat, p, s.sqd_samples_per_batch,
                                     s.sqd_batches, self.rng) if mat.shape[0] else [mat]
            sols = []
            for b in batches:
                sa, sb = bits.matrix_to_strings(b, n)
                sa = np.concatenate([sa, [hf_a], carry_a])
                sb = np.concatenate([sb, [hf_b], carry_b])
                if na == nb and s.sqd_symmetrize_spin:
                    sa = sb = np.union1d(sa, sb)
                sols.append(_sci_solve(prep["h1_mo"], eri, n, (na, nb), sa, sb,
                                       ham.spin2s, s))
            energies = [x[0] for x in sols]
            best = sols[int(np.argmin(energies))]
            occs = (np.mean([x[4][0].diagonal() for x in sols], axis=0),
                    np.mean([x[4][1].diagonal() for x in sols], axis=0))
            if s.carryover_eps > 0:
                carry_a, carry_b = bits.select_carryover(best[1], best[2], best[3],
                                                         s.carryover_eps)
            iters.append({"energy": float(best[0]),
                          "dim": int(best[2].size * best[3].size),
                          "n_carry": (int(len(carry_a)), int(len(carry_b)))})

        self._basis[ham.index] = prep["u"]
        self._carry[ham.index] = (carry_a, carry_b)

        _e, _c, sa, sb, dm1s_mo, dm2_mo = best
        u = prep["u"]
        dm1s = np.einsum("pi,sij,qj->spq", u, dm1s_mo, u)
        dm2 = np.einsum("pi,qj,rk,sl,ijkl->pqrs", u, u, u, u, dm2_mo, optimize=True)
        e_exact = (np.sum(prep["h1_mo"][0] * dm1s_mo[0])
                   + np.sum(prep["h1_mo"][1] * dm1s_mo[1])
                   + 0.5 * np.einsum("pqrs,pqrs->", dm2_mo, eri))
        full_dim = _n_strings(n, na) * _n_strings(n, nb)
        info = {
            "solver": "sqd",
            "shots_valid_fraction": n_valid / max(1, sum(counts.values())),
            "subspace_dim": int(sa.size * sb.size),
            "full_dim": int(full_dim),
            "subspace_fraction": float(sa.size * sb.size / full_dim),
            "iterations": iters,
        }
        return FragmentResult(dm1s=dm1s, dm2=dm2, energy=float(e_exact), info=info)

    def _transported_carryover(self, k, u_new):
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        if self.s.carryover_eps <= 0 or k not in self._carry:
            return empty
        perm = bits.overlap_permutation(self._basis[k], u_new)
        if perm is None:
            self.log(f"    fragment {k}: basis changed too much to transport "
                     f"carryover determinants; starting fresh this cycle")
            return empty
        ca, cb = self._carry[k]
        return bits.remap_strings(ca, perm), bits.remap_strings(cb, perm)

    # ── driver entry point ───────────────────────────────────────────────
    def solve(self, hams, cycle=0):
        preps = [self._prepare(h) for h in hams]
        counts = sample_fragment_circuits([p["circuit"] for p in preps], self.s)
        out = []
        for ham, prep, cnt in zip(hams, preps, counts):
            res = self._sqd(ham, prep, cnt)
            self.log(f"    fragment {ham.index}: SQD dim {res.info['subspace_dim']} "
                     f"({100 * res.info['subspace_fraction']:.1f}% of FCI), "
                     f"valid shots {100 * res.info['shots_valid_fraction']:.1f}%")
            out.append(res)
        return out


def _n_strings(n, k):
    from math import comb

    return comb(n, k)

