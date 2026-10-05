"""
Quantum Solver (SQD / SKQD / SqDRIFT) for QuEnAIS pipeline.

Settings are read from cfg.qiskit (QiskitSolverSettings). Until 0.2.1 this
module read cfg.ansatz, cfg.n_shots, ... -- attributes Config stopped
carrying when the settings were grouped, so every --solver sqd|skqd|sqdrift
run died at step 3 with AttributeError. No test exercised main().

OUTPUT
------
results/step3_<solver>.pkl, one file per solver, so SQD, SKQD and SqDRIFT
results can sit side by side. (They used to share step3_results.pkl behind
a bare os.path.exists() cache check, so running SKQD after SQD silently
returned the SQD result.)

SEEDS
-----
cfg.qiskit.seed controls everything stochastic *in the measurement*: the
simulator's shot sampling, configuration recovery, and SqDRIFT's random
circuit draws. The ansatz parameters (lucj_random_seed, the SU2 angles) are
deliberately NOT tied to it: the circuit is a fixed design choice, and a
seed repeat should resample the same circuit. seed=None keeps the old
behaviour (fixed recovery seeds, unseeded simulator).
"""

import os
import pickle
import time
import warnings
import numpy as np
from collections import Counter
from math import comb


def solver_result_file(cfg, solver=None):
    """results/step3_<solver>.pkl -- one file per solver."""
    return os.path.join(cfg.results_dir,
                        f"step3_{solver or cfg.quantum_solver}.pkl")


def subspace_dimension(bsm, n_orb, n_alpha, n_beta, open_shell=False):
    """
    Dimension of the determinant space solve_fermion actually diagonalises:
    (#unique alpha strings) x (#unique beta strings). For a closed-shell
    solve (open_shell=False, n_alpha == n_beta) qiskit-addon-sqd uses the
    union of the alpha and beta strings for both spins.

    This is NOT the number of bitstrings (bsm.shape[0]), which is what the
    iteration table prints as "configs".
    """
    if bsm.shape[0] == 0:
        return 0
    try:
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

        sa, sb = bitstring_matrix_to_ci_strs(bsm, open_shell=open_shell)
        return int(len(sa) * len(sb))
    except Exception:
        left = {r.tobytes() for r in np.asarray(bsm[:, :n_orb], dtype=bool)}
        right = {r.tobytes() for r in np.asarray(bsm[:, n_orb:], dtype=bool)}
        if not open_shell and n_alpha == n_beta:
            u = len(left | right)
            return u * u
        return len(left) * len(right)


def full_space_dimension(n_orb, n_alpha, n_beta):
    return comb(n_orb, n_alpha) * comb(n_orb, n_beta)


def embedding_fci_energy(step2, max_dets=5_000_000):
    """
    Exact ground-state energy of the step-2 embedding Hamiltonian
    (h1e, h2e, ecore) -- the number SQD/SKQD/SqDRIFT/GQE are approximating
    on the DMET route. Their difference from this is solver error alone.

    Returns None if the space is larger than max_dets or PySCF fails.
    """
    n = int(step2["n_emb"])
    na, nb = int(step2["n_alpha"]), int(step2["n_beta"])
    if full_space_dimension(n, na, nb) > max_dets:
        return None
    try:
        from pyscf import fci

        solver = fci.direct_spin1.FCI()
        solver.conv_tol = 1e-12
        solver.max_cycle = 500
        e, _ = solver.kernel(np.asarray(step2["h1e"]), np.asarray(step2["h2e"]),
                             n, (na, nb), ecore=float(step2["ecore"]))
        return float(e)
    except Exception as exc:
        warnings.warn(f"embedding FCI failed: {exc}", RuntimeWarning)
        return None


# ── Pure helper functions (no shared state) ───────────────────────────────────

def _fop_to_sparse_pauli(qubit_op, n_so):
    from qiskit.quantum_info import SparsePauliOp
    labels, coeffs = [], []
    for term, coeff in qubit_op.terms.items():
        arr = ['I'] * n_so
        for idx, pauli in term:
            arr[idx] = pauli
        labels.append(''.join(reversed(arr)))
        coeffs.append(complex(coeff))
    if not labels:
        return SparsePauliOp('I' * n_so, coeffs=[0.0])
    op       = SparsePauliOp(labels, coeffs=coeffs).simplify()
    max_imag = float(np.max(np.abs(np.imag(op.coeffs))))
    if max_imag > 1e-6:
        warnings.warn(
            f"Qubit Hamiltonian has imaginary coefficients up to {max_imag:.2e}.",
            RuntimeWarning,
        )
    return SparsePauliOp(op.paulis, coeffs=np.real(op.coeffs))


# ── Entry point ───────────────────────────────────────────────────────────────

def main(cfg, force=False):
    """
    Run quantum solver.
    cfg   : quenais.config.Config
    force : rerun even if cached result exists
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import efficient_su2
    from qiskit_addon_sqd.counts import counts_to_arrays
    from qiskit_addon_sqd.fermion import solve_fermion
    from qiskit_addon_sqd.configuration_recovery import recover_configurations

    os.makedirs(cfg.results_dir, exist_ok=True)

    q = cfg.qiskit
    out_path = solver_result_file(cfg)
    if not force and cfg.cached_result_is_current(out_path):
        print(f"[Step 3] Using cached result: {out_path}")
        with open(out_path, "rb") as f:
            return pickle.load(f)

    t_start = time.perf_counter()
    timings = {"sampling": 0.0, "recovery": 0.0, "diagonalisation": 0.0,
               "circuit_build": 0.0}
    stats = {"n_diagonalisations": 0, "circuit_depths": [],
             "transpiled_depths": [], "valid_shot_fraction": None,
             "unique_valid_bitstrings": None, "n_circuits": 0,
             "shots_per_circuit": None}

    for path, name in [(cfg.step1_file, "finder.main"),
                       (cfg.step2_file, "hamiltonian.main")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required input not found: {path}\nRun {name}(cfg) first."
            )

    with open(cfg.step1_file, "rb") as f:
        step1 = pickle.load(f)
    with open(cfg.step2_file, "rb") as f:
        step2 = pickle.load(f)

    h1e        = step2["h1e"]
    h2e        = step2["h2e"]
    n_emb      = step2["n_emb"]
    n_alpha    = step2["n_alpha"]
    n_beta     = step2["n_beta"]
    uhf_energy = step2["uhf_energy"]
    ecore      = step2["ecore"]
    mol_info   = step1["mol_info"]
    mp2_energy = step1.get("mp2_energy",
                            uhf_energy + step2.get("mp2_corr", 0.0))
    n_qubits   = 2 * n_emb

    print(f"\n{'='*60}")
    print(f"[Step 3] Quantum Solver — {mol_info['molecule']}")
    print(f"{'='*60}")
    print(f"  Solver       : {cfg.quantum_solver.upper()}")
    print(f"  Ansatz       : {q.ansatz.upper()}")
    print(f"  Mapping      : {q.fermion_to_qubit.upper()}")
    print(f"  Backend      : {q.backend.upper()}")
    print(f"  Embedding    : {n_emb} orbs = {n_qubits} qubits")
    print(f"  Electrons    : {n_alpha}α + {n_beta}β")
    print(f"  UHF ref      : {uhf_energy:.8f} Ha")
    print(f"  MP2 ref      : {mp2_energy:.8f} Ha")
    print(f"  Seed         : {q.seed}")

    # ── Backend dispatch ──────────────────────────────────────────────────────
    def sample_circuits(circuits, shots):
        t0 = time.perf_counter()
        stats["n_circuits"] += len(circuits)
        stats["shots_per_circuit"] = int(shots)
        stats["circuit_depths"].extend(int(c.depth()) for c in circuits)
        try:
            return _sample_circuits(circuits, shots)
        finally:
            timings["sampling"] += time.perf_counter() - t0

    def _sample_circuits(circuits, shots):
        backend = q.backend.lower()
        if backend == "local":
            from qiskit.primitives import StatevectorSampler
            res = StatevectorSampler(seed=q.seed).run(circuits, shots=shots).result()
            return [res[i].data.meas.get_counts() for i in range(len(circuits))]
        elif backend == "mps":
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            sim = AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=q.mps_max_bond_dim,
                matrix_product_state_truncation_threshold=q.mps_trunc_thresh,
            )
            tc     = transpile(circuits, backend=sim, optimization_level=1,
                               seed_transpiler=q.seed)
            stats["transpiled_depths"].extend(int(c.depth()) for c in tc)
            run_kw = {} if q.seed is None else {"seed_simulator": int(q.seed)}
            result = sim.run(tc, shots=shots, **run_kw).result()
            return [result.get_counts(i) for i in range(len(circuits))]
        elif backend == "ibm":
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            service = QiskitRuntimeService()
            hw = (service.backend(q.ibm_backend_name) if q.ibm_backend_name
                  else service.least_busy(operational=True, simulator=False,
                                          min_num_qubits=circuits[0].num_qubits))
            pm     = generate_preset_pass_manager(q.ibm_optimization_level,
                                                   backend=hw)
            tc     = pm.run(circuits)
            stats["transpiled_depths"].extend(int(c.depth()) for c in tc)
            job    = SamplerV2(mode=hw).run([(c,) for c in tc], shots=shots)
            result = job.result()
            return [result[i].data.meas.get_counts() for i in range(len(circuits))]
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Use: local | mps | ibm")

    # ── Qubit Hamiltonian ─────────────────────────────────────────────────────
    def _fop_from_integrals():
        from openfermion import FermionOperator as OF_FermionOp
        fop = OF_FermionOp()
        for p in range(n_emb):
            for q in range(n_emb):
                h = complex(h1e[p, q])
                if abs(h) < 1e-10:
                    continue
                fop += OF_FermionOp(f"{p}^ {q}",             h)
                fop += OF_FermionOp(f"{n_emb+p}^ {n_emb+q}", h)
        for p in range(n_emb):
            for q in range(n_emb):
                for r in range(n_emb):
                    for s in range(n_emb):
                        h = 0.5 * complex(h2e[p, q, r, s])
                        if abs(h) < 1e-10:
                            continue
                        pa, qa = p,         q
                        pb, qb = n_emb + p, n_emb + q
                        ra, sa = r,         s
                        rb, sb = n_emb + r, n_emb + s
                        fop += OF_FermionOp(((pa,1),(ra,1),(sa,0),(qa,0)), h)
                        fop += OF_FermionOp(((pb,1),(rb,1),(sb,0),(qb,0)), h)
                        fop += OF_FermionOp(((pa,1),(rb,1),(sb,0),(qa,0)), h)
                        fop += OF_FermionOp(((pb,1),(ra,1),(sa,0),(qb,0)), h)
        return fop

    def build_qubit_hamiltonian():
        mapping = q.fermion_to_qubit.lower()
        fop = _fop_from_integrals()
        if mapping == "jw":
            from openfermion import jordan_wigner
            op = _fop_to_sparse_pauli(jordan_wigner(fop), 2 * n_emb)
            print(f"  JW Hamiltonian: {len(op)} Pauli terms")
        elif mapping == "bk":
            from openfermion import bravyi_kitaev
            op = _fop_to_sparse_pauli(bravyi_kitaev(fop), 2 * n_emb)
            print(f"  BK Hamiltonian: {len(op)} Pauli terms")
        else:
            raise ValueError(f"Unknown mapping: '{mapping}'. Use: jw | bk")
        return op

    # ── Ansatz builders ───────────────────────────────────────────────────────
    def _build_su2_circuit():
        hf_circ = QuantumCircuit(n_qubits)
        for i in range(n_alpha): hf_circ.x(i)
        for i in range(n_beta):  hf_circ.x(n_emb + i)
        ansatz = efficient_su2(
            n_qubits,
            reps=q.ansatz_reps,
            entanglement="full",
            skip_final_rotation_layer=True,
        )
        params = np.random.default_rng(42).uniform(0, 2*np.pi,
                                                    ansatz.num_parameters)
        circ = hf_circ.compose(ansatz.assign_parameters(params))
        circ.measure_all()
        print(f"  SU2 circuit : {n_qubits}q,  depth={circ.depth()},  "
              f"params={ansatz.num_parameters}")
        print(f"  ⚠ SU2 does not conserve particle number — "
              f"~40-60% of shots will be filtered.")
        return circ

    def _build_lucj_manual():
        rng  = np.random.default_rng(q.lucj_random_seed)
        circ = QuantumCircuit(n_qubits)
        for i in range(n_alpha): circ.x(i)
        for i in range(n_beta):  circ.x(n_emb + i)

        def givens_layer(qc, qubits, offset):
            for p in range(offset, len(qubits) - 1, 2):
                q0, q1 = qubits[p], qubits[p + 1]
                theta  = rng.uniform(-np.pi / 4, np.pi / 4)
                phi    = rng.uniform(0, 2 * np.pi)
                qc.cx(q0, q1)
                qc.ry(2 * theta, q0)
                qc.rz(phi, q0)
                qc.cx(q0, q1)
                qc.rz(-phi, q1)

        def jastrow_layer(qc, qubits):
            for q in qubits:
                qc.rz(rng.uniform(-np.pi / 8, np.pi / 8), q)

        alpha_q = list(range(n_emb))
        beta_q  = list(range(n_emb, 2 * n_emb))

        for _ in range(q.lucj_num_layers):
            givens_layer(circ, alpha_q, 0)
            givens_layer(circ, alpha_q, 1)
            jastrow_layer(circ, alpha_q)
            givens_layer(circ, beta_q, 0)
            givens_layer(circ, beta_q, 1)
            jastrow_layer(circ, beta_q)

        circ.measure_all()
        print(f"  LUCJ (manual) circuit: {n_qubits}q,  depth={circ.depth()},  "
              f"layers={q.lucj_num_layers}")
        print(f"  ✓ Particle number conserved by construction")
        return circ

    def build_ansatz_circuit():
        ansatz = q.ansatz.lower()
        if ansatz == "su2":
            return _build_su2_circuit()
        elif ansatz == "lucj":
            return _build_lucj_manual()
        else:
            raise ValueError(f"Unknown ansatz: '{ansatz}'. Use: su2 | lucj")

    # ── Shared helpers ────────────────────────────────────────────────────────
    # BIT ORDER. Qubit i is alpha orbital i, qubit n_emb+i is beta orbital
    # i (see the X gates in every circuit builder). Counts keys and
    # counts_to_arrays rows are big-endian: column j is qubit 2*n_emb-1-j.
    # So columns [:n_emb] are BETA and [n_emb:] are ALPHA -- the
    # qiskit-addon-sqd convention solve_fermion assumes. Before 0.2.1 the
    # filter had them swapped (harmless when n_alpha == n_beta, zero valid
    # shots otherwise) and hf_bitstring() built the determinant occupying
    # the HIGHEST orbitals, which SqDRIFT then injected as "HF".
    def filter_bitstrings(bsm, probs):
        valid  = ((bsm[:, n_emb:].sum(axis=1) == n_alpha) &
                  (bsm[:, :n_emb].sum(axis=1) == n_beta))
        bsm_f  = bsm[valid]
        prob_f = probs[valid]
        if len(prob_f) > 0:
            prob_f = prob_f / prob_f.sum()
        return bsm_f, prob_f

    def hf_bitstring():
        row = np.zeros(2 * n_emb, dtype=bool)
        for i in range(n_alpha): row[2 * n_emb - 1 - i] = True   # qubit i
        for i in range(n_beta):  row[n_emb - 1 - i]     = True   # qubit n_emb+i
        return row

    def inject_hf_reference(bsm, probs):
        hf_row  = hf_bitstring()
        present = (bsm.shape[0] > 0 and
                   any(np.array_equal(bsm[i], hf_row)
                       for i in range(bsm.shape[0])))
        if not present:
            bsm   = (np.vstack([bsm, hf_row[np.newaxis, :]])
                     if bsm.shape[0] > 0 else hf_row[np.newaxis, :])
            probs = np.append(probs, 1.0 / max(bsm.shape[0], 1))
            probs = probs / probs.sum()
        return bsm, probs

    def _check_configs(bsm, probs, context=""):
        if bsm.shape[0] == 0:
            raise RuntimeError(
                f"No valid bitstrings after particle-number filtering"
                + (f" ({context})" if context else "") + ".\n"
                f"  Expected: {n_alpha}α + {n_beta}β in {n_emb} orbitals.\n"
                f"  If using SU2: increase n_shots or switch to lucj."
            )

    def print_iteration_header():
        print(f"\n  {'─'*84}")
        print(f"  {'Iter':>5} │ {'Energy (Ha)':>14} │ {'configs':>7} │ "
              f"{'vs UHF':>13} │ {'vs MP2':>13} │ {'ΔE(prev)':>12}")
        print(f"  {'─'*84}")

    def print_iteration(label, energy, n_configs, prev_energy=None):
        vs_uhf    = energy - uhf_energy
        vs_mp2    = energy - mp2_energy
        delta_str = (f"{energy - prev_energy:+.6f}"
                     if prev_energy is not None else "       ---")
        print(f"  {label:>5} │ {energy:>14.8f} │ {n_configs:>7d} │ "
              f"{vs_uhf:+.6f} {'↓' if vs_uhf<0 else '↑'}  │ "
              f"{vs_mp2:+.6f} {'↓' if vs_mp2<0 else '↑'}  │ {delta_str}")

    def timed_solve(bsm):
        t0 = time.perf_counter()
        try:
            return solve_fermion(
                bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
            )
        finally:
            timings["diagonalisation"] += time.perf_counter() - t0
            stats["n_diagonalisations"] += 1

    def record_valid(_raw, n_valid_unique, valid_prob):
        stats["valid_shot_fraction"] = float(valid_prob)
        stats["unique_valid_bitstrings"] = int(n_valid_unique)

    def valid_shot_fraction(raw):
        """Fraction of SHOTS (not distinct bitstrings) with the right N."""
        total = sum(raw.values())
        if total == 0:
            return 0.0
        good = 0
        for key, c in raw.items():
            bits = key.replace(" ", "")
            # big-endian: left half is beta, right half alpha (see
            # filter_bitstrings).
            if (bits[n_emb:].count("1") == n_alpha
                    and bits[:n_emb].count("1") == n_beta):
                good += c
        return good / total

    def iterative_solve(bsm, probs, n_iters):
        avg_occs = (
            np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
            np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
        )
        iterations  = []
        energy      = None
        spin_sq     = None
        prev_energy = None

        print_iteration_header()

        seed_base = 42 if q.seed is None else int(q.seed)
        for it in range(n_iters):
            t0 = time.perf_counter()
            bsm, probs = recover_configurations(
                bsm, probs, avg_occs,
                num_elec_a=n_alpha, num_elec_b=n_beta,
                rand_seed=seed_base + it,
            )
            timings["recovery"] += time.perf_counter() - t0
            if bsm.shape[0] == 0:
                warnings.warn(
                    f"recover_configurations returned 0 configs at iter {it+1}.",
                    RuntimeWarning,
                )
                break

            e_emb, _, avg_occs, spin_sq = timed_solve(bsm)
            energy = float(e_emb) + ecore

            print_iteration(f"{it+1:02d}", energy, bsm.shape[0], prev_energy)
            iterations.append({
                "iter"     : it + 1,
                "energy"   : energy,
                "e_emb"    : float(e_emb),
                "ecore"    : ecore,
                "n_configs": int(bsm.shape[0]),
                "subspace_dim": subspace_dimension(bsm, n_emb, n_alpha, n_beta),
                "vs_uhf"   : float(energy - uhf_energy),
                "vs_mp2"   : float(energy - mp2_energy),
            })
            prev_energy = energy

        print(f"  {'─'*84}")
        return energy, spin_sq, iterations

    # ── SQD ───────────────────────────────────────────────────────────────────
    def run_sqd():
        print(f"\n── SQD ({q.ansatz.upper()} ansatz) {'─'*40}")
        t0         = time.perf_counter()
        circ       = build_ansatz_circuit()
        timings["circuit_build"] += time.perf_counter() - t0
        print(f"  Shots: {q.n_shots}")
        raw        = sample_circuits([circ], q.n_shots)[0]
        bsm, probs = counts_to_arrays(raw)
        bsm, probs = filter_bitstrings(bsm, probs)
        n_total    = sum(raw.values())
        n_valid    = bsm.shape[0]
        vfrac      = valid_shot_fraction(raw)
        record_valid(raw, n_valid, vfrac)
        print(f"  Distinct valid bitstrings: {n_valid}   "
              f"valid shots: {100.*vfrac:.1f}% of {n_total}")
        _check_configs(bsm, probs, "SQD after filter")
        return iterative_solve(bsm, probs, q.sqd_iters)

    # ── SKQD ──────────────────────────────────────────────────────────────────
    def run_skqd():
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import LieTrotter

        # The measured bitstrings are fed to filter_bitstrings/solve_fermion
        # as orbital OCCUPATIONS, and the reference is prepared by X gates
        # on occupied orbitals. Both are true only under Jordan-Wigner:
        # under Bravyi-Kitaev a qubit stores a parity, so the "HF" state
        # is not HF and every sampled bitstring is a different determinant
        # from the one it is read as. The energy stays variational (any
        # subspace is), which is why this was never visible -- it just
        # samples the wrong subspace.
        if q.fermion_to_qubit.lower() != "jw":
            raise ValueError(
                f"SKQD needs --mapping jw (got {q.fermion_to_qubit!r}): "
                f"sampled bitstrings are read as occupation numbers, which "
                f"only Jordan-Wigner provides.")
        print(f"\n── SKQD ({q.fermion_to_qubit.upper()} mapping) {'─'*38}")
        print(f"  Krylov dim: {q.skqd_krylov_dim},  dt={q.skqd_dt},  "
              f"Trotter reps={q.skqd_trotter_reps}")

        t0 = time.perf_counter()
        H_qubit = build_qubit_hamiltonian()

        ref = QuantumCircuit(n_qubits)
        for i in range(n_alpha): ref.x(i)
        for i in range(n_beta):  ref.x(n_emb + i)

        evol = PauliEvolutionGate(
            H_qubit,
            time      = q.skqd_dt / q.skqd_trotter_reps,
            synthesis = LieTrotter(reps=q.skqd_trotter_reps),
        )

        circs = []
        for k in range(q.skqd_krylov_dim):
            qc = ref.copy()
            for _ in range(k):
                qc.append(evol, range(n_qubits))
            qc.measure_all()
            circs.append(qc)

        timings["circuit_build"] += time.perf_counter() - t0
        print(f"  Circuit depths: {[c.depth() for c in circs]}")
        print(f"  Sampling {len(circs)} Krylov circuits "
              f"@ {q.skqd_shots} shots each...")

        all_counts  = sample_circuits(circs, q.skqd_shots)
        merged = Counter()
        for c in all_counts:
            merged.update(c)
        _b, _p = filter_bitstrings(*counts_to_arrays(dict(merged)))
        record_valid(merged, _b.shape[0], valid_shot_fraction(merged))
        iterations  = []
        energy      = None
        spin_sq     = None
        prev_energy = None
        cumulative  = Counter()

        print_iteration_header()

        for k, raw in enumerate(all_counts):
            cumulative.update(raw)
            bsm, probs = counts_to_arrays(dict(cumulative))
            bsm, probs = filter_bitstrings(bsm, probs)

            if bsm.shape[0] < 2:
                print(f"  k={k:2d}  │  {bsm.shape[0]} valid configs — skipping")
                continue

            try:
                e_emb, _, _, spin_sq = timed_solve(bsm)
                energy = float(e_emb) + ecore
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"  k={k:2d}  │  solve_fermion failed: {e}")
                continue

            print_iteration(f"k={k:2d}", energy, bsm.shape[0], prev_energy)
            iterations.append({
                "k"        : k,
                "energy"   : float(energy),
                "n_configs": int(bsm.shape[0]),
                "subspace_dim": subspace_dimension(bsm, n_emb, n_alpha, n_beta),
                "vs_uhf"   : float(energy - uhf_energy),
                "vs_mp2"   : float(energy - mp2_energy),
            })
            prev_energy = energy

        print(f"  {'─'*84}")

        if energy is None:
            raise RuntimeError(
                "SKQD produced no valid energy estimate.\n"
                "Try increasing skqd_shots or skqd_krylov_dim."
            )
        return energy, spin_sq, iterations

    # ── SqDRIFT ───────────────────────────────────────────────────────────────
    def run_sqdrift():
        try:
            from qiskit_fermions.operators.library import FCIDump
            from qiskit_fermions.operators import FermionOperator
            from qiskit_fermions.operators.grouping import (
                group_terms_by_electronic_structure)
            from qiskit_fermions.circuit import FermionicCircuit
            from qiskit_fermions.circuit.library import Evolution
            from qiskit_fermions.transpiler.presets import (
                generate_preset_jw_pass_manager)
            from qiskit_fermions.transpiler.passes import QDriftTrotterization
            from qiskit_fermions.transpiler import FermionicPassManager
        except ImportError:
            raise ImportError(
                "qiskit-fermions required. See INSTALL.md."
            )

        import tempfile
        from pyscf.tools import fcidump as pyscf_fcidump

        print(f"\n── SqDRIFT {'─'*46}")

        t0 = time.perf_counter()
        fd, tmp = tempfile.mkstemp(suffix=".fcidump")
        os.close(fd)
        try:
            pyscf_fcidump.from_integrals(
                tmp, h1e, h2e, n_emb, n_alpha + n_beta,
                ms=abs(n_alpha - n_beta)
            )
            hamil = FermionOperator.from_fcidump(FCIDump.from_file(tmp))
        finally:
            os.unlink(tmp)

        group_terms_by_electronic_structure(hamil, n_qubits)
        evo      = Evolution(n_qubits, hamil, q.sqdrift_time)
        template = FermionicCircuit(n_qubits)
        template.append(evo, template.modes)
        pm       = generate_preset_jw_pass_manager()
        circuits = []
        drift_base = 42 if q.seed is None else int(q.seed) * 100_003

        for i in range(q.sqdrift_num_circuits):
            pm.optimization = FermionicPassManager(
                [QDriftTrotterization(q.sqdrift_num_groups, rng=drift_base + i)]
            )
            transpiled = pm.run(template)
            hf_qc = QuantumCircuit(n_qubits)
            for j in range(n_alpha): hf_qc.x(j)
            for j in range(n_beta):  hf_qc.x(n_emb + j)
            full = hf_qc.compose(transpiled)
            full.measure_all()
            circuits.append(full)

        timings["circuit_build"] += time.perf_counter() - t0
        all_counts = sample_circuits(circuits, q.sqdrift_shots)
        cumulative = Counter()
        for counts in all_counts:
            cumulative.update(counts)
        _b, _p = filter_bitstrings(*counts_to_arrays(dict(cumulative)))
        record_valid(cumulative, _b.shape[0], valid_shot_fraction(cumulative))

        bsm, probs = counts_to_arrays(dict(cumulative))
        bsm, probs = filter_bitstrings(bsm, probs)
        bsm, probs = inject_hf_reference(bsm, probs)
        _check_configs(bsm, probs, "SqDRIFT")
        print(f"  Valid configs: {bsm.shape[0]}")

        return iterative_solve(bsm, probs, q.sqdrift_iters)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    solvers = {
        "sqd"     : run_sqd,
        "skqd"    : run_skqd,
        "sqdrift" : run_sqdrift,
    }

    if cfg.quantum_solver not in solvers:
        raise ValueError(
            f"Unknown quantum_solver: '{cfg.quantum_solver}'. "
            f"Use: {list(solvers.keys())}"
        )

    energy, spin_sq, iterations = solvers[cfg.quantum_solver]()

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 3] Final Summary — {mol_info['molecule']}")
    print(f"{'='*60}")
    if energy is not None:
        print(f"  Solver  : {cfg.quantum_solver.upper()} + "
              f"{q.ansatz.upper()} + {q.fermion_to_qubit.upper()}")
        print(f"  Energy  : {energy:.8f} Ha")
        print(f"  UHF     : {uhf_energy:.8f} Ha  "
              f"(Δ = {energy-uhf_energy:+.6f})")
        print(f"  MP2     : {mp2_energy:.8f} Ha  "
              f"(Δ = {energy-mp2_energy:+.6f})")
        if len(iterations) > 1:
            print(f"  Improve : "
                  f"{iterations[-1]['energy']-iterations[0]['energy']:+.8f} Ha "
                  f"over {len(iterations)} iters")
    if spin_sq is not None:
        print(f"  <S²>    : {spin_sq:.6f}")
    print(f"{'='*60}")

    t_solver = time.perf_counter() - t_start

    # Exact energy of the SAME embedding Hamiltonian: the solver's own
    # limit. Its difference from `energy` is solver error alone, and it
    # must never be above `energy` (variational check).
    t0 = time.perf_counter()
    e_exact = embedding_fci_energy(step2)
    t_exact = time.perf_counter() - t0

    full_dim = full_space_dimension(n_emb, n_alpha, n_beta)
    final_dim = iterations[-1].get("subspace_dim") if iterations else None
    hartree_to_kcal = 627.5094740631

    if e_exact is not None and energy is not None:
        err = (energy - e_exact) * hartree_to_kcal
        print(f"  Embedding FCI : {e_exact:.8f} Ha   solver error "
              f"{err:+.4f} kcal/mol")
        if energy < e_exact - 1e-6:
            warnings.warn(
                f"{cfg.quantum_solver} energy {energy:.8f} is BELOW the exact "
                f"embedding energy {e_exact:.8f}: variational violation, "
                f"something is wrong.", RuntimeWarning)
    if final_dim is not None:
        print(f"  Subspace      : {final_dim} / {full_dim} determinants "
              f"({100. * final_dim / full_dim:.1f}%)")

    shots_pc = stats["shots_per_circuit"]
    output = {
        "solver"    : cfg.quantum_solver,
        "route"     : "dmet",
        "ansatz"    : q.ansatz,
        "mapping"   : q.fermion_to_qubit,
        "backend"   : q.backend,
        "seed"      : q.seed,
        "energy"    : float(energy)  if energy  is not None else None,
        "spin_sq"   : float(spin_sq) if spin_sq is not None else None,
        "uhf_energy": uhf_energy,
        "mp2_energy": mp2_energy,
        "e_exact_embedding": e_exact,
        "iterations": iterations,
        "n_emb"     : int(n_emb),
        "n_alpha"   : int(n_alpha),
        "n_beta"    : int(n_beta),
        "n_qubits"  : int(n_qubits),
        "subspace_dim": final_dim,
        "full_dim"  : int(full_dim),
        "subspace_fraction": (final_dim / full_dim) if final_dim else None,
        "valid_shot_fraction": stats["valid_shot_fraction"],
        "unique_valid_bitstrings": stats["unique_valid_bitstrings"],
        "n_circuits": stats["n_circuits"],
        "shots_per_circuit": shots_pc,
        "shots_total": (shots_pc * stats["n_circuits"]) if shots_pc else None,
        "circuit_depth_max": max(stats["circuit_depths"], default=None),
        "transpiled_depth_max": max(stats["transpiled_depths"], default=None),
        "n_diagonalisations": stats["n_diagonalisations"],
        "timings_s" : {**timings, "solver_total": t_solver,
                       "exact_embedding_fci": t_exact},
        "settings"  : dict(q.__dict__),
        "reproducibility": "stochastic",
        "mol_info"  : mol_info,
        "provenance": cfg.provenance(),
    }

    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\n[Step 3] ✓ Saved → {out_path}")
    return output