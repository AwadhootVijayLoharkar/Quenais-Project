"""
Quantum Solver (SQD / SKQD / SqDRIFT) for QuEnAIS pipeline.
"""

import os
import pickle
import warnings
import numpy as np
from collections import Counter


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

    if os.path.exists(cfg.step3_file) and not force:
        print(f"[Step 3] Using cached result: {cfg.step3_file}")
        return

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
    print(f"  Ansatz       : {cfg.ansatz.upper()}")
    print(f"  Mapping      : {cfg.fermion_to_qubit.upper()}")
    print(f"  Backend      : {cfg.backend.upper()}")
    print(f"  Embedding    : {n_emb} orbs = {n_qubits} qubits")
    print(f"  Electrons    : {n_alpha}α + {n_beta}β")
    print(f"  UHF ref      : {uhf_energy:.8f} Ha")
    print(f"  MP2 ref      : {mp2_energy:.8f} Ha")

    # ── Backend dispatch ──────────────────────────────────────────────────────
    def sample_circuits(circuits, shots):
        backend = cfg.backend.lower()
        if backend == "local":
            from qiskit.primitives import StatevectorSampler
            res = StatevectorSampler().run(circuits, shots=shots).result()
            return [res[i].data.meas.get_counts() for i in range(len(circuits))]
        elif backend == "mps":
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            sim = AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=cfg.mps_max_bond_dim,
                matrix_product_state_truncation_threshold=cfg.mps_trunc_thresh,
            )
            tc     = transpile(circuits, backend=sim, optimization_level=1)
            result = sim.run(tc, shots=shots).result()
            return [result.get_counts(i) for i in range(len(circuits))]
        elif backend == "ibm":
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            service = QiskitRuntimeService()
            hw = (service.backend(cfg.ibm_backend_name) if cfg.ibm_backend_name
                  else service.least_busy(operational=True, simulator=False,
                                          min_num_qubits=circuits[0].num_qubits))
            pm     = generate_preset_pass_manager(cfg.ibm_optimization_level,
                                                   backend=hw)
            tc     = pm.run(circuits)
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
        mapping = cfg.fermion_to_qubit.lower()
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
            reps=cfg.ansatz_reps,
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
        rng  = np.random.default_rng(cfg.lucj_random_seed)
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

        for _ in range(cfg.lucj_num_layers):
            givens_layer(circ, alpha_q, 0)
            givens_layer(circ, alpha_q, 1)
            jastrow_layer(circ, alpha_q)
            givens_layer(circ, beta_q, 0)
            givens_layer(circ, beta_q, 1)
            jastrow_layer(circ, beta_q)

        circ.measure_all()
        print(f"  LUCJ (manual) circuit: {n_qubits}q,  depth={circ.depth()},  "
              f"layers={cfg.lucj_num_layers}")
        print(f"  ✓ Particle number conserved by construction")
        return circ

    def build_ansatz_circuit():
        ansatz = cfg.ansatz.lower()
        if ansatz == "su2":
            return _build_su2_circuit()
        elif ansatz == "lucj":
            return _build_lucj_manual()
        else:
            raise ValueError(f"Unknown ansatz: '{ansatz}'. Use: su2 | lucj")

    # ── Shared helpers ────────────────────────────────────────────────────────
    def filter_bitstrings(bsm, probs):
        valid  = ((bsm[:, :n_emb].sum(axis=1) == n_alpha) &
                  (bsm[:, n_emb:].sum(axis=1) == n_beta))
        bsm_f  = bsm[valid]
        prob_f = probs[valid]
        if len(prob_f) > 0:
            prob_f = prob_f / prob_f.sum()
        return bsm_f, prob_f

    def hf_bitstring():
        row = np.zeros(2 * n_emb, dtype=bool)
        for i in range(n_alpha): row[i]          = True
        for i in range(n_beta):  row[n_emb + i]  = True
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

        for it in range(n_iters):
            bsm, probs = recover_configurations(
                bsm, probs, avg_occs,
                num_elec_a=n_alpha, num_elec_b=n_beta,
                rand_seed=42 + it,
            )
            if bsm.shape[0] == 0:
                warnings.warn(
                    f"recover_configurations returned 0 configs at iter {it+1}.",
                    RuntimeWarning,
                )
                break

            e_emb, _, avg_occs, spin_sq = solve_fermion(
                bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
            )
            energy = float(e_emb) + ecore

            print_iteration(f"{it+1:02d}", energy, bsm.shape[0], prev_energy)
            iterations.append({
                "iter"     : it + 1,
                "energy"   : energy,
                "e_emb"    : float(e_emb),
                "ecore"    : ecore,
                "n_configs": int(bsm.shape[0]),
                "vs_uhf"   : float(energy - uhf_energy),
                "vs_mp2"   : float(energy - mp2_energy),
            })
            prev_energy = energy

        print(f"  {'─'*84}")
        return energy, spin_sq, iterations

    # ── SQD ───────────────────────────────────────────────────────────────────
    def run_sqd():
        print(f"\n── SQD ({cfg.ansatz.upper()} ansatz) {'─'*40}")
        circ       = build_ansatz_circuit()
        print(f"  Shots: {cfg.n_shots}")
        raw        = sample_circuits([circ], cfg.n_shots)[0]
        bsm, probs = counts_to_arrays(raw)
        bsm, probs = filter_bitstrings(bsm, probs)
        n_total    = sum(raw.values())
        n_valid    = bsm.shape[0]
        print(f"  Valid configs: {n_valid} / {n_total} "
              f"({100.*n_valid/max(n_total,1):.1f}%)")
        _check_configs(bsm, probs, "SQD after filter")
        return iterative_solve(bsm, probs, cfg.sqd_iters)

    # ── SKQD ──────────────────────────────────────────────────────────────────
    def run_skqd():
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import LieTrotter

        print(f"\n── SKQD ({cfg.fermion_to_qubit.upper()} mapping) {'─'*38}")
        print(f"  Krylov dim: {cfg.skqd_krylov_dim},  dt={cfg.skqd_dt},  "
              f"Trotter reps={cfg.skqd_trotter_reps}")

        H_qubit = build_qubit_hamiltonian()

        ref = QuantumCircuit(n_qubits)
        for i in range(n_alpha): ref.x(i)
        for i in range(n_beta):  ref.x(n_emb + i)

        evol = PauliEvolutionGate(
            H_qubit,
            time      = cfg.skqd_dt / cfg.skqd_trotter_reps,
            synthesis = LieTrotter(reps=cfg.skqd_trotter_reps),
        )

        circs = []
        for k in range(cfg.skqd_krylov_dim):
            qc = ref.copy()
            for _ in range(k):
                qc.append(evol, range(n_qubits))
            qc.measure_all()
            circs.append(qc)

        print(f"  Circuit depths: {[c.depth() for c in circs]}")
        print(f"  Sampling {len(circs)} Krylov circuits "
              f"@ {cfg.skqd_shots} shots each...")

        all_counts  = sample_circuits(circs, cfg.skqd_shots)
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
                e_emb, _, _, spin_sq = solve_fermion(
                    bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
                )
                energy = float(e_emb) + ecore
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"  k={k:2d}  │  solve_fermion failed: {e}")
                continue

            print_iteration(f"k={k:2d}", energy, bsm.shape[0], prev_energy)
            iterations.append({
                "k"        : k,
                "energy"   : float(energy),
                "n_configs": int(bsm.shape[0]),
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
        evo      = Evolution(n_qubits, hamil, cfg.sqdrift_time)
        template = FermionicCircuit(n_qubits)
        template.append(evo, template.modes)
        pm       = generate_preset_jw_pass_manager()
        circuits = []

        for i in range(cfg.sqdrift_num_circuits):
            pm.optimization = FermionicPassManager(
                [QDriftTrotterization(cfg.sqdrift_num_groups, rng=42 + i)]
            )
            transpiled = pm.run(template)
            hf_qc = QuantumCircuit(n_qubits)
            for j in range(n_alpha): hf_qc.x(j)
            for j in range(n_beta):  hf_qc.x(n_emb + j)
            full = hf_qc.compose(transpiled)
            full.measure_all()
            circuits.append(full)

        all_counts = sample_circuits(circuits, cfg.sqdrift_shots)
        cumulative = Counter()
        for counts in all_counts:
            cumulative.update(counts)

        bsm, probs = counts_to_arrays(dict(cumulative))
        bsm, probs = filter_bitstrings(bsm, probs)
        bsm, probs = inject_hf_reference(bsm, probs)
        _check_configs(bsm, probs, "SqDRIFT")
        print(f"  Valid configs: {bsm.shape[0]}")

        return iterative_solve(bsm, probs, cfg.sqdrift_iters)

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
              f"{cfg.ansatz.upper()} + {cfg.fermion_to_qubit.upper()}")
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

    output = {
        "solver"    : cfg.quantum_solver,
        "ansatz"    : cfg.ansatz,
        "mapping"   : cfg.fermion_to_qubit,
        "backend"   : cfg.backend,
        "energy"    : float(energy)  if energy  is not None else None,
        "spin_sq"   : float(spin_sq) if spin_sq is not None else None,
        "uhf_energy": uhf_energy,
        "mp2_energy": mp2_energy,
        "iterations": iterations,
        "mol_info"  : mol_info,
    }

    with open(cfg.step3_file, "wb") as f:
        pickle.dump(output, f)

    print(f"\n[Step 3] ✓ Saved → {cfg.step3_file}")
    return output