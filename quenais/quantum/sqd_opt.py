"""
DMET+SQD with an optimised LUCJ circuit (--solver sqd --ansatz lucj_opt).

WHY
---
The original DMET+SQD samples a random-angle LUCJ-like circuit
(solver._build_lucj_manual). On N2 it ends 12-42 kcal/mol above the
embedding FCI. LASSQD had the same problem with its CCSD-initialised LUCJ
(~1.5 % of the correlation kept) and went 42 -> 0 kcal/mol on H8 once the
LUCJ parameters were optimised. This module gives DMET+SQD that same
circuit and the same SQD.

HOW
---
The step-2 embedding Hamiltonian is handed to the LASSQD fragment solver
(quenais.las.sqd_solver.LassqdFragmentSolver) as ONE fragment:

    h1s = [h1e, h1e]   eri = h2e   nelec = (n_alpha, n_beta)
    2S  = n_alpha - n_beta         const = ecore

and the solver does what it does for every LASSQD fragment:

  1. mean field on the embedding Hamiltonian -> sampling basis
  2. LUCJ from UCCSD amplitudes, optimised with ffsim's linear method
  3. Qiskit/Aer sampling of the circuit
  4. SQD: K batches x d samples, configuration recovery, carryover

Energy = SQD energy + ecore. Nothing here duplicates LASSQD code; this
file only adapts inputs and outputs to the step-3 result format, so the
benchmark driver writes the same CSV columns as for the other Qiskit
solvers.

EXTRA DIAGNOSTICS (in result["lucj_opt"])
-----------------------------------------
lucj_energy        <psi_LUCJ|H|psi_LUCJ> + ecore (statevector, exact)
lucj_off_hf_weight 1 - |<HF|psi_LUCJ>|^2 in the sampling basis
cx_depth, n_2q     depth and two-qubit gate count after transpiling to
                   {cx, rz, sx, x} (all-to-all, no device layout). The Aer
                   simulator accepts composite gates, so its own
                   transpiled depth says nothing about hardware cost.

If lucj_energy is already within 1 kcal/mol of the embedding FCI, the
circuit alone (a VQE) is accurate and SQD is not doing the work: report
both numbers.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["run_lucj_opt_sqd", "las_settings_from_qiskit", "embedding_fragment"]

_BACKENDS = {"mps": "aer_mps", "local": "aer_statevector", "ibm": "ibm"}


def las_settings_from_qiskit(q):
    """LasSettings for the fragment solver from cfg.qiskit (lucj_opt fields)."""
    from quenais.settings import LasSettings

    if q.backend not in _BACKENDS:
        raise ValueError(f"lucj_opt: unsupported backend {q.backend!r}")
    return LasSettings(
        shots=int(q.n_shots),
        # None = exact MPS. The embedding is <= ~24 qubits, where the exact
        # MPS is cheap; truncating it would change what is sampled.
        backend=_BACKENDS[q.backend],
        mps_max_bond_dim=None,
        ibm_backend_name=q.ibm_backend_name,
        ibm_optimization_level=q.ibm_optimization_level,
        lucj_n_reps=int(q.lucj_opt_n_reps),
        lucj_pairs=q.lucj_opt_pairs,
        lucj_optimize=True,
        lucj_opt_maxiter=int(q.lucj_opt_maxiter),
        sqd_iterations=int(q.sqd_opt_iterations),
        sqd_batches=int(q.sqd_batches),
        sqd_samples_per_batch=int(q.sqd_samples_per_batch),
        carryover_eps=float(q.sqd_carryover_eps),
        seed=1234 if q.seed is None else int(q.seed),
    )


def embedding_fragment(step2):
    """The step-2 embedding Hamiltonian as a single LAS fragment."""
    from quenais.las.energy import FragmentHamiltonian

    h1 = np.asarray(step2["h1e"], dtype=float)
    n = h1.shape[0]
    eri = np.asarray(step2["h2e"], dtype=float)
    if eri.ndim != 4:            # tolerate a packed eri
        from pyscf import ao2mo

        eri = ao2mo.restore(1, eri, n)
    na, nb = int(step2["n_alpha"]), int(step2["n_beta"])
    return FragmentHamiltonian(index=0, h1s=np.stack([h1, h1]), eri=eri,
                               const=float(step2["ecore"]), nelec=(na, nb),
                               spin2s=na - nb)


def _lucj_diagnostics(prep, n, nelec):
    """LUCJ state energy (without ecore) and weight off HF; None on failure."""
    op = prep.get("op")
    if op is None:
        return None, None
    try:
        import ffsim

        h1 = 0.5 * (prep["h1_mo"][0] + prep["h1_mo"][1])
        ham = ffsim.MolecularHamiltonian(h1, prep["eri_mo"], 0.0)
        linop = ffsim.linear_operator(ham, norb=n, nelec=nelec)
        hf = ffsim.hartree_fock_state(n, nelec)
        vec = ffsim.apply_unitary(hf, op, norb=n, nelec=nelec)
        e = float(np.real(np.vdot(vec, linop @ vec)))
        w_off = float(1.0 - abs(np.vdot(hf, vec)) ** 2)
        return e, w_off
    except Exception:            # a diagnostic must never fail the run
        return None, None


def _hardware_cost(circuit):
    """(depth, two-qubit gate count) in the {cx, rz, sx, x} basis."""
    try:
        import ffsim
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )

        pm = generate_preset_pass_manager(optimization_level=1,
                                          basis_gates=["cx", "rz", "sx", "x"])
        pm.pre_init = ffsim.qiskit.PRE_INIT
        tqc = pm.run(circuit)
        return int(tqc.depth()), int(tqc.count_ops().get("cx", 0))
    except Exception:
        return None, None


def run_lucj_opt_sqd(step2, q, log=print):
    """
    Optimised-LUCJ SQD on the embedding Hamiltonian.

    Returns a dict: energy (total, incl. ecore), iterations (step-3 format),
    timings, stats (same keys as solver.main's), lucj_opt (diagnostics).
    """
    from quenais.las import sqd_solver

    s = las_settings_from_qiskit(q)
    ham = embedding_fragment(step2)
    n, nelec = ham.norb, ham.nelec
    ecore = ham.const
    solver = sqd_solver.LassqdFragmentSolver(s, log=log)

    log(f"  LUCJ (optimised): pairs={s.lucj_pairs}, reps={s.lucj_n_reps}, "
        f"linear-method iterations={s.lucj_opt_maxiter}")
    log(f"  SQD: K={s.sqd_batches} batches x d={s.sqd_samples_per_batch}, "
        f"{s.sqd_iterations} iterations, carryover eps={s.carryover_eps}, "
        f"shots={s.shots}, sampler={s.backend}, seed={s.seed}")

    t0 = time.perf_counter()
    prep = solver._prepare(ham)
    t_build = time.perf_counter() - t0

    e_lucj, w_off = _lucj_diagnostics(prep, n, nelec)
    if e_lucj is not None:
        log(f"  LUCJ state: E = {e_lucj + ecore:.8f} Ha, weight off HF = {w_off:.4f}")

    t0 = time.perf_counter()
    depth, n2q = _hardware_cost(prep["circuit"])
    t_cost = time.perf_counter() - t0
    if depth is not None:
        log(f"  circuit ({{cx,rz,sx,x}} basis): depth {depth}, {n2q} CX")

    t0 = time.perf_counter()
    counts = sqd_solver.sample_fragment_circuits([prep["circuit"]], s)[0]
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = solver._sqd(ham, prep, counts)
    t_diag = time.perf_counter() - t0

    info = res.info
    iterations = [{
        "iter": i + 1,
        "energy": float(it["energy"]) + ecore,
        "e_emb": float(it["energy"]),
        "ecore": ecore,
        "subspace_dim": int(it["dim"]),
        "n_carry": list(it["n_carry"]),
    } for i, it in enumerate(info["iterations"])]
    energy = float(res.energy) + ecore

    log(f"  SQD dim {info['subspace_dim']} / {info['full_dim']} "
        f"({100 * info['subspace_fraction']:.1f}%), valid shots "
        f"{100 * info['shots_valid_fraction']:.1f}% "
        f"({info['unique_valid_bitstrings']} distinct)")

    return {
        "energy": energy,
        "iterations": iterations,
        "timings": {
            # circuit_build includes mean field, UCCSD and the LUCJ
            # optimisation; recovery is inside t_diag (not separable).
            "circuit_build": t_build,
            "sampling": t_sample,
            "recovery": 0.0,
            "diagonalisation": t_diag,
            "hardware_cost_estimate": t_cost,
        },
        "stats": {
            "n_diagonalisations": s.sqd_iterations * s.sqd_batches,
            "circuit_depths": [depth] if depth is not None else [],
            "transpiled_depths": [depth] if depth is not None else [],
            "valid_shot_fraction": float(info["shots_valid_fraction"]),
            "unique_valid_bitstrings": int(info["unique_valid_bitstrings"]),
            "n_circuits": 1,
            "shots_per_circuit": int(s.shots),
        },
        "lucj_opt": {
            "lucj_energy": (e_lucj + ecore) if e_lucj is not None else None,
            "lucj_off_hf_weight": w_off,
            "cx_depth": depth,
            "n_2q_gates": n2q,
            "lucj_pairs": s.lucj_pairs,
            "lucj_n_reps": s.lucj_n_reps,
            "lucj_opt_maxiter": s.lucj_opt_maxiter,
            "sqd_batches": s.sqd_batches,
            "sqd_samples_per_batch": s.sqd_samples_per_batch,
            "sqd_iterations": s.sqd_iterations,
            "carryover_eps": s.carryover_eps,
            "seed": s.seed,
            "t_prepare_s": t_build,
        },
    }
