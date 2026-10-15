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

SUBSPACE (q.sqd_subspace)
-------------------------
"product" (default) alpha x beta product of sampled strings, K batches
"list"    only the sampled (alpha, beta) determinants, one list per
          recovery iteration (quenais/quantum/sqd_list.py); comparable
          with CIPSI at the same determinant count

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

CACHE
-----
Mean field, UCCSD and the LUCJ optimisation depend only on the embedding
Hamiltonian and the LUCJ settings -- not on shots or seed. On H8 with
all pairs they took most of ~930 s per run, and every seed/shot repeat
redid them. The optimised operator is therefore cached on disk, keyed on
a hash of (h1e, h2e, nelec, LUCJ settings, ffsim version):

    $QUENAIS_LUCJ_CACHE  (default ~/.cache/quenais/lucj)

A seed repeat then resamples the SAME circuit, which is the intended
meaning of a seed (see solver.py). Delete the folder to force
re-optimisation. result["lucj_opt"]["lucj_cache_hit"] records which
happened.
"""

from __future__ import annotations

import os
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
        lucj_optimize=int(q.lucj_opt_maxiter) > 0,
        lucj_opt_maxiter=max(1, int(q.lucj_opt_maxiter)),
        sqd_iterations=int(q.sqd_opt_iterations),
        sqd_batches=int(q.sqd_batches),
        sqd_samples_per_batch=int(q.sqd_samples_per_batch),
        carryover_eps=float(q.sqd_carryover_eps),
        seed=1234 if q.seed is None else int(q.seed),
    )


#: Above this many qubits the statevector sampler needs > 4 GB per state
#: (2**28 x 16 B); below it, it is far faster than an exact MPS for deep,
#: entangling LUCJ circuits (N2 cc-pVDZ, 26 qubits: MPS never finished one
#: 1 068-CX circuit, statevector took minutes).
STATEVECTOR_MAX_QUBITS = 28


def choose_sampler(backend, n_qubits):
    """'mps' means 'automatic': statevector up to STATEVECTOR_MAX_QUBITS,
    exact MPS above. 'local' forces statevector, 'ibm' is hardware."""
    if backend == "mps":
        return "aer_statevector" if n_qubits <= STATEVECTOR_MAX_QUBITS else "aer_mps"
    return _BACKENDS[backend]


def _n_lucj_params(n, n_reps, pairs_kind):
    """Upper bound on UCJOpSpinUnbalanced parameters (final rotation on)."""
    try:
        import ffsim

        from quenais.las import sqd_solver

        pairs = sqd_solver._interaction_pairs(n, pairs_kind)
        return int(ffsim.UCJOpSpinUnbalanced.n_params(
            n, n_reps, interaction_pairs=pairs, with_final_orbital_rotation=True))
    except Exception:
        if pairs_kind == "all":
            npair = 3 * n * (n + 1) // 2
        else:
            npair = 2 * (n - 1) + len(range(0, n, 4))
        return n_reps * npair + 2 * n * n * (n_reps + 1)


def lucj_optimisation_memory_gb(n, nelec, n_reps, pairs_kind):
    """
    Peak memory of ffsim's linear method: it keeps the derivative of the
    state w.r.t. every parameter, i.e. ~(n_params + 4) complex vectors of
    the full (n_alpha, n_beta) sector.
    """
    from math import comb

    dim = comb(n, nelec[0]) * comb(n, nelec[1])
    return (_n_lucj_params(n, n_reps, pairs_kind) + 4) * dim * 16 / 1e9


def available_memory_gb():
    """Memory this process may use: $QUENAIS_MEM_GB, else the cgroup limit
    (a Slurm allocation), else physical RAM. None if unknown."""
    env = os.environ.get("QUENAIS_MEM_GB")
    if env:
        return float(env)
    limits = []
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            txt = open(path).read().strip()
            if txt.isdigit():
                limits.append(int(txt) / 1e9)
        except OSError:
            pass
    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9)
    except (ValueError, OSError, AttributeError):
        pass
    limits = [x for x in limits if 0 < x < 1e6]
    return min(limits) if limits else None


def plan_lucj_optimisation(s, n, nelec, log, fraction=0.7):
    """
    (settings, info): optimisation switched off, with a warning, when the
    linear method would need more than `fraction` of the available memory
    -- N2 cc-pVDZ (26 qubits) was killed for memory after 39 min instead.
    """
    import dataclasses

    need = lucj_optimisation_memory_gb(n, nelec, s.lucj_n_reps, s.lucj_pairs)
    have = available_memory_gb()
    info = {"lucj_opt_memory_gb": need, "available_memory_gb": have,
            "lucj_opt_skipped_for_memory": False}
    if s.lucj_optimize and have is not None and need > fraction * have:
        log(f"  WARNING: LUCJ optimisation needs ~{need:.1f} GB, only "
            f"{have:.1f} GB available -> using the CCSD-initialised circuit "
            f"WITHOUT optimisation (set QUENAIS_MEM_GB to override)")
        info["lucj_opt_skipped_for_memory"] = True
        s = dataclasses.replace(s, lucj_optimize=False)
    return s, info


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


def _cache_dir():
    import os

    return os.path.expanduser(os.environ.get("QUENAIS_LUCJ_CACHE",
                                             "~/.cache/quenais/lucj"))


def lucj_cache_key(ham, s):
    """Hash of everything the LUCJ preparation depends on."""
    import hashlib

    try:
        import ffsim
        ver = getattr(ffsim, "__version__", "?")
    except ImportError:
        ver = "none"
    h = hashlib.sha256()
    for arr in (ham.h1s, ham.eri):
        a = np.ascontiguousarray(np.round(np.asarray(arr, dtype=float), 12))
        h.update(a.tobytes())
        h.update(str(a.shape).encode())
    h.update(repr((tuple(ham.nelec), int(ham.spin2s), s.lucj_pairs,
                   int(s.lucj_n_reps), bool(s.lucj_optimize),
                   int(s.lucj_opt_maxiter), ver, "v1")).encode())
    return h.hexdigest()[:24]


def load_cached_prep(key):
    import pickle

    path = os.path.join(_cache_dir(), f"{key}.pkl")
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:            # missing, partial or unreadable: recompute
        return None


def save_cached_prep(key, prep):
    """Store everything but the circuit (rebuilt from 'op'). Never raises."""
    import pickle

    try:
        d = _cache_dir()
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".{key}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({k: v for k, v in prep.items() if k != "circuit"}, fh)
        os.replace(tmp, os.path.join(d, f"{key}.pkl"))
        return True
    except Exception:
        return False


def _prepare_cached(solver, ham, s, log):
    """solver._prepare(ham), reusing a cached optimised LUCJ if present."""
    from quenais.las import sqd_solver

    key = lucj_cache_key(ham, s)
    cached = load_cached_prep(key)
    if cached is not None and cached.get("op") is not None:
        try:
            cached["circuit"] = sqd_solver._fragment_circuit(
                ham.norb, ham.nelec, cached["op"])
            log(f"  LUCJ: reusing cached optimised circuit ({key})")
            return cached, True
        except Exception as exc:
            log(f"  LUCJ cache unusable ({exc}); re-optimising")
    prep = solver._prepare(ham)
    if prep.get("op") is not None and save_cached_prep(key, prep):
        log(f"  LUCJ: optimised circuit cached ({key})")
    return prep, False


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


def _exploration_circuit(n, nelec, s, scale, seed=12345):
    """
    Same LUCJ form as the optimised circuit, random parameters ~ N(0, scale),
    fixed seed (a seed repeat resamples the same circuit, as for the main
    one). Built in the same orbital basis, so its bitstrings mean the same
    determinants.
    """
    import ffsim

    from quenais.las import sqd_solver

    pairs = sqd_solver._interaction_pairs(n, s.lucj_pairs)
    npar = ffsim.UCJOpSpinUnbalanced.n_params(
        n, s.lucj_n_reps, interaction_pairs=pairs,
        with_final_orbital_rotation=False)
    x = np.random.default_rng(seed).normal(0.0, scale, size=npar)
    op = ffsim.UCJOpSpinUnbalanced.from_parameters(
        x, norb=n, n_reps=s.lucj_n_reps, interaction_pairs=pairs,
        with_final_orbital_rotation=False)
    return sqd_solver._fragment_circuit(n, nelec, op)


def _valid_keys(counts, n, na, nb):
    """Distinct bitstrings with the right electron numbers (Qiskit order:
    left half beta, right half alpha)."""
    out = set()
    for key in counts:
        k = key.replace(" ", "")
        if k[:n].count("1") == nb and k[n:].count("1") == na:
            out.add(k)
    return out


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

    import dataclasses

    s = las_settings_from_qiskit(q)
    ham = embedding_fragment(step2)
    n, nelec = ham.norb, ham.nelec
    ecore = ham.const
    s = dataclasses.replace(s, backend=choose_sampler(q.backend, 2 * n))
    s, mem_info = plan_lucj_optimisation(s, n, nelec, log)
    solver = sqd_solver.LassqdFragmentSolver(s, log=log)

    log(f"  LUCJ (optimised): pairs={s.lucj_pairs}, reps={s.lucj_n_reps}, "
        f"linear-method iterations={s.lucj_opt_maxiter}")
    log(f"  SQD: K={s.sqd_batches} batches x d={s.sqd_samples_per_batch}, "
        f"{s.sqd_iterations} iterations, carryover eps={s.carryover_eps}, "
        f"shots={s.shots}, sampler={s.backend}, seed={s.seed}")

    t0 = time.perf_counter()
    prep, cache_hit = _prepare_cached(solver, ham, s, log)
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
    f = float(getattr(q, "sqd_explore_fraction", 0.0) or 0.0)
    n_explore = int(round(f * s.shots)) if f > 0 else 0
    n_main = s.shots - n_explore
    import dataclasses

    counts = sqd_solver.sample_fragment_circuits(
        [prep["circuit"]], dataclasses.replace(s, shots=n_main))[0]
    na, nb = nelec
    keys_main = _valid_keys(counts, n, na, nb)
    keys_explore = set()
    if n_explore:
        circ = _exploration_circuit(n, nelec, s, float(q.sqd_explore_scale))
        c_exp = sqd_solver.sample_fragment_circuits(
            [circ], dataclasses.replace(s, shots=n_explore))[0]
        keys_explore = _valid_keys(c_exp, n, na, nb)
        merged = dict(counts)
        for k, v in c_exp.items():
            merged[k] = merged.get(k, 0) + v
        counts = merged
        log(f"  exploration: {n_explore} of {s.shots} shots, scale "
            f"{q.sqd_explore_scale}; distinct valid bitstrings main "
            f"{len(keys_main)}, exploration {len(keys_explore)}, new "
            f"{len(keys_explore - keys_main)}")
    t_sample = time.perf_counter() - t0

    mode = getattr(q, "sqd_subspace", "product") or "product"
    t0 = time.perf_counter()
    if mode == "list":
        from quenais.quantum.sqd_list import run_list_sqd

        log("  SQD subspace: determinant LIST (sampled (alpha, beta) pairs, "
            "no product, no batches)")
        info = run_list_sqd(prep["h1_mo"], prep["eri_mo"], n, nelec, counts,
                            s, solver.rng, log=log)
        e_sub = info["energy"]
        n_diag = s.sqd_iterations
    else:
        res = solver._sqd(ham, prep, counts)
        info = res.info
        e_sub = res.energy
        n_diag = s.sqd_iterations * s.sqd_batches
    t_diag = time.perf_counter() - t0

    iterations = [{
        "iter": i + 1,
        "energy": float(it["energy"]) + ecore,
        "e_emb": float(it["energy"]),
        "ecore": ecore,
        "subspace_dim": int(it["dim"]),
        "n_carry": list(it["n_carry"]),
        **({"n_sampled": int(it["n_sampled"])} if "n_sampled" in it else {}),
    } for i, it in enumerate(info["iterations"])]
    energy = float(e_sub) + ecore

    log(f"  SQD dim {info['subspace_dim']} / {info['full_dim']} "
        f"({100 * info['subspace_fraction']:.1f}%), valid shots "
        f"{100 * info['shots_valid_fraction']:.1f}% "
        f"({info['unique_valid_bitstrings']} distinct)")

    return {
        "energy": energy,
        "spin_sq": info.get("spin_sq"),
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
            "n_diagonalisations": n_diag,
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
            "lucj_cache_hit": cache_hit,
            "sampler": s.backend,
            **mem_info,
            "explore_fraction": f,
            "explore_scale": float(q.sqd_explore_scale) if n_explore else None,
            "explore_shots": n_explore,
            "unique_valid_main": len(keys_main),
            "unique_valid_explore_new": len(keys_explore - keys_main),
            "sqd_subspace": mode,
            "list_n_sampled_dets": info.get("n_sampled_dets"),
        },
    }
