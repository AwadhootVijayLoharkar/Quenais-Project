"""
DMET+SQD with the optimised LUCJ (--solver sqd --ansatz lucj_opt).

The bookkeeping test runs without Qiskit, ffsim or PySCF: it reuses the
exact stand-ins of test_las_sqd_logic (circuit = exact ground state,
sampler = |psi|^2, selected CI = brute force), so everything between the
step-2 pickle and the step-3 energy is production code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from quenais.settings import QiskitSolverSettings
from tests._las_toy import ToyIntegrals, hamiltonian_matrix, sector_states
from tests.test_las_sqd_logic import _patched

ROOT = Path(__file__).resolve().parents[1]


def _toy_step2(n=4, na=2, nb=2, seed=5, ecore=-3.25):
    ints = ToyIntegrals(n, seed=seed)
    c = np.eye(n)
    eri = ints.ao2mo(c, c, c, c).reshape(n, n, n, n)
    return {"h1e": ints.hcore, "h2e": eri, "n_emb": n, "n_alpha": na,
            "n_beta": nb, "ecore": ecore}


def _exact(step2):
    n, na, nb = step2["n_emb"], step2["n_alpha"], step2["n_beta"]
    h1 = step2["h1e"]
    st = sector_states(n, na, nb)
    w = np.linalg.eigvalsh(hamiltonian_matrix([h1, h1], step2["h2e"], n, st))
    return float(w[0]) + step2["ecore"]


def _run(step2, **kw):
    from quenais.quantum.sqd_opt import run_lucj_opt_sqd

    q = QiskitSolverSettings(ansatz="lucj_opt", n_shots=20000, seed=3,
                             sqd_batches=3, sqd_opt_iterations=2, **kw)
    return _patched(lambda: run_lucj_opt_sqd(step2, q, log=lambda *_: None))


def test_full_subspace_reproduces_embedding_fci():
    """Samples span the embedding space: energy must be the exact one,
    with ecore added exactly once."""
    step2 = _toy_step2()
    r = _run(step2, sqd_samples_per_batch=60)
    assert abs(r["energy"] - _exact(step2)) < 1e-8, (r["energy"], _exact(step2))
    assert r["iterations"][-1]["subspace_dim"] == 36          # C(4,2)^2
    assert abs(r["iterations"][-1]["energy"] - r["energy"]) < 1e-6
    assert r["stats"]["n_diagonalisations"] == 3 * 2
    assert r["stats"]["n_circuits"] == 1
    assert 0.99 < r["stats"]["valid_shot_fraction"] <= 1.0


def test_open_shell_embedding():
    step2 = _toy_step2(n=4, na=2, nb=1, seed=9)
    r = _run(step2, sqd_samples_per_batch=60)
    assert abs(r["energy"] - _exact(step2)) < 1e-8


def test_small_batches_stay_variational():
    step2 = _toy_step2(n=5, na=2, nb=2, seed=2)
    r = _run(step2, sqd_samples_per_batch=3, sqd_carryover_eps=0.0)
    assert r["energy"] >= _exact(step2) - 1e-9


def test_las_settings_mapping():
    from quenais.quantum.sqd_opt import las_settings_from_qiskit

    q = QiskitSolverSettings(ansatz="lucj_opt", n_shots=1234, seed=None,
                             backend="mps", lucj_opt_maxiter=7,
                             lucj_opt_pairs="all", sqd_carryover_eps=0.0)
    s = las_settings_from_qiskit(q)
    assert (s.shots, s.seed, s.backend, s.lucj_opt_maxiter, s.lucj_pairs,
            s.carryover_eps, s.lucj_optimize) == (
        1234, 1234, "aer_mps", 7, "all", 0.0, True)
    assert s.mps_max_bond_dim is None


def test_cli_flags_reach_settings(tmp_path):
    from quenais.cli import build_config, build_parser

    args = build_parser().parse_args([
        "--molecule", "LiH", "--solver", "sqd", "--ansatz", "lucj_opt",
        "--shots", "1000", "--seed", "2", "--sqd-lucj-maxiter", "20",
        "--sqd-batches", "8", "--sqd-samples-per-batch", "100",
        "--sqd-iterations", "4", "--sqd-carryover-eps", "0",
        "--sqd-lucj-pairs", "all", "--sqd-lucj-reps", "2",
        "--project-dir", str(tmp_path)])
    q = build_config(args).qiskit
    assert (q.ansatz, q.lucj_opt_maxiter, q.sqd_batches, q.sqd_samples_per_batch,
            q.sqd_opt_iterations, q.sqd_carryover_eps, q.lucj_opt_pairs,
            q.lucj_opt_n_reps) == ("lucj_opt", 20, 8, 100, 4, 0.0, "all", 2)


def test_driver_keeps_sqd_flags_only_for_lucj_opt():
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", ROOT / "tools" / "run_benchmark.py")
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    p, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj_opt",
                               "sqd_batches": 5, "seed": 1})
    assert p["sqd_batches"] == 5 and p["seed"] == 1
    p, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj", "sqd_batches": 5})
    assert "sqd_batches" not in p
    p, _ = d.normalise_params({"solver": "gqe", "sqd_batches": 5})
    assert "sqd_batches" not in p


def test_lucj_cache_reused_across_seeds(tmp_path):
    """Second run with another seed must reuse the optimised circuit (no
    second _prepare) and still give the exact energy."""
    import os

    from quenais.las import sqd_solver
    from quenais.quantum import sqd_opt
    from tests import test_las_sqd_logic as lsl

    os.environ["QUENAIS_LUCJ_CACHE"] = str(tmp_path / "cache")
    calls = []

    def counting_prepare(self, ham):
        calls.append(1)
        prep = lsl._fake_prepare(self, ham)
        prep["op"] = prep["circuit"]     # stand-in: the "operator" IS the circuit
        return prep

    orig_fc = sqd_solver._fragment_circuit
    sqd_solver._fragment_circuit = lambda n, nelec, op: op
    try:
        step2 = _toy_step2(seed=4)
        energies = []
        for seed in (1, 2):
            q = QiskitSolverSettings(ansatz="lucj_opt", n_shots=20000, seed=seed,
                                     sqd_batches=3, sqd_opt_iterations=2,
                                     sqd_samples_per_batch=60)

            def go():
                sqd_solver.LassqdFragmentSolver._prepare = counting_prepare
                return sqd_opt.run_lucj_opt_sqd(step2, q, log=lambda *_: None)
            r = lsl._patched(go)
            energies.append(r["energy"])
            assert r["lucj_opt"]["lucj_cache_hit"] is (seed == 2)
    finally:
        sqd_solver._fragment_circuit = orig_fc
        os.environ.pop("QUENAIS_LUCJ_CACHE", None)
    assert len(calls) == 1
    for e in energies:
        assert abs(e - _exact(step2)) < 1e-8


def test_cache_key_depends_on_hamiltonian_and_settings():
    from quenais.quantum.sqd_opt import (embedding_fragment, las_settings_from_qiskit,
                                         lucj_cache_key)

    a, b = _toy_step2(seed=1), _toy_step2(seed=2)
    s1 = las_settings_from_qiskit(QiskitSolverSettings(ansatz="lucj_opt", seed=1))
    s2 = las_settings_from_qiskit(QiskitSolverSettings(ansatz="lucj_opt", seed=7))
    s3 = las_settings_from_qiskit(QiskitSolverSettings(ansatz="lucj_opt",
                                                       lucj_opt_pairs="all"))
    ka = lucj_cache_key(embedding_fragment(a), s1)
    assert ka == lucj_cache_key(embedding_fragment(a), s2)      # seed-independent
    assert ka != lucj_cache_key(embedding_fragment(b), s1)
    assert ka != lucj_cache_key(embedding_fragment(a), s3)
