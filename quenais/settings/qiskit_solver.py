"""
Qiskit solver-family settings: SQD, SKQD, SqDRIFT.

These carry over unchanged from the 0.1 package. They are grouped here so
the Qiskit and CUDA-Q stacks stay separable -- nothing in this module
imports Qiskit, and nothing in the GQE settings appears here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["QiskitSolverSettings", "ANSATZE", "MAPPINGS", "BACKENDS"]

#: "lucj"     random-angle LUCJ-like circuit (the original DMET+SQD baseline)
#: "lucj_opt" ffsim LUCJ from UCCSD amplitudes, optimised with the linear
#:            method, sampled and diagonalised with the LASSQD machinery
#:            (K x d batches + carryover), the embedding Hamiltonian treated
#:            as one fragment. See quenais/quantum/sqd_opt.py.
ANSATZE = ("su2", "lucj", "lucj_opt")
LUCJ_OPT_PAIRS = ("heavy_hex", "all")
MAPPINGS = ("jw", "bk")
BACKENDS = ("local", "mps", "ibm")


@dataclass
class QiskitSolverSettings:
    """Ansatz, sampling and backend options for the in-process Qiskit path."""

    # ── Ansatz and mapping ───────────────────────────────────────────────
    ansatz: str = "lucj"
    fermion_to_qubit: str = "bk"
    ansatz_reps: int = 3

    # ── Sampling ─────────────────────────────────────────────────────────
    n_shots: int = 8192
    sqd_iters: int = 10

    # ── LUCJ ─────────────────────────────────────────────────────────────
    lucj_num_layers: int = 3
    lucj_random_seed: int = 42
    lucj_regularization: float = 1e-2

    # ── ansatz="lucj_opt" only (defaults = the LASSQD defaults) ──────────
    #: linear-method iterations for the LUCJ parameters
    lucj_opt_maxiter: int = 10
    lucj_opt_pairs: str = "heavy_hex"
    lucj_opt_n_reps: int = 1
    #: SQD batches K and samples per batch d
    sqd_batches: int = 15
    sqd_samples_per_batch: int = 170
    #: configuration-recovery iterations
    sqd_opt_iterations: int = 6
    #: carryover threshold on |c|; 0 disables carryover
    sqd_carryover_eps: float = 1e-5

    # ── SKQD (sample-based Krylov quantum diagonalisation) ───────────────
    skqd_krylov_dim: int = 5
    skqd_dt: float = 0.9
    skqd_trotter_reps: int = 1
    skqd_shots: int = 8192

    # ── SqDRIFT ──────────────────────────────────────────────────────────
    sqdrift_num_circuits: int = 70
    sqdrift_num_groups: int = 100
    sqdrift_time: float = 2.0
    sqdrift_iters: int = 10
    sqdrift_shots: int = 8192

    # ── Seed ─────────────────────────────────────────────────────────────
    #: Seeds the measurement: simulator shot sampling, configuration
    #: recovery and SqDRIFT's random circuit draws. NOT the ansatz angles
    #: (lucj_random_seed), so a seed repeat resamples the same circuit.
    #: None = previous behaviour (fixed recovery seeds, unseeded sampler).
    seed: int | None = None

    # ── Backend ──────────────────────────────────────────────────────────
    backend: str = "mps"
    mps_max_bond_dim: int = 256
    mps_trunc_thresh: float = 1e-6

    ibm_backend_name: str | None = None
    ibm_optimization_level: int = 1
    ibm_max_circuit_depth: int = 3000

    def validate(self) -> "QiskitSolverSettings":
        if self.ansatz not in ANSATZE:
            raise ValueError(f"ansatz must be one of {ANSATZE}, got {self.ansatz!r}")
        if self.fermion_to_qubit not in MAPPINGS:
            raise ValueError(
                f"fermion_to_qubit must be one of {MAPPINGS}, "
                f"got {self.fermion_to_qubit!r}"
            )
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.backend == "ibm" and not self.ibm_backend_name:
            raise ValueError("backend='ibm' requires ibm_backend_name to be set")
        for name in ("n_shots", "skqd_shots", "sqdrift_shots", "sqd_iters",
                     "ansatz_reps", "skqd_krylov_dim", "skqd_trotter_reps",
                     "sqdrift_num_circuits", "sqdrift_num_groups", "sqdrift_iters",
                     "mps_max_bond_dim", "lucj_num_layers",
                     "lucj_opt_maxiter", "lucj_opt_n_reps", "sqd_batches",
                     "sqd_samples_per_batch", "sqd_opt_iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.seed is not None and (int(self.seed) != self.seed or self.seed < 0):
            raise ValueError(f"seed must be a non-negative integer or None, "
                             f"got {self.seed!r}")
        if self.lucj_opt_pairs not in LUCJ_OPT_PAIRS:
            raise ValueError(f"lucj_opt_pairs must be one of {LUCJ_OPT_PAIRS}, "
                             f"got {self.lucj_opt_pairs!r}")
        if self.sqd_carryover_eps < 0:
            raise ValueError("sqd_carryover_eps must be >= 0 (0 disables carryover)")
        if self.mps_trunc_thresh <= 0:
            raise ValueError("mps_trunc_thresh must be > 0")
        if not 0 <= self.ibm_optimization_level <= 3:
            raise ValueError("ibm_optimization_level must lie in [0, 3]")
        return self
