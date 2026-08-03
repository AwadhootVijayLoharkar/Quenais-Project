"""
Qiskit solver-family settings: SQD, SKQD, SqDRIFT.

These carry over unchanged from the 0.1 package. They are grouped here so
the Qiskit and CUDA-Q stacks stay separable -- nothing in this module
imports Qiskit, and nothing in the GQE settings appears here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["QiskitSolverSettings", "ANSATZE", "MAPPINGS", "BACKENDS"]

ANSATZE = ("su2", "lucj")
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
                     "mps_max_bond_dim", "lucj_num_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.mps_trunc_thresh <= 0:
            raise ValueError("mps_trunc_thresh must be > 0")
        if not 0 <= self.ibm_optimization_level <= 3:
            raise ValueError("ibm_optimization_level must lie in [0, 3]")
        return self
