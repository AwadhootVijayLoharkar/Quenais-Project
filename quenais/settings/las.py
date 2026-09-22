"""
Settings for the LAS solver family: "lasscf" (exact fragments) and
"lassqd" (quantum-sampled fragments). Nothing here imports PySCF or Qiskit.

Defaults for the SQD part follow Wang et al., PNAS 2026 (K = 15 batches,
d = 170 samples per batch, 6 recovery iterations, carryover, spin penalty
lambda = 0.2, energy convergence 1e-5 Eh over two consecutive cycles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LasSettings", "LAS_BACKENDS", "LUCJ_PAIRS"]

LAS_BACKENDS = ("aer_mps", "aer_statevector", "ibm")
LUCJ_PAIRS = ("heavy_hex", "all")


@dataclass
class LasSettings:
    # ── Fragments ────────────────────────────────────────────────────────
    #: One spec per fragment, 'ATOMS:NORB:NA,NB[:2S]', e.g. 'Sc:6:1,1'.
    #: Their orbitals must add up to step 1's active space. See
    #: quenais/las/fragments.py.
    fragments: list = field(default_factory=list)

    # ── LAS optimisation ─────────────────────────────────────────────────
    orbital_optimization: bool = True
    max_macro: int = 50
    #: None -> 1e-8 Eh for lasscf, 1e-5 Eh for lassqd
    conv_tol_energy: float | None = None
    conv_tol_grad: float = 1e-4
    #: consecutive cycles below conv_tol_energy (lassqd only)
    n_consecutive: int = 2
    orb_max_iter: int = 30
    orb_gtol: float = 1e-6
    #: fragment-consistency passes per macro cycle for lasscf; lassqd always
    #: uses 1, because each pass is a sampling job
    lasci_max_cycle: int = 50
    density_fit: bool = False

    # ── Exact fragment solver (lasscf) ───────────────────────────────────
    fci_spin_penalty: bool = True

    # ── SQD (lassqd) ─────────────────────────────────────────────────────
    sqd_iterations: int = 6
    sqd_batches: int = 15
    sqd_samples_per_batch: int = 170
    #: carryover threshold eps on |CI coefficient|; 0 disables carryover
    carryover_eps: float = 1e-5
    sqd_nroots: int = 3
    sqd_davidson_tol: float = 1e-10
    sqd_max_davidson: int = 200
    sqd_spin_shift: float = 0.2
    #: closed-shell fragments: use the union of alpha and beta strings
    sqd_symmetrize_spin: bool = True
    sqd_max_dim: int = 50_000_000

    # ── Circuits and sampling ────────────────────────────────────────────
    shots: int = 100_000
    backend: str = "aer_mps"
    mps_max_bond_dim: int | None = None
    ibm_backend_name: str | None = None
    ibm_optimization_level: int = 3
    lucj_n_reps: int = 1
    lucj_pairs: str = "heavy_hex"
    #: refine the CCSD-initialised LUCJ with ffsim's linear method (costly:
    #: statevector simulation of the fragment)
    lucj_optimize: bool = False
    lucj_opt_maxiter: int = 10
    seed: int = 1234

    def energy_tol(self, solver):
        if self.conv_tol_energy is not None:
            return self.conv_tol_energy
        return 1e-5 if solver == "lassqd" else 1e-8

    def validate(self, solver="lassqd") -> "LasSettings":
        from quenais.las.fragments import parse_fragment_spec

        if not self.fragments:
            raise ValueError(
                "LAS solvers need fragments: set cfg.las.fragments or pass "
                "--las-fragments, e.g. --las-fragments 'Sc:6:1,1' 'F:3:3,3'"
            )
        for spec in self.fragments:
            parse_fragment_spec(spec)
        if self.backend not in LAS_BACKENDS:
            raise ValueError(f"las.backend must be one of {LAS_BACKENDS}, "
                             f"got {self.backend!r}")
        if self.backend == "ibm" and not self.ibm_backend_name:
            raise ValueError("las.backend='ibm' requires las.ibm_backend_name")
        if self.lucj_pairs not in LUCJ_PAIRS:
            raise ValueError(f"las.lucj_pairs must be one of {LUCJ_PAIRS}")
        for name in ("max_macro", "orb_max_iter", "lasci_max_cycle", "n_consecutive",
                     "sqd_iterations", "sqd_batches", "sqd_samples_per_batch",
                     "sqd_nroots", "sqd_max_davidson", "shots", "lucj_n_reps",
                     "lucj_opt_maxiter", "sqd_max_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"las.{name} must be > 0, got {getattr(self, name)}")
        if self.carryover_eps < 0:
            raise ValueError("las.carryover_eps must be >= 0 (0 disables carryover)")
        if self.conv_tol_energy is not None and self.conv_tol_energy <= 0:
            raise ValueError("las.conv_tol_energy must be > 0")
        return self
