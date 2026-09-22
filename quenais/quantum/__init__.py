"""
Quantum solvers.

Three families:

  Qiskit  -- sqd, skqd, sqdrift. In-process, needs quenais[qiskit].
  CUDA-Q  -- gqe. Subprocess against the external gqe-for-qsci trainer,
             needs quenais[cudaq] and a patched checkout.
  LAS     -- lasscf, lassqd. In-process (quenais.las). Read step 1, not
             step 2; lassqd needs quenais[qiskit].

dispatch() is the single place that maps a solver name to an
implementation. Config.validate() and the CLI both derive their accepted
values from quenais.config's SOLVERS tuple, so adding a solver means
touching one list and one branch here -- not three lists that drift apart.

Nothing heavy is imported at module scope: the stage module is imported
inside dispatch(), after the name has been resolved. That is what lets a
Qiskit-only install import this package at all.
"""

from __future__ import annotations

__all__ = ["dispatch"]


def dispatch(cfg, force=False, **kwargs):
    """
    Run the configured quantum solver.

    Raises ValueError for an unknown name, and a clear ImportError if the
    selected stack is not installed.
    """
    from quenais.config import GQE_SOLVERS, LAS_SOLVERS, QISKIT_SOLVERS, SOLVERS

    solver = cfg.quantum_solver

    if solver in QISKIT_SOLVERS:
        try:
            from quenais.quantum import solver as qiskit_solver
        except ImportError as exc:
            raise ImportError(
                f"quantum_solver={solver!r} needs the Qiskit stack. "
                f"Install it with: pip install 'quenais[qiskit]'\n"
                f"  (original error: {exc})"
            ) from exc
        return qiskit_solver.main(cfg, force=force, **kwargs)

    if solver in GQE_SOLVERS:
        try:
            from quenais.quantum import gqe_runner
        except ImportError as exc:
            raise ImportError(
                f"quantum_solver={solver!r} needs the CUDA-Q stack. "
                f"Install it with: pip install 'quenais[cudaq]'\n"
                f"  (original error: {exc})"
            ) from exc
        return gqe_runner.main(cfg, force=force, **kwargs)

    if solver in LAS_SOLVERS:
        # Reads step 1 only; see quenais/las/stage.py.
        try:
            from quenais.las import stage as las_stage
        except ImportError as exc:
            raise ImportError(
                f"quantum_solver={solver!r} needs PySCF"
                + (" and the Qiskit stack: pip install 'quenais[qiskit]'"
                   if solver == "lassqd" else "")
                + f"\n  (original error: {exc})"
            ) from exc
        return las_stage.main(cfg, force=force, **kwargs)

    raise ValueError(
        f"Unknown quantum_solver {solver!r}. Choose one of {SOLVERS}."
    )
