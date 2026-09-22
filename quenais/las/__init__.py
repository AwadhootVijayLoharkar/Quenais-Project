"""
Localized active space (LAS) methods: LASSCF and LASSQD.

An independent, Apache-2.0 implementation written from the published
equations. It contains no code from mrh (GPL) or from the LASSQD authors'
repository (unlicensed); those were used only as local numerical references.

References
----------
- M. R. Hermes, L. Gagliardi, JCTC 15, 972 (2019)         -- LASSCF
- M. R. Hermes et al., JCTC 16, 4923 (2020)               -- LASSCF
- Q. Wang et al., PNAS 123, e2603914123 (2026)            -- LASSQD, carryover SQD

Layout
------
energy.py      pure NumPy: fragment Hamiltonians, LAS energy, orbital gradient
optimizer.py   pure NumPy: orbital optimisation (moving-frame L-BFGS)
driver.py      pure NumPy: the LASCI / LASSCF macro loop, solver-agnostic
sqd_bits.py    pure NumPy: bitstring handling, subsampling, carryover
integrals.py   PySCF integral provider (lazy import)
fragments.py   fragment specs and active-orbital localisation
fci_solver.py  exact fragment solver (PySCF FCI)  -> "lasscf"
sqd_solver.py  LUCJ + sampling + SQD with carryover -> "lassqd"
stage.py       pipeline step-3 entry point

Nothing here imports PySCF, Qiskit or ffsim at module import time.
"""

__all__ = ["run_las", "LasOptions", "LasResult"]


def __getattr__(name):
    if name in __all__:
        from quenais.las import driver

        return getattr(driver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
