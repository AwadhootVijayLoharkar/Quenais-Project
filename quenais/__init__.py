"""
QuEnAIS — Quantum Embedding for Strongly Correlated Molecules
"""

# ─────────────────────────────────────────────────────────────────────────
# MUST BE FIRST. _threads sets OPENBLAS/MKL/NUMEXPR/OMP thread counts, and
# they only take effect if they are set before NumPy is first imported
# anywhere in the process. Python runs this file before any quenais
# submodule, so importing it here covers every `import quenais.*` path.
# Do not move, and do not add imports above it.
# ─────────────────────────────────────────────────────────────────────────
from quenais import _threads as _threads  # noqa: F401  (imported for side effect)

__version__ = "0.2.0"
__author__ = "Awadhoot Loharkar"

__all__ = ["__version__", "Config", "provenance"]


def __getattr__(name):
    """
    Lazy re-exports.

    Config sits behind __getattr__ rather than a module-level import so that
    `import quenais` stays cheap and does not immediately pull in NumPy and
    PySCF during package initialisation. `from quenais import Config` still
    works exactly as before.
    """
    if name == "Config":
        from quenais.config import Config

        return Config
    if name == "provenance":
        from quenais.provenance import provenance

        return provenance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
