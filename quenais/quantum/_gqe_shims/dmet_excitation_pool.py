"""
Shim: re-exports the canonical operator pools under the top-level module
name the external gqe-for-qsci repo imports.

See dmet_molecule_adapter.py in this directory for the rationale.

Note that quenais.quantum.gqe_pools builds its classes lazily via a module
__getattr__, so `import *` here triggers construction -- and therefore the
cudaq and gqe_qsci imports -- only when this shim is actually imported,
which is inside the training subprocess.
"""

from quenais.quantum.gqe_pools import *          # noqa: F401,F403
from quenais.quantum.gqe_pools import __all__    # noqa: F401
