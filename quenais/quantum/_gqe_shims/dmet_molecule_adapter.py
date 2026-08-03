"""
Shim: re-exports the canonical adapter under the top-level module name the
external gqe-for-qsci repo imports.

That repo's factory.py and configs/molecule/dmet_embedding.yaml refer to
`dmet_molecule_adapter` as a top-level absolute import. The modules
therefore only need to be IMPORTABLE, not physically present inside
someone else's checkout -- so the real implementation lives in the package
and this directory is prepended to PYTHONPATH for the training subprocess.

Upstream stays pristine: no files written into the submodule, no
duplicated implementation, and the classes remain importable from the
package for testing.

This directory is deliberately NOT a package -- there is no __init__.py.
It is a sys.path entry, not something to import from.
"""

from quenais.quantum.gqe_adapter import *          # noqa: F401,F403
from quenais.quantum.gqe_adapter import __all__    # noqa: F401
