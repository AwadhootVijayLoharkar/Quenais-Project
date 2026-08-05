"""
Test configuration.

The import below is load-bearing and must stay first. quenais._threads
pins the OpenBLAS/MKL/NUMEXPR thread counts, and those only take effect if
they are set before NumPy is first imported in the process. pytest collects
test modules in alphabetical order and several of them import NumPy at the
top, so without this conftest the guard would lose the race and the suite
would silently run in the oversubscribed configuration it exists to
prevent -- while the test asserting the guard works still passed.

Nothing may be imported above it.
"""

import quenais  # noqa: F401  isort:skip

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

for _p in (ROOT / "tools", ROOT / "tests" / "regression"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: end-to-end runs; deselect with -m 'not slow'"
    )
    config.addinivalue_line(
        "markers", "needs_pyscf: requires PySCF"
    )
    config.addinivalue_line(
        "markers", "needs_qiskit: requires the [qiskit] extra"
    )
    config.addinivalue_line(
        "markers", "needs_cudaq: requires the [cudaq] extra and a patched submodule"
    )


def _have(name):
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip stack-specific tests rather than failing them on a partial install."""
    missing = {
        "needs_pyscf": ("pyscf", _have("pyscf")),
        "needs_qiskit": ("qiskit", _have("qiskit")),
        "needs_cudaq": ("cudaq", _have("cudaq")),
    }
    for item in items:
        for marker, (dist, present) in missing.items():
            if marker in item.keywords and not present:
                item.add_marker(pytest.mark.skip(reason=f"{dist} not installed"))


@pytest.fixture(scope="session")
def golden_dir():
    return ROOT / "tests" / "regression" / "golden"


# The DMET pools import `gqe_qsci` from the submodule. The runner puts that
# directory on PYTHONPATH for the training subprocess; pytest runs in-process,
# so add it here too when the submodule is present.
import os as _os
import sys as _sys

_GQE_REPO = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "gqe-for-qsci"
)
if _os.path.isdir(_GQE_REPO) and _GQE_REPO not in _sys.path:
    _sys.path.insert(0, _GQE_REPO)
