"""
Thread-count environment variables. MUST be imported before NumPy.

This module has NO imports beyond `os` and does nothing but call
os.environ.setdefault. Keep it that way -- the moment it imports anything
that pulls in NumPy, it stops working.

WHY
---
block2 (used by the active-space finder) is OpenMP-threaded. OpenBLAS is
also multi-threaded, via pthreads, by default. When OpenBLAS is called from
inside a live OpenMP parallel region the two threading runtimes fight over
CPU affinity. OpenBLAS says so itself:

    OpenBLAS Warning : Detect OpenMP Loop and this application may hang.
    Please rebuild the library with USE_OPENMP=1 option.

That is a genuine hang risk, not noise -- it is the same failure family
behind the FeN6 DMRG hang. The fix is not rebuilding OpenBLAS; it is making
sure only ONE of the two libraries spawns threads. PySCF/block2 integral
code already parallelises via OpenMP, so OpenBLAS is pinned to a single
thread and never spawns its own pool inside an OpenMP region.

OMP_NUM_THREADS is left generous so block2 keeps its own parallelism, but
never beyond what the scheduler actually allocated -- see _OMP_THREADS.

TIMING IS THE WHOLE POINT
-------------------------
OpenBLAS reads these variables once, lazily, the first time it needs a
thread pool. Setting them after NumPy's first import does not reliably take
effect. So this module is imported as the FIRST statement of
quenais/__init__.py, which Python guarantees runs before any submodule --
`import quenais.config` executes quenais/__init__.py first, and only then
config.py's own `import numpy`.

The one thing that defeats this is a process that imports NumPy before it
imports quenais at all. `verify()` below exists to detect exactly that.
Test suites are the usual offender, which is why tests/conftest.py imports
quenais before anything else.

setdefault, not assignment: a user who deliberately exports
OPENBLAS_NUM_THREADS=4 keeps their value.
"""

import os

__all__ = ["THREAD_VARS", "applied", "verify", "snapshot",
           "CPU_SOURCE"]

#: Schedulers that tell us how many cores were actually allocated. Reading
#: these matters on a shared HPC node: os.cpu_count() reports the whole
#: machine, so a job allocated 8 cores would otherwise start ~95 OpenMP
#: threads and fight every other job on the node.
_SCHEDULER_VARS = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "PBS_NP",
    "NSLOTS",             # Grid Engine
    "LSB_DJOB_NUMPROC",   # LSF
)


def _allocated_cpus():
    """Cores this process may use: the scheduler's allocation, else the box."""
    for name in _SCHEDULER_VARS:
        raw = os.environ.get(name)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                return value, name
    return (os.cpu_count() or 4), "os.cpu_count"


_CPUS, CPU_SOURCE = _allocated_cpus()

#: OpenMP threads for block2.
#:
#: When a scheduler told us the allocation, those cores are ours
#: exclusively -- use all of them. When we are guessing from cpu_count we
#: are probably sharing the machine, so leave one core free. The old
#: unconditional cpu_count - 1 was wrong in both directions: it wasted a
#: core inside a batch job, and grabbed ~95 threads on a login node.
_OMP_THREADS = _CPUS if CPU_SOURCE != "os.cpu_count" else max(1, _CPUS - 1)

#: Variables this module controls, and the values it sets when unset.
THREAD_VARS = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": str(max(1, _OMP_THREADS)),
}

#: True for each variable this module actually set (i.e. it was unset before).
applied = {}

for _name, _value in THREAD_VARS.items():
    applied[_name] = _name not in os.environ
    os.environ.setdefault(_name, _value)

#: True if NumPy was already imported when this module ran -- meaning the
#: settings above may not have reached OpenBLAS's thread pool.
too_late = "numpy" in __import__("sys").modules


def verify(strict=False):
    """
    Check that the thread settings were applied in time.

    Returns a list of human-readable problems; empty means all good.
    With strict=True, raises RuntimeError instead of returning.

    Called by quenais-selftest and recorded in every run's provenance
    block, because a silent thread-oversubscription hang looks like a slow
    calculation, not like a bug.
    """
    problems = []
    if too_late:
        problems.append(
            "NumPy was already imported when quenais._threads ran, so the "
            "thread-count variables may not have reached OpenBLAS. Import "
            "quenais (or quenais._threads) before NumPy."
        )
    for name, value in THREAD_VARS.items():
        current = os.environ.get(name)
        if current is None:
            problems.append(f"{name} is unset")
        elif current != value and applied.get(name):
            problems.append(f"{name} changed after being set ({current} != {value})")

    if strict and problems:
        raise RuntimeError("Thread environment is not correctly configured:\n  - "
                           + "\n  - ".join(problems))
    return problems


def snapshot():
    """Current values, for the provenance block written into every result."""
    return {
        "vars": {name: os.environ.get(name) for name in THREAD_VARS},
        "set_by_quenais": dict(applied),
        "numpy_imported_first": too_late,
        "cpu_count": os.cpu_count(),
        "allocated_cpus": _CPUS,
        "cpu_source": CPU_SOURCE,
    }
