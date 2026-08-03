"""
Tests for the thread-count guard.

The interesting assertions run in a SUBPROCESS. Testing this in-process is
meaningless: by the time pytest reaches this file, NumPy and quenais are
both long since imported, so any in-process check trivially passes whether
or not the guard works. Only a fresh interpreter can observe the ordering.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from quenais import _threads

VARS = ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]

# Also cleared from the child environment. Running the suite inside a real
# SLURM job otherwise leaks the allocation into tests that are asserting
# the no-scheduler fallback.
SCHEDULER_VARS = list(_threads._SCHEDULER_VARS)


def run_snippet(code, env=None):
    """Run code in a fresh interpreter with the repo on sys.path."""
    stripped = set(VARS) | set(SCHEDULER_VARS)
    child_env = {k: v for k, v in os.environ.items() if k not in stripped}
    child_env["PYTHONPATH"] = os.pathsep.join(sys.path)
    if env:
        child_env.update(env)
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, env=child_env, timeout=120)


def test_vars_are_set_before_numpy_is_imported():
    """The core property: quenais sets the vars, and NumPy is untouched."""
    out = run_snippet(
        "import sys\n"
        "import quenais\n"
        "import os, json\n"
        "print(json.dumps({\n"
        "  'numpy_loaded': 'numpy' in sys.modules,\n"
        "  'vars': {k: os.environ.get(k) for k in "
        f"{VARS!r}"
        "},\n"
        "}))\n"
    )
    assert out.returncode == 0, out.stderr
    import json

    data = json.loads(out.stdout.strip().splitlines()[-1])

    assert not data["numpy_loaded"], (
        "importing quenais pulled in NumPy during package init -- the guard "
        "still ran first, but the lazy-import discipline has regressed"
    )
    assert data["vars"]["OPENBLAS_NUM_THREADS"] == "1"
    assert data["vars"]["MKL_NUM_THREADS"] == "1"
    assert data["vars"]["NUMEXPR_NUM_THREADS"] == "1"
    assert int(data["vars"]["OMP_NUM_THREADS"]) >= 1


@pytest.mark.parametrize(
    "module",
    ["quenais.config", "quenais.utils.cif_parser", "quenais.visualization"],
)
def test_guard_applies_via_any_submodule_import(module):
    """
    Importing a submodule directly must still trigger the guard, because
    Python runs the parent package's __init__ first. This is what makes a
    single import in quenais/__init__.py sufficient.
    """
    out = run_snippet(
        f"import {module}\n"
        "import os\n"
        "assert os.environ['OPENBLAS_NUM_THREADS'] == '1', os.environ.get('OPENBLAS_NUM_THREADS')\n"
        "print('ok')\n"
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_user_values_are_respected():
    """setdefault, not assignment -- a deliberate export must survive."""
    out = run_snippet(
        "import quenais, os\n"
        "print(os.environ['OPENBLAS_NUM_THREADS'])\n",
        env={"OPENBLAS_NUM_THREADS": "4"},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "4"


def test_verify_reports_numpy_imported_first():
    """
    The failure this guard cannot prevent -- NumPy imported before quenais --
    must at least be detected and reported.
    """
    out = run_snippet(
        "import numpy\n"
        "import quenais\n"
        "from quenais import _threads\n"
        "problems = _threads.verify()\n"
        "assert problems, 'verify() failed to notice NumPy was imported first'\n"
        "assert _threads.snapshot()['numpy_imported_first'] is True\n"
        "print('detected')\n"
    )
    assert out.returncode == 0, out.stderr
    assert "detected" in out.stdout


def test_verify_is_clean_in_this_process():
    """conftest.py imports quenais first, so the suite itself must be clean."""
    assert _threads.verify() == []


def test_threads_module_has_no_heavy_imports():
    """
    Guard against the guard being broken by a future edit. _threads must not
    import anything that could pull in NumPy.
    """
    import ast
    import pathlib

    src = pathlib.Path(_threads.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"os"}, f"_threads must import only os, found: {imported}"


def test_snapshot_is_serialisable():
    """snapshot() goes into every provenance block, so it must be JSON-safe."""
    import json

    json.dumps(_threads.snapshot())


def test_scheduler_allocation_is_respected():
    """
    On a shared node os.cpu_count() reports the whole machine. A job given
    8 cores must not start ~95 OpenMP threads and fight everything else on
    the node.
    """
    import json

    out = run_snippet(
        "import quenais, os, json\n"
        "from quenais import _threads\n"
        "print(json.dumps({'omp': os.environ['OMP_NUM_THREADS'],\n"
        "                  'src': _threads.snapshot()['cpu_source']}))\n",
        env={"SLURM_CPUS_PER_TASK": "8"},
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    # A scheduler allocation is exclusive, so use all 8 rather than 7.
    assert data["omp"] == "8", f"expected 8 (the full allocation), got {data['omp']}"
    assert data["src"] == "SLURM_CPUS_PER_TASK"


def test_falls_back_to_cpu_count_without_a_scheduler():
    import json

    out = run_snippet(
        "import quenais, json\n"
        "from quenais import _threads\n"
        "print(json.dumps(_threads.snapshot()['cpu_source']))\n"
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == "os.cpu_count"


def test_malformed_scheduler_value_is_ignored():
    """A non-numeric allocation must not crash the guard."""
    out = run_snippet(
        "import quenais, os\n"
        "assert int(os.environ['OMP_NUM_THREADS']) >= 1\n"
        "print('ok')\n",
        env={"SLURM_CPUS_PER_TASK": "not-a-number"},
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_cpu_count_fallback_leaves_one_core_free():
    """
    Without a scheduler we are probably sharing the machine, so reserve a
    core. With one, the allocation is exclusive and we take all of it --
    the two cases genuinely differ.
    """
    import json

    out = run_snippet(
        "import quenais, os, json\n"
        "from quenais import _threads\n"
        "print(json.dumps({'omp': int(os.environ['OMP_NUM_THREADS']),\n"
        "                  'cpus': _threads.snapshot()['cpu_count']}))\n"
    )
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["omp"] == max(1, data["cpus"] - 1)
