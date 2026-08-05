"""
quenais-doctor -- check that this machine and environment can actually run
the pipeline, and say exactly what to do when they cannot.

WHY THIS EXISTS
---------------
Every check below corresponds to a failure that has already cost real time,
and every one of them presents as something other than what it is:

  * A CPU without AVX-512 kills PySCF with `Illegal instruction (core
    dumped)` and no Python traceback at all.
  * A GPU below compute capability 8.0 fails cuQuantum with
    "architecture mismatch" partway into a run.
  * mpi4py and CUDA-Q's plugin linking different libmpi produces
    intermittent hangs and corrupt gathers, not an error.
  * setuptools>=82 removed pkg_resources, so tequila fails at import with a
    ModuleNotFoundError that names a package nobody installed on purpose.
  * wandb with no API key raises UsageError only in non-interactive
    contexts, so it passes interactively and fails under sbatch.
  * An unpatched gqe-for-qsci trains to completion and emits nothing
    parseable -- the run "succeeds" and produces no result.

`quenais-selftest` answers "does the physics reproduce". This answers "will
anything run at all", and is much faster. install.sh runs both.

Exit codes: 0 all clear (warnings allowed), 1 at least one FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

__all__ = ["main", "run_checks"]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}


class Report:
    def __init__(self, verbose=True):
        self.rows = []
        self.verbose = verbose

    def add(self, status, name, detail="", fix=""):
        self.rows.append((status, name, detail, fix))
        if self.verbose:
            print(f"[{_SYMBOL[status]}] {name:<38} {detail}")
            if fix and status != PASS:
                for line in fix.strip().splitlines():
                    print(f"           -> {line}")

    def count(self, status):
        return sum(1 for s, *_ in self.rows if s == status)


# ─────────────────────────────────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────────────────────────────────

def _cpu_flags():
    try:
        return set(re.findall(r"\b(avx512[a-z0-9]*)\b",
                              Path("/proc/cpuinfo").read_text()))
    except OSError:
        return set()


def check_avx512(rep):
    """
    PySCF's prebuilt libcgto.so uses AVX-512 without runtime CPU dispatch
    (numpy/OpenBLAS do dispatch; PySCF does not). On a CPU without it the
    first two-electron integral call is SIGILL.
    """
    if platform.system() != "Linux":
        rep.add(WARN, "CPU AVX-512", f"cannot check on {platform.system()}")
        return

    flags = _cpu_flags()
    if flags:
        rep.add(PASS, "CPU AVX-512", f"present ({len(flags)} flags)")
        return

    # No AVX-512. That is only a problem if PySCF is the prebuilt wheel.
    kind = _pyscf_build_kind()
    if kind == "source":
        rep.add(PASS, "CPU AVX-512",
                "absent, but PySCF was built from source")
    elif kind == "absent":
        # check_imports reports the missing package; don't double-count it
        # as a hardware failure.
        rep.add(WARN, "CPU AVX-512", "absent, and PySCF is not installed",
                "When you do install it, it must be the source build:\n"
                "pip install pyscf==2.11.0 --no-binary pyscf "
                "--force-reinstall --no-deps")
    else:
        rep.add(FAIL, "CPU AVX-512",
                f"absent, and PySCF looks like a {kind} build "
                f"({'prebuilt manylinux wheel' if kind == 'wheel' else 'origin undetermined'})",
                "PySCF will die with 'Illegal instruction (core dumped)'.\n"
                "pip install pyscf==2.11.0 --no-binary pyscf "
                "--force-reinstall --no-deps")


def _pyscf_build_kind():
    """'source', 'wheel', 'absent', or 'unknown' -- best-effort."""
    try:
        import pyscf
    except Exception:
        return "absent"
    dist_info = Path(pyscf.__file__).resolve().parent.parent
    for wheel in dist_info.glob("pyscf-*.dist-info/WHEEL"):
        text = wheel.read_text()
        # A local source build tags as linux_x86_64; a PyPI wheel is
        # manylinux*. That is the cleanest signal available without
        # objdumping libcgto.so.
        if "manylinux" in text:
            return "wheel"
        return "source"
    return "unknown"


def check_gpu(rep):
    """cuQuantum/cuStateVec requires compute capability >= 8.0 (Ampere)."""
    if not shutil.which("nvidia-smi"):
        rep.add(PASS, "GPU", "none detected -- CPU simulator path")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        rep.add(WARN, "GPU", f"nvidia-smi failed: {exc}")
        return
    if not out:
        rep.add(PASS, "GPU", "none detected -- CPU simulator path")
        return

    target = os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "")
    for line in out.splitlines():
        name, _, cap = line.partition(",")
        name, cap = name.strip(), cap.strip()
        try:
            usable = float(cap) >= 8.0
        except ValueError:
            usable = False
        if usable:
            rep.add(PASS, "GPU", f"{name} (cc {cap})")
        elif target == "qpp-cpu":
            rep.add(PASS, "GPU",
                    f"{name} (cc {cap}) unsupported, already on qpp-cpu")
        else:
            rep.add(FAIL, "GPU",
                    f"{name} (cc {cap}) is below cuQuantum's 8.0 minimum",
                    "The nvidia target will raise "
                    "'RuntimeError: architecture mismatch'.\n"
                    "export CUDAQ_DEFAULT_SIMULATOR=qpp-cpu  "
                    "(must be set before python starts)")


# ─────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────

def check_setuptools(rep):
    try:
        import setuptools
    except Exception as exc:
        rep.add(FAIL, "setuptools", str(exc), "pip install 'setuptools<82'")
        return
    ver = setuptools.__version__
    major = int(ver.split(".")[0])
    if major >= 82:
        rep.add(FAIL, "setuptools", f"{ver} (>=82 removed pkg_resources)",
                "tequila and others import pkg_resources at load time.\n"
                "pip install 'setuptools<82'")
    elif importlib.util.find_spec("pkg_resources") is None:
        rep.add(FAIL, "setuptools", f"{ver}, but pkg_resources is missing",
                "pip install --force-reinstall 'setuptools<82'")
    else:
        rep.add(PASS, "setuptools", f"{ver}, pkg_resources importable")


def check_cudaq_target(rep):
    target = os.environ.get("CUDAQ_DEFAULT_SIMULATOR")
    if not target:
        rep.add(WARN, "CUDAQ_DEFAULT_SIMULATOR", "not set",
                "CUDA-Q reads this at import; setting it inside a script is\n"
                "too late. quenais sets it for the training subprocess from\n"
                "cfg.gqe.cudaq_target, but set it in the shell for anything\n"
                "you launch by hand.")
        return
    from quenais.settings.gqe import CUDAQ_SIMULATOR_TARGETS
    if target in CUDAQ_SIMULATOR_TARGETS:
        rep.add(PASS, "CUDAQ_DEFAULT_SIMULATOR", target)
    else:
        rep.add(FAIL, "CUDAQ_DEFAULT_SIMULATOR", f"{target!r} unrecognised",
                f"expected one of {CUDAQ_SIMULATOR_TARGETS}")


def check_wandb(rep):
    mode = os.environ.get("WANDB_MODE")
    if mode in ("offline", "disabled"):
        rep.add(PASS, "WANDB_MODE", mode)
        return
    has_key = bool(os.environ.get("WANDB_API_KEY")) or \
        Path("~/.netrc").expanduser().exists()
    if has_key:
        rep.add(PASS, "WANDB_MODE", f"{mode or 'online'}, credentials found")
    else:
        rep.add(FAIL, "WANDB_MODE", f"{mode or 'online'}, no API key",
                "Batch jobs and piped output cannot show wandb's login\n"
                "prompt; the run dies with UsageError: No API key configured.\n"
                "export WANDB_MODE=offline    (or =disabled to skip wandb)")


# ─────────────────────────────────────────────────────────────────────────
# MPI
# ─────────────────────────────────────────────────────────────────────────

def _ldd_libmpi(path):
    try:
        out = subprocess.run(["ldd", str(path)], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "libmpi.so" in line and "=>" in line:
            resolved = line.split("=>", 1)[1].strip().split(" ")[0]
            if resolved and resolved != "not":
                return os.path.realpath(resolved)
    return None


def check_mpi(rep):
    """
    mpi4py and CUDA-Q's plugin must resolve to the SAME libmpi. If one
    comes from the conda env and the other from /usr/lib, the mismatch
    shows up as hangs and corrupt gathers rather than a clean error.
    """
    plugin = Path(sysconfig.get_paths()["purelib"]) / \
        "distributed_interfaces" / "libcudaq_distributed_interface_mpi.so"

    if not plugin.exists():
        rep.add(FAIL, "CUDA-Q MPI plugin", "not built",
                "RuntimeError: Unable to open distributed interface library.\n"
                "source \"$CONDA_PREFIX/lib/python*/site-packages/"
                "distributed_interfaces/activate_custom_mpi.sh\"\n"
                "or re-run install.sh")
        return

    try:
        import mpi4py
        from mpi4py import MPI  # noqa: F401
        candidates = list(Path(mpi4py.__file__).parent.glob("MPI*.so"))
    except Exception as exc:
        rep.add(FAIL, "mpi4py", f"{type(exc).__name__}: {exc}",
                "mamba install -c conda-forge mpi4py openmpi")
        return

    a = _ldd_libmpi(plugin)
    b = _ldd_libmpi(candidates[0]) if candidates else None

    if a and b and a == b:
        rep.add(PASS, "MPI ABI", f"both link {Path(a).name}")
    elif a and b:
        rep.add(FAIL, "MPI ABI", "mpi4py and the CUDA-Q plugin differ",
                f"mpi4py  -> {b}\nplugin  -> {a}\n"
                "Install both from conda-forge into the same env and rebuild\n"
                "the plugin with activate_custom_mpi.sh.")
    else:
        rep.add(WARN, "MPI ABI", "could not resolve libmpi via ldd")


# ─────────────────────────────────────────────────────────────────────────
# Packages and the submodule
# ─────────────────────────────────────────────────────────────────────────

_REQUIRED = [
    ("pyscf", "pyscf"), ("block2", "block2"),
    ("qiskit", "qiskit"), ("qiskit_aer", "qiskit-aer"),
    ("qiskit_addon_sqd", "qiskit-addon-sqd"), ("ffsim", "ffsim"),
    ("openfermion", "openfermion"), ("asf.wrapper", "asf"),
    ("qiskit_fermions.circuit", "qiskit-fermions"),
    ("pyci", "pyci (theochem)"),
    ("cudaq", "cudaq"), ("torch", "torch"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("hydra", "hydra-core"), ("tequila", "tequila"), ("wandb", "wandb"),
    ("gqe_qsci", "gqe-for-qsci"),
]


def check_imports(rep):
    missing = []
    for mod, label in _REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing.append(f"{label} ({type(exc).__name__})")
    if missing:
        rep.add(FAIL, "required packages", f"{len(missing)} missing",
                "\n".join(missing) + "\nRe-run install.sh")
    else:
        rep.add(PASS, "required packages", f"all {len(_REQUIRED)} importable")


def check_shims(rep):
    """
    The patched factory.py imports `dmet_excitation_pool` by top-level
    absolute name. The shim directory supplies it via PYTHONPATH.
    """
    shim = Path(__file__).resolve().parent / "quantum" / "_gqe_shims"
    if not shim.is_dir():
        rep.add(FAIL, "GQE shims", f"missing: {shim}",
                "package_data must include quantum/_gqe_shims/*.py")
        return
    if (shim / "__init__.py").exists():
        rep.add(FAIL, "GQE shims", "__init__.py present",
                "This directory is a sys.path entry, not a package. An\n"
                "__init__.py turns it into one and the top-level import\n"
                "`import dmet_excitation_pool` stops resolving.")
        return
    names = sorted(p.stem for p in shim.glob("*.py"))
    expected = {"dmet_excitation_pool", "dmet_molecule_adapter"}
    if expected.issubset(names):
        rep.add(PASS, "GQE shims", ", ".join(names))
    else:
        rep.add(FAIL, "GQE shims", f"found {names}",
                f"expected {sorted(expected)}")


def check_patch_shipped(rep):
    from quenais.quantum import gqe_setup
    if gqe_setup.PATCH_PATH.exists():
        rep.add(PASS, "shipped patch", str(gqe_setup.PATCH_PATH))
    else:
        rep.add(FAIL, "shipped patch", f"not found: {gqe_setup.PATCH_PATH}",
                "The patch is package_data under quenais/patches/. A missing\n"
                "file means the install did not ship it -- reinstall from a\n"
                "checkout that has quenais/patches/gqe_dmet_source.patch.")


def check_gqe_repo(rep, repo=None):
    from quenais.quantum.gqe_setup import resolve_repo, verify_gqe_repo
    try:
        path = resolve_repo(repo)
    except FileNotFoundError as exc:
        rep.add(FAIL, "gqe-for-qsci checkout", str(exc).splitlines()[0],
                "git submodule update --init --recursive\n"
                "quenais-gqe-setup --repo ./gqe-for-qsci")
        return
    problems = verify_gqe_repo(path)
    if problems:
        rep.add(FAIL, "gqe-for-qsci checkout", f"{len(problems)} problem(s)",
                "\n".join(problems) +
                f"\nquenais-gqe-setup --repo {path}")
    else:
        rep.add(PASS, "gqe-for-qsci checkout", f"patched at {path}")


# ─────────────────────────────────────────────────────────────────────────

def run_checks(repo=None, verbose=True):
    rep = Report(verbose=verbose)
    for fn in (check_avx512, check_gpu, check_setuptools,
               check_cudaq_target, check_wandb, check_mpi,
               check_imports, check_shims, check_patch_shipped):
        try:
            fn(rep)
        except Exception as exc:                      # never crash the doctor
            rep.add(WARN, fn.__name__, f"check errored: "
                                       f"{type(exc).__name__}: {exc}")
    try:
        check_gqe_repo(rep, repo)
    except Exception as exc:
        rep.add(WARN, "check_gqe_repo", f"{type(exc).__name__}: {exc}")
    return rep


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check that this environment can run the pipeline.")
    parser.add_argument("--repo", default=None,
                        help="path to the gqe-for-qsci checkout "
                             "(default: ./gqe-for-qsci or "
                             "$GQE_QSCI_REPO_PATH)")
    args = parser.parse_args(argv)

    repo = args.repo
    if repo is None and Path("gqe-for-qsci/train.py").exists():
        repo = "gqe-for-qsci"

    print(f"quenais-doctor  --  python {platform.python_version()} "
          f"on {platform.platform()}")
    print()
    rep = run_checks(repo)
    print()
    print(f"{rep.count(PASS)} ok, {rep.count(WARN)} warnings, "
          f"{rep.count(FAIL)} failures")
    if rep.count(FAIL):
        print("\nSee docs/gqe_setup.md for the full background on each of "
              "these.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
