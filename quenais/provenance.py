"""
Provenance block written into every stage result.

WHY
---
This package is meant to be run by people other than its author, who will
report numbers back. A number without provenance costs a conversation to
interpret and is sometimes uninterpretable after the fact: "I got -752.68"
does not say whether the GQE submodule was patched, whether OpenBLAS was
oversubscribed, or which PySCF version produced it.

So every pickle and every results_summary.csv carries one of these. It is
cheap to collect and it turns a bug report into a reproducible one.

Nothing here may raise. A provenance block that crashes the run it is
describing would be worse than no provenance at all, so every probe is
wrapped and degrades to None.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["provenance", "git_describe", "file_sha256"]

_PACKAGE_ROOT = Path(__file__).resolve().parent


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _run(cmd, cwd=None):
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5)
    return out.stdout.strip() if out.returncode == 0 else None


def git_describe(path=None):
    """Short SHA and dirty flag for a git checkout, or None."""
    path = Path(path) if path else _PACKAGE_ROOT
    sha = _safe(lambda: _run(["git", "rev-parse", "--short=10", "HEAD"], cwd=path))
    if sha is None:
        return None
    status = _safe(lambda: _run(["git", "status", "--porcelain"], cwd=path), default="")
    return {"sha": sha, "dirty": bool(status)}


def file_sha256(path):
    """SHA-256 of a file, or None if unreadable."""
    def _hash():
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    return _safe(_hash)


def _versions():
    names = ["numpy", "scipy", "pyscf", "matplotlib", "qiskit", "ffsim",
             "openfermion", "cudaq", "tequila", "torch"]
    out = {}
    for name in names:
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = _safe(lambda m=mod: getattr(m, "__version__", "unknown"))
        else:
            # Do not import it just to read a version -- importing cudaq or
            # torch here would defeat the lazy-import discipline the package
            # relies on. Report only what the process already loaded.
            out[name] = None
    return out


def _gpu():
    out = _safe(lambda: _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"]
    ))
    return out.splitlines() if out else None


def _gqe_repo(cfg):
    """SHA, patch stamp and patch hash for the external gqe-for-qsci checkout."""
    repo = None
    if cfg is not None:
        repo = getattr(getattr(cfg, "gqe", None), "repo_path", None)
    repo = repo or os.environ.get("GQE_QSCI_REPO_PATH")
    if not repo or not Path(repo).is_dir():
        return {"path": repo, "present": False}

    stamp = Path(repo) / ".quenais_patch_applied"
    return {
        "path": str(repo),
        "present": True,
        "git": git_describe(repo),
        "patch_applied": stamp.exists(),
        "patch_hash": _safe(lambda: stamp.read_text().strip()) if stamp.exists() else None,
    }


def provenance(cfg=None, extra=None):
    """
    Collect a provenance block. Safe to call anywhere; never raises.

    Pass the Config so molecule identity and the GQE repo path are captured
    alongside the environment.
    """
    from quenais import __version__, _threads

    block = {
        "quenais_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_describe(),
        "python": sys.version.split()[0],
        "platform": _safe(lambda: platform.platform()),
        "processor": _safe(lambda: platform.processor()) or None,
        "cpu_count": os.cpu_count(),
        "gpu": _gpu(),
        "threads": _safe(_threads.snapshot),
        "thread_warnings": _safe(lambda: _threads.verify(), default=[]),
        "versions": _versions(),
        "gqe_repo": _safe(lambda: _gqe_repo(cfg)),
    }

    if cfg is not None:
        block["config"] = {
            "molecule": getattr(cfg, "molecule", None),
            "basis": getattr(cfg, "basis", None),
            "charge": getattr(cfg, "charge", None),
            "spin": getattr(cfg, "spin", None),
            "quantum_solver": getattr(cfg, "quantum_solver", None),
        }
    if extra:
        block.update(extra)
    return block
