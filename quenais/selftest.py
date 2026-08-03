"""
quenais-selftest -- run the pipeline against known-good values and report
pass/fail per quantity.

Run this first, before trusting any result from a new install. It takes
seconds on LiH and it is the right thing to attach to a bug report: if it
passes, the environment is sound and later numbers mean something; if it
fails, the output says exactly which quantity drifted.

Checks, in order of diagnostic value:
  - thread environment applied before NumPy (a silent oversubscription
    hang looks like a slow calculation, not a bug)
  - classical reference energies
  - the active space
  - bath count and embedded electron count -- the two silent physics bugs
  - the embedded SCF against full UHF, which tests the whole embedding
    Hamiltonian at once
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

__all__ = ["main", "run_selftest"]

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "tests" / "regression"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Results:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append((status, name, detail))
        symbol = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
        print(f"[{symbol}] {name:<44} {detail}")

    @property
    def failed(self):
        return sum(1 for s, _, _ in self.rows if s == FAIL)

    @property
    def passed(self):
        return sum(1 for s, _, _ in self.rows if s == PASS)

    @property
    def skipped(self):
        return sum(1 for s, _, _ in self.rows if s == SKIP)


def _load_reference():
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    try:
        import reference_values

        return reference_values
    except ImportError:
        return None


def _check_threads(results):
    from quenais import _threads

    problems = _threads.verify()
    if problems:
        results.add(FAIL, "thread environment", problems[0])
    else:
        snap = _threads.snapshot()
        vals = snap["vars"]
        results.add(PASS, "thread environment",
                    f"OPENBLAS={vals['OPENBLAS_NUM_THREADS']} "
                    f"OMP={vals['OMP_NUM_THREADS']} "
                    f"(from {snap['cpu_source']})")


def run_selftest(system="LiH", project_dir=None, verbose=False):
    """Run the pipeline for one system and compare against references."""
    from quenais.config import Config
    from quenais.settings import AsfSettings

    results = Results()
    print(f"\nQuEnAIS self-test -- {system}")
    print("=" * 66)

    _check_threads(results)

    ref_module = _load_reference()
    if ref_module is None or system not in ref_module.SYSTEMS:
        results.add(FAIL, "reference values",
                    f"no reference data for {system}")
        return results
    ref = ref_module.SYSTEMS[system]

    try:
        import pyscf  # noqa: F401
    except ImportError:
        results.add(SKIP, "pipeline", "PySCF not installed")
        return results

    tmp = project_dir or tempfile.mkdtemp(prefix="quenais_selftest_")
    cfg = Config(
        molecule=system,
        basis=ref["basis"],
        project_dir=str(tmp),
        classical_methods=["HF", "MP2", "CCSD"],
        asf=AsfSettings(force_active_space=ref["force_active_space"]),
    )
    cfg.validate().make_dirs().load_geometry()
    if verbose:
        print(f"  working directory: {tmp}\n")

    # ── Step 0 ───────────────────────────────────────────────────────────
    try:
        from quenais.classical import runner

        step0 = _quiet(runner.main, cfg, force=True, verbose=verbose)
        for method in ("HF", "MP2", "CCSD"):
            expected, tier = ref["energies"][method]
            got = step0["methods"][method]["energy"]
            delta = abs(got - expected)
            tol = ref_module.TOL[tier]
            results.add(PASS if delta <= tol else FAIL, f"{method} energy",
                        f"{got:.9f} Ha  (delta {delta:.1e})")
    except Exception as exc:
        results.add(FAIL, "classical stage", f"{type(exc).__name__}: {exc}")
        return results

    # ── Step 1 ───────────────────────────────────────────────────────────
    try:
        from quenais.active_space import finder

        step1 = _quiet(finder.main, cfg, force=True, verbose=verbose)
        ok = list(step1["mo_list"]) == ref["structure"]["mo_list"]
        results.add(PASS if ok else FAIL, "active space",
                    f"({step1['nel']}e, {step1['n_active_orbs']}o) "
                    f"MOs={list(step1['mo_list'])}")
        for key in ("dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
            results.add(PASS if key in step1 else FAIL,
                        f"step 1 provides {key}",
                        "required by the embedding stage")
    except Exception as exc:
        results.add(FAIL, "active-space stage", f"{type(exc).__name__}: {exc}")
        return results

    # ── Step 2 ───────────────────────────────────────────────────────────
    try:
        from quenais.embedding import hamiltonian

        step2 = _quiet(hamiltonian.main, cfg, force=True, verbose=verbose)
    except Exception as exc:
        results.add(FAIL, "embedding stage", f"{type(exc).__name__}: {exc}")
        return results

    struct = ref["structure"]
    results.add(PASS if step2["n_bath"] == struct["n_bath"] else FAIL,
                "bath orbital count",
                f"{step2['n_bath']} (expected {struct['n_bath']})")
    results.add(
        PASS if (step2["n_alpha"], step2["n_beta"])
        == (struct["n_alpha"], struct["n_beta"]) else FAIL,
        "embedded electron count",
        f"({step2['n_alpha']}a, {step2['n_beta']}b) "
        f"-- from the reference density, not the active space",
    )

    expected, tol = ref["scalars"]["ecore"]
    delta = abs(float(step2["ecore"]) - expected)
    results.add(PASS if delta <= tol else FAIL, "ecore",
                f"{step2['ecore']:.9f} Ha  (delta {delta:.1e})")

    check = step2.get("embedded_scf_check")
    if check is None:
        results.add(FAIL, "embedded SCF vs full UHF", "check did not run")
    else:
        tol = ref_module.EMBEDDED_SCF_VS_UHF_TOL
        results.add(PASS if abs(check["delta"]) <= tol else FAIL,
                    "embedded SCF vs full UHF",
                    f"delta {check['delta']:+.2e} Ha  (tol {tol:.0e})")

    return results


def _quiet(fn, cfg, verbose=False, **kwargs):
    """
    Run a stage with its output suppressed unless verbose.

    Redirects at the FILE DESCRIPTOR level, not via
    contextlib.redirect_stdout. Two sources write past a Python-level
    redirect: PySCF's own logger, which holds its own stream reference, and
    the ASF library, whose internal CASCI prints from code we do not own.
    Silencing objects one at a time cannot cover third-party internals;
    dup2 on fd 1 covers everything.
    """
    import os as _os
    import warnings

    if verbose:
        return fn(cfg, **kwargs)

    saved_fd = _os.dup(1)
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    try:
        sys.stdout.flush()
        _os.dup2(devnull, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(cfg, **kwargs)
    finally:
        sys.stdout.flush()
        _os.dup2(saved_fd, 1)
        _os.close(saved_fd)
        _os.close(devnull)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="quenais-selftest",
        description="Verify a QuEnAIS install against known-good values.",
    )
    parser.add_argument("--system", default="LiH",
                        help="LiH (fast, default), N2, or ScH")
    parser.add_argument("--full", action="store_true",
                        help="run LiH, N2 and ScH")
    parser.add_argument("--project-dir", default=None,
                        help="keep outputs here instead of a temp directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show each stage's own output")
    args = parser.parse_args(argv)

    systems = ["LiH", "N2", "ScH"] if args.full else [args.system]

    total_failed = total_passed = total_skipped = 0
    for system in systems:
        results = run_selftest(system, args.project_dir, args.verbose)
        total_failed += results.failed
        total_passed += results.passed
        total_skipped += results.skipped

    print("\n" + "=" * 66)
    print(f"  {total_passed} passed, {total_failed} failed, "
          f"{total_skipped} skipped")
    if total_failed:
        print("\n  Something drifted. Attach this output to a bug report --\n"
              "  it identifies which quantity, which is most of the diagnosis.")
    elif total_skipped:
        # Do not claim a clean bill of health when the physics never ran.
        print("\n  No failures, but the pipeline was SKIPPED -- see the skip\n"
              "  reasons above. This does not verify the physics. Install the\n"
              "  missing dependency and re-run before trusting any result.")
    else:
        print("\n  Install looks sound.")
    print("=" * 66)
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
