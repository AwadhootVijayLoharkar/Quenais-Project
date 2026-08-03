"""
Stage 0 (classical reference methods) tests.

Split in two:
  - structural tests, which need no PySCF and guard the defect that made the
    0.1 module silently do nothing;
  - numerical tests, marked needs_pyscf, which run the real thing and
    compare against the validated golden step 0 pickles.
"""

from __future__ import annotations

import ast
import inspect
import os
import pickle
from pathlib import Path

import pytest

from quenais.classical import runner
from quenais.config import Config

import sys

sys.path.insert(0, str(Path(__file__).parent / "regression"))
from reference_values import SYSTEMS, Tier  # noqa: E402


# ── Structural: the 0.1 defect must not come back ────────────────────────

def test_main_actually_contains_the_pipeline():
    """
    The 0.1 module had an indentation error: main() ended after the banner
    and the entire run-and-save body was absorbed into _run_nevpt2(). It
    parsed, imported and ran cleanly, returned None and wrote no pickle.

    Assert the shape that failure would break.
    """
    tree = ast.parse(Path(runner.__file__).read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    main = funcs["main"]
    body = ast.dump(main)
    assert "pickle" in body or "dump" in body, "main() must write the step 0 pickle"
    assert any(
        isinstance(n, ast.Return) and n.value is not None for n in ast.walk(main)
    ), "main() must return results"

    nevpt = funcs["_run_nevpt2"]
    assert nevpt.end_lineno - nevpt.lineno < 60, (
        "_run_nevpt2 has swallowed code that belongs to main() -- this is "
        "exactly the 0.1 defect"
    )


def test_main_signature_matches_stage_convention():
    params = list(inspect.signature(runner.main).parameters)
    assert params[:2] == ["cfg", "force"]


def test_no_pyscf_import_at_module_level():
    """
    PySCF must be imported inside functions. A module-level import makes
    `import quenais.classical.runner` fail on an install without it, which
    breaks CLI dispatch and test collection.
    """
    tree = ast.parse(Path(runner.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or node.names[0].name
            assert not name.startswith("pyscf"), f"module-level pyscf import: {name}"


def test_every_method_has_a_tier():
    for system in SYSTEMS.values():
        for method, (_energy, _tier) in system["energies"].items():
            if method.startswith("DMET"):
                continue
            assert method in runner.METHOD_TIERS, f"{method} has no tier"


def test_tiers_agree_with_the_reference_table():
    """The module's tier map and the regression table must not drift apart."""
    for system in SYSTEMS.values():
        for method, (_energy, tier) in system["energies"].items():
            if method.startswith("DMET"):
                continue
            assert runner.METHOD_TIERS[method] == tier.value, method


def test_casscf_and_nevpt2_are_not_marked_deterministic():
    """They moved ~1-4 mHa between runs; asserting them tightly is wrong."""
    assert runner.METHOD_TIERS["CASSCF"] == "optimizer-dependent"
    assert runner.METHOD_TIERS["NEVPT2"] == "optimizer-dependent"


def test_nevpt_class_is_resolved_by_the_right_name():
    """
    Guards the API-name fix. PySCF's class is NEVPT; "NEVPT2" is the
    literature name for the method and does not exist as an attribute.
    Asking for it produced a silent 'NEVPT2 FAILED' row with the real cause
    (an AttributeError) buried in a stderr warning.

    Checks the getattr targets in the AST rather than scanning the file for
    strings -- the docstrings legitimately mention the wrong name while
    explaining the bug.
    """
    tree = ast.parse(Path(runner.__file__).read_text())
    resolver = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_resolve_nevpt_class"
    )

    requested = [
        node.args[1].value
        for node in ast.walk(resolver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
    ]

    assert requested, "_resolve_nevpt_class must look the solver up by name"
    assert set(requested) == {"NEVPT"}, (
        f"solver must be resolved as 'NEVPT', got {sorted(set(requested))}"
    )


# ── Numerical: needs PySCF ───────────────────────────────────────────────

def _stage_golden_step1(golden_dir, cfg, system):
    """
    Copy the validated step 1 pickle into the run's results directory.

    Without it, CASSCF falls back to a guessed active space and the
    comparison is meaningless -- the golden CASSCF/NEVPT2 numbers were
    produced with ASF's space. Concretely: LiH's golden space is (2e,2o)
    but the fallback picks (4e,3o); N2's is (4e,4o) but the fallback picks
    (10e,5o), which is completely filled, leaves no correlating degrees of
    freedom, and makes CASSCF return exactly the HF energy.

    This is a test-fixture concern, not a pipeline one. In a real run,
    step 1 has already written this file.
    """
    import shutil

    os.makedirs(cfg.results_dir, exist_ok=True)
    shutil.copy(golden_dir / system / "step1_asf.pkl", cfg.step1_file)


@pytest.mark.needs_pyscf
@pytest.mark.slow
@pytest.mark.parametrize("system", ["LiH", "N2"])
def test_energies_match_golden(system, tmp_path, golden_dir):
    """
    Run step 0 for real and compare against the validated pickle.

    Deterministic methods are held to 1e-9 Ha. CASSCF and NEVPT2 get a
    2 mHa window because they depend on which solution the optimiser finds.
    ScH is excluded: it takes minutes and its CASSCF is the least stable of
    the three.
    """
    ref = SYSTEMS[system]
    cfg = Config(
        molecule=system,
        basis="sto-3g",
        project_dir=str(tmp_path),
        classical_methods=["HF", "MP2", "CCSD", "CCSD_T", "CASSCF", "NEVPT2"],
    )
    cfg.validate().make_dirs().load_geometry()
    _stage_golden_step1(golden_dir, cfg, system)

    results = runner.main(cfg, force=True)

    # The active space must be the one the golden numbers came from.
    assert results["methods"]["CASSCF"]["nel"] == ref["structure"]["nel"]
    assert results["methods"]["CASSCF"]["norb"] == ref["structure"]["n_imp"]

    for method, (expected, tier) in ref["energies"].items():
        if method.startswith("DMET"):
            continue
        got = results["methods"][method]["energy"]
        assert got is not None, f"{method} failed to run"
        tol = 1e-9 if tier is Tier.DETERMINISTIC else 2e-3
        assert abs(got - expected) <= tol, (
            f"{system} {method}: got {got:.9f}, expected {expected:.9f} "
            f"(tol {tol:g}, tier {tier.value})"
        )


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_result_pickle_is_comparable_to_golden(tmp_path, golden_dir):
    """The written pickle must line up key-for-key with the golden one."""
    from compare_pickles import compare

    cfg = Config(
        molecule="LiH",
        basis="sto-3g",
        project_dir=str(tmp_path),
        classical_methods=["HF", "MP2", "CCSD", "CCSD_T", "CASSCF", "NEVPT2"],
    )
    cfg.validate().make_dirs().load_geometry()
    _stage_golden_step1(golden_dir, cfg, "LiH")
    runner.main(cfg, force=True)

    with open(golden_dir / "LiH" / "step0_classical.pkl", "rb") as fh:
        golden = pickle.load(fh)
    with open(cfg.step0_file, "rb") as fh:
        produced = pickle.load(fh)

    # Everything is held to 1e-9 except the two optimizer-dependent
    # energies, which get the same 2 mHa window as the tier table. Loosening
    # them by full dotted path rather than by the bare name "energy" keeps
    # the deterministic energies tight.
    report = compare(
        golden, produced, tol=1e-9,
        key_tol={
            "methods.CASSCF.energy": 2e-3,
            "methods.NEVPT2.energy": 2e-3,
        },
    )
    # 'tier' and 'provenance' are new in 0.2 and absent from golden; the
    # comparator reports extra keys as informational, not failures.
    assert report.ok, report.render(verbose=True)


@pytest.mark.needs_pyscf
def test_cache_is_not_reused_across_molecules(tmp_path):
    """
    Write a LiH step 0, then ask for ScH in the same project dir. The stale
    LiH pickle must be ignored rather than silently returned.
    """
    cfg_lih = Config(molecule="LiH", basis="sto-3g", project_dir=str(tmp_path),
                     classical_methods=["HF"])
    cfg_lih.validate().make_dirs().load_geometry()
    runner.main(cfg_lih, force=True)

    cfg_sch = Config(molecule="ScH", basis="sto-3g", project_dir=str(tmp_path),
                     classical_methods=["HF"])
    cfg_sch.validate().load_geometry()
    assert not cfg_sch.cached_result_is_current(cfg_sch.step0_file, verbose=False)
