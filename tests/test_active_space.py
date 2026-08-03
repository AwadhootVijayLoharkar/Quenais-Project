"""
Stage 1 (active-space finder) tests.

The pure-numpy tests cover the two silent bugs this stage carries:
degenerate orbital pairs being split, and occupations being read off the
wrong basis. Neither needs PySCF, so both run everywhere.

The needs_pyscf tests run the real stage. Note that LiH and N2 exercise the
automatic ASF path, which additionally requires block2 and the asf package;
ScH's forced path does not, which is why it is the one marked needs_pyscf
alone.
"""

from __future__ import annotations

import ast
import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from quenais.active_space import finder
from quenais.config import Config
from quenais.settings import AsfSettings

import sys

sys.path.insert(0, str(Path(__file__).parent / "regression"))
from reference_values import SYSTEMS  # noqa: E402


# ── find_gap_cutoff: the degenerate-pair fix ─────────────────────────────

# N2's two pi orbitals, identical deviation, sitting at sorted positions
# 2 and 3. The orbital-count bounds are what force a cut between them: the
# gap *inside* a degenerate block is zero, so adaptive selection would never
# choose it on merit. min_n == max_n == 3 reproduces that.
DEGENERATE_VALUES = np.array([0.9, 0.5, 0.246, 0.246, 0.01])


def test_degenerate_pair_is_not_split():
    """
    The invariant is that the pair is not SPLIT -- keeping both or dropping
    both are equally valid. Asserting they must be kept would be wrong.
    """
    # The extension warning is the expected behaviour here and is asserted
    # by test_extension_warns; suppress it so `pytest -q` stays clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _k, _gap, selected = finder.find_gap_cutoff(
            DEGENERATE_VALUES, min_n=3, max_n=3, degeneracy_tol=1e-3
        )
    assert (2 in selected) == (3 in selected), (
        f"degenerate pair split: kept indices {sorted(selected)}"
    )


def test_degenerate_pair_would_be_split_without_the_tolerance():
    """Confirms the test above actually exercises the fix."""
    _k, _gap, selected = finder.find_gap_cutoff(
        DEGENERATE_VALUES, min_n=3, max_n=3, degeneracy_tol=0.0
    )
    assert (2 in selected) != (3 in selected), (
        "with zero tolerance the pair should split -- if it does not, this "
        "input no longer tests the degeneracy logic"
    )


def test_extension_reports_a_consistent_count():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k, _gap, selected = finder.find_gap_cutoff(
            DEGENERATE_VALUES, min_n=3, max_n=3, degeneracy_tol=1e-3
        )
    assert k == len(selected) == 4, "cutoff extended from 3 to 4"


def test_extension_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_gap_cutoff(
            DEGENERATE_VALUES, min_n=3, max_n=3, degeneracy_tol=1e-3
        )
    assert any("degenerate" in str(w.message) for w in caught)


def test_extension_past_max_norb_warns_too():
    """Extending beyond gap_max_norb is legal but must be surfaced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finder.find_gap_cutoff(
            DEGENERATE_VALUES, min_n=3, max_n=3, degeneracy_tol=1e-3
        )
    assert any("gap_max_norb" in str(w.message) for w in caught)


def test_non_degenerate_values_are_untouched():
    values = np.array([0.9, 0.5, 0.2, 0.01])
    k, _gap, selected = finder.find_gap_cutoff(
        values, min_n=1, max_n=4, degeneracy_tol=1e-3
    )
    assert k == len(selected) == len(set(selected))


def test_cutoff_respects_bounds_when_min_equals_max():
    values = np.array([0.9, 0.5, 0.2, 0.01])
    k, _gap, selected = finder.find_gap_cutoff(values, min_n=2, max_n=2)
    assert k == 2 and len(selected) == 2


def test_selected_indices_are_the_largest_values():
    values = np.array([0.1, 0.9, 0.05, 0.7])
    _k, _gap, selected = finder.find_gap_cutoff(values, min_n=2, max_n=2)
    assert set(selected) == {1, 3}


# ── project_occupations: the wrong-basis fix ─────────────────────────────

def _orthonormal(n, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def test_occupations_are_basis_correct():
    """
    Build a density that is diagonal in basis A, then project it onto a
    different basis B. Reading A's diagonal while claiming to be in B is
    the bug; projecting properly is the fix.
    """
    n = 6
    S = np.eye(n)
    A = _orthonormal(n, seed=1)
    occ_true = np.array([2.0, 2.0, 1.4, 0.6, 0.0, 0.0])
    dm_ao = A @ np.diag(occ_true) @ A.T

    _dev_a, no_occ_a = finder.project_occupations(A, dm_ao, S)
    assert np.allclose(no_occ_a, occ_true, atol=1e-10)

    B = _orthonormal(n, seed=2)
    _dev_b, no_occ_b = finder.project_occupations(B, dm_ao, S)
    assert not np.allclose(no_occ_b, occ_true, atol=1e-6), (
        "a different basis must give different occupations -- otherwise "
        "this test cannot detect the basis mix-up"
    )
    assert abs(no_occ_b.sum() - occ_true.sum()) < 1e-8, "trace must be preserved"


def test_degenerate_subspace_gives_equal_occupations():
    """
    The property that makes the basis matter: within a degenerate subspace
    the density is proportional to the identity, so any orthonormal basis
    of that subspace must report equal occupations.
    """
    n = 4
    S = np.eye(n)
    A = _orthonormal(n, seed=3)
    occ = np.array([2.0, 0.5, 0.5, 0.0])       # a degenerate pair
    dm_ao = A @ np.diag(occ) @ A.T

    rot = np.eye(n)
    theta = 0.7
    rot[1:3, 1:3] = [[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]]
    B = A @ rot                                 # rotate within the subspace

    _dev, no_occ = finder.project_occupations(B, dm_ao, S)
    assert abs(no_occ[1] - no_occ[2]) < 1e-10


def test_deviation_is_symmetric_about_half_filling():
    S = np.eye(3)
    C = np.eye(3)
    dm = np.diag([2.0, 1.0, 0.0])
    dev, no_occ = finder.project_occupations(C, dm, S)
    assert np.allclose(dev, np.minimum(no_occ, 2.0 - no_occ))
    assert dev[0] == pytest.approx(0.0)
    assert dev[1] == pytest.approx(1.0)


# ── Structural ───────────────────────────────────────────────────────────

def test_main_signature_matches_stage_convention():
    assert list(inspect.signature(finder.main).parameters)[:2] == ["cfg", "force"]


def test_no_heavy_imports_at_module_level():
    tree = ast.parse(Path(finder.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or node.names[0].name
            root = name.split(".")[0]
            assert root not in {"pyscf", "asf"}, f"module-level import: {name}"


def test_mp2_density_is_spin_resolved():
    """
    The embedding stage needs alpha and beta separately; the 0.1 version
    returned only a total and step 2 could not run.
    """
    src = inspect.getsource(finder.compute_mp2_density)
    assert "dm_ao_alpha" in src and "dm_ao_beta" in src
    assert "deviation" not in src.split('"""')[2], (
        "compute_mp2_density must not compute occupations -- it only has "
        "the canonical UHF basis"
    )


def test_step1_contract_keys_are_written():
    """The three keys the embedding stage hard-requires must be in main()."""
    tree = ast.parse(Path(finder.__file__).read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    literals = {n.value for n in ast.walk(main)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for key in ("dm_ao_alpha_mp2", "dm_ao_beta_mp2", "dm_ao_total_mp2",
                "mo_coeff", "no_occ", "nel", "mo_list"):
        assert key in literals, f"step 1 pickle must contain {key}"


def test_golden_pickles_satisfy_the_contract(golden_dir):
    for system in ("LiH", "N2", "ScH"):
        with open(golden_dir / system / "step1_asf.pkl", "rb") as fh:
            data = pickle.load(fh)
        for key in ("dm_ao_alpha_mp2", "dm_ao_beta_mp2", "dm_ao_total_mp2"):
            assert key in data, f"{system} golden step 1 missing {key}"
        assert np.allclose(
            data["dm_ao_total_mp2"],
            data["dm_ao_alpha_mp2"] + data["dm_ao_beta_mp2"],
            atol=1e-12,
        )


@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
def test_occupations_in_golden_match_the_saved_basis(system, golden_dir):
    """
    The basis-consistency invariant, checked against real data.

    no_occ must be reproducible from the mo_coeff and MP2 density stored
    beside it. If step 1 ever regresses to saving canonical-basis
    occupations next to an ASF basis, this fails.

    The overlap matrix is not in the pickle, but it is recoverable: the
    saved mo_coeff is orthonormal with respect to S, so C^T S C = I, and
    for square invertible C that gives S = (C C^T)^-1. No PySCF needed.
    """
    with open(golden_dir / system / "step1_asf.pkl", "rb") as fh:
        data = pickle.load(fh)

    C = np.asarray(data["mo_coeff"])
    assert C.shape[0] == C.shape[1], "expected a square MO coefficient matrix"

    S = np.linalg.inv(C @ C.T)
    # Sanity: recovered S must actually orthonormalise the saved basis.
    assert np.allclose(C.T @ S @ C, np.eye(C.shape[0]), atol=1e-8), (
        f"{system}: saved mo_coeff is not S-orthonormal; the recovered "
        f"overlap is unreliable and this invariant cannot be checked"
    )

    _dev, no_occ = finder.project_occupations(C, data["dm_ao_total_mp2"], S)
    assert np.allclose(no_occ, data["no_occ"], atol=1e-8), (
        f"{system}: saved no_occ does not match the density projected onto "
        f"the saved mo_coeff -- occupations and basis have drifted apart"
    )

    dev_expected = np.minimum(no_occ, 2.0 - no_occ)
    assert np.allclose(dev_expected, data["deviation"], atol=1e-8)


def test_force_active_space_rejects_out_of_range_indices():
    """ScH's indices applied to LiH must fail loudly, not silently truncate."""
    src = Path(finder.__file__).read_text()
    assert "out_of_range" in src
    assert "do not exist for" in src


# ── Real runs ────────────────────────────────────────────────────────────

@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_sch_forced_space_matches_golden(tmp_path, golden_dir):
    """
    ScH with a forced active space. Does not need block2 or the asf
    package, so it is the cheapest end-to-end check of this stage.
    """
    ref = SYSTEMS["ScH"]
    cfg = Config(
        molecule="ScH",
        basis="sto-3g",
        project_dir=str(tmp_path),
        asf=AsfSettings(force_active_space=ref["force_active_space"]),
    )
    cfg.validate().make_dirs().load_geometry()

    results = finder.main(cfg, force=True)

    assert list(results["mo_list"]) == ref["structure"]["mo_list"]
    assert results["nel"] == ref["structure"]["nel"]
    assert results["n_active_orbs"] == ref["structure"]["n_imp"]
    assert results["forced_active_space"] is True

    with open(golden_dir / "ScH" / "step1_asf.pkl", "rb") as fh:
        golden = pickle.load(fh)
    assert results["uhf_energy"] == pytest.approx(golden["uhf_energy"], abs=1e-8)
    assert results["corr_strength"] == pytest.approx(
        golden["corr_strength"], abs=1e-8
    )
    for key in ("dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
        assert np.allclose(results[key], golden[key], atol=1e-9), key


@pytest.mark.needs_pyscf
def test_force_active_space_out_of_range_raises(tmp_path):
    """LiH has 6 AOs; ScH's MO indices must be rejected."""
    cfg = Config(
        molecule="LiH",
        basis="sto-3g",
        project_dir=str(tmp_path),
        asf=AsfSettings(force_active_space=[9, 10, 11, 12, 13, 14]),
    )
    cfg.validate().make_dirs().load_geometry()
    with pytest.raises(ValueError, match="do not exist"):
        finder.main(cfg, force=True)
