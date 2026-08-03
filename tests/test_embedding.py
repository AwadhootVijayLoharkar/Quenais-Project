"""
Stage 2 (DMET embedding) tests.

This is the highest-risk stage: three of the five known physics bugs live
here, and every one of them is silent -- right shape, plausible magnitude,
wrong value. The adaptive_bath tests below are pure numpy and run
everywhere; they are the cheapest guard against the worst bug (~20 Ha).
"""

from __future__ import annotations

import ast
import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from quenais.config import Config
from quenais.embedding import dmet_lib, hamiltonian
from quenais.settings import AsfSettings, DmetSettings

import sys

sys.path.insert(0, str(Path(__file__).parent / "regression"))
from reference_values import EMBEDDED_SCF_VS_UHF_TOL, SYSTEMS  # noqa: E402


# ── adaptive_bath: the fabricated-bath bug ───────────────────────────────

def test_all_zero_singular_values_give_no_bath():
    """
    N2's case. Every Schmidt singular value is numerically zero, so the
    correct answer is zero bath orbitals. The pre-fix code returned the top
    max_bath values regardless and produced ~20 Ha of error.
    """
    sv = np.zeros(4)
    n_bath, gap, cov = dmet_lib.adaptive_bath(sv, n_imp=4, max_embed=18,
                                              bath_tol=1e-8)
    assert n_bath == 0
    assert gap == 0.0 and cov == 0.0


def test_noise_level_singular_values_give_no_bath():
    """Values below tolerance are noise, not physics -- even if nonzero."""
    sv = np.array([5.4e-15, 3.1e-15, 1.0e-16, 0.0])
    n_bath, _gap, _cov = dmet_lib.adaptive_bath(sv, n_imp=4, max_embed=18,
                                                bath_tol=1e-8)
    assert n_bath == 0


def test_real_singular_values_do_give_a_bath():
    """The guard must not be so strict that legitimate baths vanish."""
    sv = np.array([0.0061, 0.0061, 0.0036, 0.0010, 0.00039, 0.0])
    n_bath, _gap, cov = dmet_lib.adaptive_bath(sv, n_imp=6, max_embed=18,
                                               bath_tol=1e-8)
    assert n_bath > 0
    assert 0.0 < cov <= 1.0


def test_bath_is_capped_by_max_embed():
    sv = np.full(8, 0.1)
    n_bath, _gap, _cov = dmet_lib.adaptive_bath(sv, n_imp=6, max_embed=8,
                                                bath_tol=1e-8)
    assert n_bath <= 8 - 6


def test_bath_never_exceeds_n_imp():
    sv = np.full(20, 0.1)
    n_bath, _gap, _cov = dmet_lib.adaptive_bath(sv, n_imp=3, max_embed=100,
                                                bath_tol=1e-8)
    assert n_bath <= 3


def test_zero_max_bath_is_handled():
    sv = np.array([0.5, 0.4])
    assert dmet_lib.adaptive_bath(sv, n_imp=6, max_embed=6, bath_tol=1e-8)[0] == 0


def test_empty_singular_values():
    assert dmet_lib.adaptive_bath(np.array([]), 4, 18, 1e-8)[0] == 0


@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
def test_adaptive_bath_reproduces_golden_n_bath(system, golden_dir):
    """Replay the real Schmidt spectra and check the bath count matches."""
    ref = SYSTEMS[system]
    with open(golden_dir / system / "step2_hamiltonian.pkl", "rb") as fh:
        data = pickle.load(fh)

    n_bath, _gap, _cov = dmet_lib.adaptive_bath(
        data["sv_all"],
        n_imp=ref["structure"]["n_imp"],
        max_embed=DmetSettings().max_embed_orbs,
        bath_tol=DmetSettings().bath_tolerance,
    )
    assert n_bath == ref["structure"]["n_bath"], (
        f"{system}: got {n_bath} bath orbitals, golden has "
        f"{ref['structure']['n_bath']}"
    )


# ── Linear-algebra helpers ───────────────────────────────────────────────

def test_lowdin_matrices_are_inverses():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    S = A @ A.T + 6 * np.eye(6)
    S_sqrt, S_invsqrt = dmet_lib.lowdin_matrices(S)
    assert np.allclose(S_sqrt @ S_sqrt, S, atol=1e-10)
    assert np.allclose(S_sqrt @ S_invsqrt, np.eye(6), atol=1e-10)


def test_symmetrize_h2e_imposes_eightfold_symmetry():
    rng = np.random.default_rng(1)
    h2e = rng.standard_normal((4, 4, 4, 4))
    sym = dmet_lib.symmetrize_h2e(h2e)
    for axes in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]:
        assert np.allclose(sym, sym.transpose(axes), atol=1e-12)


def test_symmetrize_is_idempotent():
    rng = np.random.default_rng(2)
    h2e = rng.standard_normal((4, 4, 4, 4))
    once = dmet_lib.symmetrize_h2e(h2e)
    assert np.allclose(once, dmet_lib.symmetrize_h2e(once), atol=1e-12)


# ── Chemical potential ───────────────────────────────────────────────────

# N2's situation, as recorded during debugging: the core mean-field
# potential pushed FOUR of eight embedding eigenvalues below -5 Ha while
# the target electron count was 3. A fixed (-5, 5) guess then has
# n(mu=-5) = 4 > 3, so the target is below the bracket entirely and
# bisection cannot reach it.
SHIFTED_SPECTRUM = np.diag([-40.0, -38.0, -20.0, -12.0, 1.0, 2.0, 3.0, 4.0])
SHIFTED_N_EMB = 8
SHIFTED_TARGET = 3


def test_mu_auto_range_finds_the_level():
    """
    Bisection converges onto the target level itself, so counting
    eigenvalues strictly below zero after the shift is sign-dependent at
    the boundary. The meaningful property is where mu lands: between the
    3rd and 4th levels for a 3-electron target.
    """
    _shifted, mu = dmet_lib.chemical_potential_correction(
        SHIFTED_SPECTRUM, n_emb=SHIFTED_N_EMB,
        n_alpha=SHIFTED_TARGET, n_beta=SHIFTED_TARGET, mu_range="auto"
    )
    assert np.isfinite(mu)
    assert -20.0 - 1e-6 <= mu <= -12.0 + 1e-6, (
        f"mu={mu} is outside the gap between the 3rd and 4th levels"
    )


def test_fixed_range_fails_where_auto_range_succeeds():
    """
    The reason the auto bracket exists. A fixed (-5, 5) guess must warn and
    no-op on this spectrum, while the auto bracket must find mu.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _shifted, mu_fixed = dmet_lib.chemical_potential_correction(
            SHIFTED_SPECTRUM, n_emb=SHIFTED_N_EMB,
            n_alpha=SHIFTED_TARGET, n_beta=SHIFTED_TARGET,
            mu_range=(-5.0, 5.0),
        )
    assert mu_fixed == 0.0, "a fixed range should fail to bracket here"
    assert any("bracket" in str(w.message) for w in caught)

    _shifted, mu_auto = dmet_lib.chemical_potential_correction(
        SHIFTED_SPECTRUM, n_emb=SHIFTED_N_EMB,
        n_alpha=SHIFTED_TARGET, n_beta=SHIFTED_TARGET, mu_range="auto",
    )
    assert mu_auto != 0.0


def test_mu_warns_and_no_ops_when_target_cannot_be_bracketed():
    h1e = np.diag([-1.0, -1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shifted, mu = dmet_lib.chemical_potential_correction(
            h1e, n_emb=2, n_alpha=5, n_beta=5, mu_range=(-0.1, 0.1)
        )
    assert mu == 0.0
    assert np.allclose(shifted, h1e)
    assert any("bracket" in str(w.message) for w in caught)


def test_mu_warns_for_unequal_spin():
    h1e = np.diag([-3.0, -1.0, 1.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dmet_lib.chemical_potential_correction(
            h1e, n_emb=4, n_alpha=3, n_beta=1, mu_range="auto"
        )
    assert any("n_alpha == n_beta" in str(w.message) for w in caught)


# ── Consistency score ────────────────────────────────────────────────────

def test_consistency_score_flags_large_mismatch():
    step2 = {"ref_occ_alpha": np.array([1.0, 1.0, 0.0]),
             "ref_occ_beta": np.array([1.0, 1.0, 0.0])}
    result = dmet_lib.embedding_consistency_score(
        step2, (np.array([0.2, 0.2, 0.9]), np.array([0.2, 0.2, 0.9])),
        threshold=0.1,
    )
    assert result["flag"] is True


def test_consistency_score_requires_ref_occ():
    with pytest.raises(KeyError, match="ref_occ"):
        dmet_lib.embedding_consistency_score({}, (np.zeros(2), np.zeros(2)))


# ── Structural ───────────────────────────────────────────────────────────

def test_main_signature_matches_stage_convention():
    assert list(inspect.signature(hamiltonian.main).parameters)[:2] == [
        "cfg", "force"
    ]


def test_no_module_level_pyscf_import():
    for mod in (hamiltonian, dmet_lib):
        tree = ast.parse(Path(mod.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = getattr(node, "module", None) or node.names[0].name
                assert not name.startswith("pyscf"), f"{mod.__name__}: {name}"


def test_dmet_lib_has_no_module_level_side_effects():
    """
    dmet_lib must be importable from anywhere. The original script called
    sys.exit(0) at module scope on a cache hit, which killed any process
    that imported it.
    """
    tree = ast.parse(Path(dmet_lib.__file__).read_text())
    for node in tree.body:
        assert isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef,
             ast.Assign, ast.AnnAssign, ast.Expr),
        ), f"unexpected module-level statement: {type(node).__name__}"
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant), "module-level call"


def test_ecore_identity_is_not_asserted():
    """
    E_core is defined as mf.e_tot - e_hf_emb, so asserting that identity is
    tautological and gives false confidence. The real check is
    verify_embedded_scf.
    """
    tree = ast.parse(Path(hamiltonian.__file__).read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert not any(isinstance(n, ast.Assert) for n in ast.walk(main)), (
        "main() must not assert the tautological ecore identity"
    )


def test_step2_schema_is_versioned():
    assert hamiltonian.STEP2_SCHEMA_VERSION >= 2


def test_step2_contract_keys_are_written():
    tree = ast.parse(Path(hamiltonian.__file__).read_text())
    main = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    literals = {n.value for n in ast.walk(main)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for key in ("h1e", "h2e", "ecore", "mu", "n_emb", "n_alpha", "n_beta",
                "sv_all", "ref_occ_alpha", "ref_occ_beta", "uhf_energy",
                "reference_density_info", "schema_version"):
        assert key in literals, f"step 2 pickle must contain {key}"


# ── Real runs ────────────────────────────────────────────────────────────

def _cfg_for(system, tmp_path):
    ref = SYSTEMS[system]
    return Config(
        molecule=system,
        basis="sto-3g",
        project_dir=str(tmp_path),
        asf=AsfSettings(force_active_space=ref["force_active_space"]),
    ).validate().make_dirs().load_geometry()


def _stage_golden_step1(golden_dir, cfg, system):
    import shutil

    shutil.copy(golden_dir / system / "step1_asf.pkl", cfg.step1_file)


def _run_embedding(cfg):
    """
    Run step 2, silencing the zero-coupling RuntimeWarning.

    N2 legitimately has no active/non-active coupling, so that warning
    fires on every N2 run. It is asserted explicitly in
    test_zero_coupling_is_reported, and suppressed here so `pytest -q`
    stays clean for the expected case.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return hamiltonian.main(cfg, force=True)


@pytest.mark.needs_pyscf
@pytest.mark.slow
@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
def test_embedding_matches_golden(system, tmp_path, golden_dir):
    """
    Build the embedding from the golden step 1 and compare the structural
    invariants and scalars against the golden step 2.
    """
    ref = SYSTEMS[system]
    cfg = _cfg_for(system, tmp_path)
    _stage_golden_step1(golden_dir, cfg, system)

    results = _run_embedding(cfg)

    struct = ref["structure"]
    assert results["n_bath"] == struct["n_bath"]
    assert results["n_emb"] == struct["n_emb"]
    assert results["n_alpha"] == struct["n_alpha"]
    assert results["n_beta"] == struct["n_beta"]

    for key, (expected, tol) in ref["scalars"].items():
        assert abs(float(results[key]) - expected) <= tol, key

    for spin, (expected, tol) in ref["ref_occ_sums"].items():
        got = float(np.sum(results[f"ref_occ_{spin}"]))
        assert abs(got - expected) <= tol, f"ref_occ_{spin}"


@pytest.mark.needs_pyscf
@pytest.mark.slow
@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
def test_embedded_scf_matches_full_uhf(system, tmp_path, golden_dir):
    """
    The most diagnostic check in the pipeline, and the one that can
    actually fail. Validated at 1.3e-7 Ha on ScH.
    """
    cfg = _cfg_for(system, tmp_path)
    _stage_golden_step1(golden_dir, cfg, system)
    results = _run_embedding(cfg)

    check = results["embedded_scf_check"]
    assert check is not None, "the embedded SCF check did not run"
    assert abs(check["delta"]) <= EMBEDDED_SCF_VS_UHF_TOL, (
        f"{system}: embedded SCF differs from full UHF by "
        f"{check['delta']:.3e} Ha"
    )


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_n2_produces_no_bath_end_to_end(tmp_path, golden_dir):
    """The 20 Ha bug, checked on the real system that exposed it."""
    cfg = _cfg_for("N2", tmp_path)
    _stage_golden_step1(golden_dir, cfg, "N2")
    results = _run_embedding(cfg)

    assert results["n_bath"] == 0
    assert float(np.max(np.abs(results["sv_all"]))) < cfg.dmet.bath_tolerance
    assert results["n_emb"] == results["n_imp"]


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_lih_electron_count_is_not_the_active_space_count(tmp_path, golden_dir):
    """The 2x-HF bug, checked end to end."""
    cfg = _cfg_for("LiH", tmp_path)
    _stage_golden_step1(golden_dir, cfg, "LiH")
    results = _run_embedding(cfg)

    with open(cfg.step1_file, "rb") as fh:
        nel = pickle.load(fh)["nel"]

    assert results["n_bath"] > 0, "precondition: LiH must have a real bath"
    assert results["n_alpha"] == 2
    assert results["n_alpha"] != nel // 2 + nel % 2, (
        "the active-space formula must not agree here, or the test is vacuous"
    )


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_stale_step1_is_rejected(tmp_path, golden_dir):
    """Building an embedding on another molecule's active space must fail."""
    import shutil

    cfg = _cfg_for("ScH", tmp_path)
    shutil.copy(golden_dir / "LiH" / "step1_asf.pkl", cfg.step1_file)
    with pytest.raises(RuntimeError, match="different molecule"):
        hamiltonian.main(cfg, force=True)


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_zero_coupling_is_reported_for_n2(tmp_path, golden_dir):
    """
    N2's reference density genuinely has no active/non-active coupling.
    The warning must still fire -- the same signature would appear if the
    density were ever rebuilt block-diagonally on a system that does have
    a bath, which is the failure it guards.
    """
    cfg = _cfg_for("N2", tmp_path)
    _stage_golden_step1(golden_dir, cfg, "N2")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hamiltonian.main(cfg, force=True)
    assert any("no active/non-active coupling" in str(w.message) for w in caught)
