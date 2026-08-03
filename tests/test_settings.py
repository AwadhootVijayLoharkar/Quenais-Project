"""
Tests for the settings groups and the rebuilt Config.

The Hydra-override test is the important one in this file. Those strings are
consumed by an external process; if a key is misspelled or a bool is
capitalised, Hydra rejects the run -- loudly, which is fine -- but if a key
is silently DROPPED, the external trainer runs happily against its own
defaults and produces a plausible wrong answer. That is the failure this
package exists to prevent, so the override list is asserted against the
exact output of the validated test_8 script.
"""

from __future__ import annotations

import pickle
import warnings

import pytest

from quenais.config import (
    GQE_SOLVERS,
    QISKIT_SOLVERS,
    SOLVER_ALIASES,
    SOLVERS,
    Config,
)
from quenais.settings import (
    AsfSettings,
    DmetSettings,
    GqeSettings,
    QiskitSolverSettings,
    TierSettings,
)
from quenais.settings.gqe import _fmt

# Captured verbatim from test_8/config.py build_gqe_hydra_overrides() with
# its shipped ScH values. Order is part of the contract.
TEST_8_OVERRIDES = [
    "molecule=dmet_embedding",
    "trainer.max_iters=120",
    "trainer.num_samples=100",
    "trainer.batch_size=100",
    "trainer.warmup_size=100",
    "trainer.buffer_size=100",
    "trainer.load_checkpoint=false",
    "ngates=40",
    "operator_pool.spec=dmet_excitation",
    "operator_pool.remove_z_ladder=false",
    "operator_pool.only_use_first_pauli=false",
    "qsci.max_dim=10000",
]


# ── Hydra override construction ──────────────────────────────────────────

def test_default_overrides_match_test_8_exactly():
    assert GqeSettings().hydra_overrides() == TEST_8_OVERRIDES


def test_none_fields_are_skipped():
    """A None field means 'use the external repo's own default'."""
    overrides = GqeSettings(seed=None, sampler_shots=None).hydra_overrides()
    assert not any(o.startswith("trainer.seed=") for o in overrides)
    assert not any(o.startswith("sampler.shots=") for o in overrides)


def test_set_fields_appear_with_correct_prefix():
    """
    Keys from configs/trainer/default.yaml need a "trainer." prefix; keys
    from configs/default.yaml must not have one.
    """
    overrides = GqeSettings(seed=32, sampler_shots=1024).hydra_overrides()
    assert "trainer.seed=32" in overrides
    assert "sampler.shots=1024" in overrides


def test_booleans_are_lowercase():
    """Hydra rejects Python's True/False capitalisation."""
    assert _fmt(True) == "true"
    assert _fmt(False) == "false"
    assert "trainer.load_checkpoint=false" in GqeSettings().hydra_overrides()


def test_lists_use_hydra_syntax_not_json():
    """
    Hydra's override grammar wants [a,b]. json.dumps emits ["a", "b"] with
    double quotes and spaces, which it rejects.
    """
    assert _fmt(["R-CASCI", "R-CCSD"]) == "[R-CASCI,R-CCSD]"
    overrides = GqeSettings(reference_keys=["R-CASCI", "R-CCSD"]).hydra_overrides()
    assert "reference_keys=[R-CASCI,R-CCSD]" in overrides


def test_step2_path_override_is_absolute():
    overrides = GqeSettings().hydra_overrides(step2_pickle_path="results/step2.pkl")
    match = [o for o in overrides if o.startswith("molecule.step2_pickle_path=")]
    assert len(match) == 1
    assert match[0].split("=", 1)[1].startswith("/")


def test_extra_overrides_come_last():
    overrides = GqeSettings(extra_overrides=["foo=bar"]).hydra_overrides()
    assert overrides[-1] == "foo=bar"


def test_cudaq_target_is_not_a_hydra_key():
    """It goes through CUDAQ_DEFAULT_SIMULATOR; the repo has no config key."""
    overrides = GqeSettings(cudaq_target="nvidia").hydra_overrides()
    assert not any("cudaq" in o or "target" in o for o in overrides)
    assert GqeSettings(cudaq_target="nvidia").env_overlay() == {
        "CUDAQ_DEFAULT_SIMULATOR": "nvidia"
    }


# ── GQE validation ───────────────────────────────────────────────────────

@pytest.mark.parametrize("spec", ["pauli_evolution", "excitation"])
def test_stock_pools_are_rejected(spec):
    """
    Upstream's pools rebuild the molecule from its geometry. A DMET embedding
    has no geometry, so they die with "'NoneType' object is not iterable".
    Reject at config time rather than 20 minutes into a run.
    """
    with pytest.raises(ValueError, match="geometry"):
        GqeSettings(operator_pool_spec=spec).validate()


def test_unknown_pool_is_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        GqeSettings(operator_pool_spec="nonsense").validate()


def test_hardware_target_is_rejected():
    with pytest.raises(ValueError, match="shot-based"):
        GqeSettings(cudaq_target="quantinuum").validate()


def test_batch_size_must_match_num_samples():
    with pytest.raises(ValueError, match="online training"):
        GqeSettings(num_samples=100, batch_size=50).validate()


def test_empty_molecule_config_is_rejected():
    """Without it, train.py silently loads n2.yaml."""
    with pytest.raises(ValueError, match="n2.yaml"):
        GqeSettings(molecule_config="").validate()


def test_default_gqe_settings_validate():
    GqeSettings().validate()


# ── DMET / ASF / Qiskit / tier validation ────────────────────────────────

def test_max_embed_orbs_default_is_the_validated_value():
    """18 is what every reference number was produced with (0.1 shipped 24)."""
    assert DmetSettings().max_embed_orbs == 18


def test_dmet_reference_default_is_casci():
    assert DmetSettings().reference == "casci"


def test_bad_dmet_reference_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        DmetSettings(reference="ccsd").validate()


def test_mu_range_pair_is_accepted_and_ordered():
    DmetSettings(mu_search_range=(-5.0, 5.0)).validate()
    with pytest.raises(ValueError, match="lo >= hi"):
        DmetSettings(mu_search_range=(5.0, -5.0)).validate()


def test_asf_degeneracy_tolerance_present():
    """The knob that stops a degenerate orbital pair being split."""
    assert AsfSettings().gap_degeneracy_tol == 1e-3


def test_force_active_space_validation():
    AsfSettings(force_active_space=[9, 10, 11, 12, 13, 14]).validate()
    with pytest.raises(ValueError, match="duplicates"):
        AsfSettings(force_active_space=[1, 1]).validate()
    with pytest.raises(ValueError, match="non-negative"):
        AsfSettings(force_active_space=[-1]).validate()
    with pytest.raises(ValueError, match="use None"):
        AsfSettings(force_active_space=[]).validate()


def test_asf_bounds_are_checked():
    with pytest.raises(ValueError, match="gap_max_norb"):
        AsfSettings(gap_min_norb=8, gap_max_norb=4).validate()


def test_qiskit_ibm_backend_requires_a_name():
    with pytest.raises(ValueError, match="ibm_backend_name"):
        QiskitSolverSettings(backend="ibm").validate()


def test_tier_settings_detect_transition_metals():
    tiers = TierSettings()
    assert tiers.is_transition_metal_system(["Sc", "H"])
    assert not tiers.is_transition_metal_system(["N", "N"])
    assert not tiers.is_transition_metal_system([])


# ── Config ───────────────────────────────────────────────────────────────

def test_settings_are_nested_not_flattened():
    cfg = Config()
    assert cfg.dmet.bath_tolerance == 1e-8
    assert cfg.gqe.ngates == 40
    assert cfg.asf.gap_max_norb == 16
    assert not hasattr(cfg, "bath_tolerance"), "settings must not be flattened"
    assert not hasattr(cfg, "gqe_ngates"), "settings must not be flattened"


def test_settings_groups_can_be_injected():
    cfg = Config(dmet=DmetSettings(reference="mp2", max_embed_orbs=24))
    assert cfg.dmet.reference == "mp2"
    assert cfg.dmet.max_embed_orbs == 24


def test_solver_registry_is_consistent():
    assert set(SOLVERS) == set(QISKIT_SOLVERS) | set(GQE_SOLVERS)
    assert "gqe" in SOLVERS
    for name in QISKIT_SOLVERS:
        assert Config(quantum_solver=name).is_qiskit
    for name in GQE_SOLVERS:
        assert Config(quantum_solver=name).is_gqe


def test_gqe_qsci_alias_is_accepted_with_a_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = Config(quantum_solver="gqe_qsci")
    assert cfg.quantum_solver == "gqe"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert SOLVER_ALIASES["gqe_qsci"] == "gqe"


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError, match="Unknown quantum_solver"):
        Config(quantum_solver="vqe").validate()


def test_validate_only_checks_the_selected_stack():
    """A Qiskit user must not be blocked by an unrelated GQE setting."""
    cfg = Config(quantum_solver="sqd", gqe=GqeSettings(operator_pool_spec="excitation"))
    cfg.validate()          # must not raise
    cfg.quantum_solver = "gqe"
    with pytest.raises(ValueError, match="geometry"):
        cfg.validate()


def test_validate_returns_self_for_chaining():
    cfg = Config(molecule="LiH", basis="sto-3g")
    assert cfg.validate() is cfg


@pytest.mark.parametrize("molecule", ["LiH", "N2", "ScH", "H2O"])
def test_builtin_geometries_load_without_cif(molecule):
    cfg = Config(molecule=molecule, basis="sto-3g").load_geometry()
    assert cfg.n_atoms == len(cfg.geometry)
    assert cfg.atom_syms and all(isinstance(s, str) for s in cfg.atom_syms)


def test_paths_derive_from_project_dir(tmp_path):
    cfg = Config(project_dir=str(tmp_path))
    assert cfg.step2_file.endswith("step2_hamiltonian.pkl")
    assert cfg.results_dir == str(tmp_path / "results")
    cfg.make_dirs()
    assert (tmp_path / "results" / "plots").is_dir()


# ── Cache identity validation ────────────────────────────────────────────

def test_cache_rejects_another_molecules_pickle(golden_dir, tmp_path):
    """
    The stale-cache failure, reproduced against real data: point a ScH config
    at LiH's step 2 pickle and it must refuse.
    """
    cfg = Config(molecule="ScH", basis="sto-3g", project_dir=str(tmp_path))
    lih_step2 = golden_dir / "LiH" / "step2_hamiltonian.pkl"
    assert not cfg.cached_result_is_current(str(lih_step2), verbose=False)


@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
@pytest.mark.parametrize(
    "stage", ["step0_classical.pkl", "step1_asf.pkl", "step2_hamiltonian.pkl"]
)
def test_cache_accepts_its_own_pickle(golden_dir, system, stage, tmp_path):
    cfg = Config(molecule=system, basis="sto-3g", project_dir=str(tmp_path))
    assert cfg.cached_result_is_current(str(golden_dir / system / stage), verbose=False)


def test_cache_rejects_basis_mismatch(golden_dir, tmp_path):
    cfg = Config(molecule="LiH", basis="def2-svp", project_dir=str(tmp_path))
    path = golden_dir / "LiH" / "step2_hamiltonian.pkl"
    assert not cfg.cached_result_is_current(str(path), verbose=False)


def test_cache_rejects_missing_and_untagged(tmp_path):
    cfg = Config(molecule="LiH", basis="sto-3g", project_dir=str(tmp_path))
    assert not cfg.cached_result_is_current(str(tmp_path / "nope.pkl"), verbose=False)

    untagged = tmp_path / "untagged.pkl"
    with open(untagged, "wb") as fh:
        pickle.dump({"h1e": None}, fh)
    assert not cfg.cached_result_is_current(str(untagged), verbose=False)

    corrupt = tmp_path / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle")
    assert not cfg.cached_result_is_current(str(corrupt), verbose=False)
