"""
Core Config and CIF-parser tests.

Updated for the settings-group refactor: fields that were flat attributes on
Config (ansatz, n_shots, tm_elements, asf_params, ...) now live in
cfg.qiskit / cfg.tiers / cfg.asf. The assertions below track the new
locations; the group-specific behaviour is covered in test_settings.py.

Import order is deliberate: quenais must come before numpy so the
thread-count guard in quenais/_threads.py is applied before OpenBLAS
initialises its pool. tests/conftest.py does this for the suite as a whole,
but repeating it here keeps the file correct when run on its own.
"""

import quenais  # noqa: F401  isort:skip  -- must precede numpy
from quenais.config import Config
from quenais.utils.cif_parser import load_geometry

import os

import numpy as np
import pytest


# ── Config: identity and defaults ────────────────────────────────────────

def test_config_defaults():
    cfg = Config()
    assert cfg.molecule == "TiO2"
    assert cfg.basis == "def2-svp"
    assert cfg.charge == 0
    assert cfg.spin == 0
    assert cfg.quantum_solver == "sqd"
    # Solver-specific settings now live in their own group.
    assert cfg.qiskit.ansatz == "lucj"
    assert cfg.qiskit.n_shots == 8192


def test_config_custom():
    from quenais.settings import QiskitSolverSettings

    cfg = Config(
        molecule="H2",
        basis="sto-3g",
        spin=0,
        qiskit=QiskitSolverSettings(n_shots=1024),
    )
    assert cfg.molecule == "H2"
    assert cfg.basis == "sto-3g"
    assert cfg.qiskit.n_shots == 1024


def test_config_validate_passes():
    Config().validate()


# ── Config: validation ───────────────────────────────────────────────────
#
# These now raise ValueError with an explanatory message rather than a bare
# AssertionError. A config mistake should say what is wrong and what the
# valid options are, and should survive `python -O`, which strips asserts.

def test_config_validate_fails_bad_ansatz():
    cfg = Config()
    cfg.qiskit.ansatz = "invalid"
    with pytest.raises(ValueError, match="ansatz"):
        cfg.validate()


def test_config_validate_fails_bad_solver():
    cfg = Config()
    cfg.quantum_solver = "invalid"
    with pytest.raises(ValueError, match="Unknown quantum_solver"):
        cfg.validate()


def test_config_validate_fails_bad_mapping():
    cfg = Config()
    cfg.qiskit.fermion_to_qubit = "invalid"
    with pytest.raises(ValueError, match="fermion_to_qubit"):
        cfg.validate()


def test_config_validate_fails_negative_spin():
    cfg = Config()
    cfg.spin = -1
    with pytest.raises(ValueError, match="spin"):
        cfg.validate()


def test_config_validate_message_lists_valid_solvers():
    """The error must be actionable, not just a rejection."""
    cfg = Config()
    cfg.quantum_solver = "invalid"
    with pytest.raises(ValueError, match="gqe"):
        cfg.validate()


# ── Config: paths ────────────────────────────────────────────────────────

def test_config_paths():
    cfg = Config(project_dir="/tmp/quenais_test")
    assert cfg.results_dir == "/tmp/quenais_test/results"
    assert cfg.cif_dir == "/tmp/quenais_test/cif_files"
    assert cfg.plots_dir == "/tmp/quenais_test/results/plots"
    assert cfg.step0_file == "/tmp/quenais_test/results/step0_classical.pkl"
    assert cfg.step3_file == "/tmp/quenais_test/results/step3_results.pkl"


def test_config_gqe_log_lives_under_results():
    cfg = Config(project_dir="/tmp/quenais_test")
    assert cfg.gqe_log_file == "/tmp/quenais_test/results/gqe_train.log"


def test_config_make_dirs(tmp_path):
    cfg = Config(project_dir=str(tmp_path))
    cfg.make_dirs()
    assert os.path.isdir(cfg.results_dir)
    assert os.path.isdir(cfg.cif_dir)
    assert os.path.isdir(cfg.plots_dir)


# ── Config: misc ─────────────────────────────────────────────────────────

def test_config_repr():
    r = repr(Config())
    assert "TiO2" in r
    assert "sqd" in r
    assert "casci" in r          # the DMET reference method


def test_config_constants():
    cfg = Config()
    assert abs(cfg.hartree_to_ev - 27.211386245988) < 1e-6
    assert abs(cfg.hartree_to_kcal_mol - 627.5094740631) < 1e-4


def test_config_tm_elements():
    cfg = Config()
    assert "Ti" in cfg.tiers.tm_elements
    assert "Fe" in cfg.tiers.tm_elements
    assert "H" not in cfg.tiers.tm_elements
    assert "C" not in cfg.tiers.tm_elements
    assert len(cfg.tiers.tm_elements) == 50


def test_config_asf_params():
    cfg = Config()
    for tier in (1, 2, 3):
        assert tier in cfg.asf.params
        p = cfg.asf.params[tier]
        assert "entropy_threshold" in p
        assert "max_norb" in p
        assert "min_norb" in p
        assert p["max_norb"] >= p["min_norb"]


def test_asf_params_are_not_shared_between_instances():
    """Mutable defaults must not leak across Config objects."""
    a, b = Config(), Config()
    a.asf.params[1]["max_norb"] = 99
    assert b.asf.params[1]["max_norb"] == 12


def test_config_scan_distances_default():
    cfg = Config()
    assert len(cfg.scan_distances) == 20
    assert cfg.scan_distances[0] < cfg.scan_distances[-1]


def test_config_blockexe_wrapper():
    assert "block2main_wrapper.sh" in Config().blockexe_wrapper


def test_config_provenance_is_available():
    block = Config(molecule="LiH", basis="sto-3g").provenance()
    assert block["config"]["molecule"] == "LiH"
    assert "quenais_version" in block
    assert "threads" in block


# ── CIF parser ───────────────────────────────────────────────────────────

CIF_DIR = os.path.join(os.path.dirname(__file__), "..", "cif_files")


def test_tio2_geometry():
    geom = load_geometry("TiO2", CIF_DIR)
    assert len(geom) == 3
    syms = [a[0] for a in geom]
    assert "Ti" in syms
    assert syms.count("O") == 2


def test_geometry_no_duplicates():
    geom = load_geometry("TiO2", CIF_DIR)
    coords = [np.array(a[1]) for a in geom]
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            assert dist > 0.5, f"Atoms {i} and {j} too close: {dist:.3f} A"


def test_geometry_cartesian_coords():
    for _sym, coord in load_geometry("TiO2", CIF_DIR):
        assert len(coord) == 3
        for c in coord:
            assert isinstance(float(c), float)


def test_missing_cif_raises():
    with pytest.raises(FileNotFoundError):
        load_geometry("DOESNOTEXIST", CIF_DIR)
