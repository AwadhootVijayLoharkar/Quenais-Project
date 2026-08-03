"""
Stage 4 (visualisation) tests.

The epoch-log parser is checked against your real gqe_epoch_log.csv, so
the column names in the tests are the ones the trainer actually emits
rather than ones I assumed.
"""

from __future__ import annotations

import ast
import csv
import inspect
import os
import pickle
import warnings
from pathlib import Path

import pytest

from quenais.config import Config
from quenais.visualization import plots

# One real record, reproduced from a training log. Note np.float64(...)
# wrappers -- the trainer prints numpy scalars that way and they must be
# stripped before the dict can be parsed.
SAMPLE_LOG = """
some preamble the trainer prints
[epoch 0] {'GQE-optimized(best_so_far)/energy - R-CASCI': np.float64(0.0604), 'Global-refined(best_so_far)/subspace_dim': 102, 'GQE-optimized/cx_count/max': 4024}
noise in between
[epoch 1] {'GQE-optimized(best_so_far)/energy - R-CASCI': np.float64(0.0368), 'Global-refined(best_so_far)/subspace_dim': 1116, 'GQE-optimized/cx_count/max': 4024}
[epoch 2] {'GQE-optimized(best_so_far)/energy - R-CASCI': np.float64(0.02405), 'Global-refined(best_so_far)/subspace_dim': 2000, 'GQE-optimized/cx_count/max': 4024}
"""


def _write(tmp_path, text, name="gqe_train.log"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# ── Log parsing ──────────────────────────────────────────────────────────

def test_parses_epochs_and_strips_float64(tmp_path):
    rows = plots.parse_gqe_log(_write(tmp_path, SAMPLE_LOG))
    assert len(rows) == 3
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    value = rows[0]["GQE-optimized(best_so_far)/energy - R-CASCI"]
    assert isinstance(value, float) and value == pytest.approx(0.0604)


def test_rows_are_sorted_by_epoch(tmp_path):
    shuffled = "\n".join([
        "[epoch 2] {'a': 3}",
        "[epoch 0] {'a': 1}",
        "[epoch 1] {'a': 2}",
    ])
    rows = plots.parse_gqe_log(_write(tmp_path, shuffled))
    assert [r["epoch"] for r in rows] == [0, 1, 2]


def test_missing_log_is_not_an_error(tmp_path):
    assert plots.parse_gqe_log(str(tmp_path / "nope.log")) == []
    assert plots.parse_gqe_log(None) == []


def test_unparseable_record_warns_and_is_skipped(tmp_path):
    text = "[epoch 0] {'a': 1}\n[epoch 1] {this is not a dict}\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = plots.parse_gqe_log(_write(tmp_path, text))
    assert len(rows) == 1
    assert any("epoch 1" in str(w.message) for w in caught)


def test_nonempty_log_without_epochs_warns_about_the_patch(tmp_path):
    """
    The signature of an unpatched submodule: training ran, the log is full
    of output, and there is not a single parseable epoch record. The run
    looked entirely successful.
    """
    text = "Epoch 0: 100%|#####| loss=0.1\nEpoch 1: 100%|#####| loss=0.09\n"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = plots.parse_gqe_log(_write(tmp_path, text))
    assert rows == []
    assert any("train_pipeline.py" in str(w.message) for w in caught)


def test_empty_log_does_not_warn(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plots.parse_gqe_log(_write(tmp_path, ""))
    assert not any("train_pipeline" in str(w.message) for w in caught)


def test_parser_does_not_use_eval():
    """literal_eval only -- the log is machine-generated but not trusted."""
    src = inspect.getsource(plots.parse_gqe_log)
    assert "literal_eval" in src
    assert "eval(" not in src.replace("literal_eval(", "")


def test_column_extraction_skips_absent_keys():
    rows = [{"epoch": 0, "a": 1.0}, {"epoch": 1}, {"epoch": 2, "a": 3.0}]
    x, y = plots._col(rows, "a")
    assert list(x) == [0, 2]
    assert list(y) == [1.0, 3.0]


# ── Against the real golden log ──────────────────────────────────────────

@pytest.mark.parametrize("system", ["LiH", "N2", "ScH"])
def test_golden_epoch_csv_columns_are_the_ones_we_plot(system, golden_dir):
    """
    The series names the plots ask for must exist in your real logs. A
    typo here produces an empty figure with no error.
    """
    path = golden_dir / system / "gqe_epoch_log.csv"
    with open(path) as fh:
        header = set(next(csv.reader(fh)))

    required = {
        "epoch",
        "Global-refined(best_so_far)/energy - R-CASCI",
        "GQE-optimized(best_so_far)/energy - R-CASCI",
        "GQE-optimized/cx_count/max",
        "Global-refined(best_so_far)/subspace_dim",
    }
    missing = required - header
    assert not missing, f"{system}: plotted columns absent from the log: {missing}"


# ── Structural ───────────────────────────────────────────────────────────

def test_main_signature_matches_cli_expectations():
    params = list(inspect.signature(plots.main).parameters)
    assert params[:2] == ["cfg", "force"]
    for extra in ("no_scan", "no_quantum_scan"):
        assert extra in params


def test_no_matplotlib_import_at_module_level():
    """
    Importing matplotlib at module scope picks a backend before main() can
    force Agg, which breaks on a headless compute node.
    """
    tree = ast.parse(Path(plots.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or node.names[0].name
            assert not name.startswith("matplotlib"), f"module-level: {name}"


def test_agg_backend_is_forced():
    assert 'matplotlib.use("Agg")' in Path(plots.__file__).read_text()


def test_phase_b_casci_is_not_used_as_the_reported_energy():
    """
    reference_density_info["e_cas"] is the density-building CASCI, not the
    embedded solve. Reporting it as DMET+CASCI is the bug this guards.

    Checks the AST for a subscript of the literal "e_cas" rather than
    grepping -- the local variable is called e_casci, which contains
    "e_cas" as a substring and would make a text search always fail.
    """
    tree = ast.parse(inspect.getsource(plots.plot_method_comparison))
    read_keys = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    called = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert "e_cas" not in (read_keys | called), (
        "plot_method_comparison must not read the phase B CASCI"
    )

    src = inspect.getsource(plots.plot_method_comparison)
    assert "true_embedding_casci" in src


def test_casci_failure_omits_rather_than_substitutes():
    src = inspect.getsource(plots.true_embedding_casci)
    assert "energy = None" in src
    assert "would look plausible while being wrong" in src


# ── End to end on golden data ────────────────────────────────────────────

@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_runs_on_golden_data_and_matches_the_summary(tmp_path, golden_dir):
    """
    Run the stage against the real ScH pickles and check the regenerated
    summary agrees with the one your pipeline produced.
    """
    import shutil

    cfg = Config(molecule="ScH", basis="sto-3g", project_dir=str(tmp_path))
    cfg.validate().make_dirs().load_geometry()
    for name in ("step0_classical.pkl", "step1_asf.pkl", "step2_hamiltonian.pkl"):
        shutil.copy(golden_dir / "ScH" / name, Path(cfg.results_dir) / name)

    result = plots.main(cfg)

    for figure in ("fig1_asf_deviation_spectrum.png",
                   "fig2_dmet_schmidt_spectrum.png",
                   "fig5_method_comparison.png"):
        assert os.path.exists(os.path.join(cfg.plots_dir, figure)), figure

    # No GQE log was staged, so the GQE figures must be absent rather than
    # empty, and the stage must not have failed.
    assert result["n_epochs"] == 0
    assert not os.path.exists(
        os.path.join(cfg.plots_dir, "fig3_gqe_energy_convergence.png")
    )

    rows = {}
    with open(result["results_summary"]) as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0] and not row[0].startswith(("--", "#")):
                rows[row[0]] = row[1]

    assert rows["molecule"] == "ScH"
    assert int(rows["n_bath"]) == 5
    assert int(rows["n_emb"]) == 11

    # The reported DMET+CASCI must be the embedded solve, matching your
    # validated -752.699524181 rather than the phase B reference CASCI.
    labels, energies = result["comparison"]
    idx = labels.index("DMET+CASCI(active)")
    assert energies[idx] == pytest.approx(-752.699524181, abs=5e-5)


@pytest.mark.needs_pyscf
@pytest.mark.slow
def test_n2_zero_bath_spectrum_plots_without_a_log_axis(tmp_path, golden_dir):
    """An all-zero spectrum cannot go on a log axis; it must still render."""
    import shutil

    cfg = Config(molecule="N2", basis="sto-3g", project_dir=str(tmp_path))
    cfg.validate().make_dirs().load_geometry()
    shutil.copy(golden_dir / "N2" / "step2_hamiltonian.pkl",
                Path(cfg.results_dir) / "step2_hamiltonian.pkl")

    import matplotlib

    matplotlib.use("Agg")
    with open(cfg.step2_file, "rb") as fh:
        step2 = pickle.load(fh)
    plots.plot_dmet_spectrum(cfg, step2)

    assert os.path.exists(
        os.path.join(cfg.plots_dir, "fig2_dmet_schmidt_spectrum.png")
    )
