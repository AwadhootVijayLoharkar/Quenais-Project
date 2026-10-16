"""--las-init-from: LASSQD warm start from a finished LASSCF. No PySCF."""

import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quenais.las.energy import LasSpace
from quenais.las.fragments import parse_fragment_spec
from quenais.las.stage import load_init

ROOT = Path(__file__).resolve().parents[1]
SPECS = [parse_fragment_spec("0+1:2:1,1"), parse_fragment_spec("2+3:2:1,1")]


def _write_ref(path, nao=6, nmo=6, ncore=1, ncas=(2, 2), specs=SPECS, **extra):
    rng = np.random.default_rng(0)
    ref = {
        "solver": "lasscf", "energy": -2.1234, "converged": True, "n_macro": 7,
        "grad_norm": 1e-5, "ncore": ncore, "ncas_sub": list(ncas),
        "mo_coeff": rng.normal(size=(nao, nmo)),
        "dm1s": [np.eye(2)[None].repeat(2, 0) * 0.5 for _ in ncas],
        "fragments": [s.__dict__ | {"label": s.label} for s in specs],
        "settings": {"density_fit": False},
        "mol_info": {"basis": "sto-3g", "n_electrons": 6, "n_ao": nao},
        **extra,
    }
    with open(path, "wb") as fh:
        pickle.dump(ref, fh)
    return ref


def test_load_init_returns_orbitals_and_rdms(tmp_path):
    ref = _write_ref(tmp_path / "step3_lasscf.pkl")
    space = LasSpace(ncore=1, ncas_sub=(2, 2), nmo=6)
    C, dm1s, info = load_init(tmp_path / "step3_lasscf.pkl", np.zeros((6, 6)),
                              space, SPECS,
                              mol_info={"basis": "STO-3G", "n_electrons": 6, "n_ao": 6})
    assert np.array_equal(C, ref["mo_coeff"])
    assert len(dm1s) == 2 and dm1s[0].shape == (2, 2, 2)
    assert info["energy"] == -2.1234 and info["solver"] == "lasscf"
    assert info["density_fit"] is False


def test_load_init_expands_environment_variables(tmp_path, monkeypatch):
    _write_ref(tmp_path / "step3_lasscf.pkl")
    monkeypatch.setenv("QN_TEST_B", str(tmp_path))
    space = LasSpace(ncore=1, ncas_sub=(2, 2), nmo=6)
    _, _, info = load_init("$QN_TEST_B/step3_lasscf.pkl", np.zeros((6, 6)), space, SPECS)
    assert info["path"] == str(tmp_path / "step3_lasscf.pkl")


@pytest.mark.parametrize("kw, match", [
    ({"nmo": 7}, "shape"),
    ({"ncore": 0}, "core"),
    ({"ncas": (3, 1)}, "fragment sizes"),
    ({"specs": [parse_fragment_spec("0+1:2:2,0"), parse_fragment_spec("2+3:2:0,2")]},
     "fragments"),
])
def test_load_init_rejects_a_different_problem(tmp_path, kw, match):
    _write_ref(tmp_path / "r.pkl", **kw)
    space = LasSpace(ncore=1, ncas_sub=(2, 2), nmo=6)
    with pytest.raises(ValueError, match=match):
        load_init(tmp_path / "r.pkl", np.zeros((6, 6)), space, SPECS)


def test_load_init_rejects_other_basis(tmp_path):
    _write_ref(tmp_path / "r.pkl")
    space = LasSpace(ncore=1, ncas_sub=(2, 2), nmo=6)
    with pytest.raises(ValueError, match="basis"):
        load_init(tmp_path / "r.pkl", np.zeros((6, 6)), space, SPECS,
                  mol_info={"basis": "6-31g"})


def test_missing_file(tmp_path):
    space = LasSpace(ncore=1, ncas_sub=(2, 2), nmo=6)
    with pytest.raises(FileNotFoundError):
        load_init(tmp_path / "nope.pkl", np.zeros((6, 6)), space, SPECS)


def test_cli_flag_reaches_settings():
    from quenais.cli import build_las_settings, build_parser

    args = build_parser().parse_args([
        "--molecule", "H4", "--solver", "lassqd",
        "--las-fragments", "0+1:2:1,1", "2+3:2:1,1",
        "--las-init-from", "$B/x/step3_lasscf.pkl", "--las-no-orbital-opt"])
    s = build_las_settings(args)
    assert s.init_from == "$B/x/step3_lasscf.pkl"
    assert s.orbital_optimization is False
    assert build_las_settings(build_parser().parse_args(
        ["--molecule", "H4"])).init_from is None


def _driver():
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", ROOT / "tools" / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _las_row(drv, init, solver="lassqd", density_fit=False):
    r = {"solver": solver, "energy": -2.12, "history": [], "ncas_sub": [2, 2],
         "settings": {"fragments": ["a", "b"], "density_fit": density_fit,
                      "shots": 10, "sqd_batches": 1, "sqd_iterations": 1},
         "init": init}
    cfg = SimpleNamespace(las_result_file="x")
    return drv._las_part({}, cfg, lambda _p: r)


def test_warm_start_energy_is_the_solver_reference():
    drv = _driver()
    init = {"path": "/p", "solver": "lasscf", "energy": -2.1234, "density_fit": False}
    part = _las_row(drv, init)
    assert part["solver_ref_energy_Ha"] == -2.1234
    assert part["las_init_from"] == "/p"
    # different integrals: recorded, but not used as the exact limit
    part = _las_row(drv, init, density_fit=True)
    assert "solver_ref_energy_Ha" not in part
    assert part["las_init_energy_Ha"] == -2.1234
    # cold start: no init columns
    assert "las_init_from" not in _las_row(drv, None)


def test_collect_join_separates_density_fitting():
    drv = _driver()
    assert drv._truthy(True) and drv._truthy("True") and not drv._truthy(None)
    assert not drv._truthy("False") and not drv._truthy("")
