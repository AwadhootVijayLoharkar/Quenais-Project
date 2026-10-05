"""
Benchmark driver (tools/run_benchmark.py) and the step-3 Qiskit solver's
config access.

The pure parts (grid expansion, flag mapping, run identity) run anywhere.
The end-to-end H4 run is `slow` and needs PySCF + Qiskit.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", ROOT / "tools" / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the bug that made every sqd/skqd/sqdrift run crash ───────────────────

def test_qiskit_solver_reads_only_attributes_config_has():
    """solver.py read cfg.ansatz, cfg.n_shots, ... which Config does not
    carry (they live in cfg.qiskit), so step 3 died with AttributeError on
    every DMET-route Qiskit run. Check every `cfg.<name>` statically."""
    from quenais.config import Config

    cfg = Config(molecule="H2", basis="sto-3g", project_dir="/tmp/x")
    tree = ast.parse((ROOT / "quenais" / "quantum" / "solver.py").read_text())
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "cfg"}
    missing = sorted(a for a in used if not hasattr(cfg, a))
    assert not missing, f"solver.py reads cfg.{missing}, which Config lacks"


def test_cli_shots_and_seed_reach_every_qiskit_solver(tmp_path):
    """--shots used to set n_shots only: SKQD and SqDRIFT ran at 8192
    whatever was asked for."""
    from quenais.cli import build_config, build_parser

    args = build_parser().parse_args([
        "--molecule", "LiH", "--shots", "1234", "--seed", "7",
        "--solver", "skqd", "--mapping", "jw", "--project-dir", str(tmp_path)])
    q = build_config(args).qiskit
    assert (q.n_shots, q.skqd_shots, q.sqdrift_shots) == (1234, 1234, 1234)
    assert q.seed == 7


def test_seed_validation():
    from quenais.settings import QiskitSolverSettings

    QiskitSolverSettings(seed=None).validate()
    QiskitSolverSettings(seed=3).validate()
    with pytest.raises(ValueError):
        QiskitSolverSettings(seed=-1).validate()


def test_subspace_dimension_counts_strings_not_bitstrings():
    from quenais.quantum.solver import full_space_dimension, subspace_dimension

    # 2 orbitals, 1a + 1b; big-endian rows [beta | alpha]
    bsm = np.array([[0, 1, 0, 1],    # beta orb0, alpha orb0
                    [1, 0, 0, 1],    # beta orb1, alpha orb0
                    [0, 1, 0, 1]],   # duplicate
                   dtype=bool)
    # closed shell: union of {01, 10} for both spins -> 2 x 2
    assert subspace_dimension(bsm, 2, 1, 1) == 4
    assert full_space_dimension(2, 1, 1) == 4
    assert full_space_dimension(4, 2, 2) == 36


# ── grid expansion ───────────────────────────────────────────────────────

GRID = {
    "defaults": {"molecule": "H4", "basis": "sto-3g",
                 "las_fragments": ["0+1:2:1,1", "2+3:2:1,1"],
                 "mapping": "jw"},
    "studies": [{"name": "s",
                 "sweep": {"solver": ["sqd", "lassqd", "lasscf"],
                           "shots": [100, 200], "seed": [1, 2, 3]}}],
}


def test_deterministic_solver_collapses_and_flags_are_mapped():
    d = _driver()
    runs = d.expand_grid(GRID)
    by = {}
    for r in runs:
        by.setdefault(r["params"]["solver"], []).append(r)
    assert len(by["sqd"]) == 6 and len(by["lassqd"]) == 6
    assert len(by["lasscf"]) == 1                       # no shots, no seed
    sqd = by["sqd"][0]["params"]
    assert "las_fragments" not in sqd and {"shots", "seed"} <= set(sqd)
    las = by["lassqd"][0]["params"]
    assert {"lassqd_shots", "lassqd_seed"} <= set(las)
    assert not {"shots", "seed", "mapping"} & set(las)


def test_seed_repeats_share_config_and_system_ids():
    d = _driver()
    sqd = [r for r in d.expand_grid(GRID) if r["params"]["solver"] == "sqd"]
    by_shots = {}
    for r in sqd:
        by_shots.setdefault(r["params"]["shots"], set()).add(r["config_id"])
    assert all(len(v) == 1 for v in by_shots.values())
    assert len({r["system_id"] for r in d.expand_grid(GRID)}) == 1
    assert len({r["run_id"] for r in sqd}) == 6


def test_tags_do_not_change_run_identity():
    d = _driver()
    a = d.expand_grid({"studies": [{"fixed": {"solver": "sqd", "tag_R": 1.0}}]})
    b = d.expand_grid({"studies": [{"fixed": {"solver": "sqd", "tag_R": 2.0}}]})
    assert a[0]["run_id"] == b[0]["run_id"]
    assert a[0]["tags"] == {"tag_R": 1.0}


def test_geometry_changes_the_system():
    d = _driver()
    runs = d.expand_grid({"studies": [{
        "fixed": {"solver": "sqd"},
        "cases": [{"geometry": "H 0 0 0; H 0 0 0.7"},
                  {"geometry": "H 0 0 0; H 0 0 1.4"}]}]})
    assert len({r["system_id"] for r in runs}) == 2


def test_rerun_command_runs_step1_before_step0():
    d = _driver()
    cmd = d.rerun_command({"solver": "sqd", "molecule": "H4",
                           "avas_ao_labels": ["H 1s"]})
    first, second = cmd.split(" && ")
    assert first.endswith("--steps 1") and second.endswith("--steps 0 2 3")
    assert "'H 1s'" in first


def test_shipped_grids_use_only_real_flags():
    d = _driver()
    dests = d._parser_dests()
    for path in (ROOT / "benchmarks").glob("*.json"):
        runs = d.expand_grid(json.loads(path.read_text()), dests=dests)
        assert runs, path


def test_unknown_flag_rejected():
    d = _driver()
    with pytest.raises(ValueError, match="unknown"):
        d.expand_grid({"studies": [{"fixed": {"solver": "sqd", "shotz": 1}}]},
                      dests=d._parser_dests())


def test_shard_partition_covers_everything_once():
    d = _driver()
    runs = d.expand_grid(GRID)
    parts = [runs[i::3] for i in range(3)]
    ids = [r["run_id"] for p in parts for r in p]
    assert sorted(ids) == sorted(r["run_id"] for r in runs)


def test_aggregate_mean_std():
    d = _driver()
    rows = [{"status": "ok", "config_id": "c", "solver": "sqd",
             "err_solver_kcal": x, "seed": s} for s, x in ((1, 1.0), (2, 3.0))]
    (a,) = d.aggregate(rows)
    assert a["n"] == 2 and a["err_solver_kcal_mean"] == 2.0
    assert abs(a["err_solver_kcal_std"] - 2 ** 0.5) < 1e-12


# ── end to end ───────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.needs_pyscf
@pytest.mark.needs_qiskit
def test_h4_end_to_end(tmp_path):
    d = _driver()
    grid = json.loads((ROOT / "benchmarks" / "h4_smoke.json").read_text())
    grid["studies"] = [{"name": "t", "sweep": {"solver": ["sqd", "lasscf"],
                                               "shots": [1000], "seed": [1]}}]
    gpath = tmp_path / "g.json"
    gpath.write_text(json.dumps(grid))
    out = tmp_path / "bench"
    assert d.main([str(gpath), "--out", str(out)]) == 0
    assert d.main(["--out", str(out), "--check"]) == 0
    import csv

    rows = list(csv.DictReader(open(out / "results.csv")))
    assert {r["solver"] for r in rows} == {"sqd", "lasscf"}
    assert all(r["status"] == "ok" for r in rows)
    # H4/sto-3g with AVAS H 1s is the full space: CASCI must equal FCI
    r = rows[0]
    assert abs(float(r["E_CASCI_Ha"]) - float(r["E_FCI_Ha"])) < 1e-8
