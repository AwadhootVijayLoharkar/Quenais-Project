"""LAS settings, fragment specs, solver registry and CLI wiring. No PySCF."""

import pytest

from quenais.las.fragments import (
    FragmentSpec,
    check_disjoint,
    parse_fragment_spec,
    resolve_atoms,
    split_core_active_virtual,
)
from quenais.settings import LasSettings


def test_parse_fragment_specs():
    assert parse_fragment_spec("Sc:6:1,1") == FragmentSpec(("Sc",), 6, (1, 1), 0)
    assert parse_fragment_spec("0:5:4,2:2") == FragmentSpec((0,), 5, (4, 2), 2)
    assert parse_fragment_spec("0+Fe:5:2,4") == FragmentSpec((0, "Fe"), 5, (2, 4), 2)
    assert parse_fragment_spec("F:3:3,3").label == "F"


@pytest.mark.parametrize("bad", ["Sc:6", "Sc:x:1,1", "Sc:2:3,0", "Sc:4:2,2:1",
                                 "Sc:4:2,0:0", ":4:1,1", "Sc:0:0,0"])
def test_bad_fragment_specs_raise(bad):
    with pytest.raises(ValueError):
        parse_fragment_spec(bad)


def test_resolve_atoms_by_symbol_and_index():
    syms = ["Fe", "Fe", "N", "C"]
    assert resolve_atoms(parse_fragment_spec("Fe:5:3,3"), syms) == [0, 1]
    assert resolve_atoms(parse_fragment_spec("1+N:5:3,3"), syms) == [1, 2]
    with pytest.raises(ValueError):
        resolve_atoms(parse_fragment_spec("Cu:5:3,3"), syms)
    with pytest.raises(ValueError):
        resolve_atoms(parse_fragment_spec("7:5:3,3"), syms)


def test_fragments_must_be_disjoint():
    with pytest.raises(ValueError, match="must not share"):
        check_disjoint([[0, 1], [1]], ["A", "B"])


def test_core_active_virtual_split():
    core, act, virt = split_core_active_virtual(10, [4, 5, 6], 12, 4)
    assert core == [0, 1, 2, 3] and act == [4, 5, 6] and virt == [7, 8, 9]
    with pytest.raises(ValueError):
        split_core_active_virtual(10, [4, 5, 6], 12, 3)      # odd core


def test_las_settings_validation():
    with pytest.raises(ValueError, match="need fragments"):
        LasSettings().validate()
    s = LasSettings(fragments=["Sc:6:1,1", "F:3:3,3"]).validate()
    assert s.energy_tol("lassqd") == 1e-5 and s.energy_tol("lasscf") == 1e-8
    with pytest.raises(ValueError):
        LasSettings(fragments=["Sc:6:1,1"], backend="nope").validate()
    with pytest.raises(ValueError):
        LasSettings(fragments=["Sc:6:1,1"], backend="ibm").validate()
    with pytest.raises(ValueError):
        LasSettings(fragments=["Sc:6:1,1"], carryover_eps=-1).validate()


def test_solver_registry_and_dispatch_names():
    from quenais.config import LAS_SOLVERS, SOLVERS

    assert set(LAS_SOLVERS) == {"lasscf", "lassqd"}
    assert set(LAS_SOLVERS) <= set(SOLVERS)


def test_cli_builds_las_settings():
    from quenais.cli import build_las_settings, build_parser

    args = build_parser().parse_args([
        "--solver", "lassqd", "--las-fragments", "Sc:6:1,1", "F:3:3,3",
        "--lassqd-carryover-eps", "0", "--lassqd-shots", "5000",
        "--las-no-orbital-opt", "--lassqd-backend", "aer_statevector",
    ])
    s = build_las_settings(args)
    assert s.fragments == ["Sc:6:1,1", "F:3:3,3"]
    assert s.carryover_eps == 0 and s.shots == 5000
    assert s.orbital_optimization is False and s.backend == "aer_statevector"
    assert s.sqd_batches == 15        # untouched default
