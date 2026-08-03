"""
Geometry input tests.

A partner arriving with their own molecule is the main case this package
has to support, and until now the only path was a CIF file. These cover
the four ways in and, more importantly, the ways they can go quietly
wrong: a truncated XYZ, ambiguous input, a mis-cased element symbol.
"""

from __future__ import annotations

import pytest

from quenais.config import Config
from quenais.utils.geometry import (
    BUILTIN_GEOMETRIES,
    normalise_geometry,
    parse_geometry_string,
    parse_xyz,
    write_xyz,
)

LIH_XYZ = """2
LiH, r = 1.5949 A
Li   0.00000000   0.00000000   0.00000000
H    0.00000000   0.00000000   1.59490000
"""


def _write(tmp_path, text, name="mol.xyz"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# ── XYZ ──────────────────────────────────────────────────────────────────

def test_parses_a_standard_xyz(tmp_path):
    geom = parse_xyz(_write(tmp_path, LIH_XYZ))
    assert geom == [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))]


def test_truncated_xyz_is_rejected(tmp_path):
    """
    A file claiming 5 atoms but holding 2 is almost always truncated.
    Reading the 2 silently would give a plausible energy for the wrong
    molecule -- the exact failure mode this package exists to avoid.
    """
    text = LIH_XYZ.replace("2\n", "5\n", 1)
    with pytest.raises(ValueError, match="truncated"):
        parse_xyz(_write(tmp_path, text))


def test_blank_lines_are_tolerated(tmp_path):
    geom = parse_xyz(_write(tmp_path, LIH_XYZ + "\n\n"))
    assert len(geom) == 2


def test_missing_xyz_reports_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.xyz"):
        parse_xyz(str(tmp_path / "nope.xyz"))


def test_short_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="too short"):
        parse_xyz(_write(tmp_path, "2\n"))


def test_roundtrip_through_write_xyz(tmp_path):
    original = BUILTIN_GEOMETRIES["H2O"]
    path = write_xyz(original, tmp_path / "h2o.xyz", comment="water")
    assert parse_xyz(path) == normalise_geometry(original)


# ── Inline strings ───────────────────────────────────────────────────────

def test_semicolon_separated_string():
    geom = parse_geometry_string("Li 0 0 0; H 0 0 1.5949")
    assert geom == [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))]


def test_newline_separated_string():
    assert parse_geometry_string("Li 0 0 0\nH 0 0 1.5949") == \
           parse_geometry_string("Li 0 0 0; H 0 0 1.5949")


def test_incomplete_line_is_rejected():
    with pytest.raises(ValueError, match="symbol x y z"):
        parse_geometry_string("Li 0 0")


# ── Normalisation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    [("H", (0.0, 0.0, 0.0))],
    [("H", [0.0, 0.0, 0.0])],
    [("H", 0.0, 0.0, 0.0)],
    [["H", 0, 0, 0]],
])
def test_accepts_the_shapes_people_write(raw):
    assert normalise_geometry(raw) == [("H", (0.0, 0.0, 0.0))]


@pytest.mark.parametrize("written,expected", [
    ("FE", "Fe"), ("fe", "Fe"), ("Fe", "Fe"), ("Fe1", "Fe"), ("c", "C"),
])
def test_element_symbols_are_cleaned(written, expected):
    assert normalise_geometry([(written, (0, 0, 0))])[0][0] == expected


def test_empty_geometry_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        normalise_geometry([])


def test_wrong_coordinate_count_is_rejected():
    with pytest.raises(ValueError):
        normalise_geometry([("H", (0.0, 0.0))])


# ── Config integration ───────────────────────────────────────────────────

def test_builtin_still_works(tmp_path):
    cfg = Config(molecule="LiH", project_dir=str(tmp_path)).load_geometry()
    assert cfg.geometry_source == "built-in"
    assert cfg.n_atoms == 2
    assert cfg.atom_syms == ["Li", "H"]


def test_inline_geometry(tmp_path):
    cfg = Config(molecule="MyMol", geometry="Li 0 0 0; H 0 0 1.5949",
                 project_dir=str(tmp_path)).load_geometry()
    assert cfg.geometry_source == "geometry string"
    assert cfg.n_atoms == 2


def test_geometry_list(tmp_path):
    cfg = Config(molecule="MyMol", project_dir=str(tmp_path),
                 geometry=[("Fe", (0, 0, 0)), ("N", (0, 0, 2.1))]).load_geometry()
    assert cfg.atom_syms == ["Fe", "N"]


def test_xyz_file(tmp_path):
    cfg = Config(molecule="MyMol", xyz=_write(tmp_path, LIH_XYZ),
                 project_dir=str(tmp_path)).load_geometry()
    assert "xyz file" in cfg.geometry_source
    assert cfg.n_atoms == 2


def test_explicit_input_overrides_the_builtin(tmp_path):
    """
    A partner studying LiH at a different bond length should not have to
    invent a new name to avoid the bundled geometry.
    """
    cfg = Config(molecule="LiH", geometry="Li 0 0 0; H 0 0 2.0",
                 project_dir=str(tmp_path)).load_geometry()
    assert cfg.geometry[1][1][2] == 2.0
    assert cfg.geometry_source == "geometry string"


def test_geometry_and_xyz_together_are_rejected(tmp_path):
    """Silently preferring one would make the run unreproducible."""
    cfg = Config(molecule="X", geometry="H 0 0 0", xyz="whatever.xyz",
                 project_dir=str(tmp_path))
    with pytest.raises(ValueError, match="not both"):
        cfg.load_geometry()


def test_unknown_molecule_explains_the_options(tmp_path):
    cfg = Config(molecule="Unobtainium", project_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError) as exc:
        cfg.load_geometry()
    message = str(exc.value)
    for hint in ("xyz=", "geometry=", ".cif", "Built-in names"):
        assert hint in message


# ── CLI ──────────────────────────────────────────────────────────────────

def test_cli_accepts_xyz(tmp_path):
    from quenais import cli

    args = cli.build_parser().parse_args(
        ["--molecule", "MyMol", "--basis", "sto-3g",
         "--xyz", _write(tmp_path, LIH_XYZ),
         "--project-dir", str(tmp_path)]
    )
    cfg = cli.build_config(args)
    assert cfg.n_atoms == 2


def test_cli_accepts_inline_geometry(tmp_path):
    from quenais import cli

    args = cli.build_parser().parse_args(
        ["--molecule", "MyMol", "--basis", "sto-3g",
         "--geometry", "Li 0 0 0; H 0 0 1.5949",
         "--project-dir", str(tmp_path)]
    )
    cfg = cli.build_config(args)
    assert cfg.atom_syms == ["Li", "H"]


# ── The reference geometries must not drift ──────────────────────────────

@pytest.mark.parametrize("system,expected", [
    ("LiH", 1.5949), ("N2", 1.0977), ("ScH", 1.7800),
])
def test_builtin_bond_lengths_match_the_reference_runs(system, expected):
    """
    The golden pickles were produced at these geometries. Changing one
    invalidates every reference number without any test noticing unless
    this one exists.
    """
    assert BUILTIN_GEOMETRIES[system][1][1][2] == expected
