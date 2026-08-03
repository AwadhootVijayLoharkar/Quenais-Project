"""
Geometry input.

Four ways to specify a molecule, in the order load_geometry() tries them:

  1. an explicit geometry passed to Config(geometry=...)
  2. an XYZ file passed to Config(xyz=...)
  3. a built-in name (LiH, N2, ScH, H2O)
  4. a CIF file in cfg.cif_dir named <molecule>.cif

Most people have an XYZ file or a geometry from a paper. Before this
module the only supported path for a new molecule was CIF, which meant
converting a two-line diatomic into a crystallographic format for no
reason.

Units are Angstrom throughout, matching PySCF's default.
"""

from __future__ import annotations

import os

__all__ = [
    "BUILTIN_GEOMETRIES",
    "parse_xyz",
    "parse_geometry_string",
    "normalise_geometry",
    "write_xyz",
]

#: Validated built-in geometries, in Angstrom.
#:
#: LiH, N2 and ScH are the systems the reference values in
#: tests/regression/ were produced with -- do not change them without
#: regenerating the golden data.
BUILTIN_GEOMETRIES = {
    # X(1)Sigma+ ground state, r_e ~ 1.78 A.
    "ScH": [("Sc", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7800))],
    "LiH": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))],
    "N2": [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
    "H2O": [
        ("O", (0.0, 0.0, 0.1173)),
        ("H", (0.0, 0.7572, -0.4692)),
        ("H", (0.0, -0.7572, -0.4692)),
    ],
}


def _clean_symbol(raw):
    """'FE' or 'fe' or 'Fe1' -> 'Fe'."""
    letters = ""
    for char in raw:
        if char.isalpha():
            letters += char
        else:
            break
    if not letters:
        raise ValueError(f"could not read an element symbol from {raw!r}")
    return letters[0].upper() + letters[1:].lower()


def normalise_geometry(geometry):
    """
    Accept the shapes people actually write and return the canonical one:
    [(symbol, (x, y, z)), ...] with float coordinates.

    Handles [("H", (0,0,0))], [("H", [0,0,0])], [("H", 0, 0, 0)] and
    [["H", 0, 0, 0]].
    """
    out = []
    for entry in geometry:
        if len(entry) == 2:
            symbol, coords = entry
        elif len(entry) == 4:
            symbol, coords = entry[0], entry[1:]
        else:
            raise ValueError(
                f"geometry entry {entry!r} should be (symbol, (x, y, z)) "
                f"or (symbol, x, y, z)"
            )
        coords = tuple(float(c) for c in coords)
        if len(coords) != 3:
            raise ValueError(f"{symbol} needs 3 coordinates, got {coords}")
        out.append((_clean_symbol(str(symbol)), coords))

    if not out:
        raise ValueError("geometry is empty")
    return out


def parse_xyz(path):
    """
    Read a standard XYZ file.

    Line 1 is the atom count, line 2 is a comment, then one
    `symbol x y z` per line. A wrong count in line 1 is a hard error --
    it usually means the file was truncated, and silently reading fewer
    atoms than intended would produce a plausible wrong answer.
    """
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"XYZ file not found: {path}")

    with open(path) as fh:
        lines = [ln.rstrip() for ln in fh]

    if len(lines) < 3:
        raise ValueError(
            f"{path} is too short to be an XYZ file (expected a count line, "
            f"a comment line, then one line per atom)"
        )

    try:
        declared = int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise ValueError(
            f"{path}: first line should be the atom count, got {lines[0]!r}"
        ) from None

    atoms = []
    for lineno, line in enumerate(lines[2:], start=3):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}:{lineno}: expected 'symbol x y z', "
                             f"got {line!r}")
        atoms.append((parts[0], parts[1:4]))

    if len(atoms) != declared:
        raise ValueError(
            f"{path} declares {declared} atoms but contains {len(atoms)}. "
            f"The file is probably truncated -- refusing to guess which "
            f"count is right."
        )

    return normalise_geometry(atoms)


def parse_geometry_string(text):
    """
    Parse a PySCF-style geometry string.

        "Li 0 0 0; H 0 0 1.5949"
        "Li 0 0 0\\nH 0 0 1.5949"

    Convenient for pasting a geometry out of a paper.
    """
    atoms = []
    for chunk in text.replace(";", "\n").splitlines():
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if len(parts) < 4:
            raise ValueError(f"expected 'symbol x y z', got {chunk!r}")
        atoms.append((parts[0], parts[1:4]))
    return normalise_geometry(atoms)


def write_xyz(geometry, path, comment=""):
    """Write a geometry as XYZ. Useful for recording what was actually run."""
    geometry = normalise_geometry(geometry)
    with open(path, "w") as fh:
        fh.write(f"{len(geometry)}\n{comment}\n")
        for symbol, (x, y, z) in geometry:
            fh.write(f"{symbol:<3} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
    return path
