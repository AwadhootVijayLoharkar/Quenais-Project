"""
CIF file parser for QuEnAIS.
Parses fractional coordinates and cell parameters to Cartesian coordinates.
"""

import os
import numpy as np


def load_geometry(molecule_name, cif_dir):
    """
    Load geometry from a CIF file.
    Parses fractional coordinates + cell parameters to Cartesian coords.
    Only parses loops containing _atom_site_fract_x/y/z.
    """
    cif_path = os.path.join(cif_dir, f"{molecule_name}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"CIF file not found: {cif_path}\n"
            f"Place your .cif files in: {cif_dir}/"
        )

    cell_a = cell_b = cell_c = 1.0
    cell_alpha = cell_beta = cell_gamma = 90.0
    atoms = []

    FRAC_KEYS    = {"_atom_site_fract_x",
                    "_atom_site_fract_y",
                    "_atom_site_fract_z"}
    in_atom_loop = False
    loop_has_frac= False
    atom_keys    = []
    in_multiline = False

    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(";"):
                in_multiline = not in_multiline
                continue
            if in_multiline:
                continue
            if not line or line.startswith("#"):
                continue

            if line.startswith("_cell_length_a"):
                cell_a = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_length_b"):
                cell_b = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_length_c"):
                cell_c = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_angle_alpha"):
                cell_alpha = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_angle_beta"):
                cell_beta = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_angle_gamma"):
                cell_gamma = _parse_cif_number(line.split()[-1])
            elif line == "loop_":
                in_atom_loop = False
                loop_has_frac = False
                atom_keys = []
            elif line.startswith("_atom_site_"):
                atom_keys.append(line)
                if line in FRAC_KEYS:
                    loop_has_frac = True
                in_atom_loop = True
            elif in_atom_loop and line and not line.startswith("_"):
                if line.startswith("loop_"):
                    in_atom_loop = False
                    loop_has_frac = False
                    atom_keys = []
                    continue
                if not loop_has_frac:
                    continue
                tokens = line.split()
                if len(tokens) < len(atom_keys):
                    continue
                row    = dict(zip(atom_keys, tokens))
                symbol = _extract_element(
                    row.get("_atom_site_type_symbol",
                            row.get("_atom_site_label", "X"))
                )
                if symbol in ("X", ""):
                    continue
                fx = _parse_cif_number(row.get("_atom_site_fract_x", "0"))
                fy = _parse_cif_number(row.get("_atom_site_fract_y", "0"))
                fz = _parse_cif_number(row.get("_atom_site_fract_z", "0"))
                atoms.append((symbol, fx, fy, fz))

    if not atoms:
        raise ValueError(
            f"No atoms parsed from {cif_path}\n"
            f"Check that the CIF contains _atom_site_fract_x/y/z fields."
        )

    # Check for duplicate fractional coordinates
    for i, (s1, fx1, fy1, fz1) in enumerate(atoms):
        for j, (s2, fx2, fy2, fz2) in enumerate(atoms):
            if i >= j:
                continue
            if abs(fx1-fx2) + abs(fy1-fy2) + abs(fz1-fz2) < 1e-4:
                raise ValueError(
                    f"Atoms {i}({s1}) and {j}({s2}) have identical "
                    f"fractional coordinates — likely a CIF parsing error."
                )

    frac_to_cart = _build_cell_matrix(
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma
    )
    geometry = []
    for symbol, fx, fy, fz in atoms:
        cart = frac_to_cart @ np.array([fx, fy, fz])
        geometry.append((symbol, tuple(cart)))

    # Check for atoms too close in Cartesian space
    for i, (s1, c1) in enumerate(geometry):
        for j, (s2, c2) in enumerate(geometry):
            if i >= j:
                continue
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist < 0.5:
                raise ValueError(
                    f"Atoms {i}({s1}) and {j}({s2}) are {dist:.3f} Å apart "
                    f"— likely a CIF parsing error."
                )
    return geometry


def _parse_cif_number(s):
    return float(s.split("(")[0])


def _extract_element(s):
    elem = ""
    for ch in s:
        if ch.isalpha():
            elem += ch
        else:
            break
    if not elem:
        return "X"
    return (elem[0].upper() + elem[1:].lower()
            if len(elem) > 1 else elem.upper())


def _build_cell_matrix(a, b, c, alpha, beta, gamma):
    alpha_r = np.radians(alpha)
    beta_r  = np.radians(beta)
    gamma_r = np.radians(gamma)
    cos_a   = np.cos(alpha_r)
    cos_b   = np.cos(beta_r)
    cos_g   = np.cos(gamma_r)
    sin_g   = np.sin(gamma_r)
    ax = a
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(0.0, c**2 - cx**2 - cy**2))
    return np.array([[ax, bx, cx],
                     [0., by, cy],
                     [0., 0., cz]])