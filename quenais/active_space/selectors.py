"""
Alternative active-space selectors: AVAS and APC.

WHY THIS MODULE EXISTS
----------------------
ASF's entanglement-entropy thresholds are calibrated on main-group systems
and under-select for the d-block (docs/limitations.md). The workaround has
been cfg.asf.force_active_space -- an explicit list of MO indices. That
works at one geometry and is silently wrong at any other, because MO
indices are positions in an energy-ordered list and that list reorders as a
bond stretches. Nothing validates it beyond an out-of-range check.

AVAS removes both problems at once: you name the atomic valence shell
('Sc 3d', 'Sc 4s', 'H 1s') and it finds the MOs carrying that character at
whatever geometry it is handed. APC is kept alongside it as the automatic
option for systems where nobody has decided what the valence shell is.

THREE THINGS TO KNOW BEFORE EDITING
-----------------------------------
1. Both selectors return their OWN orbital basis, not the canonical UHF
   one. That is already true of ASF, and project_occupations() in finder.py
   exists for exactly this reason. Anything that indexes mo_list must index
   the matching mo_coeff.

2. Take nel from Selection.nel. Do NOT call count_active_electrons() on an
   AVAS/APC space -- it reads occupations out of mf.mo_occ, which is the
   canonical UHF basis, and indexes them with mo_list, which after AVAS is
   not. That is the same silent basis mismatch project_occupations() was
   written to kill, and it would not raise.

3. Phase C must not run on these spaces. It narrows by MP2 occupation
   deviation and can only ever shrink a selection; on ScH it dropped an
   orbital ASF had deliberately chosen. Applied to AVAS it would discard
   orbitals the user named explicitly.
"""

from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np

__all__ = [
    "Selection",
    "SELECTORS",
    "select_avas",
    "select_apc",
    "default_ao_labels",
    "active_block_indices",
    "check_space_is_usable",
]


#: mo_list  : 0-based indices into mo_coeff (NOT into the UHF basis)
#: mo_coeff : the selector's orbital coefficients, AO basis, C^T S C = I
#: nel      : active electrons, as reported by the selector itself
#: meta     : what it took to reproduce this selection; goes into the pickle
Selection = namedtuple("Selection", "mo_list mo_coeff nel meta")


# ═════════════════════════════════════════════════════════════════════════
# Valence-shell defaults
# ═════════════════════════════════════════════════════════════════════════

#: Element -> AVAS AO labels for its valence shell. Only what the pipeline
#: is plausibly pointed at; anything else raises rather than guessing, since
#: a wrong label set is a wrong active space with no warning attached.
_VALENCE_SHELLS = {
    # period 1
    "H": ("1s",), "He": ("1s",),
    # period 2
    "Li": ("2s",), "Be": ("2s",),
    "B": ("2s", "2p"), "C": ("2s", "2p"), "N": ("2s", "2p"),
    "O": ("2s", "2p"), "F": ("2s", "2p"), "Ne": ("2s", "2p"),
    # period 3
    "Na": ("3s",), "Mg": ("3s",),
    "Al": ("3s", "3p"), "Si": ("3s", "3p"), "P": ("3s", "3p"),
    "S": ("3s", "3p"), "Cl": ("3s", "3p"), "Ar": ("3s", "3p"),
    # period 4 s-block
    "K": ("4s",), "Ca": ("4s",),
    # 3d block -- the whole reason this module exists
    "Sc": ("3d", "4s"), "Ti": ("3d", "4s"), "V": ("3d", "4s"),
    "Cr": ("3d", "4s"), "Mn": ("3d", "4s"), "Fe": ("3d", "4s"),
    "Co": ("3d", "4s"), "Ni": ("3d", "4s"), "Cu": ("3d", "4s"),
    "Zn": ("3d", "4s"),
    # period 4 p-block
    "Ga": ("4s", "4p"), "Ge": ("4s", "4p"), "As": ("4s", "4p"),
    "Se": ("4s", "4p"), "Br": ("4s", "4p"), "Kr": ("4s", "4p"),
    # 4d block
    "Y": ("4d", "5s"), "Zr": ("4d", "5s"), "Nb": ("4d", "5s"),
    "Mo": ("4d", "5s"), "Tc": ("4d", "5s"), "Ru": ("4d", "5s"),
    "Rh": ("4d", "5s"), "Pd": ("4d", "5s"), "Ag": ("4d", "5s"),
    "Cd": ("4d", "5s"),
}


def default_ao_labels(mol):
    """
    Derive AVAS AO labels from the elements present.

    Convenience only. The label list IS the physics of an AVAS selection --
    if you care about the answer, pass cfg.asf.avas_ao_labels explicitly and
    record it. The defaults here are the neutral-atom valence shell, which
    is right for ScH and wrong for, say, a system where you want to correlate
    a ligand pi manifold as well.
    """
    syms = {mol.atom_symbol(i) for i in range(mol.natm)}
    unknown = sorted(s for s in syms if s not in _VALENCE_SHELLS)
    if unknown:
        raise ValueError(
            f"No default AVAS valence shell for {unknown}. Pass "
            f"cfg.asf.avas_ao_labels explicitly, e.g. ['Sc 3d', 'Sc 4s', "
            f"'H 1s']. Guessing here would produce a plausible-looking "
            f"active space chosen by nobody."
        )
    labels = []
    for sym in sorted(syms):
        labels.extend(f"{sym} {shell}" for shell in _VALENCE_SHELLS[sym])
    return labels


# ═════════════════════════════════════════════════════════════════════════
# The one piece of real adapter logic
# ═════════════════════════════════════════════════════════════════════════

def check_space_is_usable(mol, ncas, nel, selector, hint=""):
    """
    Warn when the selected space is degenerate rather than merely small.

    THE ScF LESSON
    --------------
    On ScF/sto-3g, AVAS with ['Sc 3d', 'Sc 4s'] returned (2e, 6o): one
    occupied orbital and five empty 3d orbitals. The labels described the
    metal correctly and described where the ELECTRONS are incorrectly --
    ScF is ionic, so its occupied valence manifold sits on the fluorine.

    Nothing downstream complains about this. CASCI runs, DMET runs, the
    Schmidt values come back small but finite, and the first sign of
    trouble is an embedded-SCF check failing by a number too small to
    look like a catastrophe. Catch it here, where the cause is visible.

    A space is degenerate in either direction:
      - nearly empty: almost no electrons to correlate
      - nearly full : almost no holes to excite into
    Both give a CASCI that is close to a single determinant.
    """
    n_occ = nel / 2.0
    n_holes = ncas - n_occ
    if ncas >= 4 and n_occ <= 1.0:
        warnings.warn(
            f"{selector.upper()} returned ({nel}e, {ncas}o): only "
            f"{n_occ:.0f} of {ncas} orbitals is occupied. That is not a "
            f"small active space, it is very likely the WRONG one -- the "
            f"selection named orbitals that are essentially empty, so "
            f"there is almost nothing to correlate. {hint}",
            RuntimeWarning,
        )
    elif ncas >= 4 and n_holes <= 1.0:
        warnings.warn(
            f"{selector.upper()} returned ({nel}e, {ncas}o): only "
            f"{n_holes:.0f} of {ncas} orbitals is unoccupied, so there is "
            f"almost nowhere to excite into. The space is nearly closed and "
            f"the CASCI will be close to a single determinant. {hint}",
            RuntimeWarning,
        )


def active_block_indices(mol, ncas, nelecas):
    """
    Map PySCF's (ncas, nelecas, mo) onto our (mo_list, mo_coeff) contract.

    Both avas.kernel() and APC.kernel() return

        mo = hstack((mofreeze, mocore, mocas, movir))

    so the active orbitals are one contiguous block starting after the
    doubly occupied core. APC's docstring states it follows the AVAS
    convention, which is what lets one adapter serve both.
    """
    nel = int(np.sum(nelecas))          # nelecas may be int or (na, nb)
    n_core_electrons = mol.nelectron - nel
    if n_core_electrons < 0:
        raise ValueError(
            f"selector reported {nel} active electrons but the molecule has "
            f"only {mol.nelectron}. Refusing to build an orbital list from "
            f"this."
        )
    if n_core_electrons % 2:
        raise ValueError(
            f"core electron count {n_core_electrons} is odd, so the core is "
            f"not doubly occupied and the contiguous-block assumption in "
            f"active_block_indices() does not hold. This is expected to be "
            f"reachable only for open-shell references -- check "
            f"avas_openshell_option before working around it."
        )
    ncore = n_core_electrons // 2
    return list(range(ncore, ncore + int(ncas))), nel


# ═════════════════════════════════════════════════════════════════════════
# AVAS
# ═════════════════════════════════════════════════════════════════════════

def select_avas(mf, mol, cfg):
    """
    Project the MOs onto a named set of atomic valence orbitals.

    Recommended for transition metals. Unlike an entropy threshold it cannot
    decide a 3d shell is uninteresting, and unlike force_active_space it is
    re-derived at every geometry, so an active space stays the same physical
    object along a dissociation curve.
    """
    from pyscf.mcscf import avas as avas_mod

    # PySCF exposes this as avas.avas in some versions and avas.kernel in
    # others. Resolving it here rather than importing one name keeps the
    # supported pyscf range wider than the pin in requirements-quantum.txt.
    kernel = getattr(avas_mod, "avas", None) or avas_mod.kernel

    labels = cfg.asf.avas_ao_labels or default_ao_labels(mol)

    # Fail loudly on a label that matches nothing. PySCF's search_ao_label
    # returns an empty selection for a typo, and AVAS on an empty reference
    # set returns an empty active space several steps later, where the cause
    # is no longer visible.
    for label in labels:
        if len(mol.search_ao_label(label)) == 0:
            raise ValueError(
                f"AVAS label {label!r} matches no AO in "
                f"{cfg.molecule}/{cfg.basis}. Available labels: "
                f"{sorted(set(mol.ao_labels()))}"
            )

    print(f"  AVAS labels    : {labels}")
    print(f"  AVAS threshold : {cfg.asf.avas_threshold}  "
          f"minao={cfg.asf.avas_minao}")

    ncas, nelecas, mo_coeff = kernel(
        mf,
        labels,
        threshold=cfg.asf.avas_threshold,
        minao=cfg.asf.avas_minao,
        openshell_option=cfg.asf.avas_openshell_option,
        canonicalize=True,
    )

    if ncas == 0:
        raise RuntimeError(
            f"AVAS selected 0 orbitals at threshold="
            f"{cfg.asf.avas_threshold}. Lower it, or check the labels "
            f"{labels} describe the shell you meant."
        )

    mo_list, nel = active_block_indices(mol, ncas, nelecas)

    check_space_is_usable(
        mol, ncas, nel, "avas",
        hint=f"Labels used: {labels}. For an ionic ligand the occupied "
             f"valence density sits on the LIGAND, so metal-only labels "
             f"select the empty d manifold. Add the ligand shell (e.g. "
             f"'F 2p') and re-run. tools/inspect_mo_character.py shows "
             f"which MOs actually hold the valence electrons.",
    )

    # In a minimal basis the projection onto minao is close to an identity,
    # so the eigenvalue spectrum AVAS thresholds is much less structured
    # than in a polarised basis. Not an error, but the threshold means less
    # here than the literature default implies.
    if cfg.basis.lower().replace("-", "") in ("sto3g", "minao", "sto6g"):
        warnings.warn(
            f"AVAS is running in a minimal basis ({cfg.basis}). The "
            f"projection onto {cfg.asf.avas_minao} is near-trivial there, so "
            f"avas_threshold is weakly discriminating. Scan the threshold, "
            f"and confirm in a polarised basis before concluding the "
            f"selection is wrong.",
            RuntimeWarning,
        )

    meta = {
        "selector": "avas",
        "ao_labels": list(labels),
        "threshold": float(cfg.asf.avas_threshold),
        "minao": cfg.asf.avas_minao,
        "openshell_option": int(cfg.asf.avas_openshell_option),
        "ncas": int(ncas),
        "nelecas": nel,
    }
    return Selection(mo_list, np.asarray(mo_coeff), nel, meta)


# ═════════════════════════════════════════════════════════════════════════
# APC
# ═════════════════════════════════════════════════════════════════════════

def select_apc(mf, mol, cfg):
    """
    PySCF's ranked-orbital selector: entropy from HF Fock/exchange coupling.

    Fully automatic -- no AO labels. Use it where nobody has decided what
    the valence shell is. Note its published validation is on organic
    pi systems and vertical excitations, not the d-block; on a transition
    metal, prefer AVAS and treat APC as a cross-check.

    On a UHF reference APC averages the Fock matrices, sums the exchange
    matrices and works in the alpha basis. Singly occupied orbitals are
    given an inflated entropy so they are always selected, which is the
    property that makes it worth keeping for the open-shell systems in
    docs/limitations.md.
    """
    from pyscf.mcscf import apc as apc_mod

    print(f"  APC max_size : {cfg.asf.apc_max_size}  n={cfg.asf.apc_n}")

    chooser = apc_mod.APC(mf, max_size=cfg.asf.apc_max_size, n=cfg.asf.apc_n)
    ncas, nelecas, mo_coeff = chooser.kernel()

    if ncas == 0:
        raise RuntimeError(
            f"APC selected 0 orbitals at max_size={cfg.asf.apc_max_size}."
        )

    mo_list, nel = active_block_indices(mol, ncas, nelecas)

    check_space_is_usable(
        mol, ncas, nel, "apc",
        hint=f"Try a different apc_max_size (currently "
             f"{cfg.asf.apc_max_size}) or apc_n (currently {cfg.asf.apc_n}).",
    )

    meta = {
        "selector": "apc",
        "max_size": int(cfg.asf.apc_max_size),
        "n": int(cfg.asf.apc_n),
        "ncas": int(ncas),
        "nelecas": nel,
    }
    return Selection(mo_list, np.asarray(mo_coeff), nel, meta)


#: Registry. "asf" is handled inline in finder.py because it needs the tier
#: and the block2 environment; the two PySCF selectors need neither.
SELECTORS = {
    "asf": None,
    "avas": select_avas,
    "apc": select_apc,
}