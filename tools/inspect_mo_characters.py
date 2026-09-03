#!/usr/bin/env python3
"""
Pick AVAS labels from data instead of from chemical intuition.

WHY THIS EXISTS
---------------
On ScF, the label set ['Sc 3d', 'Sc 4s'] returned a (2e, 6o) active space:
one occupied orbital and five empty 3d orbitals. The labels were not wrong
as a description of the metal -- they were wrong as a description of where
ScF's valence ELECTRONS live, which is on the fluorine. AVAS selects MOs
that carry the character you name; if you name only virtual character you
get a nearly empty active space and every downstream number is meaningless.

That failure is invisible from the formula and obvious from the orbitals.
This script prints the orbitals.

WHAT IT PRINTS
--------------
1. Every valence MO with its energy, occupation, and Loewdin population
   broken down by (atom, shell). The occupied ones are what an active space
   has to cover.
2. For each candidate AVAS label set: the (nelecas, ncas) it would produce
   and how many of those orbitals are occupied.

Read table 2 and pick the label set whose electron count matches the
valence electron count in table 1.

USAGE
-----
    python inspect_mo_character.py --atom "Sc 0 0 0; F 0 0 1.7877"

    python inspect_mo_character.py --atom "Ti 0 0 0; O 0 0 1.62022" --spin 2 \
        --labels "Ti 3d,Ti 4s" "Ti 3d,Ti 4s,O 2p" "Ti 3d,Ti 4s,O 2p,O 2s"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from pyscf import gto, scf


# ─────────────────────────────────────────────────────────────────────────

def lowdin_by_shell(mol, mo_coeff):
    """
    Loewdin population of every MO, grouped by (atom index, symbol, shell).

    Returns (labels, weights) where weights[i] maps a "Sc 3d"-style key to
    that MO's fraction on it. Loewdin rather than Mulliken so the weights
    are non-negative and sum to 1.
    """
    S = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_half = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    # (atom_index, symbol, nl, ml) -- the same shape finder.py unpacks.
    raw = mol.ao_labels(fmt=None)
    keys = []
    for entry in raw:
        atom_idx, sym, nl = entry[0], entry[1], entry[2]
        # nl is like '3d'; strip any ml suffix a build might have merged in.
        shell = str(nl)[:2] if len(str(nl)) >= 2 else str(nl)
        keys.append(f"{sym} {shell}")

    out = []
    for i in range(mo_coeff.shape[1]):
        c = S_half @ mo_coeff[:, i]
        w = c ** 2
        total = w.sum()
        grouped = defaultdict(float)
        for k, wi in zip(keys, w):
            grouped[k] += float(wi / total)
        out.append(dict(grouped))
    return keys, out


def print_mo_table(mol, mf, occ_total, n_show_virt):
    """Table 1: what each valence MO is made of."""
    mo = mf.mo_coeff[0] if isinstance(mf.mo_coeff, (tuple, list)) else mf.mo_coeff
    energies = (mf.mo_energy[0] if isinstance(mf.mo_energy, (tuple, list))
                else mf.mo_energy)
    _, pops = lowdin_by_shell(mol, mo)

    n_occ = int(np.count_nonzero(occ_total > 0.5))
    # A core orbital is doubly occupied AND essentially localised on one
    # shell that is not the valence shell -- rather than guess, show a
    # window: the last 8 occupied and the first n_show_virt virtual.
    lo = max(0, n_occ - 8)
    hi = min(len(energies), n_occ + n_show_virt)

    print(f"\n  Total electrons: {mol.nelectron}   AOs: {mol.nao_nr()}   "
          f"occupied MOs: {n_occ}")
    print(f"  Showing MOs {lo}-{hi - 1} (window around the HOMO-LUMO gap)\n")
    print(f"  {'MO':>4} {'occ':>5} {'E (Ha)':>10}  character (Loewdin, >5%)")
    print("  " + "-" * 68)
    for i in range(lo, hi):
        marker = "occ " if occ_total[i] > 0.5 else "virt"
        top = sorted(pops[i].items(), key=lambda kv: -kv[1])
        parts = [f"{k} {v:.0%}" for k, v in top if v > 0.05][:4]
        sep = "  <-- HOMO" if i == n_occ - 1 else ""
        print(f"  {i:>4} {marker:>5} {energies[i]:>10.4f}  "
              f"{', '.join(parts)}{sep}")
    return n_occ


def try_labels(mf, mol, label_sets, thresholds):
    """Table 2: what each candidate label set would actually give AVAS."""
    from pyscf.mcscf import avas as avas_mod
    kernel = getattr(avas_mod, "avas", None) or avas_mod.kernel

    print(f"\n  {'labels':<38}{'thr':>6}{'space':>12}{'occ orbs':>10}  verdict")
    print("  " + "-" * 84)

    n_core_min = None
    rows = []
    for labels in label_sets:
        bad = [l for l in labels if len(mol.search_ao_label(l)) == 0]
        if bad:
            print(f"  {','.join(labels)[:37]:<38}{'--':>6}{'--':>12}"
                  f"{'--':>10}  no such AO: {bad}")
            continue
        for thr in thresholds:
            try:
                ncas, nelecas, _mo = kernel(mf, list(labels), threshold=thr,
                                            canonicalize=True)
            except Exception as exc:                      # noqa: BLE001
                print(f"  {','.join(labels)[:37]:<38}{thr:>6}{'FAILED':>12}"
                      f"{'':>10}  {exc}")
                continue
            nel = int(np.sum(nelecas))
            n_occ_orbs = nel // 2 + (nel % 2)
            # An active space whose electron count is a small fraction of
            # the valence electron count is the ScF failure: the labels
            # named virtual character only.
            frac = nel / max(1, mol.nelectron - 2 * ((mol.nelectron - nel) // 2))
            verdict = ""
            if ncas == 0:
                verdict = "empty"
            elif nel <= 2 and ncas >= 4:
                verdict = "<-- nearly EMPTY space, labels miss the electrons"
            elif nel >= 2 * ncas - 2:
                verdict = "<-- nearly FULL space, little room to correlate"
            rows.append((labels, thr, ncas, nel))
            print(f"  {','.join(labels)[:37]:<38}{thr:>6}"
                  f"{f'({nel}e,{ncas}o)':>12}{n_occ_orbs:>10}  {verdict}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atom", default="Sc 0 0 0; F 0 0 1.7877")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--labels", nargs="+", default=None,
                   help="candidate label sets, comma-separated within a set: "
                        "\"Sc 3d,Sc 4s\" \"Sc 3d,Sc 4s,F 2p\"")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.2, 0.1, 0.02])
    p.add_argument("--n-virt", type=int, default=10,
                   help="how many virtual MOs to show in table 1")
    args = p.parse_args(argv)

    mol = gto.M(atom=args.atom, basis=args.basis, spin=args.spin,
                charge=args.charge, verbose=0)
    mf = scf.RHF(mol) if args.spin == 0 else scf.ROHF(mol)
    mf.max_cycle, mf.level_shift = 400, 0.5
    mf.kernel()
    print("=" * 74)
    print(f"  {args.atom}   basis={args.basis}  spin={args.spin}")
    print(f"  SCF = {mf.e_tot:.8f} Ha  (converged={mf.converged})")
    print("=" * 74)

    occ = mf.mo_occ
    occ_total = (np.asarray(occ[0]) + np.asarray(occ[1])
                 if isinstance(occ, (tuple, list)) else np.asarray(occ))

    print("\n--- 1. What each MO is made of "
          "---------------------------------------")
    print_mo_table(mol, mf, occ_total, args.n_virt)

    if args.labels:
        label_sets = [tuple(s.strip() for s in grp.split(",")) for grp in args.labels]
    else:
        syms = sorted({mol.atom_symbol(i) for i in range(mol.natm)})
        label_sets = _default_candidates(syms)

    print("\n--- 2. What each label set would give AVAS "
          "-------------------------")
    try_labels(mf, mol, label_sets, args.thresholds)

    print("\n  Pick the set whose electron count matches the valence "
          "electrons in table 1.")
    print("  A space with 2 electrons in 6 orbitals is not a small active "
          "space -- it is\n  the wrong one: the labels named orbitals that "
          "are empty.\n")
    return 0


_TM = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split())
_SHELLS = {"H": "1s", "F": "2p", "O": "2p", "N": "2p", "C": "2p", "Cl": "3p"}


def _default_candidates(syms):
    """Metal-only, metal+ligand-p, and metal+ligand-p+s, in that order."""
    metals = [s for s in syms if s in _TM]
    ligs = [s for s in syms if s not in _TM]
    if not metals:
        return [tuple(f"{s} {_SHELLS.get(s, '2p')}" for s in syms)]
    m = [f"{s} 3d" for s in metals] + [f"{s} 4s" for s in metals]
    lp = [f"{s} {_SHELLS.get(s, '2p')}" for s in ligs]
    ls = [f"{s} {_SHELLS.get(s, '2p')[0]}s" for s in ligs]
    return [tuple(m), tuple(m + lp), tuple(m + lp + ls)]


if __name__ == "__main__":
    sys.exit(main())