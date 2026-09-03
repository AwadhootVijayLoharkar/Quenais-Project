#!/usr/bin/env python3
"""
Probe AVAS and APC against the forced ScH active space.

Standalone -- imports nothing from quenais, so it can be run before any of
the integration in docs/.../09_active_space_alternatives.md is written.

WHAT IT ANSWERS
---------------
ASF under-selects for the d-block, so ScH has needed
force_active_space=[9,10,11,12,13,14] since 0.2. Can a PySCF selector find
that space by itself?

The pass/fail criterion is the project's own, from docs/limitations.md:

    a well-chosen active space should put CASSCF+NEVPT2 at or below
    CCSD(T). If NEVPT2 lands above CCSD(T), the active space is too small.

ScH's forced space is itself ~7.2 mHa above CCSD(T), so the honest target
is "no worse than forced", not "below CCSD(T)".

HOW TO READ THE OUTPUT
----------------------
Compare on energy and on d-character. Do NOT compare mo_list against
[9,...,14]: AVAS returns a rotated basis, so identical physics gives
different indices. Comparing index lists across bases is the error
finder.py's project_occupations() docstring is about.

USAGE
-----
    python probe_avas_apc.py                          # ScH, sto-3g
    python probe_avas_apc.py --basis ccpvdz           # if sto-3g is too coarse
    python probe_avas_apc.py --scan-geometry 1.78 2.2 2.6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pyscf import gto, scf, mcscf, mrpt, cc

# ScH bond length shipped in quenais.utils.geometry.BUILTIN_GEOMETRIES
SCH_R = 1.78
FORCED_SCH = [9, 10, 11, 12, 13, 14]


# ─────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────

def build(mol_spec, basis, spin, charge):
    mol = gto.M(atom=mol_spec, basis=basis, spin=spin, charge=charge,
                verbose=0)
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.max_cycle = 400
    mf.level_shift = 0.5
    mf.kernel()
    if not mf.converged:
        print("  !! SCF did not converge -- every number below is suspect")
    return mol, mf


def lowdin_shell_weight(mol, mo_coeff, mo_list, shell_substr):
    """
    Fraction of each selected orbital sitting on AOs whose label contains
    shell_substr (e.g. '3d'). Loewdin, so the weights are basis-orthogonal
    and sum to 1 across all AOs.
    """
    S = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_half = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    labels = mol.ao_labels()
    hit = np.array([shell_substr in lab for lab in labels])

    out = []
    for idx in mo_list:
        c = S_half @ mo_coeff[:, idx]
        w = c ** 2
        out.append(float(w[hit].sum() / w.sum()))
    return np.array(out)


def cas_energies(mf, mol, ncas, nelecas, mo, label):
    """CASSCF then NEVPT2 on a given orbital set. Returns (e_cas, e_nevpt2)."""
    try:
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.verbose = 0
        mc.max_cycle_macro = 200
        mc.kernel(mo)
        e_cas = float(mc.e_tot)
    except Exception as exc:                       # noqa: BLE001
        print(f"    CASSCF failed for {label}: {exc}")
        return None, None
    try:
        e_pt2 = float(mc.e_tot + mrpt.NEVPT(mc).kernel())
    except Exception as exc:                       # noqa: BLE001
        print(f"    NEVPT2 failed for {label}: {exc}")
        e_pt2 = None
    return e_cas, e_pt2


def active_block(mol, ncas, nelecas):
    """PySCF returns mo = [freeze, core, cas, vir]; find the cas block."""
    nel = int(np.sum(nelecas))
    ncore = (mol.nelectron - nel) // 2
    return list(range(ncore, ncore + int(ncas))), nel


def report(name, mol, mf, mo, mo_list, nel, ref_ccsd_t, dshell):
    ncas = len(mo_list)
    dchar = lowdin_shell_weight(mol, mo, mo_list, dshell)
    e_cas, e_pt2 = cas_energies(mf, mol, ncas, nel, mo, name)

    verdict = "?"
    if e_pt2 is not None and ref_ccsd_t is not None:
        delta = (e_pt2 - ref_ccsd_t) * 1e3
        verdict = f"NEVPT2 - CCSD(T) = {delta:+8.3f} mHa"
        verdict += "   OK" if delta <= 0 else "   space likely too small"

    print(f"\n  {name}")
    print(f"    space        : ({nel}e, {ncas}o)   mo_list={mo_list}")
    print(f"    {dshell}-character : " +
          " ".join(f"{x:.2f}" for x in dchar) +
          f"   (sum {dchar.sum():.2f})")
    print(f"    CASSCF       : {e_cas if e_cas is None else f'{e_cas:.8f}'}")
    print(f"    NEVPT2       : {e_pt2 if e_pt2 is None else f'{e_pt2:.8f}'}")
    print(f"    verdict      : {verdict}")
    return {"name": name, "ncas": ncas, "nel": nel, "e_cas": e_cas,
            "e_pt2": e_pt2, "dchar_sum": float(dchar.sum())}


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def run_one_geometry(atom, basis, spin, charge, dshell, ao_labels,
                     thresholds, apc_sizes, forced, forced_nel):
    mol, mf = build(atom, basis, spin, charge)
    print(f"  SCF          : {mf.e_tot:.8f} Ha  "
          f"(converged={mf.converged})")
    print(f"  electrons={mol.nelectron}  AOs={mol.nao_nr()}")

    try:
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.kernel()
        ref = float(mycc.e_tot + mycc.ccsd_t())
        print(f"  CCSD(T)      : {ref:.8f} Ha   <- the answer key")
    except Exception as exc:                       # noqa: BLE001
        print(f"  CCSD(T) failed: {exc}")
        ref = None

    rows = []

    # ── control: the forced space ────────────────────────────────────────
    if forced:
        caslst = [i for i in forced if i < mol.nao_nr()]
        if len(caslst) != len(forced):
            print(f"  !! forced indices out of range for this basis; using "
                  f"{caslst}")
        if caslst:
            ncas = len(caslst)
            nel = forced_nel
            mc = mcscf.CASSCF(mf, ncas, nel)
            # sort_mo moves the chosen columns into the active block, so the
            # returned mo has the same [core, cas, vir] layout AVAS/APC use
            # and active_block() locates the cas block the same way.
            mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, caslst, base=0)
            mo_list, _ = active_block(mol, ncas, nel)
            rows.append(report(f"FORCED {caslst}", mol, mf, mo,
                               mo_list, nel, ref, dshell))

    # ── AVAS threshold scan ──────────────────────────────────────────────
    try:
        from pyscf.mcscf import avas as avas_mod
        avas_kernel = getattr(avas_mod, "avas", None) or avas_mod.kernel
    except ImportError as exc:
        print(f"  !! pyscf.mcscf.avas unavailable: {exc}")
        avas_kernel = None

    if avas_kernel is not None:
        for label in ao_labels:
            if len(mol.search_ao_label(label)) == 0:
                print(f"  !! AVAS label {label!r} matches no AO -- available: "
                      f"{sorted(set(mol.ao_labels()))}")
                avas_kernel = None
                break

    if avas_kernel is not None:
        for thr in thresholds:
            try:
                ncas, nelecas, mo = avas_kernel(
                    mf, ao_labels, threshold=thr, canonicalize=True)
            except Exception as exc:               # noqa: BLE001
                print(f"\n  AVAS thr={thr}: failed -- {exc}")
                continue
            if ncas == 0:
                print(f"\n  AVAS thr={thr}: selected 0 orbitals")
                continue
            mo_list, nel = active_block(mol, ncas, nelecas)
            rows.append(report(f"AVAS thr={thr} {ao_labels}", mol, mf, mo,
                               mo_list, nel, ref, dshell))

    # ── APC size scan ────────────────────────────────────────────────────
    try:
        from pyscf.mcscf import apc as apc_mod
    except ImportError as exc:
        print(f"  !! pyscf.mcscf.apc unavailable: {exc}")
        apc_mod = None

    if apc_mod is not None:
        for size in apc_sizes:
            try:
                ncas, nelecas, mo = apc_mod.APC(mf, max_size=size).kernel()
            except Exception as exc:               # noqa: BLE001
                print(f"\n  APC max_size={size}: failed -- {exc}")
                continue
            if ncas == 0:
                print(f"\n  APC max_size={size}: selected 0 orbitals")
                continue
            mo_list, nel = active_block(mol, ncas, nelecas)
            rows.append(report(f"APC max_size={size}", mol, mf, mo,
                               mo_list, nel, ref, dshell))

    return rows, ref


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atom", default=None,
                   help="PySCF geometry string. Default: ScH at 1.78 A")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--ao-labels", nargs="+",
                   default=["Sc 3d", "Sc 4s", "H 1s"],
                   help="AVAS labels. Quote them: 'Sc 3d'")
    p.add_argument("--d-shell", default="3d",
                   help="AO-label substring used for the character report")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.5, 0.2, 0.1, 0.05, 0.01])
    p.add_argument("--apc-sizes", nargs="+", type=int, default=[4, 6, 8, 10])
    p.add_argument("--forced", nargs="*", type=int, default=FORCED_SCH,
                   help="control space, MO indices in the SCF basis. Pass "
                        "--forced with no values to skip the control. It is "
                        "skipped automatically whenever --atom is given, "
                        "since [9..14] means nothing for another molecule.")
    p.add_argument("--forced-nel", type=int, default=4,
                   help="active electrons in the control space (ScH: 4)")
    p.add_argument("--scan-geometry", nargs="+", type=float, default=None,
                   help="bond lengths (A) to repeat the whole probe at. This "
                        "is the test that matters: a forced index list is "
                        "only valid at one geometry, AVAS should not be.")
    args = p.parse_args(argv)

    geoms = args.scan_geometry or [SCH_R]

    for r in geoms:
        atom = args.atom or f"Sc 0 0 0; H 0 0 {r}"
        print("\n" + "=" * 72)
        print(f"  {atom}   basis={args.basis}  spin={args.spin}")
        print("=" * 72)
        rows, ref = run_one_geometry(
            atom, args.basis, args.spin, args.charge, args.d_shell,
            args.ao_labels, args.thresholds, args.apc_sizes,
            args.forced if args.atom is None else None,
            args.forced_nel,
        )

        print("\n  " + "-" * 70)
        header = f"  {'selection':<34}{'space':>10}{'d-char':>9}{'NEVPT2-CCSD(T)':>17}"
        print(header)
        print("  " + "-" * 70)
        for row in rows:
            if row["e_pt2"] is None or ref is None:
                delta = "n/a"
            else:
                delta = f"{(row['e_pt2'] - ref) * 1e3:+.3f} mHa"
            space = "({}e,{}o)".format(row["nel"], row["ncas"])
            name = row["name"][:33]
            print(f"  {name:<34}{space:>10}"
                  f"{row['dchar_sum']:>9.2f}{delta:>17}")

    print("\n  Read this on energy and d-character, not on index equality:")
    print("  AVAS returns a rotated basis, so the same physical space gives")
    print("  different mo_list values than the forced [9..14].\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())