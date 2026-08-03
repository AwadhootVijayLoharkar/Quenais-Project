"""
Step 1 -- active-space selection.

Five phases: UHF and tier classification, MP2 density plus ASF's
entanglement-entropy selection, adaptive gap detection, Loewdin population
analysis, and the electron count.

TWO THINGS TO KNOW BEFORE EDITING
---------------------------------
1. Occupation numbers must be computed in the SAME basis as the mo_coeff
   that gets saved. See project_occupations(). Getting this wrong is silent
   and it corrupts the DMET reference density downstream, not just the
   printed active space.

2. This stage owns the step 1 pickle contract. quenais.embedding requires
   dm_ao_alpha_mp2 / dm_ao_beta_mp2 / dm_ao_total_mp2 from it -- the CASCI
   reference-density path raises KeyError without them. The 0.1 package
   collapsed MP2 to a single total density and never spin-resolved it,
   which is why step 2 could not run in its default mode.

Known limitation: ASF's entropy thresholds are calibrated on main-group
systems and under-select for the d-block. Transition-metal systems
generally need cfg.asf.force_active_space. See docs/limitations.md.
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np

__all__ = [
    "main",
    "run_uhf",
    "classify",
    "compute_mp2_density",
    "project_occupations",
    "find_gap_cutoff",
    "lowdin_population",
    "count_active_electrons",
]


# ═════════════════════════════════════════════════════════════════════════
# Phase A -- reference determinant and tier
# ═════════════════════════════════════════════════════════════════════════

def run_uhf(mol):
    """UHF with a DIIS attempt and a Newton fallback."""
    from pyscf import scf

    mf = scf.UHF(mol)
    mf.max_cycle = 400
    mf.level_shift = 0.5
    mf.kernel()

    if not mf.converged:
        print("  DIIS did not converge -> trying Newton solver...")
        newton = mf.newton()
        newton.max_cycle = 400
        newton.kernel(mf.mo_coeff)
        if newton.converged:
            for attr in ("e_tot", "mo_coeff", "mo_energy", "mo_occ", "converged"):
                setattr(mf, attr, getattr(newton, attr))
        else:
            warnings.warn(
                "UHF did not converge with DIIS or Newton. Results may be "
                "unreliable.",
                RuntimeWarning,
            )
    return mf


def classify(mol, mf, cfg):
    """Assign a correlation tier, which selects the ASF thresholds."""
    tiers = cfg.tiers
    has_tm = any(mol.atom_symbol(i) in tiers.tm_elements for i in range(mol.natm))

    s2, _ = mf.spin_square()
    spin_value = mol.spin / 2.0
    s_expected = spin_value * (spin_value + 1.0)
    is_singlet = mol.spin == 0

    if is_singlet:
        spin_cont = float(s2)
        contaminated = spin_cont > tiers.spin_contamination_singlet_threshold
    else:
        spin_cont = float(s2 / s_expected)
        contaminated = spin_cont > tiers.spin_contamination_tier2_threshold

    gaps, gap_min = {}, 10.0
    for label, channel in (("alpha", 0), ("beta", 1)):
        mo_e = np.asarray(mf.mo_energy[channel])
        mo_occ = np.asarray(mf.mo_occ[channel])
        occ_e, vir_e = mo_e[mo_occ > 0.5], mo_e[mo_occ < 0.5]
        if len(occ_e) and len(vir_e):
            gaps[label] = float((vir_e[0] - occ_e[-1]) * cfg.hartree_to_ev)
    if gaps:
        gap_min = min(gaps.values())

    indicators = {
        "has_tm": has_tm,
        "is_singlet": is_singlet,
        "s2": float(s2),
        "s_expected": float(s_expected),
        "spin_cont": float(spin_cont),
        "spin_contaminated": contaminated,
        "homo_lumo_gap_eV": float(gap_min),
        "gap_alpha_eV": gaps.get("alpha"),
        "gap_beta_eV": gaps.get("beta"),
    }

    if has_tm:
        tier = 3
    elif contaminated or gap_min < tiers.homo_lumo_tier2_threshold_ev:
        tier = 2
    else:
        tier = 1

    print(f"  Spin  : {spin_cont:.4f}  contaminated={contaminated}")
    print(f"  Gap   : min={gap_min:.4f} eV")
    print(f"  TM    : {has_tm}  ->  Tier {tier}")
    return tier, indicators


# ═════════════════════════════════════════════════════════════════════════
# Phase B -- MP2 density
# ═════════════════════════════════════════════════════════════════════════

def compute_mp2_density(mf, mol):
    """
    Returns (e_corr, mp2_ok, dm_ao_alpha, dm_ao_beta): the spin-separated
    MP2 1-RDM in the AO basis.

    Deliberately does NOT return deviation/no_occ. Those depend on which MO
    basis you want them expressed in, and this function only has the
    canonical UHF basis. The basis that actually gets saved may be ASF's
    natural-orbital basis instead. Computing occupations here and then
    indexing them with ASF's orbital indices reads values off the wrong
    orbitals -- silently. See project_occupations().

    Spin resolution matters: the embedding's CASCI reference-density path
    needs alpha and beta separately, and reconstructing them from the total
    is not possible.
    """
    from pyscf import mp as pyscf_mp

    mp2_ok, e_corr = False, 0.0
    Ca = np.asarray(mf.mo_coeff[0])
    Cb = np.asarray(mf.mo_coeff[1])

    try:
        mymp = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _ = mymp.kernel()
        dm1 = mymp.make_rdm1()

        if isinstance(dm1, (tuple, list)):
            dm_ao_alpha = Ca @ dm1[0] @ Ca.T
            dm_ao_beta = Cb @ dm1[1] @ Cb.T
        else:
            dm_ao_alpha = 0.5 * (Ca @ dm1 @ Ca.T)
            dm_ao_beta = 0.5 * (Cb @ dm1 @ Cb.T)
        mp2_ok = True

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
        warnings.warn(
            f"MP2 failed with: {exc}\nFalling back to the UHF density matrix.",
            RuntimeWarning,
        )
        dm_raw = mf.make_rdm1()
        if isinstance(dm_raw, (tuple, list)):
            dm_ao_alpha = np.asarray(dm_raw[0])
            dm_ao_beta = np.asarray(dm_raw[1])
        else:
            dm_ao_alpha = dm_ao_beta = 0.5 * np.asarray(dm_raw)

    return e_corr, mp2_ok, dm_ao_alpha, dm_ao_beta


def project_occupations(mo_coeff, dm_ao_total, S):
    """
    Project an AO-basis density matrix's occupation numbers onto whatever
    MO basis mo_coeff actually is.

    Occupations used to be computed once in the canonical UHF basis and
    then indexed with mo_list -- indices into ASF's own, different
    natural-orbital basis. Those are not the same orbitals in the same
    order, so the lookup quietly read values off the wrong orbitals.

    It bites hardest in a degenerate subspace. The true, basis-independent
    occupations of a genuinely symmetric pair ARE equal, because the
    density restricted to that subspace is proportional to the identity --
    but the canonical-basis array did not show that, because it was not the
    basis ASF was using. That is why N2's degenerate pi pair still got
    split even after the gap cutoff was made degeneracy-aware.

    It also reaches beyond the printed active space: the embedding's CASCI
    reference density fills core occupations from step1["no_occ"] indexed
    against step1["mo_coeff"]. If those are not the same basis, the
    reference density is wrong too.
    """
    dm_mo = mo_coeff.T @ S @ dm_ao_total @ S @ mo_coeff
    no_occ = np.clip(np.diag(dm_mo), 0.0, 2.0)
    deviation = np.minimum(no_occ, 2.0 - no_occ)
    return deviation, no_occ


# ═════════════════════════════════════════════════════════════════════════
# Phase C -- gap detection
# ═════════════════════════════════════════════════════════════════════════

def find_gap_cutoff(values, min_n, max_n, degeneracy_tol=1e-3):
    """
    Adaptive gap detection, degeneracy-aware.

    If the chosen cutoff would land strictly inside a block of near-equal
    values, it is extended to the end of that block rather than splitting
    it. Without this, N2's two pi orbitals -- identical entanglement
    entropy, S=0.246 each -- got split, one kept and its degenerate partner
    dropped, leaving a symmetry-broken and physically incomplete active
    space.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    min_n, max_n = max(1, min(min_n, n)), min(max_n, n)
    order = np.argsort(-values)
    sorted_v = values[order]

    if min_n >= max_n:
        k = min_n
    else:
        best_gap, best_n = -1.0, min_n
        for candidate in range(min_n, max_n + 1):
            gap = sorted_v[candidate - 1] - (
                sorted_v[candidate] if candidate < n else 0.0
            )
            if gap > best_gap:
                best_gap, best_n = gap, candidate
        k = best_n

    k_before = k
    while k < n and abs(sorted_v[k - 1] - sorted_v[k]) < degeneracy_tol:
        k += 1

    if k != k_before:
        warnings.warn(
            f"find_gap_cutoff: extended the cutoff from {k_before} to {k} "
            f"orbitals to avoid splitting a (near-)degenerate group "
            f"(values {sorted_v[k_before - 1]:.4f} vs {sorted_v[k_before]:.4f}, "
            f"tol={degeneracy_tol}).",
            RuntimeWarning,
        )
        if k > max_n:
            warnings.warn(
                f"The degeneracy-aware extension pushed the active space to "
                f"{k} orbitals, past gap_max_norb={max_n}. Consider raising "
                f"cfg.asf.gap_max_norb, or setting cfg.asf.force_active_space "
                f"for this molecule.",
                RuntimeWarning,
            )

    best_gap = sorted_v[k - 1] - (sorted_v[k] if k < n else 0.0)
    return k, float(best_gap), list(order[:k])


# ═════════════════════════════════════════════════════════════════════════
# Phase D -- population analysis and electron count
# ═════════════════════════════════════════════════════════════════════════

def lowdin_population(mo_coeff, mo_list, S, ao_labels, n_atoms):
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_sqrt = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    weights = np.zeros((len(mo_list), n_atoms))
    for k, mo_idx in enumerate(mo_list):
        c_lo = S_sqrt @ mo_coeff[:, mo_idx]
        for ao_j, (atom_idx, *_rest) in enumerate(ao_labels):
            weights[k, atom_idx] += c_lo[ao_j] ** 2
    return weights


def count_active_electrons(mol, mf, final_mo_list, cfg):
    active_set = set(final_mo_list)
    mo_occ_total = np.asarray(mf.mo_occ[0]) + np.asarray(mf.mo_occ[1])
    core_orbs = [
        i for i, occ in enumerate(mo_occ_total)
        if i not in active_set and occ > cfg.asf.core_occ_threshold
    ]
    n_core = len(core_orbs)
    nel = mol.nelectron - 2 * n_core

    print(f"  Total electrons: {mol.nelectron}  Core MOs: {n_core}  "
          f"Raw active: {nel}")

    max_nel = 2 * len(final_mo_list)
    if nel <= 0:
        raise ValueError(
            f"Active electron count is {nel} <= 0. Raise "
            f"cfg.asf.core_occ_threshold."
        )
    if nel > max_nel:
        warnings.warn(
            f"Active electrons ({nel}) exceed 2 x active orbitals ({max_nel}). "
            f"Capping.",
            RuntimeWarning,
        )
        nel = max_nel
    if nel % 2 != 0:
        nel -= 1
    nel = max(2, nel)

    print(f"  Final active electrons: {nel}")
    return nel


# ═════════════════════════════════════════════════════════════════════════
# Environment checks
# ═════════════════════════════════════════════════════════════════════════

def _validate_block2_wrapper(cfg):
    """
    Check the block2 wrapper points at the current environment. Pointing at
    a stale env is a common mistake after switching environments.

    Only relevant when ASF/DMRG will actually run -- a forced active space
    never touches block2.
    """
    import sys

    wrapper = cfg.blockexe_wrapper
    if not os.path.exists(wrapper):
        warnings.warn(
            f"block2 wrapper not found: {wrapper}\n"
            f"Run: bash install.sh  OR  "
            f"python -m quenais.utils.regenerate_wrapper",
            RuntimeWarning,
        )
        return

    current_env = os.path.dirname(sys.executable)
    with open(wrapper) as fh:
        content = fh.read()

    if current_env not in content and "block2main" in content:
        warnings.warn(
            f"block2 wrapper may point at the wrong environment.\n"
            f"  Current env : {current_env}\n"
            f"  Wrapper     : {wrapper}\n"
            f"  Fix: run bash install.sh to regenerate it.",
            RuntimeWarning,
        )


# ═════════════════════════════════════════════════════════════════════════
# Stage entry point
# ═════════════════════════════════════════════════════════════════════════

def main(cfg, force=False):
    """
    Select the active space.

    cfg   : quenais.config.Config
    force : recompute even when a valid cached result exists
    """
    from pyscf import gto

    os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.cached_result_is_current(cfg.step1_file) and not force:
        print(f"[Step 1] Using cached result: {cfg.step1_file}")
        print("         Pass force=True to recompute.")
        with open(cfg.step1_file, "rb") as fh:
            return pickle.load(fh)

    if cfg.geometry is None:
        cfg.load_geometry()

    forced = cfg.asf.force_active_space is not None

    print(f"\n{'='*60}")
    print(f"[Step 1] Active Space Finder -- {cfg.molecule}")
    print(f"{'='*60}")

    # verbose=0: the stage prints its own summary, and PySCF's SCF banner
    # is noise in a library context. Raise it when debugging convergence.
    mol = gto.M(atom=cfg.geometry, basis=cfg.basis, charge=cfg.charge,
                spin=cfg.spin, verbose=0)
    print(f"  Atoms     : {cfg.n_atoms} {cfg.atom_syms}  Basis: {cfg.basis}")
    print(f"  Electrons : {mol.nelectron}  AOs: {mol.nao_nr()}")
    S = mol.intor("int1e_ovlp")

    # ── Phase A ──────────────────────────────────────────────────────────
    print("\n-- Phase A: UHF + Classification --")
    mf = run_uhf(mol)
    print(f"  UHF energy = {mf.e_tot:.8f} Ha (converged: {mf.converged})")
    if not mf.converged:
        warnings.warn(
            "UHF did not converge. All downstream results are unreliable.",
            RuntimeWarning,
        )
    tier, indicators = classify(mol, mf, cfg)

    # ── Phase B ──────────────────────────────────────────────────────────
    print("\n-- Phase B: MP2 Density + ASF --")
    e_corr, mp2_ok, dm_ao_alpha_mp2, dm_ao_beta_mp2 = compute_mp2_density(mf, mol)
    dm_ao_total_mp2 = dm_ao_alpha_mp2 + dm_ao_beta_mp2
    print(f"  MP2 used: {mp2_ok}  E_corr: {e_corr:.6f} Ha")

    if forced:
        # No ASF, no DMRG, no block2. Worth keeping cheap: transition-metal
        # systems currently need a forced space, and requiring a working
        # block2 install to bypass block2 would be perverse.
        final_mo_list = sorted(cfg.asf.force_active_space)
        print(f"  cfg.asf.force_active_space is set -- skipping ASF/DMRG "
              f"entirely and using orbitals {final_mo_list} directly "
              f"(indexed in the UHF alpha-MO basis).")

        n_ao = mol.nao_nr()
        out_of_range = [i for i in final_mo_list if i >= n_ao]
        if out_of_range:
            raise ValueError(
                f"force_active_space contains MO indices {out_of_range} that "
                f"do not exist for {cfg.molecule}/{cfg.basis} "
                f"({n_ao} orbitals). These are typically indices copied from "
                f"a different molecule."
            )

        mo_coeff = np.asarray(mf.mo_coeff[0])
        n_final, gap_val = len(final_mo_list), 0.0
    else:
        _validate_block2_wrapper(cfg)
        os.environ["BLOCKEXE"] = cfg.blockexe_wrapper
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["MKL_DEBUG_CPU_TYPE"] = "5"

        from asf.wrapper import find_from_scf
        from pyscf.dmrgscf import dmrgci

        dmrgci.settings.BLOCKEXE = cfg.blockexe_wrapper

        asf_p = cfg.asf.params[tier]
        print(f"  ASF (Tier {tier}): "
              f"entropy_threshold={asf_p['entropy_threshold']}, "
              f"max_norb={asf_p['max_norb']}, min_norb={asf_p['min_norb']}")

        active_space = find_from_scf(
            mf,
            entropy_threshold=asf_p["entropy_threshold"],
            max_norb=asf_p["max_norb"],
            min_norb=asf_p["min_norb"],
            verbose=True,
        )
        mo_list = list(active_space.mo_list)
        mo_coeff = active_space.mo_coeff
        print(f"  ASF candidates: {len(mo_list)} orbitals -> {mo_list}")
        if not mo_list:
            raise RuntimeError(
                f"ASF returned 0 candidates with "
                f"entropy_threshold={asf_p['entropy_threshold']} (Tier {tier}). "
                f"Lower it in cfg.asf.params, or set cfg.asf.force_active_space."
            )

        final_mo_list, n_final, gap_val = _phase_c(
            cfg, mo_list, mo_coeff, dm_ao_total_mp2, S
        )

    # Recompute occupations in the SAME basis as the mo_coeff about to be
    # saved. Required for consistency with mo_list indexing everywhere
    # downstream -- corr_strength here, and the embedding's CASCI reference
    # density, which fills core occupations from no_occ indexed against
    # this mo_coeff.
    deviation, no_occ = project_occupations(mo_coeff, dm_ao_total_mp2, S)

    nel = count_active_electrons(mol, mf, final_mo_list, cfg)

    # ── Phase D ──────────────────────────────────────────────────────────
    print("\n-- Phase D: Loewdin Population --")
    ao_labels = mol.ao_labels(fmt=None)
    weights = lowdin_population(mo_coeff, final_mo_list, S, ao_labels, cfg.n_atoms)
    dominant_atoms = np.argmax(weights, axis=1).astype(int)

    print(f"\n  {'MO':>5}  {'Atom':>6}  {'Symbol':>6}  {'Weight':>8}")
    print(f"  {'-'*33}")
    for k, mo_idx in enumerate(final_mo_list):
        atom = dominant_atoms[k]
        print(f"  {mo_idx:>5}  {atom:>6}  {cfg.atom_syms[atom]:>6}  "
              f"{weights[k, atom]:>8.4f}")

    final_devs = np.array([deviation[i] for i in final_mo_list
                           if i < len(deviation)])
    corr_strength = float(np.mean(final_devs)) if len(final_devs) else 0.0

    print(f"\n{'='*60}")
    print(f"[Step 1] Summary -- {cfg.molecule}")
    print(f"  Tier: {tier}  Active space: ({nel}e, {n_final}orb)  "
          f"Orbitals: {final_mo_list}")
    print(f"  Selection: {'forced (cfg.asf.force_active_space)' if forced else 'automatic (ASF)'}")
    print(f"  Correlation strength: {corr_strength:.4f}")
    print(f"{'='*60}")

    results = {
        "nel": nel,
        "mo_list": final_mo_list,
        "mo_coeff": mo_coeff,
        "n_active_orbs": n_final,
        "no_occ": no_occ,
        "deviation": deviation,
        "lowdin_weights": weights,
        "dominant_atoms": dominant_atoms,
        "tier": tier,
        "indicators": indicators,
        "corr_strength": corr_strength,
        "mol_info": {
            "molecule": cfg.molecule,
            "basis": cfg.basis,
            "n_atoms": cfg.n_atoms,
            "atom_syms": cfg.atom_syms,
            "n_electrons": mol.nelectron,
            "n_ao": mol.nao_nr(),
        },
        "uhf_energy": float(mf.e_tot),
        "mp2_energy": float(mf.e_tot + e_corr),
        "mp2_ok": mp2_ok,
        "mo_coeff_uhf": np.asarray(mf.mo_coeff),
        "mo_energy": np.asarray(mf.mo_energy),
        "mo_occ": np.asarray(mf.mo_occ),
        "converged": mf.converged,
        # Required by the embedding stage's CASCI reference-density path.
        # Absent from the 0.1 pickle, which is why step 2 could not run.
        "dm_ao_alpha_mp2": dm_ao_alpha_mp2,
        "dm_ao_beta_mp2": dm_ao_beta_mp2,
        "dm_ao_total_mp2": dm_ao_total_mp2,
        "forced_active_space": forced,
        "gap_value": float(gap_val),
        "provenance": cfg.provenance(),
    }

    with open(cfg.step1_file, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[Step 1] Saved -> {cfg.step1_file}")
    return results


def _phase_c(cfg, mo_list, mo_coeff, dm_ao_total_mp2, S):
    """Narrow ASF's candidates by MP2 occupation deviation."""
    print("\n-- Phase C: Gap Detection --")

    if not cfg.asf.phase_c_enabled:
        print("  Phase C disabled (cfg.asf.phase_c_enabled=False) -- keeping "
              "ASF's full selection.")
        return sorted(mo_list), len(mo_list), 0.0

    # Deviations computed in ASF's own basis, not the canonical UHF basis.
    # See project_occupations() for why that distinction is what broke N2's
    # degenerate pi pair.
    deviation_asf, _ = project_occupations(mo_coeff, dm_ao_total_mp2, S)
    cand_devs = np.array([deviation_asf[i] for i in mo_list])

    print(f"\n  Candidate orbital deviations (sorted):")
    print(f"  {'MO':>5}  {'dev':>8}")
    for mo_idx, dev in sorted(zip(mo_list, cand_devs), key=lambda x: -x[1]):
        print(f"  {mo_idx:>5}  {dev:>8.4f}")

    n_final, gap_val, selected_k = find_gap_cutoff(
        cand_devs,
        cfg.asf.gap_min_norb,
        cfg.asf.gap_max_norb,
        degeneracy_tol=cfg.asf.gap_degeneracy_tol,
    )
    final_mo_list = sorted(mo_list[k] for k in selected_k)
    print(f"  Gap detected: {gap_val:.4f} at position {n_final} "
          f"-> orbitals {final_mo_list}")

    # Phase C can only ever SHRINK ASF's candidate list, and it ranks by MP2
    # occupation deviation -- a cruder signal than the entanglement entropy
    # ASF used to pick those orbitals. On ScH it silently dropped MO 13 from
    # ASF's own [10,11,12,13], and the resulting (2e,3o) CASSCF recovered
    # barely half the correlation energy of plain CCSD. Discarding orbitals
    # a better metric deliberately selected must be visible.
    dropped = sorted(set(mo_list) - set(final_mo_list))
    if dropped:
        warnings.warn(
            f"Phase C gap detection DISCARDED {len(dropped)} orbital(s) "
            f"{dropped} that ASF itself selected (ASF chose {sorted(mo_list)}, "
            f"kept {final_mo_list}). Phase C ranks by MP2 occupation "
            f"deviation, which is a weaker signal than ASF's entanglement "
            f"entropy, so it can shrink a well-chosen active space. If CASSCF "
            f"on this space underperforms CCSD, that is the likely cause; set "
            f"cfg.asf.force_active_space to override, or "
            f"cfg.asf.phase_c_enabled=False to keep ASF's full selection.",
            RuntimeWarning,
        )

    return final_mo_list, n_final, gap_val
