"""
Step 2 -- DMET embedding Hamiltonian.

Six phases: restore the UHF reference, build the reference density,
Schmidt-decompose to get the bath, build the core mean-field potential,
transform the integrals, and apply the chemical-potential shift.

REWRITTEN IN 0.2, not ported. The 0.1 module had no chemical-potential
correction at all, no mp2/casci reference switch, never computed
ref_occ_alpha/beta or dm_a_hf, and did not save the full Schmidt spectrum.
It also carried three of the five known physics bugs. The content here
comes from the validated test_8 scripts.

THE THREE BUGS THIS FILE EXISTS NOT TO HAVE
-------------------------------------------
1. A bath fabricated from numerical noise when no singular value clears the
   tolerance (~20 Ha errors on N2). See dmet_lib.adaptive_bath.
2. The embedded electron count taken from the active space rather than the
   reference density, which roughly doubles the energy whenever n_bath > 0.
   See the count right after phase C.
3. dm_a_hf built by assuming an occupation pattern instead of projecting
   the converged UHF density, which corrupts the E_core partition. See
   phase E.

The E_core self-consistency identity is deliberately NOT asserted here:
E_core is defined as that difference, so the check is tautological and can
never fail. The real check is verify_embedded_scf(), which runs an
independent SCF on the embedding Hamiltonian and compares against the
full-molecule UHF energy.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings

import numpy as np

from quenais.embedding.dmet_lib import (
    adaptive_bath,
    chemical_potential_correction,
    get_reference_density,
    lowdin_matrices,
    symmetrize_h2e,
)

__all__ = ["main", "verify_embedded_scf", "STEP2_SCHEMA_VERSION"]

#: Bumped when the step 2 pickle layout changes, so downstream consumers
#: fail loudly on an old file instead of raising KeyError mid-run.
#: Version 2 adds mu, sv_all, ref_occ_alpha/beta and
#: reference_density_info, and drops mp2_used / mp2_corr.
STEP2_SCHEMA_VERSION = 2


def verify_embedded_scf(step2, tol=2e-7, verbose=True):
    """
    Run a real SCF on the embedding Hamiltonian and compare with the
    full-molecule UHF energy.

    This is the single most diagnostic quantity in the pipeline. Unlike the
    E_core self-consistency identity, which is tautological, this one can
    actually fail: if a converged self-consistent HF on h1e_emb / h2e_emb /
    ecore does not land on the full molecule's UHF energy, the embedding
    Hamiltonian itself is wrong -- independent of mu or the choice of
    reference density.

    Validated at 1.3e-7 Ha on ScH.

    Returns {"e_scf_emb", "e_uhf_full", "delta", "within_tol"}.
    """
    from pyscf import ao2mo, gto, scf

    n_emb = int(step2["n_emb"])
    n_alpha, n_beta = int(step2["n_alpha"]), int(step2["n_beta"])
    h1e, h2e, ecore = step2["h1e"], step2["h2e"], float(step2["ecore"])

    mol = gto.M(verbose=0)
    mol.nelectron = n_alpha + n_beta
    mol.spin = n_alpha - n_beta
    mol.incore_anyway = True
    mol.build(dump_input=False, parse_arg=False)

    mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    mf.get_hcore = lambda *a, **k: h1e
    mf.get_ovlp = lambda *a, **k: np.eye(n_emb)
    mf._eri = ao2mo.restore(8, h2e, n_emb)
    mf.energy_nuc = lambda *a, **k: ecore
    mf.max_cycle, mf.conv_tol = 200, 1e-10
    # PySCF's logger writes to its own stream, past contextlib.redirect_stdout,
    # so callers that capture stdout still see "converged SCF energy = ...".
    # Silence the object instead of the stream.
    mf.verbose = 0
    mf.kernel()

    if not mf.converged:
        raise RuntimeError(
            "The embedding-space SCF did not converge. Check h1e_emb/h2e_emb "
            "for numerical problems before trusting anything downstream."
        )

    e_uhf_full = float(step2["uhf_energy"])
    delta = float(mf.e_tot) - e_uhf_full
    within = abs(delta) <= tol

    if verbose:
        print(f"  Embedded SCF (independent check) = {mf.e_tot:.10f} Ha")
        print(f"  Full-molecule UHF                = {e_uhf_full:.10f} Ha")
        print(f"  delta = {delta:+.3e} Ha  (tol {tol:.0e})  "
              f"{'OK' if within else 'FAIL'}")
    if not within:
        warnings.warn(
            f"The embedded SCF differs from full UHF by {delta:.3e} Ha, "
            f"beyond the {tol:.0e} tolerance. The embedding Hamiltonian is "
            f"probably wrong; do not trust downstream energies.",
            RuntimeWarning,
        )

    return {"e_scf_emb": float(mf.e_tot), "e_uhf_full": e_uhf_full,
            "delta": delta, "within_tol": bool(within)}


def main(cfg, force=False):
    """
    Build the DMET embedding Hamiltonian.

    cfg   : quenais.config.Config
    force : recompute even when a valid cached result exists
    """
    from pyscf import ao2mo, gto, scf
    from pyscf.scf import hf as pyscf_hf

    os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.cached_result_is_current(cfg.step2_file) and not force:
        print(f"[Step 2] Using cached result: {cfg.step2_file}")
        with open(cfg.step2_file, "rb") as fh:
            return pickle.load(fh)

    if not os.path.exists(cfg.step1_file):
        raise FileNotFoundError(f"Run step 1 first: {cfg.step1_file} not found.")
    if not cfg.cached_result_is_current(cfg.step1_file, verbose=False):
        raise RuntimeError(
            f"{cfg.step1_file} was generated for a different molecule or "
            f"basis than the config specifies ({cfg.molecule}/{cfg.basis}). "
            f"Re-run step 1 first -- silently building an embedding on "
            f"another molecule's active space is exactly the failure this "
            f"check exists to stop."
        )

    with open(cfg.step1_file, "rb") as fh:
        step1 = pickle.load(fh)

    if cfg.geometry is None:
        cfg.load_geometry()

    nel = step1["nel"]
    mo_list, mo_coeff = step1["mo_list"], step1["mo_coeff"]
    n_imp, mol_info = step1["n_active_orbs"], step1["mol_info"]

    print(f"\n{'='*60}")
    print(f"[Step 2] DMET Embedding -- {mol_info['molecule']}")
    print(f"{'='*60}")
    print(f"  Active space (step 1): ({nel}e, {n_imp}orb) MOs={mo_list}")
    print(f"  reference={cfg.dmet.reference}   "
          f"mu_correction={cfg.dmet.mu_correction}")

    mol = gto.M(atom=cfg.geometry, basis=cfg.basis, charge=cfg.charge,
                spin=cfg.spin, verbose=0)
    n_ao = mol.nao_nr()
    S = mol.intor("int1e_ovlp")

    # ── Phase A: restore the UHF reference ───────────────────────────────
    print("\n-- Phase A: Restore UHF from step 1 --")
    mf = scf.UHF(mol)
    mf.mo_coeff = step1["mo_coeff_uhf"]
    mf.mo_energy = step1["mo_energy"]
    mf.mo_occ = step1["mo_occ"]
    mf.e_tot, mf.converged = step1["uhf_energy"], step1["converged"]
    print(f"  UHF energy = {mf.e_tot:.8f} Ha (restored, not recomputed)")
    if not mf.converged:
        warnings.warn("The restored UHF was not converged in step 1.",
                      RuntimeWarning)

    # ── Phase B: reference density ───────────────────────────────────────
    print(f"\n-- Phase B: Reference Density ({cfg.dmet.reference}) --")
    dm_ao_total, dm_ao_alpha, dm_ao_beta, ref_info = get_reference_density(
        mf, mol, step1, mo_list, mo_coeff, cfg.dmet.reference
    )
    print(f"  {ref_info}")
    n_elec_ref = float(np.trace(dm_ao_alpha @ S) + np.trace(dm_ao_beta @ S))
    print(f"  [diag] reference density electron count: {n_elec_ref:.6f}  "
          f"(mol.nelectron = {mol.nelectron})")

    # ── Phase C: Schmidt decomposition ───────────────────────────────────
    print("\n-- Phase C: Schmidt Decomposition --")
    S_sqrt, S_invsqrt = lowdin_matrices(S)

    C_imp = mo_coeff[:, mo_list].copy()
    Q_imp = S_sqrt @ C_imp
    dm_lo = S_sqrt @ dm_ao_total @ S_sqrt
    P_env = np.eye(n_ao) - Q_imp @ Q_imp.T
    F = P_env @ dm_lo @ Q_imp
    U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

    print(f"  [diag] Schmidt singular values (all {len(sv)}): "
          f"{np.array2string(sv, precision=6, suppress_small=True)}")

    n_bath, sv_gap, sv2_cov = adaptive_bath(
        sv, n_imp, cfg.dmet.max_embed_orbs, cfg.dmet.bath_tolerance
    )
    if n_bath < cfg.dmet.min_bath_orbs:
        warnings.warn(f"Only {n_bath} bath orbital(s) found.", RuntimeWarning)

    if n_bath > 0:
        Q_bath = U_env[:, :n_bath]
        Q_emb = np.hstack([Q_imp, Q_bath])
        # Check only the bath vectors actually kept. Checking n_imp columns
        # regardless of how many were used manufactures false alarms: a
        # column whose singular value is exactly 0.0 has a numerically
        # arbitrary direction, free to overlap Q_imp, and is discarded.
        overlap = float(np.max(np.abs(Q_imp.T @ Q_bath)))
        print(f"  [diag] max |Q_imp.T @ Q_bath| over the {n_bath} kept bath "
              f"orbital(s): {overlap:.2e}  (should be ~1e-10)")
    else:
        Q_emb = Q_imp.copy()
        print("  No bath orbital cleared the tolerance -- the embedding is "
              "the active space alone. This is valid, not a failure.")

    n_emb = n_imp + n_bath
    C_emb = S_invsqrt @ Q_emb
    orthonorm_err = float(np.max(np.abs(C_emb.T @ S @ C_emb - np.eye(n_emb))))
    print(f"  Impurity: {n_imp}  Bath: {n_bath}  Embedding: {n_emb} "
          f"-> {2 * n_emb} qubits")
    print(f"  sv^2 coverage: {sv2_cov:.4f}")
    print(f"  [diag] C_emb orthonormality error: {orthonorm_err:.2e}")

    # ── Embedded electron count, from the reference density ──────────────
    # NOT from the active-space count. That is only correct when
    # n_bath == 0; once bath orbitals exist the embedding holds whatever
    # the reference density actually puts there. On LiH the active-space
    # count gives (1, 1) while the reference-density trace gives
    # (2.0000076, 2.0000076) -> (2, 2), and the wrong count roughly doubles
    # the energy.
    dm_emb_alpha = C_emb.T @ S @ dm_ao_alpha @ S @ C_emb
    dm_emb_beta = C_emb.T @ S @ dm_ao_beta @ S @ C_emb
    ref_occ_alpha = np.clip(np.diag(dm_emb_alpha), 0.0, 1.0)
    ref_occ_beta = np.clip(np.diag(dm_emb_beta), 0.0, 1.0)
    n_alpha = int(round(float(np.sum(ref_occ_alpha))))
    n_beta = int(round(float(np.sum(ref_occ_beta))))
    print(f"  [diag] embedding electron count from the reference density: "
          f"alpha={np.sum(ref_occ_alpha):.6f} -> {n_alpha}, "
          f"beta={np.sum(ref_occ_beta):.6f} -> {n_beta}  "
          f"(the active-space count would give "
          f"{nel // 2 + nel % 2}, {nel // 2})")

    # ── Phase D: core mean-field potential ───────────────────────────────
    print("\n-- Phase D: Core Mean-Field Potential --")
    h1e_bare = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    P_core_lo = np.eye(n_ao) - Q_emb @ Q_emb.T

    dm_core_alpha = S_invsqrt @ (
        P_core_lo @ (S_sqrt @ dm_ao_alpha @ S_sqrt) @ P_core_lo
    ) @ S_invsqrt
    dm_core_beta = S_invsqrt @ (
        P_core_lo @ (S_sqrt @ dm_ao_beta @ S_sqrt) @ P_core_lo
    ) @ S_invsqrt
    dm_core_alpha = 0.5 * (dm_core_alpha + dm_core_alpha.T)
    dm_core_beta = 0.5 * (dm_core_beta + dm_core_beta.T)

    n_core_elec = float(np.trace(dm_core_alpha @ S) + np.trace(dm_core_beta @ S))
    n_emb_elec = n_alpha + n_beta
    print(f"  [diag] core density electron count: {n_core_elec:.6f}  "
          f"(expect mol.nelectron - embedding electrons = "
          f"{mol.nelectron} - {n_emb_elec} = {mol.nelectron - n_emb_elec})")

    vj_a, vk_a = pyscf_hf.get_jk(mol, dm_core_alpha, hermi=1)
    vj_b, vk_b = pyscf_hf.get_jk(mol, dm_core_beta, hermi=1)
    h1e_eff = h1e_bare + (vj_a + vj_b) - 0.5 * (vk_a + vk_b)

    # ── Phase E: integral transformation ─────────────────────────────────
    print("\n-- Phase E: Integral Transformation --")
    t0 = time.time()
    h1e_emb = C_emb.T @ h1e_eff @ C_emb
    h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)
    h2e_raw = ao2mo.kernel(mol, C_emb, compact=False).reshape(
        n_emb, n_emb, n_emb, n_emb
    )
    h2e_emb = symmetrize_h2e(h2e_raw)

    sym_err = float(np.max(np.abs(h2e_emb - h2e_emb.transpose(1, 0, 2, 3))))
    print(f"  [diag] h2e symmetry error: {sym_err:.2e}")

    # dm_a_hf from the PROJECTED converged UHF density.
    #
    # This used to be a naive aufbau filling: occupy the first n_alpha
    # columns of C_emb -- which are the impurity orbitals, then the bath
    # orbitals -- regardless of h1e_emb's actual eigenvalue ordering. Since
    # ecore is defined as mf.e_tot - e_hf_emb, that invented density
    # directly determines how much of the true HF energy is folded into
    # ecore rather than the embedding space. Projecting the real UHF
    # solution removes the guess.
    dm_hf_ao_alpha, dm_hf_ao_beta = mf.make_rdm1()
    dm_a_hf = C_emb.T @ S @ dm_hf_ao_alpha @ S @ C_emb
    dm_b_hf = C_emb.T @ S @ dm_hf_ao_beta @ S @ C_emb
    dm_t_hf = dm_a_hf + dm_b_hf

    e1_hf = float(np.einsum("ij,ji->", h1e_emb, dm_t_hf))
    J_hf = np.einsum("pqrs,rs->pq", h2e_emb, dm_t_hf)
    Ka_hf = np.einsum("prqs,rs->pq", h2e_emb, dm_a_hf)
    Kb_hf = np.einsum("prqs,rs->pq", h2e_emb, dm_b_hf)
    e2_hf = 0.5 * (
        float(np.einsum("pq,qp->", J_hf, dm_t_hf))
        - float(np.einsum("pq,qp->", Ka_hf, dm_a_hf))
        - float(np.einsum("pq,qp->", Kb_hf, dm_b_hf))
    )
    ecore = float(mf.e_tot) - (e1_hf + e2_hf)
    print(f"  ecore = {ecore:.8f} Ha")

    # ── Phase F: chemical potential ──────────────────────────────────────
    mu = 0.0
    if cfg.dmet.mu_correction:
        print("\n-- Phase F: Chemical Potential Correction --")
        h1e_emb, mu = chemical_potential_correction(
            h1e_emb, n_emb, n_alpha, n_beta,
            cfg.dmet.mu_search_range, cfg.dmet.mu_max_iter, cfg.dmet.mu_tol,
        )
        ecore += mu * (n_alpha + n_beta)
        print(f"  mu = {mu:.6f} Ha   ecore -> {ecore:.8f} Ha")
    else:
        print("\n-- Phase F: Chemical Potential Correction -- SKIPPED")

    print(f"\n  h1e {h1e_emb.shape}  h2e {h2e_emb.shape}  "
          f"Time: {time.time() - t0:.1f}s")

    results = {
        "schema_version": STEP2_SCHEMA_VERSION,
        "h1e": h1e_emb,
        "h2e": h2e_emb,
        "ecore": ecore,
        "mu": mu,
        "n_emb": n_emb,
        "n_imp": n_imp,
        "n_bath": n_bath,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "sv": sv[:n_bath],
        "sv_all": sv,
        "sv_gap": sv_gap,
        "sv2_cov": sv2_cov,
        "uhf_energy": float(mf.e_tot),
        "reference_density_info": ref_info,
        "ref_occ_alpha": ref_occ_alpha,
        "ref_occ_beta": ref_occ_beta,
        "mol_info": mol_info,
        "provenance": cfg.provenance(),
    }

    # The real check, run before saving so its verdict is stored with the
    # data it describes.
    print("\n-- Verification: independent embedded SCF --")
    try:
        results["embedded_scf_check"] = verify_embedded_scf(results)
    except Exception as exc:
        warnings.warn(f"The embedded SCF check could not run: {exc}",
                      RuntimeWarning)
        results["embedded_scf_check"] = None

    with open(cfg.step2_file, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[Step 2] Saved -> {cfg.step2_file}")

    print(f"\n{'='*60}")
    print(f"[Step 2] Summary -- {mol_info['molecule']}")
    print(f"  Embedding : {n_imp} imp + {n_bath} bath = {n_emb} orbitals "
          f"({2 * n_emb} qubits)")
    print(f"  Electrons : {n_alpha} alpha + {n_beta} beta")
    print(f"  ecore     : {ecore:.8f} Ha    mu: {mu:.6f} Ha")
    print(f"{'='*60}")
    return results
