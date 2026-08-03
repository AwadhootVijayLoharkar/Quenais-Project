"""
Step 0 -- classical reference methods on the full molecule.

These energies are the answer key the DMET/GQE pipeline is validated
against, so this stage is worth more care than its size suggests. Every
bug found during the N2 and LiH work was caught by a disagreement with a
number produced here.

CASSCF and NEVPT2 reuse step 1's active space when it is available. Run
the active-space finder FIRST for a fair comparison with the embedding
pipeline: the fallback guess can be meaningless (cramming N electrons into
N/2 orbitals leaves no correlating degrees of freedom, so CASSCF trivially
returns exactly the HF energy).

REWRITTEN in 0.2. The 0.1 version of this module was structurally broken:
an indentation error left main() ending after the banner, with the entire
run-and-save body absorbed into _run_nevpt2()'s scope. It parsed, imported
and ran without error -- and did nothing, returning None and writing no
pickle. Physics content is ported from the validated test_8 script.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings

__all__ = ["main", "METHOD_TIERS"]


#: Reproducibility class per method -- written into the results so a partner
#: comparing numbers knows which differences are expected.
#:
#: CASSCF is an optimisation and can converge to different valid solutions:
#: two runs of identical ScH input gave -752.680677 and -752.681604, 0.93 mHa
#: apart. NEVPT2 is built on the CASSCF reference and moves with it (3.6 mHa
#: on the same pair of runs). The single-determinant methods reproduce to
#: ~1e-10 across machines. See docs/limitations.md.
METHOD_TIERS = {
    "HF": "deterministic",
    "MP2": "deterministic",
    "CCSD": "deterministic",
    "CCSD_T": "deterministic",
    "CASSCF": "optimizer-dependent",
    "NEVPT2": "optimizer-dependent",
}


# ═════════════════════════════════════════════════════════════════════════
# Helpers -- module level, no side effects
# ═════════════════════════════════════════════════════════════════════════

def _timer(name):
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            print(f"  [{name}] done in {time.time() - self.t0:.1f}s")

    return Timer()


def _run_hf(mol):
    """RHF for closed shell, UHF otherwise, with a Newton fallback."""
    from pyscf import scf

    print("\n-- HF --")
    is_restricted = mol.spin == 0
    mf = scf.RHF(mol) if is_restricted else scf.UHF(mol)
    mf.max_cycle, mf.level_shift, mf.verbose = 400, 0.3, 0
    with _timer("HF"):
        mf.kernel()

    if not mf.converged:
        # Second-order fallback. Level shifting gets most cases close but
        # can stall just short of the tolerance.
        newton = mf.newton()
        newton.verbose = 0
        newton.kernel(mf.mo_coeff)
        if newton.converged:
            mf.mo_coeff, mf.mo_energy = newton.mo_coeff, newton.mo_energy
            mf.mo_occ, mf.e_tot, mf.converged = newton.mo_occ, newton.e_tot, True

    label = "RHF" if is_restricted else "UHF"
    print(f"  {label} energy: {mf.e_tot:.8f} Ha (converged: {mf.converged})")
    if not mf.converged:
        warnings.warn(
            "HF did not converge, even after the Newton fallback. Every "
            "downstream method builds on this reference, so treat all of "
            "step 0 as unreliable.",
            RuntimeWarning,
        )
    return mf, float(mf.e_tot)


def _run_mp2(mf):
    from pyscf import mp

    print("\n-- MP2 --")
    try:
        mymp = mp.MP2(mf)
        mymp.verbose = 0
        with _timer("MP2"):
            e_corr, _ = mymp.kernel()
        e_mp2 = float(mf.e_tot + e_corr)
        print(f"  E_corr: {e_corr:.8f} Ha   MP2 energy: {e_mp2:.8f} Ha")
        return e_mp2, float(e_corr), mymp
    except Exception as exc:
        warnings.warn(f"MP2 failed: {exc}", RuntimeWarning)
        return None, None, None


def _run_ccsd(mf):
    from pyscf import cc

    print("\n-- CCSD --")
    try:
        mycc = cc.CCSD(mf)
        mycc.verbose, mycc.max_cycle = 0, 200
        with _timer("CCSD"):
            mycc.kernel()
        e_ccsd = float(mf.e_tot + mycc.e_corr)
        print(f"  E_corr: {mycc.e_corr:.8f} Ha   CCSD energy: {e_ccsd:.8f} Ha   "
              f"Converged: {mycc.converged}")
        if not mycc.converged:
            warnings.warn("CCSD did not converge.", RuntimeWarning)
        return e_ccsd, float(mycc.e_corr), mycc
    except Exception as exc:
        warnings.warn(f"CCSD failed: {exc}", RuntimeWarning)
        return None, None, None


def _run_ccsd_t(mf, mycc):
    print("\n-- CCSD(T) --")
    if mycc is None:
        print("  Skipped -- CCSD not available.")
        return None, None
    try:
        with _timer("CCSD(T)"):
            e_t = mycc.ccsd_t()
        e_ccsdt = float(mf.e_tot + mycc.e_corr + e_t)
        print(f"  (T) correction: {e_t:.8f} Ha   CCSD(T) energy: {e_ccsdt:.8f} Ha")
        return e_ccsdt, float(e_t)
    except Exception as exc:
        warnings.warn(f"CCSD(T) failed: {exc}", RuntimeWarning)
        return None, None


def _run_casscf(mf, nel, norb, mo_guess=None):
    from pyscf import mcscf

    print(f"\n-- CASSCF({nel}e,{norb}o) --")
    try:
        mc = mcscf.CASSCF(mf, norb, nel)
        mc.verbose, mc.max_cycle, mc.conv_tol = 0, 500, 1e-8
        mo = (mcscf.addons.sort_mo(mc, mf.mo_coeff, mo_guess, base=0)
              if mo_guess is not None else mf.mo_coeff)
        with _timer(f"CASSCF({nel}e,{norb}o)"):
            mc.kernel(mo)
        print(f"  CASSCF energy: {mc.e_tot:.8f} Ha   CI energy: {mc.e_cas:.8f} Ha   "
              f"Converged: {mc.converged}")
        if not mc.converged:
            warnings.warn("CASSCF did not converge.", RuntimeWarning)
        return float(mc.e_tot), mc
    except Exception as exc:
        warnings.warn(f"CASSCF failed: {exc}", RuntimeWarning)
        return None, None


def _resolve_nevpt_class():
    """
    Locate PySCF's strongly-contracted NEVPT2 solver.

    The class is named `NEVPT`, exported as pyscf.mrpt.NEVPT. "NEVPT2" is
    the method's name in the literature, not the class name -- asking for
    pyscf.mrpt.nevpt2.NEVPT2 raises AttributeError, which surfaced as a
    bare "NEVPT2 FAILED" row in the results table with the real cause
    buried in a stderr warning. A wrong API name, not a convergence or
    memory problem.
    """
    from pyscf import mrpt

    solver = getattr(mrpt, "NEVPT", None)
    if solver is None:  # very old or unusual PySCF layouts
        from pyscf.mrpt import nevpt2 as nevpt2_mod

        solver = getattr(nevpt2_mod, "NEVPT", None)
    if solver is None:
        raise AttributeError(
            "Could not find PySCF's NEVPT solver class (tried pyscf.mrpt.NEVPT "
            "and pyscf.mrpt.nevpt2.NEVPT)."
        )
    return solver


def _run_nevpt2(mc):
    print("\n-- NEVPT2 --")
    if mc is None:
        print("  Skipped -- CASSCF not available.")
        return None

    if not getattr(mc, "converged", True):
        # NEVPT2 is a perturbative correction on top of the CASSCF
        # reference, so a poorly converged CASSCF yields a number that is
        # untrustworthy rather than obviously broken. That matters here:
        # this energy is meant to be the answer key.
        warnings.warn(
            "Running NEVPT2 on a CASSCF reference that did NOT converge. "
            "Treat the resulting energy as unreliable -- it is not a valid "
            "answer key for validating the DMET/GQE pipeline.",
            RuntimeWarning,
        )

    try:
        solver_cls = _resolve_nevpt_class()
        with _timer("NEVPT2"):
            e_corr = solver_cls(mc).kernel()
        e_total = float(mc.e_tot + e_corr)
        print(f"  E_corr(NEVPT2): {e_corr:.8f} Ha   NEVPT2 energy: {e_total:.8f} Ha")
        return e_total
    except Exception as exc:
        warnings.warn(f"NEVPT2 failed: {exc}", RuntimeWarning)
        return None


def _choose_active_space(cfg, mol, step1):
    """Active space for CASSCF: step 1's if available, else a fallback."""
    if step1 is not None:
        nel, norb, mo_guess = step1["nel"], step1["n_active_orbs"], step1["mo_list"]
        print(f"\n  Using step 1 active space: ({nel}e, {norb}o)")
        return nel, norb, mo_guess

    nel = min(mol.nelectron, 10)
    norb = min(mol.nao_nr() // 2, 8)
    print(f"\n  No step 1 found. Fallback: ({nel}e, {norb}o)")
    if nel >= 2 * norb:
        warnings.warn(
            f"Fallback active space ({nel}e, {norb}o) leaves no correlating "
            f"degrees of freedom -- CASSCF will trivially return the HF "
            f"energy. Run step 1 first.",
            RuntimeWarning,
        )
    return nel, norb, None


# ═════════════════════════════════════════════════════════════════════════
# Stage entry point
# ═════════════════════════════════════════════════════════════════════════

def main(cfg, force=False):
    """
    Run the classical reference methods.

    cfg   : quenais.config.Config
    force : recompute even when a valid cached result exists
    """
    from pyscf import gto

    os.makedirs(cfg.results_dir, exist_ok=True)

    # Content-validated, not just existence-checked: every stage writes to a
    # fixed filename shared across molecules, so a plain exists() check
    # silently reuses the previous system's results.
    if cfg.cached_result_is_current(cfg.step0_file) and not force:
        print(f"[Step 0] Using cached result: {cfg.step0_file}")
        with open(cfg.step0_file, "rb") as fh:
            return pickle.load(fh)

    print(f"\n{'='*60}")
    print(f"[Step 0] Classical Methods -- {cfg.molecule}")
    print(f"{'='*60}")
    print(f"  Basis     : {cfg.basis}")
    print(f"  Charge    : {cfg.charge}   Spin (2S): {cfg.spin}")
    print(f"  Methods   : {cfg.classical_methods}")

    if cfg.geometry is None:
        cfg.load_geometry()

    mol = gto.M(atom=cfg.geometry, basis=cfg.basis, charge=cfg.charge,
                spin=cfg.spin, verbose=0)
    print(f"  Electrons : {mol.nelectron}   AOs: {mol.nao_nr()}")

    step1 = None
    if cfg.cached_result_is_current(cfg.step1_file, verbose=False):
        with open(cfg.step1_file, "rb") as fh:
            step1 = pickle.load(fh)
        print(f"  Step 1 loaded: ({step1['nel']}e, {step1['n_active_orbs']}orb)")
    elif os.path.exists(cfg.step1_file):
        print("  Step 1 exists but was built for a different molecule/basis "
              "-- ignoring it; CASSCF will use the fallback active space.")
    else:
        print("  Step 1 not found -- CASSCF/NEVPT2 will use a fallback active "
              "space. Run step 1 first for a meaningful comparison.")

    results = {"molecule": cfg.molecule, "basis": cfg.basis, "methods": {}}
    t_total = time.time()

    mf, e_hf = _run_hf(mol)
    results["methods"]["HF"] = {"energy": e_hf, "converged": bool(mf.converged)}

    if "MP2" in cfg.classical_methods:
        e_mp2, e_corr, _ = _run_mp2(mf)
        results["methods"]["MP2"] = {
            "energy": e_mp2, "e_corr": e_corr, "success": e_mp2 is not None,
        }

    mycc = None
    if "CCSD" in cfg.classical_methods:
        e_ccsd, e_corr, mycc = _run_ccsd(mf)
        results["methods"]["CCSD"] = {
            "energy": e_ccsd, "e_corr": e_corr, "success": e_ccsd is not None,
            "converged": bool(mycc.converged) if mycc else False,
        }

    if "CCSD_T" in cfg.classical_methods:
        e_ccsdt, e_t = _run_ccsd_t(mf, mycc)
        results["methods"]["CCSD_T"] = {
            "energy": e_ccsdt, "e_t_correction": e_t,
            "success": e_ccsdt is not None,
        }

    mc = None
    if "CASSCF" in cfg.classical_methods:
        nel_cas, norb_cas, mo_guess = _choose_active_space(cfg, mol, step1)
        e_casscf, mc = _run_casscf(mf, nel_cas, norb_cas, mo_guess)
        results["methods"]["CASSCF"] = {
            "energy": e_casscf, "nel": nel_cas, "norb": norb_cas,
            "success": e_casscf is not None,
            "converged": bool(mc.converged) if mc else False,
        }

    if "NEVPT2" in cfg.classical_methods:
        e_nevpt2 = _run_nevpt2(mc)
        results["methods"]["NEVPT2"] = {
            "energy": e_nevpt2, "success": e_nevpt2 is not None,
        }

    # Tag each method with its reproducibility class.
    for name, data in results["methods"].items():
        data["tier"] = METHOD_TIERS.get(name, "unknown")

    results["total_time"] = time.time() - t_total
    results["provenance"] = cfg.provenance()

    _print_table(cfg, results, e_hf)

    with open(cfg.step0_file, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[Step 0] Saved -> {cfg.step0_file}")
    return results


def _print_table(cfg, results, e_hf):
    print(f"\n{'='*72}")
    print(f"[Step 0] Results -- {cfg.molecule} / {cfg.basis}")
    print(f"{'='*72}")
    print(f"\n  {'Method':<10} {'Energy (Ha)':>17} {'vs HF (Ha)':>13} "
          f"{'kcal/mol':>11}  {'reproducibility':<20}")
    print(f"  {'-'*70}")

    for method, data in results["methods"].items():
        energy = data.get("energy")
        tier = data.get("tier", "")
        if energy is None:
            print(f"  {method:<10} {'FAILED':>17} {'':>13} {'':>11}  {tier:<20}")
            continue
        vs_hf = energy - e_hf
        print(f"  {method:<10} {energy:>17.8f} {vs_hf:>+13.6f} "
              f"{vs_hf * cfg.hartree_to_kcal_mol:>+11.2f}  {tier:<20}")

    print(f"\n  'optimizer-dependent' values can differ between runs and "
          f"machines; see docs/limitations.md.")
    print(f"  Total time: {results['total_time']:.1f}s")
    print(f"{'='*72}")
