"""
Exact and near-exact reference methods: FCI, CASCI, DMRG.

Step 0 already had HF, MP2, CCSD, CCSD(T), CASSCF and NEVPT2. All six are
either single-reference or perturbative, which means they degrade exactly
where this package is pointed: stretched bonds, transition-metal centres,
antiferromagnetic coupling. A dissociation curve benchmarked against
CCSD(T) shows the two methods disagreeing at long range without saying
which one is wrong -- CCSD(T) is usually the one that is wrong there.

This module adds the three references that stay correct across a curve:

  FCI    exact in the full basis. Feasible to roughly 16 orbitals. It is
         the only number here that validates the active-space CHOICE as
         well as the solver, because it never made one.

  CASCI  exact within step 1's active space, at step 1's orbitals. This is
         the apples-to-apples reference for every step-3 solver: SQD,
         SKQD, SQDrift, GQE and LASSQD are all approximating this exact
         number in this exact space, so the difference is solver error
         with nothing else mixed in. Cheap, and the single most useful
         addition here.

  DMRG   near-exact to roughly 50 orbitals via block2, controlled by one
         parameter (bond dimension M). Runs in stages of increasing M and
         extrapolates to zero discarded weight, which is how a DMRG energy
         should be quoted.

Open-shell note: the other step-0 methods build on UHF. CASCI and DMRG
need a spin-free core potential, so this module converges its own ROHF
reference when mol.spin != 0 and says so in the log. The energies are
therefore ROHF-based and are not comparable orbital-for-orbital with the
UHF-based CCSD numbers in the same table -- they are comparable with each
other and with the step-3 solvers, which is what they are for.

Licensing: block2 is GPL-3.0 and is imported ONLY when DMRG is requested.
Reference energies it produces are data, not a derived work, so storing
and publishing them is fine. Shipping a container or packed environment
containing block2 is not. See docs/licensing.md.
"""

from __future__ import annotations

import math
import os
import shutil
import time
import warnings

import numpy as np

__all__ = [
    "REFERENCE_TIERS",
    "restricted_reference",
    "fci_determinant_count",
    "run_fci",
    "run_casci",
    "run_dmrg",
]


#: Reproducibility class, merged into runner.METHOD_TIERS.
#:
#: FCI is unique given (geometry, basis): there is nothing to converge to a
#: different answer. CASCI inherits step 1's orbitals, so two runs agree
#: only if step 1 did. DMRG is variational in M and its residual error is
#: the truncation error, which the extrapolation estimates.
REFERENCE_TIERS = {
    "FCI": "deterministic",
    "CASCI": "reference-dependent",
    "DMRG": "bond-dimension-limited",
}


def _timer(name):
    # Local copy rather than an import from runner: runner imports this
    # module, and a back-import would make the pair circular.
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            self.dt = time.time() - self.t0
            print(f"  [{name}] done in {self.dt:.1f}s")

    return Timer()


# ═════════════════════════════════════════════════════════════════════════
# Shared machinery
# ═════════════════════════════════════════════════════════════════════════

def fci_determinant_count(norb, nelec):
    """Determinants in the FCI space. Exact integer -- no overflow."""
    na, nb = nelec
    return math.comb(int(norb), int(na)) * math.comb(int(norb), int(nb))


def restricted_reference(mol, mf):
    """
    A spin-free SCF reference for CASCI/DMRG.

    Closed shell: reuse the caller's RHF, which is already the right thing.
    Open shell: step 0 ran UHF, whose alpha and beta orbitals differ, and a
    CASCI core potential built from them is not spin free. Converge an ROHF
    instead. It costs seconds next to the methods that follow, and it makes
    the reference reproducible rather than dependent on which UHF solution
    the SCF happened to find.
    """
    from pyscf import scf

    if mol.spin == 0 and not isinstance(mf, scf.uhf.UHF):
        return mf, False

    print("  Open shell: converging an ROHF reference for CASCI/DMRG "
          "(step 0's UHF orbitals are not spin free).")
    rohf = scf.ROHF(mol)
    rohf.max_cycle, rohf.level_shift, rohf.verbose = 400, 0.3, 0
    rohf.kernel()
    if not rohf.converged:
        newton = rohf.newton()
        newton.verbose = 0
        newton.kernel(rohf.mo_coeff)
        if newton.converged:
            rohf.mo_coeff, rohf.mo_energy = newton.mo_coeff, newton.mo_energy
            rohf.mo_occ, rohf.e_tot, rohf.converged = (
                newton.mo_occ, newton.e_tot, True)
    print(f"  ROHF energy: {rohf.e_tot:.8f} Ha (converged: {rohf.converged})")
    if not rohf.converged:
        warnings.warn(
            "ROHF reference for CASCI/DMRG did not converge. Those energies "
            "are built on it and should be treated as unreliable.",
            RuntimeWarning,
        )
    return rohf, True


def _casci_object(mf, nel, norb, mo_guess=None):
    """CASCI in the requested space, with step 1's orbitals selected."""
    from pyscf import mcscf

    mc = mcscf.CASCI(mf, norb, nel)
    mc.verbose = 0
    if mo_guess is not None:
        mc.mo_coeff = mcscf.addons.sort_mo(mc, mf.mo_coeff, mo_guess, base=0)
    else:
        mc.mo_coeff = mf.mo_coeff
    return mc


def _active_hamiltonian(mc):
    """
    (h1e, g2e, ecore, nelec) for the active space, in the active orbitals.

    get_h1eff folds the doubly occupied core into a one-body potential and
    returns the core energy alongside, so the active-space problem is
    self-contained -- which is exactly the form block2 wants.
    """
    from pyscf import ao2mo

    h1e, ecore = mc.get_h1eff()
    g2e = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
    return np.asarray(h1e), np.asarray(g2e), float(ecore), tuple(mc.nelecas)


# ═════════════════════════════════════════════════════════════════════════
# FCI
# ═════════════════════════════════════════════════════════════════════════

def run_fci(mol, mf, settings, active=None):
    """
    Full configuration interaction.

    settings.fci_in_active_space=False (default) means the full basis --
    the true answer, and the only reference here that also validates the
    active-space choice. True routes to the active space instead, for
    molecules whose full space is out of reach.

    active : (nel, norb, mo_guess) from step 1, or None.

    Returns (energy, info). energy is None when the calculation was
    refused or failed; info always explains why.
    """
    from pyscf import fci

    print("\n-- FCI --")

    in_cas = bool(settings.fci_in_active_space)
    if in_cas and active is None:
        print("  Skipped -- fci_in_active_space is set but step 1 was not found.")
        return None, {"skipped": "no active space available"}

    if in_cas:
        nel, norb, mo_guess = active
        mc = _casci_object(mf, nel, norb, mo_guess)
        nelec = tuple(mc.nelecas)
        space = f"active space ({nel}e,{norb}o)"
    else:
        norb = int(mol.nao_nr())
        nelec = tuple(mol.nelec)
        space = f"full basis ({mol.nelectron}e,{norb}o)"

    ndet = fci_determinant_count(norb, nelec)
    print(f"  Space: {space} -> {ndet:,} determinants")

    if ndet > settings.fci_max_dets:
        # Refused BEFORE the integral transformation. FCI has no graceful
        # degradation: past the memory limit the job is killed, having
        # produced nothing, after however long the transformation took.
        msg = (f"FCI space has {ndet:,} determinants, over the "
               f"{settings.fci_max_dets:,} limit (ref.fci_max_dets). "
               f"Use DMRG for this system, or raise the limit if you are "
               f"certain the node has the memory.")
        print(f"  Refused -- {msg}")
        warnings.warn(msg, RuntimeWarning)
        return None, {"skipped": "too large", "n_determinants": ndet,
                      "limit": int(settings.fci_max_dets)}

    try:
        if in_cas:
            mc.fcisolver.conv_tol = 1e-12
            with _timer("FCI") as t:
                mc.kernel()
            energy, ci = float(mc.e_tot), mc.ci
            solver, nrb = mc.fcisolver, mc.ncas
        else:
            cisolver = fci.FCI(mf)
            cisolver.conv_tol = 1e-12
            cisolver.max_cycle = 300
            with _timer("FCI") as t:
                energy, ci = cisolver.kernel()
            energy = float(energy)
            solver, nrb = cisolver, norb

        s2, mult = _spin_square(solver, ci, nrb, nelec)
        print(f"  FCI energy: {energy:.8f} Ha   <S^2>: {s2:.4f} "
              f"(2S+1 = {mult:.2f})")
        return energy, {
            "space": space, "n_determinants": ndet, "in_active_space": in_cas,
            "norb": norb, "nelec": list(nelec), "s_squared": s2,
            "multiplicity": mult, "time": getattr(t, "dt", None),
        }
    except Exception as exc:
        warnings.warn(f"FCI failed: {exc}", RuntimeWarning)
        return None, {"error": str(exc), "n_determinants": ndet}


def _spin_square(solver, ci, norb, nelec):
    """<S^2> and multiplicity, tolerating solvers that cannot report them."""
    try:
        s2, mult = solver.spin_square(ci, norb, nelec)
        return float(s2), float(mult)
    except Exception:
        try:
            from pyscf import fci

            s2, mult = fci.spin_op.spin_square0(ci, norb, nelec)
            return float(s2), float(mult)
        except Exception:
            return float("nan"), float("nan")


# ═════════════════════════════════════════════════════════════════════════
# CASCI
# ═════════════════════════════════════════════════════════════════════════

def run_casci(mf, active, settings):
    """
    Exact diagonalisation inside step 1's active space, at fixed orbitals.

    This is the number every step-3 solver is approximating. Comparing a
    quantum solver against CASSCF instead mixes solver error with the
    difference between optimised and step-1 orbitals; comparing against
    this isolates the solver.

    active : (nel, norb, mo_guess) from step 1.
    """
    print("\n-- CASCI --")
    if active is None:
        print("  Skipped -- no active space. Run step 1 first.")
        return None, {"skipped": "no active space available"}

    nel, norb, mo_guess = active
    ndet = None
    try:
        mc = _casci_object(mf, nel, norb, mo_guess)
        mc.fcisolver.conv_tol = 1e-12
        if settings.casci_nroots > 1:
            mc.fcisolver.nroots = int(settings.casci_nroots)
        ndet = fci_determinant_count(norb, mc.nelecas)
        print(f"  Space: ({nel}e,{norb}o) -> {ndet:,} determinants"
              + (f", {settings.casci_nroots} roots"
                 if settings.casci_nroots > 1 else ""))

        with _timer(f"CASCI({nel}e,{norb}o)") as t:
            mc.kernel()

        # With nroots > 1, e_tot is an array and the ground state is [0].
        energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))
        ci0 = mc.ci[0] if settings.casci_nroots > 1 else mc.ci
        s2, mult = _spin_square(mc.fcisolver, ci0, mc.ncas, mc.nelecas)

        print(f"  CASCI energy: {energies[0]:.8f} Ha   "
              f"CI energy: {float(np.atleast_1d(mc.e_cas)[0]):.8f} Ha   "
              f"<S^2>: {s2:.4f}")
        if len(energies) > 1:
            gaps = (energies[1:] - energies[0]) * 27.211386245988
            print("  Excited roots (eV above ground): "
                  + ", ".join(f"{g:.3f}" for g in gaps))

        return float(energies[0]), {
            "nel": nel, "norb": norb, "n_determinants": ndet,
            "e_cas": float(np.atleast_1d(mc.e_cas)[0]),
            "roots": energies.tolist(), "s_squared": s2, "multiplicity": mult,
            "time": getattr(t, "dt", None),
        }
    except Exception as exc:
        warnings.warn(f"CASCI failed: {exc}", RuntimeWarning)
        return None, {"error": str(exc), "n_determinants": ndet}


# ═════════════════════════════════════════════════════════════════════════
# DMRG (block2)
# ═════════════════════════════════════════════════════════════════════════

def run_dmrg(mol, mf, active, settings, scratch_root):
    """
    DMRG via block2's pyblock2 driver.

    Runs one stage per bond dimension, each restarted from the previous
    MPS, and optionally extrapolates the energy to zero discarded weight.

    The integrals come from a PySCF CASCI object rather than block2's own
    PySCF adapters, so the active space is bit-identical to the one every
    other stage of the pipeline uses.
    """
    print("\n-- DMRG --")

    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        msg = ("DMRG requested but block2 is not importable. Install it with "
               "`pip install block2` (GPL-3.0 -- fine to run, do not bundle; "
               "see docs/licensing.md).")
        print(f"  Skipped -- {msg}")
        warnings.warn(msg, RuntimeWarning)
        return None, {"skipped": "block2 not installed"}

    if settings.dmrg_in_active_space and active is None:
        print("  Skipped -- dmrg_in_active_space is set but step 1 was not "
              "found. Run step 1, or set dmrg_in_active_space=False.")
        return None, {"skipped": "no active space available"}

    if settings.dmrg_in_active_space:
        nel, norb, mo_guess = active
    else:
        nel, norb, mo_guess = int(mol.nelectron), int(mol.nao_nr()), None

    scratch = settings.dmrg_scratch or os.path.join(scratch_root, "dmrg_scratch")
    os.makedirs(scratch, exist_ok=True)

    try:
        mc = _casci_object(mf, nel, norb, mo_guess)
        h1e, g2e, ecore, nelecas = _active_hamiltonian(mc)
        n_elec, spin = int(sum(nelecas)), int(nelecas[0] - nelecas[1])

        dims = [int(m) for m in settings.dmrg_bond_dims]
        noises = [float(x) for x in settings.dmrg_noises]
        thrds = [float(x) for x in settings.dmrg_thrds]
        sweeps = int(settings.dmrg_sweeps_per_stage)
        reorder = None if settings.dmrg_reorder == "none" else settings.dmrg_reorder

        print(f"  Space: ({n_elec}e,{norb}o)   2S = {spin}")
        print(f"  Bond dimensions: {dims}   {sweeps} sweeps each   "
              f"reorder: {settings.dmrg_reorder}")
        print(f"  Scratch: {scratch}")

        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=(settings.dmrg_threads or None),
        )
        driver.initialize_system(n_sites=norb, n_elec=n_elec, spin=spin)
        mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore,
                                reorder=reorder, iprint=0)
        ket = driver.get_random_mps(tag="KET", bond_dim=dims[0], nroots=1)

        stages = []
        with _timer("DMRG") as t:
            for m, noise, thrd in zip(dims, noises, thrds):
                energy = driver.dmrg(
                    mpo, ket, n_sweeps=sweeps,
                    bond_dims=[m] * sweeps,
                    noises=[noise] * sweeps,
                    thrds=[thrd] * sweeps,
                    tol=float(settings.dmrg_tol),
                    iprint=0,
                )
                dw = _last_discarded_weight(driver)
                stages.append({"bond_dim": m, "energy": float(energy),
                               "discarded_weight": dw, "noise": noise})
                dw_txt = "n/a" if dw is None else f"{dw:.3e}"
                print(f"    M = {m:>5}   E = {float(energy):.8f} Ha   "
                      f"discarded weight = {dw_txt}")

        energy = stages[-1]["energy"]
        info = {
            "nel": n_elec, "norb": norb, "spin": spin,
            "in_active_space": bool(settings.dmrg_in_active_space),
            "stages": stages, "reorder": settings.dmrg_reorder,
            "max_bond_dim": dims[-1], "time": getattr(t, "dt", None),
        }

        if settings.dmrg_extrapolate:
            e_ext, err, note = _extrapolate(stages)
            info["extrapolated_energy"] = e_ext
            info["truncation_error_estimate"] = err
            info["extrapolation_note"] = note
            if e_ext is not None:
                print(f"  Extrapolated to zero discarded weight: "
                      f"{e_ext:.8f} Ha   (truncation error ~ {err:.2e} Ha, "
                      f"{abs(err) * 627.5094740631:.3f} kcal/mol)")
                print(f"  Quote the extrapolated value; the gap above is the "
                      f"honest error bar.")
            else:
                print(f"  Extrapolation skipped -- {note}")

        print(f"  DMRG energy (M = {dims[-1]}): {energy:.8f} Ha")
        return float(energy), info

    except Exception as exc:
        warnings.warn(f"DMRG failed: {exc}", RuntimeWarning)
        return None, {"error": str(exc)}
    finally:
        if not settings.dmrg_keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _last_discarded_weight(driver):
    """
    Largest discarded weight of the final sweep.

    block2 has moved this attribute around between releases, so every
    plausible location is tried and a miss degrades to None (extrapolation
    is then skipped with a note) rather than killing a finished run.
    """
    for owner in (getattr(driver, "_dmrg", None), driver):
        if owner is None:
            continue
        for attr in ("discarded_weights", "sweep_discarded_weights"):
            values = getattr(owner, attr, None)
            if values is None:
                continue
            try:
                flat = [float(v) for v in np.ravel(np.asarray(values, dtype=float))
                        if np.isfinite(v)]
            except (TypeError, ValueError):
                continue
            if flat:
                return max(flat)
    return None


def _extrapolate(stages):
    """
    Linear fit of E against discarded weight, evaluated at zero.

    The standard way to quote a DMRG energy. Returns
    (extrapolated_energy, estimated_truncation_error, note); the energy is
    None when the fit is not defensible, and the note says why.
    """
    points = [(s["discarded_weight"], s["energy"]) for s in stages
              if s["discarded_weight"] is not None and s["discarded_weight"] > 0]
    if len(points) < 3:
        return None, None, (
            f"needs 3 stages with a positive discarded weight, got "
            f"{len(points)} (this block2 build may not expose them)")

    dw = np.array([p[0] for p in points], dtype=float)
    e = np.array([p[1] for p in points], dtype=float)
    if np.ptp(dw) <= 0:
        return None, None, "all discarded weights identical -- nothing to fit"

    slope, intercept = np.polyfit(dw, e, 1)
    e_ext = float(intercept)
    err = float(e[-1] - e_ext)

    note = "linear fit of E against discarded weight"
    if slope <= 0:
        # E must fall as truncation is removed. A positive-going fit means
        # the stages are not converged and the number is not a DMRG limit.
        note = (f"WARNING: fitted slope {slope:.3e} <= 0. The energy is not "
                f"decreasing with bond dimension as it must -- the sweeps are "
                f"probably not converged. Treat the extrapolation as invalid "
                f"and rerun with more sweeps per stage.")
    return e_ext, err, note
