"""
The LASCI / LASSCF macro loop. Pure NumPy; the fragment solver is injected.

One macro cycle:

  1. build every fragment Hamiltonian from the current orbitals and the
     current RDMs of all other fragments (energy.fragment_hamiltonians)
  2. solve all fragments together (solver.solve) -- together, because the
     quantum solver runs all fragment circuits as one job
  3. optionally repeat 1-2 until the fragments are mutually consistent
     (the "LASCI" loop; one pass for a sampling solver, where every pass
     costs a hardware job)
  4. evaluate the LAS energy and orbital gradient
  5. rotate the orbitals at fixed RDMs (optimizer.optimize_orbitals)

A solver is any object with

    stochastic : bool
    name       : str
    solve(hams: list[FragmentHamiltonian], cycle: int) -> list[FragmentResult]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quenais.las.energy import (
    LasSpace,
    diagonal_hessian,
    fragment_hamiltonians,
    las_energy,
    las_energy_and_gradient,
    nonredundant_mask,
)
from quenais.las.optimizer import optimize_orbitals

__all__ = ["FragmentResult", "LasOptions", "LasResult", "run_las", "guess_dm1s"]


@dataclass
class FragmentResult:
    dm1s: np.ndarray            # (2, n, n) in the fragment's LAS orbitals
    dm2: np.ndarray             # (n, n, n, n) spin summed
    energy: float = np.nan      # <H_K> as reported by the solver (info only)
    info: dict = field(default_factory=dict)


@dataclass
class LasOptions:
    max_macro: int = 50
    orbital_optimization: bool = True
    orb_max_iter: int = 30
    orb_gtol: float = 1e-6
    lasci_max_cycle: int = 50
    lasci_conv_tol: float = 1e-10
    conv_tol_energy: float = 1e-8
    conv_tol_grad: float = 1e-4
    #: stochastic solvers: converge on |dE| alone, this many cycles in a row
    n_consecutive: int = 2


@dataclass
class LasResult:
    energy: float
    converged: bool
    n_macro: int
    C: np.ndarray
    dm1s: list
    dm2: list
    grad_norm: float
    history: list


def guess_dm1s(space, nelec_sub):
    """Aufbau occupation of each fragment's own orbital order."""
    out = []
    for n, (na, nb) in zip(space.ncas_sub, nelec_sub):
        d = np.zeros((2, n, n))
        d[0, range(na), range(na)] = 1.0
        d[1, range(nb), range(nb)] = 1.0
        out.append(d)
    return out


def _check_inputs(space, nelec_sub, spin2s_sub):
    if len(nelec_sub) != space.nfrag or len(spin2s_sub) != space.nfrag:
        raise ValueError("nelec_sub / spin2s_sub must have one entry per fragment")
    for k, ((na, nb), n, s2) in enumerate(zip(nelec_sub, space.ncas_sub, spin2s_sub)):
        if not (0 <= na <= n and 0 <= nb <= n):
            raise ValueError(f"fragment {k}: nelec ({na},{nb}) does not fit {n} orbitals")
        if s2 < abs(na - nb) or (s2 - abs(na - nb)) % 2:
            raise ValueError(f"fragment {k}: 2S={s2} incompatible with ({na},{nb})")


def run_las(ints, C, space, nelec_sub, spin2s_sub, solver, options=None,
            dm1s_guess=None, log=print):
    """
    Run LASCI (options.orbital_optimization=False) or LASSCF.

    ints        integral provider: hcore, energy_nuc, get_jk(dms), ao2mo(c1..c4)
    C           (nao, nmo) orbitals ordered [core | frags | virtual]
    space       energy.LasSpace
    nelec_sub   [(na, nb), ...] per fragment
    spin2s_sub  [2S, ...] per fragment
    """
    opt = options or LasOptions()
    _check_inputs(space, nelec_sub, spin2s_sub)
    stochastic = bool(getattr(solver, "stochastic", False))
    n_inner = 1 if stochastic else max(1, opt.lasci_max_cycle)
    mask = nonredundant_mask(space)

    dm1s = [np.array(d, dtype=float) for d in
            (dm1s_guess if dm1s_guess is not None else guess_dm1s(space, nelec_sub))]
    dm2 = None
    history = []
    e_prev, n_ok, converged, gnorm = None, 0, False, np.inf
    energy = np.nan
    macro = 0

    for macro in range(opt.max_macro):
        # ── fragment solves (LASCI) ───────────────────────────────────────
        e_inner = None
        for inner in range(n_inner):
            hams = fragment_hamiltonians(ints, C, space, dm1s, nelec_sub,
                                         spin2s_sub, dm2=dm2)
            results = solver.solve(hams, cycle=macro)
            dm1s = [np.asarray(r.dm1s) for r in results]
            dm2 = [np.asarray(r.dm2) for r in results]
            e_new = las_energy(ints, C, space, dm1s, dm2)
            if e_inner is not None and abs(e_new - e_inner) < opt.lasci_conv_tol:
                e_inner = e_new
                break
            e_inner = e_new

        energy, gmat = las_energy_and_gradient(ints, C, space, dm1s, dm2)
        gnorm = float(np.linalg.norm(gmat[mask])) if mask.any() else 0.0
        de = None if e_prev is None else energy - e_prev
        entry = {
            "cycle": macro,
            "energy": energy,
            "delta_e": de,
            "grad_norm": gnorm,
            "lasci_iters": inner + 1,
            "fragments": [r.info for r in results],
            "fragment_energies": [float(r.energy) for r in results],
        }
        history.append(entry)
        log(f"  LAS cycle {macro:3d}  E = {energy:.10f}"
            + ("" if de is None else f"  dE = {de:+.2e}")
            + f"  |g_orb| = {gnorm:.2e}")

        # ── convergence ───────────────────────────────────────────────────
        if de is not None and abs(de) < opt.conv_tol_energy:
            n_ok += 1
        else:
            n_ok = 0
        if stochastic:
            converged = n_ok >= opt.n_consecutive
        elif not opt.orbital_optimization:
            converged = True          # LASCI: the inner loop already converged
        else:
            converged = (gnorm < opt.conv_tol_grad and de is not None
                         and abs(de) < opt.conv_tol_energy)
        if converged:
            break
        e_prev = energy

        # ── orbital step at fixed RDMs ────────────────────────────────────
        if opt.orbital_optimization and mask.any():
            fixed1, fixed2 = dm1s, dm2
            step = optimize_orbitals(
                lambda Cx: las_energy_and_gradient(ints, Cx, space, fixed1, fixed2),
                C, mask, diagonal_hessian(ints, C, space, dm1s),
                max_iter=opt.orb_max_iter, gtol=opt.orb_gtol, log=log,
            )
            entry["orbital_step"] = {
                "iters": step.n_iter, "e_after": step.energy,
                "grad_after": step.grad_norm,
            }
            C = step.C

    return LasResult(energy=float(energy), converged=converged, n_macro=macro + 1,
                     C=C, dm1s=dm1s, dm2=dm2, grad_norm=gnorm, history=history)
