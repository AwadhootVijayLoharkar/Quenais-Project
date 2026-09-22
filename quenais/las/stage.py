"""
Step 3 for the LAS solver family ("lasscf", "lassqd").

Reads step 1 (orbitals and active space), never step 2: LAS does its own
partitioning, so the DMET embedding is not an input. Writes
results/step3_<solver>.pkl, so an exact LASSCF reference and a LASSQD run
can sit side by side for step 4 to compare.

    quenais-run --molecule ScF --basis sto-3g \\
        --active-space-method avas --avas-ao-labels 'Sc 3d' 'Sc 4s' 'F 2p' \\
        --solver lasscf --las-fragments 'Sc:6:1,1' 'F:3:3,3' --steps 0 1 3

The fragments must add up to step 1's active space in orbitals and
electrons; that is checked before anything expensive starts.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

__all__ = ["main", "build_problem"]


def _load_step1(cfg):
    if not os.path.exists(cfg.step1_file):
        raise FileNotFoundError(
            f"LAS needs step 1 (active space): {cfg.step1_file} not found. "
            f"Run --steps 1 first (AVAS is recommended: --active-space-method avas)."
        )
    if not cfg.cached_result_is_current(cfg.step1_file):
        raise RuntimeError(f"{cfg.step1_file} belongs to a different molecule or "
                           f"basis; rerun step 1 with --force")
    with open(cfg.step1_file, "rb") as fh:
        return pickle.load(fh)


def build_problem(cfg, mol, step1, log=print):
    """
    Everything run_las needs, from the step 1 pickle:
    (C ordered [core|frags|virt], LasSpace, nelec_sub, spin2s_sub, specs,
     localisation singular values, initial fragment 1-RDMs).
    """
    from quenais.las.energy import LasSpace
    from quenais.las.fragments import (
        check_disjoint,
        fragment_ao_indices,
        localize_active,
        parse_fragment_spec,
        resolve_atoms,
        split_core_active_virtual,
    )

    specs = [parse_fragment_spec(s) for s in cfg.las.fragments]
    atom_lists = [resolve_atoms(s, cfg.atom_syms) for s in specs]
    check_disjoint(atom_lists, [s.label for s in specs])

    mo = np.asarray(step1["mo_coeff"])
    nel = int(step1["nel"])
    ncas = len(step1["mo_list"])
    norbs = [s.norb for s in specs]
    nelec_sub = [s.nelec for s in specs]
    if sum(norbs) != ncas:
        raise ValueError(f"fragments hold {sum(norbs)} orbitals but step 1's active "
                         f"space has {ncas}: {step1['mo_list']}")
    if sum(na + nb for na, nb in nelec_sub) != nel:
        raise ValueError(f"fragments hold {sum(na + nb for na, nb in nelec_sub)} "
                         f"electrons but step 1's active space has {nel}")
    if sum(na - nb for na, nb in nelec_sub) != cfg.spin:
        raise ValueError(f"fragment M_S values add up to 2M_S = "
                         f"{sum(na - nb for na, nb in nelec_sub)}, but the molecule "
                         f"has spin={cfg.spin}")
    method = step1.get("selection_method")
    if method == "asf":
        log("  WARNING: step 1 used ASF; its orbitals may not split cleanly by "
            "atom. AVAS (--active-space-method avas) is the intended input.")

    core, act, virt = split_core_active_virtual(mo.shape[1], step1["mo_list"],
                                                mol.nelectron, nel)
    frag_aos = [fragment_ao_indices(mol, a) for a in atom_lists]
    blocks, svals = localize_active(mol, mo[:, act], frag_aos, norbs)
    C = np.hstack([mo[:, core], *blocks, mo[:, virt]])
    space = LasSpace(ncore=len(core), ncas_sub=tuple(norbs), nmo=C.shape[1])

    for spec, atoms, sv in zip(specs, atom_lists, svals):
        log(f"  fragment {spec.label:<10} atoms {atoms}  ({spec.norb}o, "
            f"{spec.nelec[0]}a+{spec.nelec[1]}b, 2S={spec.spin2s})  "
            f"localisation: min singular value {sv.min():.3f}")
        if sv.min() < 0.5:
            log(f"  WARNING: fragment {spec.label} orbitals are poorly localised "
                f"(singular value {sv.min():.2f} < 0.5); check the atom "
                f"assignment and the step 1 labels")

    dm1s0 = _initial_dm1s(step1, blocks, mol, nelec_sub)
    return C, space, nelec_sub, [s.spin2s for s in specs], specs, svals, dm1s0


def _initial_dm1s(step1, blocks, mol, nelec_sub):
    """Step 1's UHF density projected onto each fragment, rescaled to N_K."""
    s = mol.intor_symmetric("int1e_ovlp")
    cu = np.asarray(step1["mo_coeff_uhf"])
    occ = np.asarray(step1["mo_occ"])
    out = []
    for c, (na, nb) in zip(blocks, nelec_sub):
        d = []
        for spin, n_el in ((0, na), (1, nb)):
            dao = (cu[spin] * occ[spin]) @ cu[spin].T
            dk = c.T @ s @ dao @ s @ c
            tr = np.trace(dk)
            d.append(dk * (n_el / tr) if tr > 1e-8 else dk)
        out.append(np.stack(d))
    return out


def main(cfg, force=False):
    from pyscf import gto

    from quenais.las.driver import LasOptions, run_las
    from quenais.las.integrals import PyscfIntegrals

    solver_name = cfg.quantum_solver
    out_path = cfg.las_result_file
    os.makedirs(cfg.results_dir, exist_ok=True)
    if os.path.exists(out_path) and not force and cfg.cached_result_is_current(out_path):
        print(f"[Step 3] Using cached result: {out_path}")
        with open(out_path, "rb") as fh:
            return pickle.load(fh)

    print(f"\n{'=' * 60}")
    print(f"[Step 3] {solver_name.upper()} -- {cfg.molecule}")
    print(f"{'=' * 60}")

    step1 = _load_step1(cfg)
    mol = gto.M(atom=cfg.geometry, basis=cfg.basis, charge=cfg.charge,
                spin=cfg.spin, verbose=0)
    C, space, nelec_sub, spin2s_sub, specs, svals, dm1s0 = build_problem(cfg, mol, step1)
    ints = PyscfIntegrals(mol, density_fit=cfg.las.density_fit)

    s = cfg.las
    if solver_name == "lassqd":
        from quenais.las.sqd_solver import LassqdFragmentSolver

        solver = LassqdFragmentSolver(s)
        print(f"  SQD: K={s.sqd_batches} batches x d={s.sqd_samples_per_batch}, "
              f"{s.sqd_iterations} iterations, carryover eps={s.carryover_eps:g}, "
              f"{s.shots} shots on {s.backend}")
    else:
        from quenais.las.fci_solver import FCIFragmentSolver

        solver = FCIFragmentSolver(spin_penalty=s.fci_spin_penalty)

    opts = LasOptions(
        max_macro=s.max_macro,
        orbital_optimization=s.orbital_optimization,
        orb_max_iter=s.orb_max_iter,
        orb_gtol=s.orb_gtol,
        lasci_max_cycle=s.lasci_max_cycle,
        conv_tol_energy=s.energy_tol(solver_name),
        conv_tol_grad=s.conv_tol_grad,
        n_consecutive=s.n_consecutive,
    )
    res = run_las(ints, C, space, nelec_sub, spin2s_sub, solver, opts,
                  dm1s_guess=dm1s0)

    status = "converged" if res.converged else "NOT converged"
    print(f"\n  {solver_name.upper()} {status} after {res.n_macro} cycles: "
          f"E = {res.energy:.10f} Ha  |g_orb| = {res.grad_norm:.2e}")

    output = {
        "solver": solver_name,
        "energy": res.energy,
        "converged": res.converged,
        "n_macro": res.n_macro,
        "grad_norm": res.grad_norm,
        "history": res.history,
        "fragments": [s_.__dict__ | {"label": s_.label} for s_ in specs],
        "localisation_singular_values": [v.tolist() for v in svals],
        "ncore": space.ncore,
        "ncas_sub": list(space.ncas_sub),
        "mo_coeff": res.C,
        "dm1s": res.dm1s,
        "dm2": res.dm2,
        "settings": dict(cfg.las.__dict__),
        "mol_info": {"molecule": cfg.molecule, "basis": cfg.basis,
                     "n_electrons": mol.nelectron, "n_ao": mol.nao_nr()},
        "uhf_energy": step1.get("uhf_energy"),
        "reproducibility": "stochastic" if solver_name == "lassqd"
        else "optimizer-dependent",
        "provenance": cfg.provenance(),
    }
    with open(out_path, "wb") as fh:
        pickle.dump(output, fh)
    print(f"[Step 3] Saved -> {out_path}")
    return output
