#!/usr/bin/env python
"""
Where does LASSQD's sampled distribution come from? Diagnostic, read-only.

    python tools/diagnose_lassqd_circuit.py <benchmark run dir> [--shots 100000]

Takes a run directory written by tools/run_benchmark.py (it needs run.json
and results/step1_asf.pkl), rebuilds the FIRST macro cycle's fragment
Hamiltonians exactly as quenais/las/stage.py does, and for every fragment
prints the determinant weights from four sources, in the fragment's own
mean-field basis (the basis the circuit is built and measured in):

    exact      |c|^2 of the fragment FCI ground state
    LUCJ hh    |<det|LUCJ>|^2, ffsim statevector, heavy_hex pairs (default)
    LUCJ all   the same with all interaction pairs
    Aer hh     measured frequency from the real sampling path (glued
               circuit -> transpile -> AerSimulator), heavy_hex

then an ANSATZ SCAN: off-HF weight and energy error of the LUCJ state for
interaction pairs {heavy_hex, all} x n_reps {1, 2, 4} x linear-method
optimisation {off, on}. The production default is heavy_hex, n_reps=1, off.

Reading it:
    LUCJ << exact on the excited determinants -> the ansatz construction
    Aer  << LUCJ                               -> circuit / transpile / bit order
    all  >> hh                                 -> heavy-hex truncation

Nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _driver():
    path = Path(__file__).resolve().parent / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--top", type=int, default=8, help="determinants to list")
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 4],
                    help="LUCJ n_reps values to scan (default 1 2 4)")
    a = ap.parse_args(argv)

    import ffsim
    from pyscf import fci, gto
    from pyscf.fci import cistring

    from quenais.las import sqd_bits as bits
    from quenais.las import sqd_solver as sq
    from quenais.las import stage
    from quenais.las.energy import fragment_hamiltonians
    from quenais.las.integrals import PyscfIntegrals

    rdir = Path(a.run_dir).resolve()
    spec = json.loads((rdir / "run.json").read_text())
    params = spec["params"]
    if params["solver"] not in ("lassqd", "lasscf"):
        sys.exit(f"{rdir} is a {params['solver']} run; need a LAS run")
    cfg, _ = _driver()._cfg_for(params, rdir)

    step1 = stage._load_step1(cfg)
    mol = gto.M(atom=cfg.geometry, basis=cfg.basis, charge=cfg.charge,
                spin=cfg.spin, verbose=0)
    C, space, nelec_sub, spin2s_sub, specs, svals, dm1s0 = stage.build_problem(
        cfg, mol, step1, log=lambda *_: None)
    ints = PyscfIntegrals(mol, density_fit=cfg.las.density_fit)
    hams = fragment_hamiltonians(ints, C, space, dm1s0, nelec_sub, spin2s_sub)

    print(f"run {rdir.name}: {cfg.molecule}/{cfg.basis}, fragments "
          f"{cfg.las.fragments}, first macro cycle, {a.shots} shots")
    print("determinant labels: alpha|beta, printed orbital n-1 ... 0 "
          "(rightmost = lowest fragment MO)\n")

    for ham in hams:
        n, (na, nb) = ham.norb, ham.nelec
        print(f"=== fragment {ham.index}: {n} orbitals, {na}a + {nb}b ===")
        if na < nb:
            print("  na < nb: not handled by this diagnostic, skipped\n")
            continue
        h_avg = 0.5 * (ham.h1s[0] + ham.h1s[1])
        mf = sq._fragment_mean_field(h_avg, ham.eri, n, na, nb)
        u = np.asarray(mf.mo_coeff)
        print(f"  fragment mean field converged={mf.converged}  "
              f"orbital energies {np.round(mf.mo_energy, 4)}")
        h1_mo = np.einsum("pi,spq,qj->sij", u, ham.h1s, u)
        eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", ham.eri, u, u, u, u,
                           optimize=True)
        h_mo = 0.5 * (h1_mo[0] + h1_mo[1])

        amps = None
        try:
            t1, t2 = sq._uccsd_amplitudes(mf)
            amps = (t1, t2)
            print(f"  UCCSD: max|t1| = "
                  f"{max(np.abs(x).max() if np.size(x) else 0 for x in t1):.4f}  "
                  f"max|t2| (aa, ab, bb) = "
                  + ", ".join(f"{np.abs(x).max() if np.size(x) else 0:.4f}" for x in t2))
        except Exception as exc:
            print(f"  UCCSD FAILED: {exc!r}  -> LUCJ falls back to a random start")

        e_fci, civec = fci.direct_spin1.FCI().kernel(h_mo, eri_mo, n, (na, nb))
        exact = np.abs(np.asarray(civec)) ** 2

        strs_a = cistring.make_strings(range(n), na)
        strs_b = cistring.make_strings(range(n), nb)
        idx_a = {int(s): i for i, s in enumerate(strs_a)}
        idx_b = {int(s): i for i, s in enumerate(strs_b)}

        cols, msgs = {}, []
        rng = np.random.default_rng(0)
        hf = ffsim.hartree_fock_state(n, (na, nb))
        hop = ffsim.linear_operator(ffsim.MolecularHamiltonian(h_mo, eri_mo, 0.0),
                                    norb=n, nelec=(na, nb))
        hf_i, hf_j = idx_a[(1 << na) - 1], idx_b[(1 << nb) - 1]
        ops, scan = {}, []
        for pairs in ("heavy_hex", "all"):
            for n_reps in a.reps:
                for opt in (False, True):
                    s = dataclasses.replace(cfg.las, lucj_pairs=pairs,
                                            lucj_n_reps=n_reps, lucj_optimize=opt)
                    tag = f"{pairs[:3]} r={n_reps}{' opt' if opt else ''}"
                    try:
                        op = sq._lucj_operator(
                            h_mo, eri_mo, n, (na, nb), amps, s,
                            np.random.default_rng(0),
                            lambda m, t=tag: msgs.append(f"[{t}] {m.strip()}"))
                    except Exception as exc:
                        scan.append((tag, None, None, repr(exc)[:60]))
                        continue
                    vec = ffsim.apply_unitary(hf, op, norb=n, nelec=(na, nb))
                    pw = (np.abs(vec) ** 2).reshape(len(strs_a), len(strs_b))
                    e = float(np.real(np.vdot(vec, hop @ vec)))
                    scan.append((tag, 1 - pw[hf_i, hf_j], e - e_fci, ""))
                    if n_reps == cfg.las.lucj_n_reps and not opt:
                        ops[pairs] = op
                        cols[f"LUCJ {pairs[:3]}"] = pw
        for m in msgs:
            print(f"  {m}")

        s = dataclasses.replace(cfg.las, lucj_pairs="heavy_hex", shots=a.shots,
                                seed=7)
        if "heavy_hex" not in ops:
            sys.exit(f"--reps must include the production n_reps "
                     f"({cfg.las.lucj_n_reps})")
        counts = sq.sample_fragment_circuits(
            [sq._fragment_circuit(n, (na, nb), ops["heavy_hex"])], s)[0]
        mat, pr = bits.counts_to_matrix(counts, 2 * n)
        sa, sb = bits.matrix_to_strings(mat, n)
        aer = np.zeros_like(exact)
        lost = 0.0
        for x, y, p in zip(sa, sb, pr):
            if int(x) in idx_a and int(y) in idx_b:
                aer[idx_a[int(x)], idx_b[int(y)]] += p
            else:
                lost += p
        cols["Aer hh"] = aer

        order = np.argsort(-exact, axis=None)[: a.top]
        print(f"  fragment FCI energy (sampling basis, no const) {e_fci:.8f}")
        head = f"  {'det a|b':>{2 * n + 3}}  {'exact':>9}" + "".join(
            f"  {k:>9}" for k in cols)
        print(head)
        for flat in order:
            i, j = np.unravel_index(flat, exact.shape)
            lab = f"{int(strs_a[i]):0{n}b}|{int(strs_b[j]):0{n}b}"
            print(f"  {lab:>{2 * n + 3}}  {exact[i, j]:9.5f}" + "".join(
                f"  {cols[k][i, j]:9.5f}" for k in cols))
        print("  weight OFF the HF determinant:  exact "
              f"{1 - exact[hf_i, hf_j]:.5f}" + "".join(
                  f"   {k} {1 - cols[k][hf_i, hf_j]:.5f}" for k in cols))
        if lost:
            print(f"  Aer shots with wrong electron numbers: {lost:.5f}")
        print(f"\n  ansatz scan (exact off-HF weight {1 - exact[hf_i, hf_j]:.5f}):")
        print(f"  {'variant':<16} {'off-HF weight':>14} {'E - E_FCI (mHa)':>16}")
        for tag, w, de, err in scan:
            if w is None:
                print(f"  {tag:<16} {'failed':>14}   {err}")
            else:
                print(f"  {tag:<16} {w:14.5f} {1000 * de:16.3f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
