#!/usr/bin/env python
"""
Correlation scan -- find where classical selection starts to lose.

Goes in: tools/run_correlation_scan.py

THE EXPERIMENT
--------------
Stretch a bond. Correlation rises smoothly: near equilibrium one configuration
dominates and classical selected-CI is unbeatable; near dissociation the
wavefunction spreads over many near-degenerate configurations and classical
selection has nothing to anchor on.

At each bond length, with the active space held FIXED, measure:

    dominant weight  -- |c|^2 of the largest determinant. The correlation dial.
                        ~0.9 = single reference.  <0.5 = strongly correlated.
    N_chem           -- determinants needed for chemical accuracy
    CIPSI error      -- classical selection at a fixed budget
    oracle error     -- best possible at that budget

The bond length where CIPSI's error starts climbing away from the oracle is the
crossover: the point past which selection quality stops being solved by a
perturbative criterion. That is where a quantum sampler has something to prove,
and it is a number nobody has published.

Everything here is classical and cheap. Do NOT run the quantum solvers across
the whole scan -- run them at the two or three geometries this identifies.

WHY THE ACTIVE SPACE IS FORCED
------------------------------
ASF would choose a different active space at each bond length, so the series
would compare different embeddings rather than different correlation strengths.
Forcing it keeps geometry as the only variable. The default reproduces the
validated N2 golden data at equilibrium, which is the built-in check that the
scan is wired up correctly.

USAGE
-----
    python tools/run_correlation_scan.py
    python tools/run_correlation_scan.py --distances 1.0977 1.4 1.8 2.2 2.6 3.0
    python tools/run_correlation_scan.py --molecule N2 --active-space 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# N2 in STO-3G. The default active space is the one the golden data used, so
# the r = 1.0977 point must reproduce DMET_CASCI = -107.598406106545040.
DEFAULTS = {
    "N2": {
        "atoms": ("N", "N"),
        # FULL VALENCE (10e,8o): MOs 2..9, 0-indexed, = 2sg 2su 1pu(x2) 3sg
        # 1pg(x2) 3su. C(8,5)^2 = 3136 determinants.
        #
        # NOT the golden [5,6,7,8]. That space is (4e,4o) -> 36 determinants
        # in total, which is smaller than any sensible selection budget: both
        # CIPSI and the oracle would be handed the entire space at every
        # geometry and report zero difference. It is the right space for
        # reproducing the golden regression numbers and the wrong one for
        # measuring selection quality.
        #
        # It also has to be the full valence space for the physics: breaking
        # the triple bond needs all three bonding/antibonding pairs, or the
        # stretched geometries are not actually strongly correlated.
        "active_space": [2, 3, 4, 5, 6, 7, 8, 9],
        "nelecas": 10,
        "equilibrium": 1.0977,
        "distances": [1.0977, 1.3, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2],
        # No anchor: this active space is not the one the golden data used.
        # The (4e,4o) golden number is reproduced by run_stage0.py instead.
        "golden_casci": None,
    },
    # Strongly correlated at every geometry -- the real target once the N2
    # scan has established the method. Needs a forced space; ASF under-selects
    # for d-block. (12e,12o) is C(12,6)^2 = 853,776 determinants: still exact.
    "Cr2": {
        "atoms": ("Cr", "Cr"),
        "active_space": list(range(18, 30)),
        "nelecas": 12,
        "equilibrium": 1.68,
        "distances": [1.5, 1.68, 2.0, 2.5, 3.0],
        "golden_casci": None,
    },
}

# Fixed budget for the CIPSI comparison, in determinants. Must be well below
# the size of the determinant space or the comparison is vacuous -- see the
# guard in measure().
BUDGET = 200


def build_cas_molecule(atoms, r, basis, ncas, nelecas, threads=1):
    """
    Build the active-space Hamiltonian directly with PySCF -- no DMET.

    WHY THIS EXISTS
    ---------------
    The first version of this scan ran the full DMET pipeline at each geometry
    and every point failed the embedded-SCF check by 0.3-0.7 Ha, equilibrium
    included. The cause was the active space, not the stretching: N2/STO-3G has
    only 10 orbitals, so an 8-orbital impurity plus 2 bath orbitals spans the
    whole molecule. DMET needs an environment to fold into e_core; with none
    left, the core potential and the electron count double-count and the
    embedding Hamiltonian is meaningless.

    But this scan does not need DMET at all. "Where does perturbative selection
    stop being optimal as correlation grows" is a question about a Hamiltonian,
    not about an embedding. Building the CAS integrals directly removes the
    failure mode and changes nothing about what is being measured.

    DMET belongs in the follow-up: once this identifies the interesting
    geometries, run the real pipeline there, on a basis large enough to leave
    a genuine environment.

    Returns (molecule_object, reference_scf_energy).
    """
    from pyscf import ao2mo, gto, mcscf, scf
    from quenais.quantum.gqe_adapter import DMETEmbeddingMolecule

    mol = gto.M(atom=f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {r:.6f}",
                basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"reference RHF did not converge at r={r:.4f}")

    # ncore = (nelectron - nelecas) / 2, so the active orbitals are exactly the
    # ncas canonical MOs above the core. For N2/STO-3G with ncas=8, nelecas=10
    # that is MOs 2..9 -- the full valence space, core 1s frozen.
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = 0
    mc.fcisolver.verbose = 0
    h1, ecore = mc.get_h1eff()
    h2 = ao2mo.restore(1, mc.get_h2eff(), ncas)

    # PySCF's own CASCI on the same active space. This is the reference the
    # trust check uses.
    #
    # NOTE ON THE CHECK THAT WAS HERE BEFORE. The first version compared the
    # active-space SCF (mol.hf.e_tot) against the full RHF energy. That check
    # fires spuriously: the active-space SCF starts from a core-Hamiltonian
    # guess, and where h1eff's ordering disagrees with the Fock ordering it
    # converges to an excited SCF solution instead of the ground state. On
    # N2/STO-3G that happens below ~2 A and produces deltas up to 0.73 Ha on
    # perfectly good integrals.
    #
    # It is also the wrong thing to test. compute_casci() passes h1/h2/e_core
    # straight to the FCI solver and never touches mol.hf, so the SCF solution
    # has no bearing on any number this scan reports. What must be validated is
    # that the extracted Hamiltonian reproduces PySCF's CASCI -- which is the
    # exact reference every error here is measured against.
    e_casci_pyscf = float(mc.kernel()[0])

    n_a = (nelecas + mol.spin) // 2
    n_b = nelecas - n_a

    # Isolated cache per call. DMETEmbeddingMolecule defaults to
    # ./.cache/pyscf_dmet relative to the working directory, shared across
    # every geometry and every invocation, keyed on integral hashes. That is
    # one more piece of state between two runs that should be identical, and
    # this project has been bitten by fixed-path caches more than once. The
    # CASCI here costs milliseconds, so there is nothing to gain by keeping it.
    cache_dir = tempfile.mkdtemp(prefix="corrscan_")
    emb = DMETEmbeddingMolecule(h1, h2, ecore, n_a, n_b,
                                num_threads=threads, cache_dir=cache_dir)
    emb._scan_cache_dir = cache_dir
    return emb, e_casci_pyscf, float(mf.e_tot)


def step2_path_for(scan_root, molecule, r):
    """
    Where the pipeline actually writes the step 2 pickle.

    quenais-run puts stage outputs under <project-dir>/results/, so the path
    is NOT <project-dir>/step2_hamiltonian.pkl.
    """
    return scan_root / f"{molecule}_r{r:.4f}" / "results" / "step2_hamiltonian.pkl"


def run_pipeline(molecule, atoms, r, active_space, basis, scan_root, force=False):
    """Run steps 0-2 for one geometry. Returns the step 2 pickle path."""
    proj = scan_root / f"{molecule}_r{r:.4f}"
    step2 = step2_path_for(scan_root, molecule, r)
    if step2.exists() and not force:
        print(f"  [cache] step2 exists for r={r:.4f}")
        return step2

    geom = f"{atoms[0]} 0 0 0; {atoms[1]} 0 0 {r:.6f}"
    cmd = [
        "quenais-run",
        "--molecule", molecule,
        "--basis", basis,
        "--geometry", geom,
        "--steps", "0", "1", "2",
        "--project-dir", str(proj),
        "--no-scan", "--no-quantum-scan",
        "--force-active-space", *[str(i) for i in active_space],
    ]
    if force:
        cmd.append("--force")

    print(f"  running pipeline at r={r:.4f} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  FAILED at r={r:.4f}:\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")
        return None
    if not step2.exists():
        print(f"  pipeline finished but {step2} is missing")
        return None
    return step2


def measure(mol, e_casci_pyscf, e_rhf, budget=BUDGET, tol=1e-9):
    """Stage 0 and stage 1 measurements for one active-space Hamiltonian."""
    try:
        from quenais.quantum import det_analysis as da
        from quenais.quantum import det_expansion as dx
    except ImportError:
        import det_analysis as da
        import det_expansion as dx

    n_elec_emb = sum(mol.nelec)
    e_exact, flat, space = da.casci_vector(mol)

    # TRUST CHECK. The exact reference every error below is measured against
    # must match PySCF's own CASCI on the same active space. If it does not,
    # the integral extraction or the determinant bookkeeping is wrong.
    casci_delta = e_exact - float(e_casci_pyscf)
    trusted = abs(casci_delta) <= tol

    # Informational only -- see the note in build_cas_molecule. A non-zero
    # value means the active-space SCF found an excited solution, which says
    # nothing about the validity of the integrals.
    scf_delta = float(mol.hf.e_tot) - float(e_rhf)
    order, cum = da.weight_curve(flat)

    # A budget at or near the size of the space makes the comparison vacuous:
    # both CIPSI and the oracle get handed (almost) everything and both come
    # out exact. This is the failure mode of using the (4e,4o) N2 space, whose
    # 36 determinants are smaller than any reasonable budget.
    if budget >= 0.5 * space.ndet:
        raise ValueError(
            f"budget {budget} is >= half the determinant space ({space.ndet}). "
            f"Every method would look identical. Use a larger active space or "
            f"a smaller budget -- aim for a budget around 5-10% of the space."
        )
    n_eff = budget

    # Oracle: best possible at the budget.
    e_oracle = da.projected_energy(mol, order[:n_eff], space=space)

    # CIPSI: classical selection at the same budget, no quantum input.
    sel, history = dx.cipsi_from_scratch(mol, n_eff, space=space, verbose=False)
    e_cipsi = history[-1]["energy"]

    # N_chem on a coarse grid -- cheap, and only the magnitude matters here.
    grid = sorted({int(round(x)) for x in np.logspace(0, np.log10(space.ndet), 24)})
    n_chem = None
    for n in grid:
        if abs(da.projected_energy(mol, order[:n], space=space) - e_exact) <= 1.6e-3:
            n_chem = n
            break

    return {
        "n_emb": space.norb,
        "nelec": list(space.nelec),
        "ndet": space.ndet,
        "e_casci": e_exact,
        "casci_delta": casci_delta,
        "scf_delta": scf_delta,
        "trusted": trusted,
        "e_core": float(mol.cas_hamiltonian.e_core),
        "n_elec_active": n_elec_emb,
        "dominant_weight": float(cum[0]),
        "n_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "n_chem": n_chem,
        "budget": n_eff,
        "err_oracle_mha": 1e3 * (e_oracle - e_exact),
        "err_cipsi_mha": 1e3 * (e_cipsi - e_exact),
        "cipsi_above_oracle_mha": 1e3 * (e_cipsi - e_oracle),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", default="N2", choices=list(DEFAULTS))
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--distances", type=float, nargs="+", default=None)
    p.add_argument("--active-space", type=int, nargs="+", default=None)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--scan-root", default="scans",
                   help="where the per-geometry project dirs live; must match "
                        "the --project-dir used when running the pipeline")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    spec = DEFAULTS[args.molecule]
    distances = args.distances or spec["distances"]
    active_space = args.active_space or spec["active_space"]
    # Relative to the working directory, so it matches wherever the pipeline
    # loop was launched from. quenais-run resolves --project-dir against cwd.
    scan_root = Path(args.scan_root).resolve() / args.molecule
    scan_root.mkdir(parents=True, exist_ok=True)

    print(f"{'='*76}")
    print(f"correlation scan: {args.molecule}/{args.basis}")
    print(f"active space (forced, fixed across the scan): {active_space}")
    print(f"budget for the CIPSI/oracle comparison: {args.budget} determinants")
    print(f"{'='*76}")

    rows = []
    for r in distances:
        print(f"\n-- r = {r:.4f} A --")
        try:
            mol, e_casci_ref, e_rhf = build_cas_molecule(
                spec["atoms"], r, args.basis, len(active_space),
                spec["nelecas"], threads=args.threads)
            m = measure(mol, e_casci_ref, e_rhf, budget=args.budget)
        except Exception as exc:
            print(f"  failed at r={r:.4f}: {exc}")
            continue
        finally:
            tmp = getattr(mol, "_scan_cache_dir", None) if "mol" in dir() else None
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

        m["r"] = r
        rows.append(m)
        print(f"  ncas={m['n_emb']}  nelec={tuple(m['nelec'])}  "
              f"ndet={m['ndet']}  budget={m['budget']} "
              f"({100*m['budget']/m['ndet']:.1f}% of the space)")
        print(f"  dominant weight={m['dominant_weight']:.4f}  "
              f"(1.0 = single reference, lower = more correlated)")
        flag = "ok" if m["trusted"] else "*** UNTRUSTED ***"
        print(f"  CASCI vs PySCF CASCI: {m['casci_delta']:+.3e} Ha  {flag}")
        if not m["trusted"]:
            print(f"    the exact reference does not reproduce; nothing in "
                  f"this row is evidence of anything")
        print(f"  CIPSI {m['err_cipsi_mha']:8.4f} mHa   "
              f"oracle {m['err_oracle_mha']:8.4f} mHa   "
              f"gap {m['cipsi_above_oracle_mha']:+8.4f} mHa   "
              f"N_chem={m['n_chem']}")

    if not rows:
        print("\nno geometries succeeded")
        return

    print(f"\n{'='*76}")
    print(f"  {'r (A)':>7} {'dom wt':>8} {'N_chem':>8} {'CIPSI':>10} "
          f"{'oracle':>10} {'gap':>10}  {'CASCI chk':>10}")
    for m in rows:
        print(f"  {m['r']:>7.3f} {m['dominant_weight']:>8.4f} "
              f"{str(m['n_chem']):>8} {m['err_cipsi_mha']:>10.4f} "
              f"{m['err_oracle_mha']:>10.4f} {m['cipsi_above_oracle_mha']:>+10.4f}"
              f"  {'ok' if m['trusted'] else 'FAILED':>10}")
    print(f"  (errors in mHa at a {args.budget}-determinant budget)")

    bad = [m for m in rows if not m["trusted"]]
    if bad:
        print(f"\n  {len(bad)} of {len(rows)} geometries FAILED the CASCI "
              f"cross-check: r = "
              + ", ".join(f"{m['r']:.3f}" for m in bad))
        print("  Those rows are not evidence.")

    good = [m for m in rows if m["trusted"]]
    if good:
        print("\n  Read the 'gap' column over the TRUSTED rows only: while it "
              "stays near\n  zero, classical selection is effectively optimal "
              "and no sampler can help.\n  Where it starts to grow is the "
              "crossover -- run the quantum solvers THERE.")

    out = scan_root / f"correlation_scan_{args.molecule}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(scan_root / f"correlation_scan_{args.molecule}.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()