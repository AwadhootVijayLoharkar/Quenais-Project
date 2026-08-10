#!/usr/bin/env python
"""
Stage 0 driver -- Experiments A and B on LiH and ScH.

Goes in: tools/run_stage0.py

Usage, from the repo root with quenais-env active:

    python tools/run_stage0.py                 # LiH then ScH
    python tools/run_stage0.py --system LiH    # one system
    python tools/run_stage0.py --threads 24

Writes stage0_<system>_curve.csv and stage0_<system>_summary.json beside the
golden data. Reads reference energies from tests/regression/reference_values.py
rather than hard-coding them.

Expected wall time: LiH seconds, ScH minutes. If ScH takes hours something is
wrong -- most likely eigsh is being handed a bad starting vector, or threads
are not set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GOLDEN = REPO / "tests" / "regression" / "golden"


def load_reference(system):
    from tests.regression.reference_values import SYSTEMS, Tier

    ref = SYSTEMS[system]
    energy, tier = ref["energies"]["DMET_CASCI"]
    if tier is not Tier.DETERMINISTIC:
        raise RuntimeError(
            f"{system} DMET_CASCI is tiered {tier}, not DETERMINISTIC. "
            f"It cannot be used as an exact target."
        )
    return energy, ref


def run_system(system, threads=1, verbose=True):
    from quenais.quantum.gqe_adapter import load_from_dmet_pickle
    import det_analysis as da          # quenais.quantum.det_analysis once installed

    pickle_path = GOLDEN / system / "step2_hamiltonian.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(pickle_path)

    e_ref, ref = load_reference(system)

    print(f"\n{'='*72}\n{system}\n{'='*72}")
    mol = load_from_dmet_pickle(str(pickle_path), num_threads=threads)

    # Guard the electron-count bug documented in reference_values: derived from
    # the active-space count these come out wrong whenever bath orbitals exist.
    assert mol.nelec == (ref["structure"]["n_alpha"], ref["structure"]["n_beta"]), (
        f"nelec {mol.nelec} does not match the validated structure "
        f"{(ref['structure']['n_alpha'], ref['structure']['n_beta'])}"
    )
    assert mol.norb == ref["structure"]["n_emb"]

    print("\n-- gates --")
    da.run_gates(mol, e_ref)

    print("\n-- Experiment A: weight capture --")
    _, flat, space = da.casci_vector(mol)
    order, cum = da.weight_curve(flat)
    need = da.n_for_weight(cum)
    for target, n in need.items():
        print(f"  weight >= {target:<9} needs {n:>8d} determinants "
              f"({100*n/space.ndet:6.3f}% of {space.ndet})")

    print("\n-- Experiment B: oracle bound --")
    # GQE's actual counts from the golden log, so the comparison lands exactly
    # on the measured baseline rather than near it.
    extra = {"ScH": (46, 87, 423, 2430, 2439), "LiH": (32, 34, 36), "N2": ()}
    rows = da.oracle_curve(mol, extra_points=extra.get(system, ()))

    n_chem = da.n_for_accuracy(rows)
    summary = {
        "system": system,
        "n_emb": space.norb,
        "nelec": list(space.nelec),
        "ndet": space.ndet,
        "e_casci": e_ref,
        "weight_targets": {str(k): v for k, v in need.items()},
        "n_chem": n_chem,
        "n_chem_fraction": (n_chem / space.ndet) if n_chem else None,
    }

    if n_chem:
        print(f"\n  N_chem (<= 1.6 mHa) = {n_chem} determinants "
              f"({100*n_chem/space.ndet:.4f}% of the space)")
    else:
        print("\n  N_chem: not reached on this grid")

    # Headroom against the GQE baseline, for ScH only.
    if system == "ScH":
        gqe_best_mha = 20.9      # best of three STOCHASTIC runs; see reference_values
        at_2439 = next((r for r in rows if r["n_det"] == 2439), None)
        if at_2439:
            summary["oracle_at_2439_mha"] = at_2439["err_projected_mha"]
            summary["headroom_mha"] = gqe_best_mha - at_2439["err_projected_mha"]
            print(f"  oracle at N=2439    = {at_2439['err_projected_mha']:.4f} mHa")
            print(f"  GQE best observed   = {gqe_best_mha:.4f} mHa "
                  f"(spread 20.9-30.7, STOCHASTIC)")
            print(f"  headroom            = {summary['headroom_mha']:.4f} mHa")

    out_csv = GOLDEN / system / f"stage0_{system}_curve.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_json = GOLDEN / system / f"stage0_{system}_summary.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  wrote {out_csv}")
    print(f"  wrote {out_json}")
    return summary


def main(cfg=None, force=False):
    p = argparse.ArgumentParser()
    p.add_argument("--system", default=None, choices=["LiH", "N2", "ScH"])
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    systems = [args.system] if args.system else ["LiH", "ScH"]
    out = {}
    for s in systems:
        out[s] = run_system(s, threads=args.threads)

    print(f"\n{'='*72}\nSCALING POINTS SO FAR\n{'='*72}")
    print(f"  {'system':<8} {'n_emb':>6} {'ndet':>12} {'N_chem':>10} {'fraction':>12}")
    for s, d in out.items():
        frac = f"{100*d['n_chem_fraction']:.4f}%" if d["n_chem_fraction"] else "--"
        print(f"  {s:<8} {d['n_emb']:>6} {d['ndet']:>12} "
              f"{str(d['n_chem']):>10} {frac:>12}")
    return out


if __name__ == "__main__":
    main()