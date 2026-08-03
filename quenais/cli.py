"""
QuEnAIS command-line interface.

    quenais-run --molecule LiH --basis sto-3g --steps 0 1 2

Solver choices come from quenais.config's SOLVERS tuple rather than a
hardcoded list, so the CLI and Config.validate() can never disagree about
what is accepted.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

__all__ = ["run_pipeline", "build_parser", "build_config"]

STEP_NAMES = {
    0: "Classical",
    1: "Active space",
    2: "Embedding Hamiltonian",
    3: "Quantum solver",
    4: "Visualisation",
}


def build_parser():
    from quenais.config import SOLVERS
    from quenais.settings.qiskit_solver import ANSATZE, BACKENDS, MAPPINGS

    parser = argparse.ArgumentParser(
        prog="quenais-run",
        description="QuEnAIS quantum embedding pipeline",
    )
    parser.add_argument("--molecule", default="LiH")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--solver", default="sqd", choices=list(SOLVERS),
                        help="quantum solver family")
    parser.add_argument("--ansatz", default="lucj", choices=list(ANSATZE))
    parser.add_argument("--mapping", default="bk", choices=list(MAPPINGS))
    parser.add_argument("--backend", default="mps", choices=list(BACKENDS))
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument(
        "--xyz", default=None,
        help="path to an XYZ file. Overrides the built-in geometry table "
             "and any CIF of the same name.",
    )
    parser.add_argument(
        "--geometry", default=None,
        help="geometry inline, PySCF style: 'Li 0 0 0; H 0 0 1.5949'. "
             "Angstrom.",
    )
    parser.add_argument(
        "--force-active-space", nargs="+", type=int, default=None,
        help="explicit MO indices, bypassing ASF/DMRG. Transition-metal "
             "systems generally need this -- see docs/limitations.md",
    )
    parser.add_argument(
        "--dmet-reference", default="casci", choices=["casci", "mp2"],
        help="reference density for the Schmidt decomposition",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="0=classical 1=active-space 2=embedding "
                             "3=solver 4=visualise")
    parser.add_argument("--force", action="store_true",
                        help="recompute every step, ignoring caches")
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--no-quantum-scan", action="store_true")
    return parser


def build_config(args):
    from quenais.config import Config
    from quenais.settings import AsfSettings, DmetSettings, QiskitSolverSettings

    cfg = Config(
        molecule=args.molecule,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        quantum_solver=args.solver,
        project_dir=os.path.abspath(args.project_dir),
        geometry=args.geometry,
        xyz=args.xyz,
        asf=AsfSettings(force_active_space=args.force_active_space),
        dmet=DmetSettings(reference=args.dmet_reference),
        qiskit=QiskitSolverSettings(
            ansatz=args.ansatz,
            fermion_to_qubit=args.mapping,
            backend=args.backend,
            n_shots=args.shots,
        ),
    )
    return cfg.validate().make_dirs().load_geometry()


def run_step(step, cfg, args):
    """Run one stage. Step 3 goes through the solver dispatch."""
    if step == 3:
        from quenais.quantum import dispatch

        return dispatch(cfg, force=args.force)

    module_path = {
        0: "quenais.classical.runner",
        1: "quenais.active_space.finder",
        2: "quenais.embedding.hamiltonian",
        4: "quenais.visualization.plots",
    }[step]
    module = importlib.import_module(module_path)

    if step == 4:
        return module.main(cfg, force=args.force, no_scan=args.no_scan,
                           no_quantum_scan=args.no_quantum_scan)
    return module.main(cfg, force=args.force)


def run_pipeline(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    unknown = set(args.steps) - set(STEP_NAMES)
    if unknown:
        parser.error(f"unknown step(s): {sorted(unknown)}. "
                     f"Valid: {sorted(STEP_NAMES)}")

    cfg = build_config(args)

    for step in sorted(args.steps):
        print(f"\n{'='*60}")
        print(f"  Step {step}: {STEP_NAMES[step]}")
        print(f"{'='*60}")
        run_step(step, cfg, args)

    print("\n[QuEnAIS] Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
