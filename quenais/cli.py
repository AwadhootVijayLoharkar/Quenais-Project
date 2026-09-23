"""
QuEnAIS command-line interface.

    quenais-run --molecule LiH --basis sto-3g --steps 0 1 2

Solver choices come from quenais.config's SOLVERS tuple rather than a
hardcoded list, so the CLI and Config.validate() can never disagree about
what is accepted. The CUDA-Q simulator targets come from
quenais.settings.gqe for the same reason.

THE GQE FLAGS ARE NOT DECORATION
--------------------------------
Until 0.3 this file built Config with asf=, dmet= and qiskit= but nothing
for gqe=, so every `quenais-run --solver gqe` invocation silently used
GqeSettings() defaults. There was no way to choose the simulator backend,
the number of epochs, or the circuit depth from the command line at all --
the only route was the Python API. On an A100 that meant a full training
run printing "backend : qpp-cpu" and simulating circuits on the CPU, while
the log's "GPU available: True, used: True" (Lightning, for the
transformer) made it look like the GPU was in play.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

__all__ = ["run_pipeline", "build_parser", "build_config", "build_asf_settings",
           "build_las_settings", "build_ref_settings"]

STEP_NAMES = {
    0: "Classical",
    1: "Active space",
    2: "Embedding Hamiltonian",
    3: "Quantum solver",
    4: "Visualisation",
}


def build_parser():
    from quenais.config import CLASSICAL_METHODS, SOLVERS
    from quenais.settings.asf import SELECTION_METHODS
    from quenais.settings.gqe import CUDAQ_SIMULATOR_TARGETS, DMET_POOL_SPECS
    from quenais.settings.las import LAS_BACKENDS, LUCJ_PAIRS
    from quenais.settings.qiskit_solver import ANSATZE, BACKENDS, MAPPINGS
    from quenais.settings.reference import DMRG_REORDERINGS

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
        help="explicit MO indices, bypassing every selector. Valid only at "
             "the geometry it was calibrated at -- MO indices reorder as "
             "bonds stretch. Prefer --active-space-method avas.",
    )

    # ── Active-space selection ───────────────────────────────────────────
    sel = parser.add_argument_group(
        "Active-space selection",
        "--force-active-space overrides all of these.",
    )
    sel.add_argument(
        "--active-space-method", default="asf",
        choices=list(SELECTION_METHODS),
        help="'asf' (default) is ASF/DMRG entanglement entropy and needs "
             "block2; it under-selects for the d-block. 'avas' projects "
             "onto named atomic valence orbitals and is the recommended "
             "route for transition metals. 'apc' is PySCF's ranked-orbital "
             "selector -- automatic, no AO labels. Neither PySCF selector "
             "needs block2.",
    )
    sel.add_argument(
        "--avas-ao-labels", nargs="+", default=None, metavar="LABEL",
        help="AO labels for --active-space-method avas. QUOTE EACH ONE: "
             "--avas-ao-labels 'Sc 3d' 'Sc 4s'. Unquoted, 'Sc 3d' becomes "
             "two arguments and a bare 'Sc' matches every Sc orbital "
             "including the core (rejected by AsfSettings.validate). "
             "Default: the valence shell of each element present.",
    )
    sel.add_argument(
        "--avas-threshold", type=float, default=None,
        help="AVAS projector eigenvalue cutoff (default 0.2). Weakly "
             "discriminating in a minimal basis -- scan it.",
    )
    sel.add_argument(
        "--apc-max-size", type=int, default=None,
        help="largest active space APC may return (default 8)",
    )
    parser.add_argument(
        "--dmet-reference", default="casci", choices=["casci", "mp2"],
        help="reference density for the Schmidt decomposition",
    )
    # Mirrors METHOD_TIERS in quenais.classical.runner. Hardcoded rather
    # than imported because that module pulls in PySCF, and building the
    # parser must stay cheap enough for --help to work anywhere.
    #
    # Config defaults to ["HF", "MP2"] and there was no way to change that
    # from the command line at all, so every quenais-run produced a
    # two-row results table while the README advertised six methods.
    #
    # Note "CCSD_T", not "CCSD(T)" -- parentheses would need quoting.
    parser.add_argument(
        "--classical-methods", nargs="+", default=None,
        choices=list(CLASSICAL_METHODS),
        help="step 0 reference methods (default: HF MP2). CASSCF, NEVPT2, "
             "CASCI and (by default) DMRG reuse step 1's active space IF "
             "step 1 has already run -- on a first pass they fall back to a "
             "guess and warn, or skip. For meaningful values, run steps 0 1 "
             "first, then re-run step 0 with --force. CASSCF/NEVPT2 are "
             "labelled 'optimizer-dependent' in results_summary.csv: not "
             "reproducible to tight tolerance across machines. "
             "CASCI/FCI/DMRG are the exact tier: CASCI is exact in step 1's "
             "active space and is the reference every step-3 solver is "
             "approximating, FCI is exact in the full basis, DMRG reaches "
             "what FCI cannot. Use these, not CCSD_T, to benchmark "
             "stretched bonds -- CCSD(T) is the method that breaks there.",
    )

    # ── Exact references (cfg.ref) ───────────────────────────────────────
    ref_group = parser.add_argument_group(
        "exact references",
        "FCI / CASCI / DMRG settings. Only read when the matching method is "
        "in --classical-methods.",
    )
    ref_group.add_argument(
        "--fci-max-dets", type=int, default=None,
        help="refuse a full-space FCI larger than this many determinants "
             "(default 5000000). The check runs before the integral "
             "transformation, so an impossible request costs a message "
             "rather than a killed job.",
    )
    ref_group.add_argument(
        "--fci-in-active-space", action="store_true",
        help="run FCI inside step 1's active space instead of the full "
             "basis. Makes it a synonym for CASCI; only useful when the "
             "full space is out of reach.",
    )
    ref_group.add_argument(
        "--casci-nroots", type=int, default=None,
        help="CASCI roots (default 1). >1 shows excited states, which is "
             "how you catch a curve where the solver tracked the wrong one.",
    )
    ref_group.add_argument(
        "--dmrg-bond-dims", type=int, nargs="+", default=None,
        help="DMRG bond dimensions, non-decreasing (default 250 500 1000). "
             "Each is a stage restarted from the previous MPS, and the list "
             "is also the extrapolation data -- three points minimum.",
    )
    ref_group.add_argument(
        "--dmrg-sweeps", type=int, default=None,
        help="sweeps per bond-dimension stage (default 8).",
    )
    ref_group.add_argument(
        "--dmrg-full-space", action="store_true",
        help="run DMRG in the full basis instead of step 1's active space. "
             "The active space is the default because that is what the "
             "step-3 solvers work in.",
    )
    ref_group.add_argument(
        "--dmrg-no-extrapolate", action="store_true",
        help="report only the largest-M energy, with no fit to zero "
             "discarded weight.",
    )
    ref_group.add_argument(
        "--dmrg-reorder", default=None, choices=list(DMRG_REORDERINGS),
        help="orbital ordering before the MPO is built (default fiedler). "
             "DMRG accuracy depends strongly on this.",
    )
    ref_group.add_argument(
        "--dmrg-threads", type=int, default=None,
        help="OpenMP threads for block2. 0 (default) leaves it to block2 "
             "and OMP_NUM_THREADS, which is right under a scheduler that "
             "already pinned cores.",
    )
    ref_group.add_argument(
        "--dmrg-scratch", default=None,
        help="scratch directory for the MPS (default: under results/). "
             "Point this at node-local fast storage -- block2 writes every "
             "renormalised operator to disk, and NFS turns a ten-minute run "
             "into an hour.",
    )
    ref_group.add_argument(
        "--dmrg-keep-scratch", action="store_true",
        help="keep block2's scratch files after the run (GB-sized).",
    )

    # ── GQE solver (--solver gqe) ────────────────────────────────────────
    gqe = parser.add_argument_group(
        "GQE solver",
        "Only used with --solver gqe. Anything left unset keeps the "
        "GqeSettings default.",
    )
    gqe.add_argument(
        "--cudaq-target", default=None, choices=list(CUDAQ_SIMULATOR_TARGETS),
        help="CUDA-Q circuit simulator. Default: $CUDAQ_DEFAULT_SIMULATOR, "
             "which install.sh sets from the GPU's compute capability "
             "('nvidia' needs cc>=8.0), else qpp-cpu. NOTE this selects the "
             "backend for CIRCUIT SIMULATION only -- the transformer runs "
             "on the GPU via Lightning either way, which is why a CPU "
             "simulator run still logs 'GPU available: True, used: True'.",
    )
    gqe.add_argument("--gqe-repo", default=None,
                     help="path to the gqe-for-qsci checkout. Default: "
                          "$GQE_QSCI_REPO_PATH")
    gqe.add_argument("--gqe-max-iters", type=int, default=None,
                     help="training epochs (default 120). Use 2 for a smoke "
                          "test.")
    gqe.add_argument("--gqe-ngates", type=int, default=None,
                     help="circuit depth (default 40). Larger embeddings "
                          "need more: on ScH, 10 stalled at HF, 20 "
                          "plateaued, 40 recovered 60%% of the correlation "
                          "energy.")
    gqe.add_argument("--gqe-num-samples", type=int, default=None,
                     help="samples per epoch (default 100). batch_size is "
                          "set to match, which GqeSettings.validate() "
                          "requires.")
    gqe.add_argument("--gqe-pool", default=None, choices=list(DMET_POOL_SPECS),
                     help="operator pool (default dmet_excitation). "
                          "dmet_pauli_evolution cannot conserve particle "
                          "number -- see docs/gqe_integration.md.")
    gqe.add_argument("--gqe-qsci-max-dim", type=int, default=None,
                     help="QSCI subspace cap (default 10000). The upstream "
                          "default of 2000 became the binding constraint "
                          "on ScH.")
    gqe.add_argument("--gqe-seed", type=int, default=None,
                     help="trainer seed. The external repo's "
                          "configs/trainer/default.yaml pins seed=32, so runs "
                          "are otherwise bit-identical and repeats are not "
                          "independent samples.")

    # ── LAS solvers (--solver lasscf | lassqd) ────────────────────────────
    las = parser.add_argument_group(
        "LAS solvers",
        "Only used with --solver lasscf or lassqd. They read step 1 (use "
        "--active-space-method avas) and skip step 2. Anything left unset "
        "keeps the LasSettings default.",
    )
    las.add_argument(
        "--las-fragments", nargs="+", default=None, metavar="SPEC",
        help="one per fragment, ATOMS:NORB:NA,NB[:2S]; ATOMS are '+'-joined "
             "atom indices or element symbols. QUOTE each: "
             "--las-fragments 'Sc:6:1,1' 'F:3:3,3'. Orbitals and electrons "
             "must add up to step 1's active space.",
    )
    las.add_argument("--las-max-cycles", type=int, default=None,
                     help="LAS macro cycles (default 50)")
    las.add_argument("--las-no-orbital-opt", action="store_true",
                     help="LASCI: keep the localised step-1 orbitals fixed")
    las.add_argument("--las-conv-tol", type=float, default=None,
                     help="energy convergence (default 1e-8 lasscf, 1e-5 lassqd)")
    las.add_argument("--las-density-fit", action="store_true",
                     help="density-fitted integrals (large basis sets)")
    las.add_argument("--lassqd-backend", default=None, choices=list(LAS_BACKENDS),
                     help="sampler (default aer_mps)")
    las.add_argument("--lassqd-ibm-backend", default=None,
                     help="IBM device name for --lassqd-backend ibm")
    las.add_argument("--lassqd-shots", type=int, default=None,
                     help="shots per macro cycle, all fragments in one job "
                          "(default 100000)")
    las.add_argument("--lassqd-batches", type=int, default=None,
                     help="SQD batches K (default 15)")
    las.add_argument("--lassqd-samples-per-batch", type=int, default=None,
                     help="SQD samples per batch d (default 170)")
    las.add_argument("--lassqd-iterations", type=int, default=None,
                     help="configuration-recovery iterations (default 6)")
    las.add_argument("--lassqd-carryover-eps", type=float, default=None,
                     help="carryover threshold on |c| (default 1e-5; 0 "
                          "disables carryover = conventional SQD)")
    las.add_argument("--lassqd-nroots", type=int, default=None,
                     help="Davidson roots per subspace (default 3)")
    las.add_argument("--lassqd-lucj-pairs", default=None, choices=list(LUCJ_PAIRS),
                     help="LUCJ interaction pairs (default heavy_hex)")
    las.add_argument("--lassqd-lucj-optimize", action="store_true",
                     help="refine LUCJ parameters with ffsim's linear method")
    las.add_argument("--lassqd-seed", type=int, default=None)

    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="0=classical 1=active-space 2=embedding "
                             "3=solver 4=visualise")
    parser.add_argument("--force", action="store_true",
                        help="recompute every step, ignoring caches")
    parser.add_argument("--no-scan", action="store_true")
    parser.add_argument("--no-quantum-scan", action="store_true")
    return parser


def build_gqe_settings(args):
    """
    GqeSettings from the --gqe-* flags, omitting anything left unset so the
    dataclass default applies.

    num_samples and batch_size move together: GqeSettings.validate()
    rejects them differing, because the trainer is online.
    """
    from quenais.settings import GqeSettings

    kwargs = {}
    if args.cudaq_target is not None:
        kwargs["cudaq_target"] = args.cudaq_target
    if args.gqe_repo is not None:
        kwargs["repo_path"] = args.gqe_repo
    if args.gqe_max_iters is not None:
        kwargs["max_iters"] = args.gqe_max_iters
    if args.gqe_ngates is not None:
        kwargs["ngates"] = args.gqe_ngates
    if args.gqe_num_samples is not None:
        kwargs["num_samples"] = args.gqe_num_samples
        kwargs["batch_size"] = args.gqe_num_samples
    if args.gqe_pool is not None:
        kwargs["operator_pool_spec"] = args.gqe_pool
    if args.gqe_qsci_max_dim is not None:
        kwargs["qsci_max_dim"] = args.gqe_qsci_max_dim
    if args.gqe_seed is not None:
        kwargs["seed"] = args.gqe_seed
    return GqeSettings(**kwargs)


def build_ref_settings(args):
    """ReferenceSettings from the --fci-*/--casci-*/--dmrg-* flags."""
    from quenais.settings import ReferenceSettings

    kwargs = {}
    simple = {
        "fci_max_dets": "fci_max_dets",
        "casci_nroots": "casci_nroots",
        "dmrg_sweeps": "dmrg_sweeps_per_stage",
        "dmrg_reorder": "dmrg_reorder",
        "dmrg_threads": "dmrg_threads",
        "dmrg_scratch": "dmrg_scratch",
    }
    for arg, field_name in simple.items():
        value = getattr(args, arg, None)
        if value is not None:
            kwargs[field_name] = value

    if getattr(args, "fci_in_active_space", False):
        kwargs["fci_in_active_space"] = True
    if getattr(args, "dmrg_full_space", False):
        kwargs["dmrg_in_active_space"] = False
    if getattr(args, "dmrg_no_extrapolate", False):
        kwargs["dmrg_extrapolate"] = False
    if getattr(args, "dmrg_keep_scratch", False):
        kwargs["dmrg_keep_scratch"] = True

    dims = getattr(args, "dmrg_bond_dims", None)
    if dims:
        # The noise and Davidson schedules are per stage, so changing the
        # number of bond dimensions without rebuilding them would fail
        # validation with a length mismatch the user did not cause. Noise
        # must end at exactly zero; the ramp below is the usual shape.
        n = len(dims)
        kwargs["dmrg_bond_dims"] = tuple(dims)
        if n == 1:
            noises = (0.0,)
        else:
            noises = tuple([1e-4] * (n - 2) + [1e-5, 0.0])
        kwargs["dmrg_noises"] = noises
        kwargs["dmrg_thrds"] = tuple(
            max(1e-12, 1e-8 / (10 ** i)) for i in range(n)
        )
        if n < 3:
            kwargs["dmrg_extrapolate"] = False

    return ReferenceSettings(**kwargs)


def build_las_settings(args):
    """LasSettings from the --las-*/--lassqd-* flags; unset flags keep defaults."""
    from quenais.settings import LasSettings

    kwargs = {}
    simple = {
        "las_fragments": "fragments",
        "las_max_cycles": "max_macro",
        "las_conv_tol": "conv_tol_energy",
        "lassqd_backend": "backend",
        "lassqd_ibm_backend": "ibm_backend_name",
        "lassqd_shots": "shots",
        "lassqd_batches": "sqd_batches",
        "lassqd_samples_per_batch": "sqd_samples_per_batch",
        "lassqd_iterations": "sqd_iterations",
        "lassqd_carryover_eps": "carryover_eps",
        "lassqd_nroots": "sqd_nroots",
        "lassqd_lucj_pairs": "lucj_pairs",
        "lassqd_seed": "seed",
    }
    for arg, field_name in simple.items():
        value = getattr(args, arg, None)
        if value is not None:
            kwargs[field_name] = value
    if getattr(args, "las_no_orbital_opt", False):
        kwargs["orbital_optimization"] = False
    if getattr(args, "las_density_fit", False):
        kwargs["density_fit"] = True
    if getattr(args, "lassqd_lucj_optimize", False):
        kwargs["lucj_optimize"] = True
    return LasSettings(**kwargs)


def build_asf_settings(args):
    """
    AsfSettings from the selection flags, omitting anything left unset so
    the dataclass default applies.

    Same pattern as build_gqe_settings: passing None explicitly would
    overwrite a real default with None for fields whose default is not None
    (avas_threshold, apc_max_size).
    """
    from quenais.settings import AsfSettings

    kwargs = {
        "force_active_space": args.force_active_space,
        "method": args.active_space_method,
    }
    if args.avas_ao_labels is not None:
        kwargs["avas_ao_labels"] = args.avas_ao_labels
    if args.avas_threshold is not None:
        kwargs["avas_threshold"] = args.avas_threshold
    if args.apc_max_size is not None:
        kwargs["apc_max_size"] = args.apc_max_size
    return AsfSettings(**kwargs)


def build_config(args):
    from quenais.config import Config
    from quenais.settings import DmetSettings, QiskitSolverSettings

    cfg = Config(
        molecule=args.molecule,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        quantum_solver=args.solver,
        project_dir=os.path.abspath(args.project_dir),
        geometry=args.geometry,
        xyz=args.xyz,
        # None lets Config apply its own ["HF", "MP2"] default.
        classical_methods=args.classical_methods,
        ref=build_ref_settings(args),
        asf=build_asf_settings(args),
        dmet=DmetSettings(reference=args.dmet_reference),
        qiskit=QiskitSolverSettings(
            ansatz=args.ansatz,
            fermion_to_qubit=args.mapping,
            backend=args.backend,
            n_shots=args.shots,
        ),
        gqe=build_gqe_settings(args),
        las=build_las_settings(args),
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