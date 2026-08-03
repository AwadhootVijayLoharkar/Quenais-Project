"""
Central configuration for the QuEnAIS pipeline.

Config carries molecule identity, paths and the solver choice. Everything
else lives in a settings group (quenais.settings) and is reached as
cfg.asf.*, cfg.dmet.*, cfg.qiskit.*, cfg.gqe.*, cfg.tiers.*.

Grouping is deliberate. The flat version of this configuration grew to
roughly 800 lines mixing paths, constants, CIF parsing, thresholds and
training hyperparameters. Adding a nested group costs one line here; adding
a flat field costs three, and it never stops.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from quenais.settings import (
    AsfSettings,
    DmetSettings,
    GqeSettings,
    QiskitSolverSettings,
    TierSettings,
)
from quenais.utils.cif_parser import load_geometry as _load_geometry

__all__ = ["Config", "SOLVERS", "QISKIT_SOLVERS", "GQE_SOLVERS", "SOLVER_ALIASES"]


# ─────────────────────────────────────────────────────────────────────────
# Solver registry -- the single source of truth.
#
# Config.validate(), the CLI's --solver choices and quenais.quantum.dispatch
# all read these. Three hand-maintained lists is how a solver ends up
# accepted in one place and rejected in another.
# ─────────────────────────────────────────────────────────────────────────

#: In-process Qiskit solvers.
QISKIT_SOLVERS = ("sqd", "skqd", "sqdrift")

#: Subprocess CUDA-Q solvers.
GQE_SOLVERS = ("gqe",)

#: Everything selectable via quantum_solver.
SOLVERS = QISKIT_SOLVERS + GQE_SOLVERS

#: Deprecated spellings, accepted with a warning. "gqe_qsci" was the name
#: used in the test_8 scripts.
SOLVER_ALIASES = {"gqe_qsci": "gqe"}


#: Explicit finite geometries, in Angstrom. Preferred over CIF lookup.
BUILTIN_GEOMETRIES = {
    "ScH": [("Sc", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7800))],
    "LiH": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))],
    "N2": [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
    "H2O": [
        ("O", (0.0, 0.0, 0.1173)),
        ("H", (0.0, 0.7572, -0.4692)),
        ("H", (0.0, -0.7572, -0.4692)),
    ],
}

HARTREE_TO_EV = 27.211386245988          # NIST 2018 CODATA
HARTREE_TO_KCAL_MOL = 627.5094740631


class Config:
    """Pipeline configuration. Passed explicitly to every stage."""

    def __init__(
        self,
        # ── Molecule ─────────────────────────────────────────────────────
        molecule="TiO2",
        charge=0,
        spin=0,
        basis="def2-svp",
        project_dir=None,
        # ── Solver selection ─────────────────────────────────────────────
        quantum_solver="sqd",
        # ── Classical reference methods ──────────────────────────────────
        classical_methods=None,
        # ── Geometry scan ────────────────────────────────────────────────
        geometry_scan=True,
        scan_atom_pair=(0, 1),
        scan_distances=None,
        scan_method="MP2",
        quantum_scan=True,
        quantum_scan_fast=True,
        quantum_scan_shots=2048,
        quantum_scan_iters=4,
        # ── Settings groups ──────────────────────────────────────────────
        asf=None,
        dmet=None,
        qiskit=None,
        gqe=None,
        tiers=None,
        # ── External tools ───────────────────────────────────────────────
        blockexe_wrapper=None,
    ):
        # Molecule
        self.molecule = molecule
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.project_dir = project_dir or os.getcwd()

        # Solver
        self.quantum_solver = self._normalise_solver(quantum_solver)

        # Classical
        self.classical_methods = classical_methods or ["HF", "MP2"]

        # Geometry scan
        self.geometry_scan = geometry_scan
        self.scan_atom_pair = scan_atom_pair
        self.scan_distances = (
            scan_distances if scan_distances is not None
            else np.linspace(0.9, 4.0, 20)
        )
        self.scan_method = scan_method
        self.quantum_scan = quantum_scan
        self.quantum_scan_fast = quantum_scan_fast
        self.quantum_scan_shots = quantum_scan_shots
        self.quantum_scan_iters = quantum_scan_iters

        # Settings groups
        self.asf = asf if asf is not None else AsfSettings()
        self.dmet = dmet if dmet is not None else DmetSettings()
        self.qiskit = qiskit if qiskit is not None else QiskitSolverSettings()
        self.gqe = gqe if gqe is not None else GqeSettings()
        self.tiers = tiers if tiers is not None else TierSettings()

        # External tools
        self.blockexe_wrapper = blockexe_wrapper or os.path.expanduser(
            "~/block2main_wrapper.sh"
        )

        # Constants
        self.hartree_to_ev = HARTREE_TO_EV
        self.hartree_to_kcal_mol = HARTREE_TO_KCAL_MOL

        # Geometry, populated by load_geometry()
        self.geometry = None
        self.atom_syms = None
        self.n_atoms = None

    # ── Solver name handling ─────────────────────────────────────────────
    @staticmethod
    def _normalise_solver(name):
        if name in SOLVER_ALIASES:
            import warnings

            canonical = SOLVER_ALIASES[name]
            warnings.warn(
                f"quantum_solver={name!r} is deprecated; use {canonical!r}.",
                DeprecationWarning,
                stacklevel=3,
            )
            return canonical
        return name

    @property
    def is_gqe(self):
        return self.quantum_solver in GQE_SOLVERS

    @property
    def is_qiskit(self):
        return self.quantum_solver in QISKIT_SOLVERS

    # ── Derived paths ────────────────────────────────────────────────────
    @property
    def results_dir(self):
        return os.path.join(self.project_dir, "results")

    @property
    def cif_dir(self):
        return os.path.join(self.project_dir, "cif_files")

    @property
    def plots_dir(self):
        return os.path.join(self.results_dir, "plots")

    @property
    def step0_file(self):
        return os.path.join(self.results_dir, "step0_classical.pkl")

    @property
    def step1_file(self):
        return os.path.join(self.results_dir, "step1_asf.pkl")

    @property
    def step2_file(self):
        return os.path.join(self.results_dir, "step2_hamiltonian.pkl")

    @property
    def step3_file(self):
        return os.path.join(self.results_dir, "step3_results.pkl")

    @property
    def gqe_log_file(self):
        return os.path.join(self.results_dir, "gqe_train.log")

    # ── Methods ──────────────────────────────────────────────────────────
    def load_geometry(self):
        """Resolve geometry from the built-in table, else from a CIF file."""
        if self.molecule in BUILTIN_GEOMETRIES:
            self.geometry = BUILTIN_GEOMETRIES[self.molecule]
        else:
            import warnings

            warnings.warn(
                f"{self.molecule!r} has no built-in geometry -- falling back to "
                f"a CIF lookup in {self.cif_dir}. Prefer an explicit geometry.",
                RuntimeWarning,
            )
            self.geometry = _load_geometry(self.molecule, self.cif_dir)

        self.atom_syms = [a[0] for a in self.geometry]
        self.n_atoms = len(self.geometry)
        return self

    def make_dirs(self):
        for path in (self.results_dir, self.cif_dir, self.plots_dir):
            os.makedirs(path, exist_ok=True)
        return self

    def cached_result_is_current(self, path, verbose=True):
        """
        True only if `path` exists AND its pickle was produced for the
        current molecule and basis.

        Every stage writes to a fixed filename shared across molecules
        (step0_classical.pkl, step1_asf.pkl, step2_hamiltonian.pkl). A plain
        os.path.exists() cache check therefore reuses the previous system's
        results, silently, with no error. That exact failure cost real
        debugging time three separate times in this project.

        The paths stay fixed on purpose -- the external gqe-for-qsci repo's
        molecule config references the step 2 path, so making the filenames
        molecule-specific would trade one stale-path bug for another.
        Validating the contents is the fix that does not.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as exc:
            if verbose:
                print(f"  [cache] {os.path.basename(path)} unreadable ({exc}); "
                      f"recomputing.")
            return False

        # step0 stores identity at top level; step1/step2 nest it in mol_info.
        info = data.get("mol_info", data) if isinstance(data, dict) else {}
        cached_mol = info.get("molecule")
        cached_basis = info.get("basis")

        if cached_mol is None:
            if verbose:
                print(f"  [cache] {os.path.basename(path)} has no molecule tag "
                      f"(predates this check); recomputing to be safe.")
            return False

        if cached_mol != self.molecule or (
            cached_basis is not None and cached_basis != self.basis
        ):
            if verbose:
                print(f"  [cache] {os.path.basename(path)} was built for "
                      f"{cached_mol}/{cached_basis}, but config says "
                      f"{self.molecule}/{self.basis} -- ignoring stale cache "
                      f"and recomputing.")
            return False
        return True

    def validate(self):
        """Catch configuration mistakes before any expensive work starts."""
        if self.spin < 0:
            raise ValueError(f"spin must be >= 0, got {self.spin}")
        if self.quantum_solver not in SOLVERS:
            raise ValueError(
                f"Unknown quantum_solver {self.quantum_solver!r}. "
                f"Choose one of {SOLVERS}."
            )
        if not self.classical_methods:
            raise ValueError("classical_methods must not be empty")

        self.asf.validate()
        self.dmet.validate()
        self.tiers.validate()
        # Only validate the stack actually selected, so a Qiskit-only user is
        # never blocked by a GQE setting they have not touched.
        if self.is_qiskit:
            self.qiskit.validate()
        if self.is_gqe:
            self.gqe.validate()
        return self

    def provenance(self, extra=None):
        from quenais.provenance import provenance as _provenance

        return _provenance(self, extra=extra)

    def __repr__(self):
        return (f"Config(molecule={self.molecule}, basis={self.basis}, "
                f"solver={self.quantum_solver}, "
                f"reference={self.dmet.reference})")
