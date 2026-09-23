"""
High-accuracy reference settings (FCI, CASCI, DMRG).

These three methods answer a different question from the ones already in
step 0. MP2/CCSD/CCSD(T) are single-reference and lose accuracy exactly
where this package is aimed -- stretched bonds, transition-metal centres,
anything multireference. Benchmarking LASSQD against CCSD(T) on a
dissociation curve tells you the two disagree, not which one is right.

  FCI   -- exact in the full basis. The true answer; feasible to ~16
           orbitals. Benchmarks the active space AND the solver.
  CASCI -- exact within step 1's active space, at fixed orbitals. The
           apples-to-apples reference for every step-3 solver: it is the
           number SQD, SKQD, SQDrift and LASSQD are all trying to
           reproduce, so the difference is solver error alone.
  DMRG  -- near-exact to ~50 orbitals, controlled by one parameter (the
           bond dimension M). The reference for systems FCI cannot reach.

Licensing note: DMRG runs through block2 (GPL-3.0), which install.sh
already fetches for ASF. Running it to produce reference numbers is fine
-- energies are data, not derived works. What is not fine is shipping a
Docker image or packed environment that CONTAINS block2. See
docs/licensing.md. Nothing here imports block2 unless DMRG is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ReferenceSettings", "DMRG_REORDERINGS"]

#: Orbital orderings block2 can apply before building the MPO.
#:
#: DMRG cost and accuracy depend strongly on the ordering of orbitals along
#: the chain, because the method truncates entanglement between the left and
#: right halves of whatever order you give it. "fiedler" sorts by the Fiedler
#: vector of the exchange matrix and is the standard default; "gaopt" runs a
#: genetic optimisation on top and is slower but occasionally better on
#: transition-metal systems. "none" keeps the canonical order and is mainly
#: useful when you need bit-reproducibility against an external code.
DMRG_REORDERINGS = ("fiedler", "gaopt", "none")


@dataclass
class ReferenceSettings:
    """Exact and near-exact reference methods for step 0."""

    # ── FCI ──────────────────────────────────────────────────────────────
    #: Refuse to start a full-space FCI larger than this many determinants.
    #:
    #: Load-bearing. FCI has no graceful degradation: it either fits in
    #: memory or the process is killed by the OOM reaper after however long
    #: the integral transformation took. 5e6 determinants is roughly 40 MB
    #: for the CI vector alone, and PySCF needs several copies plus the
    #: Davidson subspace. The check runs before any work, so an impossible
    #: request costs a printed message rather than a dead cluster job.
    fci_max_dets: int = 5_000_000

    #: Run FCI inside step 1's active space instead of the full basis.
    #:
    #: False (default) is the honest reading of "FCI": exact in the given
    #: basis. True makes it a synonym for CASCI and is only here so a curve
    #: can be produced for a molecule whose full space is out of reach.
    fci_in_active_space: bool = False

    # ── CASCI ────────────────────────────────────────────────────────────
    #: Roots to solve for. >1 is useful when a curve crosses states and you
    #: need to see whether the solver is tracking the wrong one.
    casci_nroots: int = 1

    # ── DMRG (block2) ────────────────────────────────────────────────────
    #: Bond dimensions, in order. Each is a separate DMRG stage restarted
    #: from the previous MPS, which is both faster and more reliable than
    #: starting at the final M from a random state.
    #:
    #: The list is also the extrapolation data: three points is the minimum
    #: for a meaningful fit of E against discarded weight, which is why the
    #: default has three rather than one.
    dmrg_bond_dims: tuple = (250, 500, 1000)

    #: Noise per stage, same length as dmrg_bond_dims. Noise keeps the
    #: sweep out of local minima while the bond dimension is still growing
    #: and MUST reach zero in the final stage -- a non-zero noise biases the
    #: energy upward, which would then be extrapolated as if it were
    #: truncation error.
    dmrg_noises: tuple = (1e-4, 1e-5, 0.0)

    #: Davidson tolerance per stage, same length as dmrg_bond_dims.
    dmrg_thrds: tuple = (1e-8, 1e-9, 1e-10)

    #: Sweeps per bond-dimension stage.
    dmrg_sweeps_per_stage: int = 8

    #: Energy convergence between sweeps within a stage.
    dmrg_tol: float = 1e-8

    #: Run DMRG in step 1's active space (True) or the full basis (False).
    #:
    #: True by default because the comparison that matters is against the
    #: step-3 solvers, which all work in that active space.
    dmrg_in_active_space: bool = True

    #: Extrapolate E to zero discarded weight and report both numbers.
    #:
    #: The linear fit of E against discarded weight is the standard way to
    #: quote a DMRG energy, and the gap between the raw and extrapolated
    #: values is the honest error bar. Silently skipped, with a note in the
    #: results, when block2 does not expose per-sweep discarded weights.
    dmrg_extrapolate: bool = True

    #: Orbital ordering; see DMRG_REORDERINGS.
    dmrg_reorder: str = "fiedler"

    #: OpenMP threads for block2. 0 leaves it to block2/OMP_NUM_THREADS,
    #: which is what you want under a scheduler that already pinned cores.
    dmrg_threads: int = 0

    #: Scratch directory for the MPS. None puts it under cfg.results_dir.
    #:
    #: On a cluster this should point at node-local fast storage: block2
    #: writes every renormalised operator to disk, and doing that over NFS
    #: turns a ten-minute run into an hour.
    dmrg_scratch: str | None = None

    #: Keep the block2 scratch files after the run. They are large (GB for
    #: a real system) but are what a restart would need.
    dmrg_keep_scratch: bool = False

    #: Extra note written into the results dict, e.g. a run label.
    tags: dict = field(default_factory=dict)

    def validate(self) -> "ReferenceSettings":
        if self.fci_max_dets < 1:
            raise ValueError("ref.fci_max_dets must be >= 1")
        if self.casci_nroots < 1:
            raise ValueError("ref.casci_nroots must be >= 1")

        dims = tuple(self.dmrg_bond_dims)
        if not dims:
            raise ValueError("ref.dmrg_bond_dims must not be empty")
        if any(int(m) < 1 for m in dims):
            raise ValueError(f"ref.dmrg_bond_dims must all be >= 1, got {dims}")
        if list(dims) != sorted(dims):
            raise ValueError(
                f"ref.dmrg_bond_dims must be non-decreasing, got {dims}. Each "
                f"stage restarts from the previous MPS, so shrinking M throws "
                f"away the information the previous stage paid for."
            )

        for name in ("dmrg_noises", "dmrg_thrds"):
            seq = tuple(getattr(self, name))
            if len(seq) != len(dims):
                raise ValueError(
                    f"ref.{name} has {len(seq)} entries but "
                    f"ref.dmrg_bond_dims has {len(dims)}; block2 pairs them "
                    f"stage by stage."
                )
        if any(float(x) < 0 for x in self.dmrg_noises):
            raise ValueError("ref.dmrg_noises must all be >= 0")
        if float(tuple(self.dmrg_noises)[-1]) != 0.0:
            raise ValueError(
                "ref.dmrg_noises must end at 0.0: a non-zero noise in the "
                "final stage raises the energy, and extrapolating that as "
                "truncation error gives a number that is wrong in a way no "
                "convergence check would catch."
            )
        if any(float(x) <= 0 for x in self.dmrg_thrds):
            raise ValueError("ref.dmrg_thrds must all be > 0")

        if self.dmrg_sweeps_per_stage < 1:
            raise ValueError("ref.dmrg_sweeps_per_stage must be >= 1")
        if self.dmrg_tol <= 0:
            raise ValueError("ref.dmrg_tol must be > 0")
        if self.dmrg_reorder not in DMRG_REORDERINGS:
            raise ValueError(
                f"ref.dmrg_reorder must be one of {DMRG_REORDERINGS}, "
                f"got {self.dmrg_reorder!r}"
            )
        if self.dmrg_threads < 0:
            raise ValueError("ref.dmrg_threads must be >= 0 (0 = let block2 decide)")
        if self.dmrg_extrapolate and len(dims) < 3:
            raise ValueError(
                f"ref.dmrg_extrapolate needs at least 3 bond dimensions to fit "
                f"a line through, got {len(dims)}. Either add stages or set "
                f"dmrg_extrapolate=False and quote the raw M={dims[-1]} energy."
            )
        return self
