"""
Active-space finder settings.

Physics rationale for the non-obvious values lives in docs/physics_notes.md;
this module stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AsfSettings", "DEFAULT_ASF_PARAMS", "SELECTION_METHODS"]


#: Available active-space selectors.
#:
#:   "asf"  -- ASF/DMRG entanglement entropy. Needs block2. Under-selects
#:             for the d-block (docs/limitations.md).
#:   "avas" -- project the MOs onto named atomic valence orbitals. Needs
#:             only PySCF. Recommended for transition metals.
#:   "apc"  -- PySCF's ranked-orbital APC entropy. Automatic, no AO labels.
#:
#: force_active_space still overrides all three.
SELECTION_METHODS = ("asf", "avas", "apc")


#: Per-tier entanglement-entropy thresholds and orbital-count bounds.
#: Calibrated on main-group systems -- see docs/limitations.md for the
#: known d-block under-selection.
DEFAULT_ASF_PARAMS = {
    1: {"entropy_threshold": 0.05, "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.02, "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.005, "max_norb": 16, "min_norb": 4},
}


@dataclass
class AsfSettings:
    """Active-space selection."""

    #: Which selector produces the active space. See SELECTION_METHODS.
    #: Defaults to "asf" so every existing result reproduces bit-identically.
    method: str = "asf"

    #: Tier -> {entropy_threshold, max_norb, min_norb}. Used by "asf" only.
    params: dict = field(default_factory=lambda: {
        tier: dict(vals) for tier, vals in DEFAULT_ASF_PARAMS.items()
    })

    # ── AVAS (method="avas") ─────────────────────────────────────────────

    #: AO labels defining the valence shell, in PySCF search_ao_label
    #: syntax: ["Sc 3d", "Sc 4s"]. None derives them from the elements
    #: present via selectors.default_ao_labels().
    #:
    #: This list IS the physics of an AVAS selection. It is written into
    #: the step 1 pickle as selection_meta so a run can be reproduced from
    #: the pickle alone.
    avas_ao_labels: list | None = None

    #: Projector eigenvalue above which an MO counts as carrying the
    #: reference character. PySCF's own default.
    #:
    #: In a minimal basis (sto-3g) the projection onto minao is nearly
    #: trivial, so this threshold is weakly discriminating -- expect a
    #: plateau rather than a sharp optimum. selectors.select_avas() warns.
    avas_threshold: float = 0.2

    #: Reference minimal AO basis AVAS projects onto.
    avas_minao: str = "minao"

    #: Singly-occupied handling, PySCF's option 2 or 3. Irrelevant for
    #: closed-shell systems.
    avas_openshell_option: int = 2

    # ── APC (method="apc") ───────────────────────────────────────────────

    #: Maximum active-space size APC may return.
    apc_max_size: int = 8

    #: APC-n: how many times to strip the highest-entropy virtual before
    #: re-scoring. Higher n favours fewer doubly-occupied orbitals.
    apc_n: int = 2

    #: Bounds for Phase C gap detection.
    gap_min_norb: int = 2
    gap_max_norb: int = 16

    #: Orbitals whose deviation values differ by less than this are treated
    #: as one degenerate group and kept or dropped together, rather than
    #: split across the cutoff.
    #:
    #: Without it, N2's two pi orbitals (identical entropy, S=0.246) get
    #: split -- one kept, its degenerate partner dropped -- producing a
    #: symmetry-broken, physically incomplete active space.
    gap_degeneracy_tol: float = 1e-3

    #: Natural-orbital occupation above which an orbital counts as core.
    core_occ_threshold: float = 1.95

    #: Escape hatch: an explicit list of 0-based MO indices in the UHF
    #: alpha-MO basis, bypassing ASF/DMRG entirely.
    #:
    #: Needed for transition metals today -- ASF's thresholds under-select
    #: for the d-block (docs/limitations.md). ScH uses [9, 10, 11, 12, 13, 14].
    #: Leave None for automatic selection.
    force_active_space: list | None = None

    #: Phase C can only shrink ASF's own selection, and ranks by MP2
    #: occupation deviation -- a weaker signal than the entanglement entropy
    #: ASF used to choose those orbitals. It warns when it discards any.
    #: Set False to disable Phase C narrowing entirely.
    phase_c_enabled: bool = True

    def validate(self) -> "AsfSettings":
        if self.method not in SELECTION_METHODS:
            raise ValueError(
                f"asf.method must be one of {list(SELECTION_METHODS)}, got "
                f"{self.method!r}"
            )
        if self.avas_ao_labels is not None:
            labels = list(self.avas_ao_labels)
            if not labels:
                raise ValueError(
                    "avas_ao_labels is empty; use None to derive them from "
                    "the elements present"
                )
            if not all(isinstance(s, str) and s.strip() for s in labels):
                raise ValueError(
                    f"avas_ao_labels must be non-empty strings such as "
                    f"'Sc 3d', got {labels}"
                )
            # A bare element symbol matches every AO on that atom, core
            # included, and produces a large plausible-looking active space
            # nobody asked for. Almost always a shell-quoting accident:
            #   --avas-ao-labels Sc 3d   ->  ['Sc', '3d']
            bare = [s for s in labels if " " not in s.strip()]
            if bare:
                raise ValueError(
                    f"avas_ao_labels entries {bare} name no shell. A bare "
                    f"symbol matches every AO on that atom including the "
                    f"core. Quote each label: --avas-ao-labels 'Sc 3d' "
                    f"'Sc 4s'"
                )
        if not 0.0 < self.avas_threshold < 1.0:
            raise ValueError(
                f"avas_threshold must lie in (0, 1), got {self.avas_threshold}"
            )
        if self.avas_openshell_option not in (2, 3):
            raise ValueError(
                f"avas_openshell_option must be 2 or 3, got "
                f"{self.avas_openshell_option}"
            )
        if self.apc_max_size < 1:
            raise ValueError(f"apc_max_size must be >= 1, got {self.apc_max_size}")
        if self.apc_n < 0:
            raise ValueError(f"apc_n must be >= 0, got {self.apc_n}")
        if self.gap_min_norb < 1:
            raise ValueError(f"gap_min_norb must be >= 1, got {self.gap_min_norb}")
        if self.gap_max_norb < self.gap_min_norb:
            raise ValueError(
                f"gap_max_norb ({self.gap_max_norb}) < gap_min_norb "
                f"({self.gap_min_norb})"
            )
        if self.gap_degeneracy_tol < 0:
            raise ValueError("gap_degeneracy_tol must be >= 0")
        if not 0.0 <= self.core_occ_threshold <= 2.0:
            raise ValueError(
                f"core_occ_threshold must lie in [0, 2], got {self.core_occ_threshold}"
            )
        if self.force_active_space is not None:
            fas = list(self.force_active_space)
            if not fas:
                raise ValueError("force_active_space is empty; use None for automatic")
            if any((not isinstance(i, int)) or i < 0 for i in fas):
                raise ValueError(
                    f"force_active_space must be non-negative ints, got {fas}"
                )
            if len(set(fas)) != len(fas):
                raise ValueError(f"force_active_space has duplicates: {fas}")
        for tier, vals in self.params.items():
            missing = {"entropy_threshold", "max_norb", "min_norb"} - set(vals)
            if missing:
                raise ValueError(f"asf.params[{tier}] missing keys: {sorted(missing)}")
        return self