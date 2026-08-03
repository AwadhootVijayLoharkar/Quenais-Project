"""
Active-space finder settings.

Physics rationale for the non-obvious values lives in docs/physics_notes.md;
this module stays readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AsfSettings", "DEFAULT_ASF_PARAMS"]


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

    #: Tier -> {entropy_threshold, max_norb, min_norb}
    params: dict = field(default_factory=lambda: {
        tier: dict(vals) for tier, vals in DEFAULT_ASF_PARAMS.items()
    })

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
