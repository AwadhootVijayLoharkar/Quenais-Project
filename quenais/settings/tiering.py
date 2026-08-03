"""
Correlation-tier classification settings.

The tier decides which entropy threshold and orbital bounds the active-space
finder uses (see AsfSettings.params).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TierSettings", "TM_ELEMENTS"]


#: Transition metals, lanthanides and actinides. Presence of one of these
#: pushes a system to a higher correlation tier.
TM_ELEMENTS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
    "Ac", "Th", "Pa", "U", "Np", "Pu",
})


@dataclass
class TierSettings:
    """Thresholds for classifying how strongly correlated a system is."""

    #: <S^2> above this indicates strong static correlation.
    spin_contamination_tier2_threshold: float = 1.3

    #: <S^2> above this in a nominal singlet indicates spin contamination.
    spin_contamination_singlet_threshold: float = 0.05

    #: HOMO-LUMO gap (eV) below this indicates near-degeneracy.
    homo_lumo_tier2_threshold_ev: float = 1.0

    #: Elements that force a higher tier regardless of the other indicators.
    tm_elements: frozenset = field(default_factory=lambda: TM_ELEMENTS)

    def is_transition_metal_system(self, atom_syms) -> bool:
        return any(sym in self.tm_elements for sym in (atom_syms or []))

    def validate(self) -> "TierSettings":
        if self.spin_contamination_tier2_threshold <= 0:
            raise ValueError("spin_contamination_tier2_threshold must be > 0")
        if self.spin_contamination_singlet_threshold < 0:
            raise ValueError("spin_contamination_singlet_threshold must be >= 0")
        if self.homo_lumo_tier2_threshold_ev <= 0:
            raise ValueError("homo_lumo_tier2_threshold_ev must be > 0")
        return self
