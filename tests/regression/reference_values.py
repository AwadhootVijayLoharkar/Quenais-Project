"""
Validated reference values for the QuEnAIS regression suite.

Source: A100 / AMD EPYC 7402 run of scripts/test_8, July 2026.
Golden pickles for these systems live in tests/regression/golden/<system>/.

WHY THIS FILE HAS TIERS
-----------------------
A single tolerance across all quantities is wrong, and asserting one would
make the suite fail for reasons unrelated to the code:

  DETERMINISTIC       reproduces to ~1e-10 across machines. Assert tightly.
  OPTIMIZER_DEPENDENT depends which solution CASSCF converges to. Two runs of
                      the same input gave ScH CASSCF -752.680677 and
                      -752.681604 (0.93 mHa apart); NEVPT2 rides on top of
                      CASSCF and moved 3.6 mHa with it. Both are valid
                      solutions. Assert loosely, or not at all.
  STOCHASTIC          varies run to run by design. ScH DMET+GQE has produced
                      -752.668847, -752.678674 and -752.677509. Assert only
                      against a pinned seed, or treat as informational.

Every bug in this project's history was "right shape, plausible magnitude,
wrong value" -- so the DETERMINISTIC tier is the one that actually catches
regressions, and it is deliberately tight.
"""

from enum import Enum


class Tier(str, Enum):
    DETERMINISTIC = "deterministic"
    OPTIMIZER_DEPENDENT = "optimizer-dependent"
    STOCHASTIC = "stochastic"


# Default tolerances in Hartree, by tier.
TOL = {
    Tier.DETERMINISTIC: 1e-9,
    Tier.OPTIMIZER_DEPENDENT: 2e-3,
    Tier.STOCHASTIC: 5e-2,
}


# ─────────────────────────────────────────────────────────────────────────
# LiH / STO-3G, r = 1.5949 Ang, ASF automatic (2e,2o), N_emb = 4, 8 qubits
# ─────────────────────────────────────────────────────────────────────────
LIH = {
    "molecule": "LiH",
    "basis": "sto-3g",
    "force_active_space": None,
    "energies": {
        "HF":     (-7.862026959374693, Tier.DETERMINISTIC),
        "MP2":    (-7.874890916658973, Tier.DETERMINISTIC),
        "CCSD":   (-7.882392916896698, Tier.DETERMINISTIC),
        "CCSD_T": (-7.882401304905777, Tier.DETERMINISTIC),
        "CASSCF": (-7.881119854628031, Tier.OPTIMIZER_DEPENDENT),
        "NEVPT2": (-7.881866437419046, Tier.OPTIMIZER_DEPENDENT),
        "DMET_CASCI": (-7.881246151999259, Tier.DETERMINISTIC),
        "DMET_GQE":   (-7.881246151999270, Tier.STOCHASTIC),
    },
    # Structural invariants -- these are the bug tripwires, not energies.
    "structure": {
        "nel": 2,
        "mo_list": [1, 2],
        "n_imp": 2,
        # Guards the adaptive_bath fix: LiH DOES have a real bath.
        "n_bath": 2,
        "n_emb": 4,
        # Guards the electron-count fix. The buggy path derives these from the
        # active-space count and gives (1, 1), producing energies ~2x HF.
        "n_alpha": 2,
        "n_beta": 2,
        "sv2_cov": 1.0,
        "reference_density_method": "casci",
    },
    "scalars": {
        "ecore": (-5.010051310495199, 1e-9),
        "mu": (-1.500452061427127, 1e-9),
        "uhf_energy": (-7.862026959343912, 1e-9),
    },
    # Sum of the reference-density occupations in the embedding space, before
    # rounding to n_alpha/n_beta. The buggy path never computes these.
    "ref_occ_sums": {"alpha": (2.000007616328369, 1e-8),
                     "beta":  (2.000007616672249, 1e-8)},
}


# ─────────────────────────────────────────────────────────────────────────
# N2 / STO-3G, r = 1.0977 Ang, ASF automatic (4e,4o)
#
# THE n_bath = 0 CASE. All Schmidt singular values come back numerically
# zero (measured max 5.4e-15), so there is no bath and the embedding is the
# active space alone. This is CORRECT BEHAVIOUR, not a failure. The buggy
# adaptive_bath fabricates 4 bath orbitals from this noise and produces
# ~20 Ha errors.
# ─────────────────────────────────────────────────────────────────────────
N2 = {
    "molecule": "N2",
    "basis": "sto-3g",
    "force_active_space": None,
    "energies": {
        "HF":     (-107.495893307834180, Tier.DETERMINISTIC),
        "MP2":    (-107.649983786559490, Tier.DETERMINISTIC),
        "CCSD":   (-107.648941288706010, Tier.DETERMINISTIC),
        "CCSD_T": (-107.650656674265040, Tier.DETERMINISTIC),
        "CASSCF": (-107.598367310094000, Tier.OPTIMIZER_DEPENDENT),
        "NEVPT2": (-107.640852538941690, Tier.OPTIMIZER_DEPENDENT),
        "DMET_CASCI": (-107.598406106545040, Tier.DETERMINISTIC),
        "DMET_GQE":   (-107.598406106545040, Tier.STOCHASTIC),
    },
    "structure": {
        "nel": 4,
        "mo_list": [5, 6, 7, 8],
        "n_imp": 4,
        "n_bath": 0,          # <-- the tripwire
        "n_emb": 4,
        "n_alpha": 2,
        "n_beta": 2,
        "sv2_cov": 0.0,
        "reference_density_method": "casci",
    },
    "scalars": {
        "ecore": (-110.786156378632280, 1e-9),
        "mu": (-2.222190266165760, 1e-9),
        "uhf_energy": (-107.495893307833610, 1e-9),
    },
    "ref_occ_sums": {"alpha": (2.0, 1e-8), "beta": (2.0, 1e-8)},
    # Every Schmidt singular value must stay below the bath tolerance.
    "max_abs_sv_all_below": 1e-8,
    # DMET+CASCI must track CASSCF closely on this system.
    "dmet_casci_vs_casscf_mha": (0.04, 0.01),   # (expected, tolerance)
}


# ─────────────────────────────────────────────────────────────────────────
# ScH / STO-3G, r = 1.78 Ang, FORCED (4e,6o) MOs [9-14], N_emb = 11, 22 qubits
#
# CAVEAT -- READ BEFORE USING ScH AS AN ANSWER KEY.
# The active space is known to be under-selected for this system. ASF's
# entropy thresholds are calibrated on main-group elements and under-select
# for d-block, which is why MOs [9..14] are forced by hand. Even so, NEVPT2
# lands 7.2 mHa ABOVE CCSD(T) (-752.702671 vs -752.709890); a well-chosen
# active space should put CASSCF+NEVPT2 at or below CCSD(T). So ScH's
# classical reference is softer than LiH's or N2's. Scheduled to be fixed
# in a later release -- see docs/limitations.md.
#
# Consequence: do not build tight regression assertions on ScH CASSCF or
# NEVPT2. The DMET quantities on this system ARE reliable -- DMET+CASCI
# reproduced to 1e-10 across two runs on different hardware.
# ─────────────────────────────────────────────────────────────────────────
SCH = {
    "molecule": "ScH",
    "basis": "sto-3g",
    "force_active_space": [9, 10, 11, 12, 13, 14],
    "energies": {
        "HF":     (-752.638702408343600, Tier.DETERMINISTIC),
        "MP2":    (-752.687689554408200, Tier.DETERMINISTIC),
        "CCSD":   (-752.708840128687400, Tier.DETERMINISTIC),
        "CCSD_T": (-752.709890151334200, Tier.DETERMINISTIC),
        # Superseded values from the earlier run, kept for provenance:
        #   CASSCF -752.680676666795, NEVPT2 -752.699028364395
        # The values below are the lower (better) CASSCF solution.
        "CASSCF": (-752.681604480450600, Tier.OPTIMIZER_DEPENDENT),
        "NEVPT2": (-752.702670521892900, Tier.OPTIMIZER_DEPENDENT),
        "DMET_CASCI": (-752.699524181160900, Tier.DETERMINISTIC),
        "DMET_GQE":   (-752.677509258033800, Tier.STOCHASTIC),
    },
    "structure": {
        "nel": 4,
        "mo_list": [9, 10, 11, 12, 13, 14],
        "n_imp": 6,
        "n_bath": 5,
        "n_emb": 11,
        # Buggy path gives (2, 2) here -- active-space count, not the
        # reference-density trace.
        "n_alpha": 4,
        "n_beta": 4,
        "sv2_cov": 1.0,
        "reference_density_method": "casci",
    },
    "scalars": {
        "ecore": (-750.010796110870000, 1e-8),
        "mu": (-2.400772052838401, 1e-8),
        "uhf_energy": (-752.638702408099300, 1e-8),
    },
    "ref_occ_sums": {"alpha": (4.008817683544550, 1e-7),
                     "beta":  (4.008818139779939, 1e-7)},
    # MOs 11 and 12 are a degenerate pair (identical entropy). They must be
    # kept together -- the pre-fix find_gap_cutoff splits pairs like this.
    "degenerate_pairs": [(11, 12)],
}


SYSTEMS = {"LiH": LIH, "N2": N2, "ScH": SCH}

# Systems cheap enough to run in the default self-test.
SELFTEST_SYSTEMS = ["LiH"]

# ─────────────────────────────────────────────────────────────────────────
# Cross-cutting checks that are not per-system energies
# ─────────────────────────────────────────────────────────────────────────

# The single most diagnostic quantity in the pipeline. mol.hf is a real,
# converged SCF run on h1e_emb/h2e_emb/ecore -- if it does not land on the
# full molecule's UHF energy, the embedding Hamiltonian itself is wrong.
# Unlike the ecore self-consistency assertion in DMET Phase E, this one can
# actually fail. Validated at 1.3e-7 Ha on ScH.
EMBEDDED_SCF_VS_UHF_TOL = 2e-7

# Jordan-Wigner excitation-generator convention vs tequila. Validated as
# exactly 1.0 on singles (both spins) and doubles, with identical Pauli sets.
EXCITATION_GENERATOR_RATIO = 1.0
EXCITATION_GENERATOR_TOL = 1e-12
