"""
Exact reference methods -- settings, guards and bookkeeping.

Runs anywhere; needs NumPy only. This half covers the parts that are easy
to get quietly wrong: the DMRG schedule guards, the FCI size refusal, the
extrapolation fit, and which reference the error column is measured
against. The physics is in tests/test_references_pyscf.py.
"""

import pytest

from quenais.classical.references import (
    REFERENCE_TIERS,
    _extrapolate,
    _last_discarded_weight,
    fci_determinant_count,
)
from quenais.classical.runner import METHOD_TIERS, best_reference
from quenais.config import CLASSICAL_METHODS, EXACT_METHODS, Config
from quenais.settings import ReferenceSettings


# ═════════════════════════════════════════════════════════════════════════
# Settings and registry
# ═════════════════════════════════════════════════════════════════════════

def test_defaults_validate():
    assert ReferenceSettings().validate() is not None


def test_exact_methods_are_registered_everywhere():
    """A method Config accepts must also carry a reproducibility tier."""
    for name in EXACT_METHODS:
        assert name in CLASSICAL_METHODS
        assert name in METHOD_TIERS
        assert name in REFERENCE_TIERS


def test_config_rejects_unknown_classical_method():
    with pytest.raises(ValueError, match="Unknown classical method"):
        Config(classical_methods=["HF", "CCSD(T)"]).validate()


def test_config_accepts_the_exact_tier():
    cfg = Config(classical_methods=["HF", "CASCI", "FCI", "DMRG"]).validate()
    assert cfg.ref.fci_max_dets > 0


@pytest.mark.parametrize("kwargs, match", [
    # Noise must reach zero: a residual noise raises the energy, and
    # extrapolating that as truncation error is wrong in a way no
    # convergence check would catch.
    (dict(dmrg_noises=(1e-4, 1e-5, 1e-6)), "must end at 0"),
    # Each stage restarts from the previous MPS.
    (dict(dmrg_bond_dims=(1000, 500, 250)), "non-decreasing"),
    # Three points minimum for a line. The per-stage lists are shortened to
    # match, so this reaches the extrapolation guard rather than tripping
    # the length check first.
    (dict(dmrg_bond_dims=(250, 500), dmrg_noises=(1e-5, 0.0),
          dmrg_thrds=(1e-8, 1e-9)), "at least 3 bond dimensions"),
    (dict(dmrg_noises=(1e-4, 0.0)), "block2 pairs them"),
    (dict(dmrg_thrds=(1e-8, 1e-9)), "block2 pairs them"),
    (dict(dmrg_reorder="banana"), "dmrg_reorder must be one of"),
    (dict(fci_max_dets=0), "fci_max_dets"),
    (dict(casci_nroots=0), "casci_nroots"),
    (dict(dmrg_sweeps_per_stage=0), "dmrg_sweeps_per_stage"),
    (dict(dmrg_threads=-1), "dmrg_threads"),
    (dict(dmrg_bond_dims=()), "must not be empty"),
])
def test_bad_settings_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ReferenceSettings(**kwargs).validate()


def test_two_stages_are_fine_without_extrapolation():
    ReferenceSettings(dmrg_bond_dims=(250, 500), dmrg_noises=(1e-5, 0.0),
                      dmrg_thrds=(1e-8, 1e-9),
                      dmrg_extrapolate=False).validate()


# ═════════════════════════════════════════════════════════════════════════
# CLI plumbing
# ═════════════════════════════════════════════════════════════════════════

def _args(*argv):
    from quenais.cli import build_parser

    return build_parser().parse_args(["--molecule", "N2", *argv])


def test_cli_builds_a_consistent_dmrg_schedule():
    """
    Changing the number of bond dimensions must rebuild the per-stage
    noise and Davidson lists too, or validation fails on a length mismatch
    the user never caused.
    """
    from quenais.cli import build_ref_settings

    ref = build_ref_settings(_args("--dmrg-bond-dims", "100", "200", "400", "800"))
    ref.validate()
    assert ref.dmrg_bond_dims == (100, 200, 400, 800)
    assert len(ref.dmrg_noises) == 4
    assert ref.dmrg_noises[-1] == 0.0
    assert len(ref.dmrg_thrds) == 4


def test_cli_single_bond_dim_disables_extrapolation():
    from quenais.cli import build_ref_settings

    ref = build_ref_settings(_args("--dmrg-bond-dims", "500"))
    ref.validate()
    assert ref.dmrg_extrapolate is False
    assert ref.dmrg_noises == (0.0,)


def test_cli_flags_reach_the_settings():
    from quenais.cli import build_ref_settings

    ref = build_ref_settings(_args(
        "--fci-in-active-space", "--fci-max-dets", "1000",
        "--casci-nroots", "3", "--dmrg-full-space", "--dmrg-no-extrapolate",
        "--dmrg-reorder", "gaopt", "--dmrg-sweeps", "12",
        "--dmrg-keep-scratch",
    ))
    ref.validate()
    assert ref.fci_in_active_space is True
    assert ref.fci_max_dets == 1000
    assert ref.casci_nroots == 3
    assert ref.dmrg_in_active_space is False
    assert ref.dmrg_extrapolate is False
    assert ref.dmrg_reorder == "gaopt"
    assert ref.dmrg_sweeps_per_stage == 12
    assert ref.dmrg_keep_scratch is True


def test_unset_flags_keep_defaults():
    from quenais.cli import build_ref_settings

    assert build_ref_settings(_args()) == ReferenceSettings()


def test_cli_accepts_the_exact_methods():
    args = _args("--classical-methods", "HF", "CASCI", "FCI", "DMRG")
    assert args.classical_methods == ["HF", "CASCI", "FCI", "DMRG"]


# ═════════════════════════════════════════════════════════════════════════
# Determinant counting -- the FCI guard depends on it being exact
# ═════════════════════════════════════════════════════════════════════════

def test_determinant_count_matches_binomials():
    assert fci_determinant_count(4, (2, 2)) == 36        # H4 / STO-3G
    assert fci_determinant_count(6, (3, 3)) == 400       # H6 / STO-3G
    assert fci_determinant_count(10, (5, 3)) == 252 * 120


def test_determinant_count_is_exact_for_huge_spaces():
    """
    Python ints, not floats. N2/cc-pVDZ is 1.4e12 determinants; a float
    count would still refuse correctly, but the number written into the
    results pickle would be wrong.
    """
    n = fci_determinant_count(28, (7, 7))
    assert isinstance(n, int)
    assert n == 1_401_950_721_600


# ═════════════════════════════════════════════════════════════════════════
# DMRG extrapolation
# ═════════════════════════════════════════════════════════════════════════

def _stages(energies, weights):
    return [{"bond_dim": 250 * 2 ** i, "energy": e, "discarded_weight": w}
            for i, (e, w) in enumerate(zip(energies, weights))]


def test_extrapolation_recovers_a_known_intercept():
    """E(dw) = E0 + slope*dw, fitted back to E0."""
    e0, slope = -109.2821, 150.0
    dw = [4e-4, 1e-4, 2.5e-5]
    stages = _stages([e0 + slope * w for w in dw], dw)
    e_ext, err, note = _extrapolate(stages)
    assert abs(e_ext - e0) < 1e-9
    assert err == pytest.approx(stages[-1]["energy"] - e0, abs=1e-9)
    assert "linear fit" in note


def test_extrapolation_flags_a_non_decreasing_energy():
    """
    The energy must fall as truncation is removed. If it does not, the
    sweeps are not converged and the "extrapolated" number is meaningless
    -- the note has to say so rather than quietly returning it.
    """
    _e, _err, note = _extrapolate(_stages([-1.0, -1.1, -1.2],
                                          [1e-5, 1e-4, 4e-4]))
    assert "WARNING" in note


def test_extrapolation_skipped_without_discarded_weights():
    e_ext, err, note = _extrapolate(_stages([-1.0, -1.1, -1.2], [None] * 3))
    assert e_ext is None and err is None
    assert "3 stages" in note


def test_extrapolation_skipped_with_identical_weights():
    e_ext, _err, note = _extrapolate(_stages([-1.0, -1.1, -1.2], [1e-5] * 3))
    assert e_ext is None
    assert "identical" in note


def test_discarded_weight_probe_never_raises():
    """block2 moves this attribute between releases; a miss must degrade."""
    assert _last_discarded_weight(object()) is None
    assert _last_discarded_weight(None) is None

    class Fake:
        discarded_weights = [1e-6, 3e-6, 2e-6]

    assert _last_discarded_weight(Fake()) == pytest.approx(3e-6)

    class Nested:
        _dmrg = Fake()

    assert _last_discarded_weight(Nested()) == pytest.approx(3e-6)

    class Junk:
        discarded_weights = "not numbers"

    assert _last_discarded_weight(Junk()) is None


# ═════════════════════════════════════════════════════════════════════════
# Which reference the error column is measured against
# ═════════════════════════════════════════════════════════════════════════

def test_reference_preference_is_accuracy_order():
    full_fci = {"energy": -2.0, "in_active_space": False}
    cas_fci = {"energy": -2.0, "in_active_space": True}
    dmrg = {"energy": -1.9, "extrapolated_energy": -1.95, "max_bond_dim": 1000}
    casci = {"energy": -1.5}

    assert best_reference({"FCI": full_fci, "DMRG": dmrg})[0] == "FCI"
    assert best_reference({"DMRG": dmrg, "CASCI": casci})[0] == "DMRG(M->inf)"
    # An active-space FCI is not exact in the basis, so DMRG outranks it.
    assert best_reference({"FCI": cas_fci, "DMRG": dmrg})[0] == "DMRG(M->inf)"
    assert best_reference({"FCI": cas_fci})[0] == "FCI(CAS)"
    assert best_reference({"CASCI": casci}) == ("CASCI", -1.5)
    assert best_reference({}) == (None, None)


def test_failed_methods_are_not_used_as_the_reference():
    assert best_reference({"FCI": {"energy": None, "error": "boom"},
                           "CASCI": {"energy": -1.5}}) == ("CASCI", -1.5)


def test_raw_dmrg_is_used_when_extrapolation_was_skipped():
    name, e = best_reference({"DMRG": {"energy": -1.9, "max_bond_dim": 500}})
    assert name == "DMRG(M=500)" and e == -1.9
