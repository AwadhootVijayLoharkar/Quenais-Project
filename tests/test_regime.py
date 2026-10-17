"""Regime diagnostics and router (quenais/quantum/regime.py). NumPy only."""

import numpy as np
import pytest

from quenais.quantum.regime import (Thresholds, noon_metrics, recommend,
                                    spin_adapted_hf_weight)


def test_hf_weight_two_electrons_two_orbitals():
    # H2 minimal basis: |0> + t |doubly excited>, norm^2 = 1 + t^2
    t = 0.3
    w = spin_adapted_hf_weight(np.zeros((1, 1)), np.full((1, 1, 1, 1), t))
    assert w == pytest.approx(1.0 / (1.0 + t * t))


def test_hf_weight_singles_count_both_spins():
    t1 = np.array([[0.2]])
    w = spin_adapted_hf_weight(t1, np.zeros((1, 1, 1, 1)))
    # singles 2 * 0.2^2 = 0.08; doubles C2 = t1 t1 = 0.04 -> 0.04 * (2*0.04 - 0.04)
    assert w == pytest.approx(1.0 / (1.0 + 0.08 + 0.04 ** 2))


def test_hf_weight_zero_amplitudes_is_one():
    assert spin_adapted_hf_weight(np.zeros((2, 3)), np.zeros((2, 2, 3, 3))) == 1.0


def test_noon_metrics():
    frac, mx = noon_metrics([2.0, 1.9, 1.0, 0.1, 0.0])
    assert frac == pytest.approx(0.0 + 0.1 + 1.0 + 0.1 + 0.0)
    assert mx == pytest.approx(1.0)
    assert noon_metrics([2.0, 0.0]) == (0.0, 0.0)


def test_recommend_regimes():
    thr = Thresholds()
    assert recommend({"w1_est": 0.95, "ccsd_ok": True, "noon_max": 0.05}, thr)[0] == "weak"
    assert recommend({"w1_est": 0.75, "ccsd_ok": True, "noon_max": 0.15}, thr)[0] == "moderate"
    assert recommend({"w1_est": 0.40, "ccsd_ok": True, "noon_max": 0.6}, thr)[0] == "strong"
    assert recommend({"w1_est": None, "ccsd_ok": False}, thr)[0] == "strong"


def test_strong_route_depends_on_fragments():
    d = {"w1_est": 0.3, "ccsd_ok": True, "noon_max": 0.9}
    assert recommend(d, fragments_separable=True)[1].startswith("LASSQD")
    assert "GENIE-MR" in recommend(d, fragments_separable=False)[1]
    assert "LASSQD" in recommend(d)[1] and "GENIE-MR" in recommend(d)[1]


def test_thresholds_are_adjustable():
    d = {"w1_est": 0.65, "ccsd_ok": True, "noon_max": 0.2}
    assert recommend(d, Thresholds())[0] == "moderate"
    assert recommend(d, Thresholds(strong_w1=0.7))[0] == "strong"
