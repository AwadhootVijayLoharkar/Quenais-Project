"""
Noisy sampler (--backend noisy, quenais/quantum/noise.py).

Pure parts (rate extraction, depolarizing conversion, fidelity estimate,
settings/CLI/driver wiring) run without Qiskit. The needs_qiskit test
samples a 4-qubit circuit on FakeSherbrooke through the production
sampler: noise_scale 0 -> only the prepared bitstring, large scale ->
other bitstrings and wrong electron numbers appear.
"""

from __future__ import annotations

import numpy as np
import pytest

from quenais.quantum import noise
from quenais.settings import LasSettings, QiskitSolverSettings


class _Props:
    def __init__(self, e):
        self.error = e


class _Target:
    """Minimal stand-in for qiskit.transpiler.Target."""

    def __init__(self, table):
        self._t = table
        self.operation_names = list(table)

    def __getitem__(self, name):
        return self._t[name]


def _target():
    return _Target({
        "sx": {(0,): _Props(2e-4), (1,): _Props(4e-4), (2,): _Props(3e-4)},
        "x": {(0,): _Props(2e-4), (1,): _Props(4e-4), (2,): _Props(3e-4)},
        "ecr": {(0, 1): _Props(6e-3), (1, 2): _Props(1e-2), (2, 1): None},
        "measure": {(0,): _Props(1e-2), (1,): _Props(3e-2), (2,): _Props(2e-2)},
        "rz": {(0,): _Props(0.0)},
    })


def test_device_rates_are_medians():
    r = noise.device_error_rates(_target())
    assert r["gates_2q"] == ["ecr"]
    assert np.isclose(r["r1"], 3e-4)
    assert np.isclose(r["r2"], 8e-3)
    assert np.isclose(r["r_ro"], 2e-2)


def test_no_two_qubit_gate_is_an_error():
    with pytest.raises(ValueError):
        noise.device_error_rates(_Target({"sx": {(0,): _Props(1e-4)}}))


def test_depolarizing_lambda():
    assert np.isclose(noise.depolarizing_lambda(1e-3, 1), 2e-3)
    assert np.isclose(noise.depolarizing_lambda(3e-3, 2), 4e-3)
    assert noise.depolarizing_lambda(0.9, 1) == 1.0


def test_estimated_fidelity():
    rates = {"gates_2q": ["ecr"], "r1": 1e-3, "r2": 1e-2, "r_ro": 2e-2}
    f, n1, n2 = noise.estimated_fidelity({"sx": 10, "x": 2, "ecr": 50, "rz": 99},
                                         rates, n_measured=4)
    assert (n1, n2) == (12, 50)
    assert np.isclose(f, 0.999 ** 12 * 0.99 ** 50 * 0.98 ** 4)
    f0, _, _ = noise.estimated_fidelity({"ecr": 50}, {**rates, "r1": 0, "r2": 0,
                                                      "r_ro": 0}, 4)
    assert f0 == 1.0


def test_settings_cli_and_driver():
    from quenais.cli import build_parser, build_qiskit_settings
    from quenais.quantum.sqd_opt import choose_sampler, las_settings_from_qiskit

    with pytest.raises(ValueError):
        QiskitSolverSettings(backend="noisy", ansatz="lucj").validate()
    with pytest.raises(ValueError):
        QiskitSolverSettings(backend="noisy", ansatz="lucj_opt",
                             noise_scale=-1).validate()
    with pytest.raises(ValueError):
        LasSettings(fragments=["0:2:1,1"], noise_model="thermal").validate()

    args = build_parser().parse_args([
        "--molecule", "LiH", "--solver", "sqd", "--ansatz", "lucj_opt",
        "--backend", "noisy", "--noise-device", "FakeTorino",
        "--noise-model", "depolarizing", "--noise-scale", "0.1"])
    q = build_qiskit_settings(args).validate()
    assert (q.backend, q.noise_device, q.noise_scale) == ("noisy", "FakeTorino", 0.1)
    s = las_settings_from_qiskit(q)
    assert (s.backend, s.noise_device, s.noise_scale) == ("aer_noisy", "FakeTorino", 0.1)
    assert choose_sampler("noisy", 20) == "aer_noisy"

    import run_benchmark as d
    p, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj_opt",
                               "backend": "noisy", "noise_scale": 0.1})
    assert p["noise_scale"] == 0.1
    p, _ = d.normalise_params({"solver": "sqd", "ansatz": "lucj_opt",
                               "backend": "mps", "noise_scale": 0.1})
    assert "noise_scale" not in p


@pytest.mark.needs_qiskit
def test_noisy_sampler_end_to_end():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("ffsim")
    from qiskit import QuantumCircuit

    from quenais.las import sqd_solver

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(2)            # one alpha (qubit 0) and one beta (qubit 2) electron
    res = {}
    for scale in (0.0, 20.0):
        s = LasSettings(fragments=["0:2:1,1"], backend="aer_noisy", shots=2000,
                        seed=1, noise_scale=scale, ibm_optimization_level=1)
        counts = sqd_solver.sample_fragment_circuits([qc], s)[0]
        res[scale] = (counts, dict(sqd_solver.LAST_SAMPLING_INFO))
    c0, i0 = res[0.0]
    assert list(c0) == ["0101"] and i0["est_fidelity"] == 1.0
    c1, i1 = res[20.0]
    assert len(c1) > 1 and c1.get("0101", 0) < 2000
    assert i1["est_fidelity"] < 1.0 and i1["noise_scale"] == 20.0
