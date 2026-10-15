"""
Hardware-like noise for the sampler (--backend noisy, step 5).

WHAT IS SIMULATED
-----------------
The circuit is transpiled to a FAKE IBM DEVICE (qiskit_ibm_runtime.
fake_provider: a snapshot of a real machine's qubit connectivity, native
gates and calibrated error rates) -- so routing SWAPs on the heavy-hex
lattice are included -- and sampled with Qiskit Aer under a noise model:

  "depolarizing" (default)
      device-AVERAGE error rates (median over qubits / qubit pairs),
      multiplied by noise_scale:
        1q gates   depolarizing, lambda = 2 r1
        2q gates   depolarizing, lambda = 4/3 r2
        readout    symmetric bit flip, p = r_ro
      (r = reported gate error = average infidelity; lambda = r d/(d-1)).
      noise_scale = 1 is today's device, 0.1 a 10x better one, 0 ideal
      gates on the device layout (isolates the effect of routing alone).
  "device"
      NoiseModel.from_backend: per-qubit errors, T1/T2 relaxation, as
      calibrated. noise_scale must be 1.

DIAGNOSTICS (noise_info)
------------------------
n_2q_device, depth_device   after routing to the device
est_fidelity                prod (1 - r) over every gate and measurement:
                            the chance that a shot sees no error at all.
                            Near 0 means the samples are mostly noise.

Cost: Aer simulates noise shot by shot (trajectories), so time grows
with shots x gates x 2^n_qubits. Everything heavy is imported inside
functions; this module imports without Qiskit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NOISE_MODELS", "fake_backend", "device_error_rates",
           "depolarizing_lambda", "build_noise_model", "estimated_fidelity",
           "noisy_sampler_setup"]

NOISE_MODELS = ("depolarizing", "device")
TWO_QUBIT_GATES = ("ecr", "cz", "cx")
ONE_QUBIT_GATES = ("sx", "x")


def fake_backend(name):
    """An instance of qiskit_ibm_runtime.fake_provider.<name>."""
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as exc:
        raise ImportError(
            "--backend noisy needs qiskit-ibm-runtime for the fake devices: "
            "pip install 'qiskit-ibm-runtime>=0.23'") from exc
    cls = getattr(fake_provider, name, None)
    if cls is None:
        have = sorted(n for n in dir(fake_provider) if n.startswith("Fake"))
        raise ValueError(f"unknown fake device {name!r}; available: {have}")
    return cls()


def _errors(target, name):
    if name not in target.operation_names:
        return []
    out = []
    for _qargs, props in target[name].items():
        if props is not None and getattr(props, "error", None) is not None:
            out.append(float(props.error))
    return out


def device_error_rates(target):
    """Median gate / readout errors of a Target (or any object with
    .operation_names and [name] -> {qargs: props})."""
    two = [g for g in TWO_QUBIT_GATES if g in target.operation_names]
    if not two:
        raise ValueError(f"device has no two-qubit gate among {TWO_QUBIT_GATES}")
    e2 = [e for g in two for e in _errors(target, g)]
    e1 = [e for g in ONE_QUBIT_GATES for e in _errors(target, g)]
    ro = _errors(target, "measure")
    med = lambda v: float(np.median(v)) if v else 0.0  # noqa: E731
    return {"gates_2q": two, "r1": med(e1), "r2": med(e2), "r_ro": med(ro)}


def depolarizing_lambda(r, n_qubits):
    """Depolarizing parameter for average gate infidelity r: r d/(d-1)."""
    d = 2 ** n_qubits
    return min(float(r) * d / (d - 1), 1.0)


def build_noise_model(backend, model="depolarizing", scale=1.0):
    """(NoiseModel, rates actually used)."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    if model not in NOISE_MODELS:
        raise ValueError(f"noise model must be one of {NOISE_MODELS}")
    rates = device_error_rates(backend.target)
    if model == "device":
        if abs(scale - 1.0) > 1e-12:
            raise ValueError("noise_model='device' uses the calibrated rates; "
                             "noise_scale must be 1")
        return NoiseModel.from_backend(backend), rates
    used = {"gates_2q": rates["gates_2q"], "r1": scale * rates["r1"],
            "r2": scale * rates["r2"], "r_ro": min(scale * rates["r_ro"], 0.5)}
    nm = NoiseModel()
    if used["r1"] > 0:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(depolarizing_lambda(used["r1"], 1), 1),
            list(ONE_QUBIT_GATES))
    if used["r2"] > 0:
        nm.add_all_qubit_quantum_error(
            depolarizing_error(depolarizing_lambda(used["r2"], 2), 2),
            list(used["gates_2q"]))
    if used["r_ro"] > 0:
        p = used["r_ro"]
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm, used


def estimated_fidelity(op_counts, rates, n_measured):
    """prod (1 - r): probability that a shot sees no gate or readout error."""
    n1 = sum(int(op_counts.get(g, 0)) for g in ONE_QUBIT_GATES)
    n2 = sum(int(op_counts.get(g, 0)) for g in rates["gates_2q"])
    logf = (n1 * np.log1p(-min(rates["r1"], 0.999999))
            + n2 * np.log1p(-min(rates["r2"], 0.999999))
            + n_measured * np.log1p(-min(rates["r_ro"], 0.999999)))
    return float(np.exp(logf)), n1, n2


def noisy_sampler_setup(circuit, settings):
    """
    Transpile `circuit` to the fake device and build the noisy simulator
    options. Returns (transpiled circuit, Aer backend options, info dict).
    """
    import ffsim
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    dev = fake_backend(settings.noise_device)
    nm, rates = build_noise_model(dev, settings.noise_model,
                                  float(settings.noise_scale))
    pm = generate_preset_pass_manager(
        optimization_level=int(settings.ibm_optimization_level), backend=dev,
        seed_transpiler=12345)
    pm.pre_init = ffsim.qiskit.PRE_INIT
    tqc = pm.run(circuit)
    ops = dict(tqc.count_ops())
    n_meas = int(ops.get("measure", 0))
    fid, n1, n2 = estimated_fidelity(ops, rates, n_meas)
    info = {
        "noise_device": settings.noise_device,
        "noise_model": settings.noise_model,
        "noise_scale": float(settings.noise_scale),
        "noise_r1": rates["r1"], "noise_r2": rates["r2"],
        "noise_r_ro": rates["r_ro"],
        "n_1q_device": n1, "n_2q_device": n2,
        "depth_device": int(tqc.depth()),
        "est_fidelity": fid,
    }
    opts = {"method": "statevector", "noise_model": nm}
    return tqc, opts, info
