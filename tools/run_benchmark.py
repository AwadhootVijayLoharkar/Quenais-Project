#!/usr/bin/env python
"""
Benchmark driver: parameter grid -> one CSV row per run.

    python tools/run_benchmark.py benchmarks/h4_smoke.json --out ~/quenais_bench/h4_smoke --dry-run
    python tools/run_benchmark.py benchmarks/h4_smoke.json --out ~/quenais_bench/h4_smoke
    python tools/run_benchmark.py benchmarks/h4_smoke.json --out ~/quenais_bench/h4_smoke --check

WHAT IT GUARANTEES
------------------
* Resumable. Each run's truth is runs/<run_id>/row.json. A run with a row
  marked "ok" is never repeated; a killed job is resumed by re-running the
  same command. results.csv is *derived* (rebuilt from every row.json), so a
  crash mid-write cannot corrupt it.
* Isolated. Every run gets its own project dir, because every stage writes
  to fixed filenames and the cache check keys on molecule+basis only -- NOT
  geometry. Two points of a dissociation curve sharing a directory would
  silently reuse each other's step 0/1/2.
* Shared prep. Steps 1 and 0 (in that order, so CASCI/FCI see step 1's
  active space) run once per *system* (molecule, basis, geometry, active
  space, references) under a file lock, and step 2 once per DMET reference
  density. Runs copy those pickles in, then run step 3.
* Re-runnable. Every row carries the seed, the full settings, the git SHA,
  and `rerun_cmd`: the two quenais-run commands that reproduce it.
* Crash-proof. Each stage runs in a child process. block2 aborting, an OOM
  kill or a timeout fails that run and the driver moves on.

GRID FORMAT (JSON)
------------------
    {
      "name": "h4_smoke",
      "defaults": { <param>: <value>, ... },
      "studies": [
        {"name": "...",
         "fixed":  { <param>: <value> },               # optional
         "cases":  [ {<param>: <value>}, ... ],        # optional, zipped axis
         "sweep":  { <param>: [<v1>, <v2>, ...] }}     # cartesian product
      ]
    }

<param> is any quenais-run flag with dashes as underscores (basis,
geometry, avas_ao_labels, las_fragments, lassqd_carryover_eps, ...), plus:

    shots   per-circuit shots; mapped to --shots (sqd/skqd/sqdrift) or
            --lassqd-shots (lassqd); dropped for lasscf and gqe
    seed    mapped to --seed / --lassqd-seed / --gqe-seed; dropped for
            lasscf (deterministic)
    tag_*   free labels copied to the CSV (e.g. tag_R for a bond length);
            not part of the run identity

Flags that do not apply to a run's solver are dropped before the run is
hashed, so one grid can sweep every solver and duplicates collapse (lasscf
crossed with 5 shot counts x 5 seeds is ONE run).

SLURM: --shard I/N runs every N-th run starting at I (use
$SLURM_ARRAY_TASK_ID). Finish with --collect-only once all shards are done.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

HARTREE_TO_KCAL = 627.5094740631

DMET_SOLVERS = ("sqd", "skqd", "sqdrift", "gqe")
QISKIT_SOLVERS = ("sqd", "skqd", "sqdrift")
LAS_SOLVERS = ("lasscf", "lassqd")
STOCHASTIC = ("sqd", "skqd", "sqdrift", "gqe", "lassqd")

#: Parameters that define the *system* (steps 0 and 1). Everything else is
#: per run (step 2 reference, step 3).
SYSTEM_KEYS = (
    "molecule", "basis", "charge", "spin", "geometry", "xyz",
    "force_active_space", "active_space_method", "avas_ao_labels",
    "avas_threshold", "apc_max_size", "classical_methods",
    "fci_max_dets", "fci_in_active_space", "casci_nroots",
    "dmrg_bond_dims", "dmrg_sweeps", "dmrg_full_space", "dmrg_no_extrapolate",
    "dmrg_reorder", "dmrg_threads", "dmrg_scratch", "dmrg_keep_scratch",
)
DRIVER_OWNED = ("steps", "force", "project_dir", "no_scan", "no_quantum_scan")

DEFAULT_CLASSICAL = ["HF", "MP2", "CCSD", "CCSD_T", "CASCI", "FCI"]


# ═════════════════════════════════════════════════════════════════════════
# Grid expansion (pure; unit-tested in tests/test_benchmark_driver.py)
# ═════════════════════════════════════════════════════════════════════════

def _parser_dests():
    from quenais.cli import build_parser

    p = build_parser()
    return {a.dest: a for a in p._actions if a.dest != "help"}


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj, n=12):
    return hashlib.sha256(_canon(obj).encode()).hexdigest()[:n]


def normalise_params(raw):
    """
    Map the generic `shots`/`seed` onto the solver's own flag and drop every
    flag the solver does not read. Returns (params, tags).
    """
    params = {k: v for k, v in raw.items() if not k.startswith("tag_")}
    tags = {k: v for k, v in raw.items() if k.startswith("tag_")}
    solver = params.get("solver", "sqd")
    params["solver"] = solver

    shots = params.pop("shots", None)
    seed = params.pop("seed", None)
    if solver in QISKIT_SOLVERS:
        if shots is not None:
            params["shots"] = shots
        if seed is not None:
            params["seed"] = seed
    elif solver == "lassqd":
        if shots is not None:
            params["lassqd_shots"] = shots
        if seed is not None:
            params["lassqd_seed"] = seed
    elif solver == "gqe":
        if seed is not None:
            params["gqe_seed"] = seed

    def drop(prefixes):
        for k in [k for k in params if k.startswith(prefixes)]:
            del params[k]

    if solver not in LAS_SOLVERS:
        drop(("las_", "lassqd_"))
    # --sqd-* configure only the optimised-LUCJ SQD; anywhere else they
    # would only split identical runs into different run_ids.
    if not (solver == "sqd" and params.get("ansatz") == "lucj_opt"):
        drop(("sqd_",))
    if solver == "lasscf":
        drop(("lassqd_",))
    if solver != "gqe":
        drop(("gqe_", "cudaq_target"))
    if solver in LAS_SOLVERS or solver == "gqe":
        for k in ("ansatz", "mapping", "backend", "shots", "seed"):
            params.pop(k, None)
    if solver in LAS_SOLVERS:
        params.pop("dmet_reference", None)

    params.setdefault("classical_methods", list(DEFAULT_CLASSICAL))
    # None means "flag not given"; keep the identity independent of it.
    params = {k: v for k, v in params.items() if v is not None and v is not False}
    return params, tags


def expand_grid(grid, dests=None):
    """List of run dicts {run_id, system_id, config_id, study, params, tags}."""
    defaults = grid.get("defaults", {})
    runs, seen = [], {}
    for study in grid["studies"]:
        fixed = {**defaults, **study.get("fixed", {})}
        cases = study.get("cases") or [{}]
        sweep = study.get("sweep", {})
        keys = list(sweep)
        for case in cases:
            for values in itertools.product(*(sweep[k] for k in keys)):
                raw = {**fixed, **case, **dict(zip(keys, values))}
                params, tags = normalise_params(raw)
                if dests is not None:
                    bad = [k for k in params
                           if k not in dests or k in DRIVER_OWNED]
                    if bad:
                        raise ValueError(
                            f"study {study.get('name')!r}: unknown or "
                            f"driver-owned parameter(s) {bad}")
                run_id = short_hash(params)
                if run_id in seen:
                    continue
                sysp = {k: params[k] for k in SYSTEM_KEYS if k in params}
                cfgp = {k: v for k, v in params.items()
                        if k not in ("seed", "lassqd_seed", "gqe_seed")}
                run = {
                    "run_id": run_id,
                    "system_id": short_hash(sysp),
                    "config_id": short_hash(cfgp),
                    "study": study.get("name", ""),
                    "params": params,
                    "system_params": sysp,
                    "tags": tags,
                }
                seen[run_id] = run
                runs.append(run)
    return runs


def params_to_argv(params, dests=None):
    """quenais-run argv for a param dict (no --steps/--project-dir)."""
    argv = []
    for key in sorted(params):
        val = params[key]
        flag = "--" + key.replace("_", "-")
        if val is True:
            argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.extend([flag, str(val)])
    return argv


def rerun_command(params):
    """The two quenais-run calls that reproduce a run in an empty directory.
    Step 1 must precede step 0, or CASCI has no active space."""
    base = shlex.join(["quenais-run", *params_to_argv(params)])
    route_steps = "0 2 3" if params["solver"] in DMET_SOLVERS else "0 3"
    return f"{base} --steps 1 && {base} --steps {route_steps}"


def parse_shard(text):
    if not text:
        return 0, 1
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"--shard {text}: need 0 <= I < N")
    return i, n


# ═════════════════════════════════════════════════════════════════════════
# Filesystem layout
# ═════════════════════════════════════════════════════════════════════════

def system_dir(out, sid):
    return Path(out) / "systems" / sid


def dmet_dir(out, sid, ref):
    return system_dir(out, sid) / f"dmet_{ref}"


def run_dir(out, rid):
    return Path(out) / "runs" / rid


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


@contextlib.contextmanager
def file_lock(path):
    """flock: released by the kernel if the holder dies, so no stale locks."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ═════════════════════════════════════════════════════════════════════════
# Child processes
# ═════════════════════════════════════════════════════════════════════════

def _spawn(args, log_path, timeout):
    cmd = [sys.executable, os.path.abspath(__file__), *args]
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, time.perf_counter() - t0


def _log_tail(path, n=15):
    try:
        with open(path, errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def _cfg_for(params, project_dir, extra_argv=()):
    from quenais.cli import build_config, build_parser

    argv = params_to_argv(params) + ["--project-dir", str(project_dir),
                                     "--force", *extra_argv]
    args = build_parser().parse_args(argv)
    return build_config(args), args


def child_prep_system(out, sid):
    """Steps 1 then 0 in systems/<sid>/."""
    from quenais.cli import run_step

    sdir = system_dir(out, sid)
    spec = _read_json(sdir / "system.json")
    cfg, args = _cfg_for(spec["system_params"] | {"solver": "sqd"}, sdir)
    t = {}
    for step in (1, 0):
        t0 = time.perf_counter()
        print(f"\n===== system {sid}: step {step} =====", flush=True)
        run_step(step, cfg, args)
        t[f"step{step}"] = time.perf_counter() - t0
    _write_json(sdir / "prep.json", {"status": "ok", "timings_s": t})


def child_prep_dmet(out, sid, ref):
    from quenais.cli import run_step

    sdir, ddir = system_dir(out, sid), dmet_dir(out, sid, ref)
    (ddir / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy2(sdir / "results" / "step1_asf.pkl",
                 ddir / "results" / "step1_asf.pkl")
    spec = _read_json(sdir / "system.json")
    params = spec["system_params"] | {"solver": "sqd", "dmet_reference": ref}
    cfg, args = _cfg_for(params, ddir)
    t0 = time.perf_counter()
    run_step(2, cfg, args)
    _write_json(ddir / "prep.json", {"status": "ok",
                                     "timings_s": {"step2": time.perf_counter() - t0}})


#: Columns a rebuilt row keeps from the original run: they describe WHEN and
#: WITH WHAT the energy was computed, which a rebuild does not change.
KEEP_ON_REBUILD = ("quenais_version", "git_sha", "git_dirty", "timestamp_utc",
                   "hostname", "python", "platform", "cpu_count",
                   "slurm_job_id", "slurm_array_task_id", "versions_json",
                   "threads_json", "t_step3_s")


def child_run(out, rid, rebuild=False):
    """Step 3 in runs/<rid>/, then row.json.

    rebuild=True re-derives row.json from the step-3 results already on disk
    (no solver run), keeping the original provenance columns. Used to add
    columns introduced after a run finished.
    """
    import pickle

    from quenais.cli import run_step

    rdir = run_dir(out, rid)
    spec = _read_json(rdir / "run.json")
    old_row = _read_json(rdir / "row.json") if rebuild else None
    params, solver = spec["params"], spec["params"]["solver"]
    sdir = system_dir(out, spec["system_id"])
    res = rdir / "results"
    res.mkdir(parents=True, exist_ok=True)
    for name in (() if rebuild else ("step0_classical.pkl", "step1_asf.pkl")):
        shutil.copy2(sdir / "results" / name, res / name)
    if solver in DMET_SOLVERS and not rebuild:
        ref = params.get("dmet_reference", "casci")
        shutil.copy2(dmet_dir(out, spec["system_id"], ref) / "results"
                     / "step2_hamiltonian.pkl", res / "step2_hamiltonian.pkl")

    cfg, args = _cfg_for(params, rdir)
    if rebuild:
        t_step3 = (old_row or {}).get("t_step3_s")
    else:
        t0 = time.perf_counter()
        run_step(3, cfg, args)
        t_step3 = time.perf_counter() - t0

    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    step0 = load(res / "step0_classical.pkl")
    step1 = load(res / "step1_asf.pkl")
    step2 = load(res / "step2_hamiltonian.pkl") if solver in DMET_SOLVERS else None
    row = build_row(spec, cfg, step0, step1, step2, rdir, t_step3, load)
    if rebuild and old_row:
        for k in KEEP_ON_REBUILD:
            if k in old_row:
                row[k] = old_row[k]
        row["row_rebuilt_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(rdir / "row.json", row)
    print(f"\n[benchmark] row written: energy={row.get('energy_Ha')}  "
          f"err_vs_best_ref={row.get('err_vs_best_ref_kcal')} kcal/mol")


# ═════════════════════════════════════════════════════════════════════════
# Row construction
# ═════════════════════════════════════════════════════════════════════════

def _kcal(e, ref):
    if e is None or ref is None:
        return None
    return (float(e) - float(ref)) * HARTREE_TO_KCAL


def _settings_snapshot(cfg):
    import dataclasses

    out = {}
    for name in ("asf", "dmet", "qiskit", "gqe", "las", "ref"):
        obj = getattr(cfg, name, None)
        if obj is None:
            continue
        try:
            out[name] = dataclasses.asdict(obj)
        except TypeError:
            out[name] = dict(getattr(obj, "__dict__", {}))
    return out


def build_row(spec, cfg, step0, step1, step2, rdir, t_step3, load):
    from quenais.classical.runner import best_reference

    params, solver = spec["params"], spec["params"]["solver"]
    route = "dmet" if solver in DMET_SOLVERS else "las"
    m = step0.get("methods", {})
    e = lambda name: (m.get(name) or {}).get("energy")  # noqa: E731
    ref_name, e_ref = best_reference(m)

    row = {
        "status": "ok", "run_id": spec["run_id"], "system_id": spec["system_id"],
        "config_id": spec["config_id"], "study": spec["study"],
        **spec["tags"],
        "molecule": cfg.molecule, "basis": cfg.basis, "charge": cfg.charge,
        "spin": cfg.spin,
        "geometry": "; ".join(f"{a} {x:.6f} {y:.6f} {z:.6f}"
                              for a, (x, y, z) in cfg.geometry),
        "route": route, "solver": solver,
        "seed": params.get("seed", params.get("lassqd_seed", params.get("gqe_seed"))),
        # step 1
        "active_space_method": step1.get("selection_method"),
        "avas_ao_labels": json.dumps(params.get("avas_ao_labels")),
        "n_active_el": step1.get("nel"), "n_active_orb": step1.get("n_active_orbs"),
        "mo_list": json.dumps(list(map(int, step1.get("mo_list", [])))),
        "corr_strength": step1.get("corr_strength"), "tier_step1": step1.get("tier"),
        # step 0 references
        **{f"E_{k}_Ha": e(k) for k in ("HF", "MP2", "CCSD", "CCSD_T", "CASSCF",
                                       "NEVPT2", "CASCI", "FCI", "DMRG")},
        "fci_in_active_space": (m.get("FCI") or {}).get("in_active_space"),
        "best_ref_name": ref_name, "E_best_ref_Ha": e_ref,
        "t_step0_s": step0.get("total_time"),
    }

    if route == "dmet":
        row.update(_dmet_part(params, cfg, step2, rdir, load))
        t0 = time.perf_counter()
        row.update(selection_metrics(cfg, row.get("subspace_dim")))
        row["t_selection_s"] = time.perf_counter() - t0
    else:
        row.update(_las_part(params, cfg, load))

    E = row.get("energy_Ha")
    row["err_vs_casci_kcal"] = _kcal(E, e("CASCI"))
    row["err_vs_fci_kcal"] = _kcal(E, e("FCI"))
    row["err_vs_best_ref_kcal"] = _kcal(E, e_ref)
    if row.get("solver_ref_energy_Ha") is not None:
        row["err_solver_kcal"] = _kcal(E, row["solver_ref_energy_Ha"])
        row["variational_ok"] = bool(E >= row["solver_ref_energy_Ha"] - 1e-6)
    if row.get("err_cipsi_kcal") is not None and row.get("err_solver_kcal") is not None:
        # > 0: the solver beat classical selection at the same budget
        row["advantage_vs_cipsi_kcal"] = row["err_cipsi_kcal"] - row["err_solver_kcal"]
    sf = row.get("subspace_fraction")
    row["subspace_fraction"] = sf
    row["t_step3_s"] = t_step3
    row["reproducibility"] = "stochastic" if solver in STOCHASTIC else "optimizer-dependent"

    # Provenance
    from quenais.provenance import provenance

    prov = provenance(cfg)
    row.update({
        "quenais_version": prov.get("quenais_version"),
        "git_sha": (prov.get("git") or {}).get("sha"),
        "git_dirty": (prov.get("git") or {}).get("dirty"),
        "timestamp_utc": prov.get("timestamp_utc"),
        "hostname": socket.gethostname(),
        "python": prov.get("python"), "platform": prov.get("platform"),
        "cpu_count": prov.get("cpu_count"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "versions_json": json.dumps(prov.get("versions"), sort_keys=True),
        "threads_json": json.dumps(prov.get("threads"), sort_keys=True,
                                   default=str),
        "params_json": _canon(params),
        "settings_json": json.dumps(_settings_snapshot(cfg), sort_keys=True,
                                    default=str),
        "rerun_cmd": rerun_command(params),
        "run_dir": str(rdir),
    })
    return row


def selection_metrics(cfg, budget, max_det=2_000_000):
    """
    The thesis' reference points for a DMET-route run, on the embedding
    Hamiltonian of THIS run (step 2 pickle):

      w1            weight |c0|^2 of the largest exact CI amplitude
      oracle        best variational energy from ANY `budget` determinants
                    (top-|c| ranking of the exact vector, re-diagonalised)
      cipsi         classical perturbative selection grown to the same budget

    budget = the solver's diagonalised subspace dimension, so the three are
    compared at an identical determinant count. Never raises: a failure is
    recorded in `selection_error` and the row still writes.
    """
    out = {}
    try:
        from quenais.quantum.det_analysis import (
            casci_vector, n_for_weight, projected_energy, weight_curve)
        from quenais.quantum.det_expansion import cipsi_from_scratch
        from quenais.quantum.gqe_adapter import load_from_dmet_pickle

        mol = load_from_dmet_pickle(cfg.step2_file)
        e_exact, flat, space = casci_vector(mol)
        out["ndet_full"] = int(space.ndet)
        if space.ndet > max_det:
            out["selection_error"] = f"space of {space.ndet} determinants > {max_det}"
            return out
        order, cum = weight_curve(flat)
        out["w1"] = float(cum[0])
        nw = n_for_weight(cum, (0.99, 0.999))
        out["n_det_99pct"], out["n_det_999pct"] = nw[0.99], nw[0.999]
        if not budget:
            return out
        n = int(min(int(budget), space.ndet))
        out["n_det_budget"] = n
        if n >= space.ndet:
            out.update(E_oracle_Ha=e_exact, E_cipsi_Ha=e_exact,
                       err_oracle_kcal=0.0, err_cipsi_kcal=0.0)
            return out
        e_or = projected_energy(mol, order[:n], space=space)
        sel, _hist = cipsi_from_scratch(mol, n, space=space, verbose=False)
        e_ci = projected_energy(mol, sel, space=space)
        out.update(E_oracle_Ha=e_or, E_cipsi_Ha=e_ci,
                   n_det_cipsi=int(len(sel)),
                   err_oracle_kcal=(e_or - e_exact) * HARTREE_TO_KCAL,
                   err_cipsi_kcal=(e_ci - e_exact) * HARTREE_TO_KCAL)
    except Exception as exc:     # a reference point must never kill a row
        out["selection_error"] = repr(exc)[:300]
    return out


def _dmet_part(params, cfg, step2, rdir, load):
    from quenais.quantum.solver import embedding_fci_energy, solver_result_file

    solver = params["solver"]
    part = {
        "dmet_reference": cfg.dmet.reference,
        "n_imp": step2.get("n_imp"), "n_bath": step2.get("n_bath"),
        "n_emb": step2.get("n_emb"), "sv2_coverage": step2.get("sv2_cov"),
        "n_qubits": 2 * int(step2.get("n_emb")),
        "embedded_scf_vs_uhf_Ha": (step2.get("embedded_scf_check") or {}).get("delta"),
        "solver_ref_name": "DMET-embedding-FCI",
    }
    if solver in QISKIT_SOLVERS:
        r = load(solver_result_file(cfg))
        t = r.get("timings_s", {})
        part.update({
            "energy_Ha": r.get("energy"),
            "solver_ref_energy_Ha": r.get("e_exact_embedding"),
            "ansatz": r.get("ansatz"), "mapping": r.get("mapping"),
            "backend": r.get("backend"),
            "shots_per_circuit": r.get("shots_per_circuit"),
            "n_circuits": r.get("n_circuits"), "shots_total": r.get("shots_total"),
            "subspace_dim": r.get("subspace_dim"), "full_dim": r.get("full_dim"),
            "subspace_fraction": r.get("subspace_fraction"),
            "valid_shot_fraction": r.get("valid_shot_fraction"),
            "unique_valid_bitstrings": r.get("unique_valid_bitstrings"),
            "circuit_depth_max": r.get("circuit_depth_max"),
            "transpiled_depth_max": r.get("transpiled_depth_max"),
            "n_diagonalisations": r.get("n_diagonalisations"),
            "n_iterations": len(r.get("iterations", [])),
            "spin_sq": r.get("spin_sq"),
            "t_sampling_s": t.get("sampling"), "t_diag_s": t.get("diagonalisation"),
            "t_recovery_s": t.get("recovery"), "t_circuit_build_s": t.get("circuit_build"),
            "energy_trace_json": json.dumps([it.get("energy") for it in r.get("iterations", [])]),
            "subspace_trace_json": json.dumps([it.get("subspace_dim") for it in r.get("iterations", [])]),
        })
        lo = r.get("lucj_opt") or {}
        if lo:
            e_exact = r.get("e_exact_embedding")
            e_lucj = lo.get("lucj_energy")
            part.update({
                "lucj_energy_Ha": e_lucj,
                "err_lucj_kcal": _kcal(e_lucj, e_exact),
                "lucj_off_hf_weight": lo.get("lucj_off_hf_weight"),
                "n_2q_gates": lo.get("n_2q_gates"),
                "lucj_pairs": lo.get("lucj_pairs"),
                "lucj_n_reps": lo.get("lucj_n_reps"),
                "lucj_opt_maxiter": lo.get("lucj_opt_maxiter"),
                "sqd_batches": lo.get("sqd_batches"),
                "sqd_samples_per_batch": lo.get("sqd_samples_per_batch"),
                "sqd_iterations": lo.get("sqd_iterations"),
                "carryover_eps": lo.get("carryover_eps"),
                "t_lucj_prepare_s": lo.get("t_prepare_s"),
                "lucj_cache_hit": lo.get("lucj_cache_hit"),
            })
    else:   # gqe
        from quenais.visualization.plots import parse_gqe_log

        e_exact = embedding_fci_energy(step2)
        rows = parse_gqe_log(cfg.gqe_log_file)
        key_e = "Global-refined(best_so_far)/energy - R-CASCI"
        key_d = "Global-refined(best_so_far)/subspace_dim"
        errs = [r_.get(key_e) for r_ in rows if r_.get(key_e) is not None]
        last = rows[-1] if rows else {}
        energy = (e_exact + errs[-1]) if (errs and e_exact is not None) else None
        from math import comb
        full = comb(int(step2["n_emb"]), int(step2["n_alpha"])) * \
            comb(int(step2["n_emb"]), int(step2["n_beta"]))
        dim = last.get(key_d)
        part.update({
            "energy_Ha": energy, "solver_ref_energy_Ha": e_exact,
            "n_epochs": len(rows), "subspace_dim": dim, "full_dim": full,
            "subspace_fraction": (dim / full) if dim else None,
            "energy_trace_json": json.dumps([e_exact + x for x in errs]
                                            if e_exact is not None else []),
        })
    return part


def _las_part(params, cfg, load):
    r = load(cfg.las_result_file)
    hist = r.get("history", [])
    last = hist[-1] if hist else {}
    frags = last.get("fragments", []) or []
    sfs = [f["subspace_fraction"] for f in frags if "subspace_fraction" in f]
    vfs = [f["shots_valid_fraction"] for f in frags if "shots_valid_fraction" in f]
    s = r.get("settings", {})
    ncas = r.get("ncas_sub", [])
    svals = r.get("localisation_singular_values", [])
    part = {
        "energy_Ha": r.get("energy"),
        "las_fragments": json.dumps(s.get("fragments")),
        "orbital_optimization": s.get("orbital_optimization"),
        "converged": r.get("converged"), "n_macro": r.get("n_macro"),
        "grad_norm": r.get("grad_norm"),
        "n_fragments": len(ncas), "frag_norb_json": json.dumps(ncas),
        "loc_sv_min": min((min(v) for v in svals if v), default=None),
        "n_qubits": 2 * sum(ncas) if r["solver"] == "lassqd" else None,
        "n_qubits_max_fragment": 2 * max(ncas) if (ncas and r["solver"] == "lassqd") else None,
        "energy_trace_json": json.dumps([h.get("energy") for h in hist]),
        "grad_trace_json": json.dumps([h.get("grad_norm") for h in hist]),
    }
    if r["solver"] == "lassqd":
        dims = [f.get("subspace_dim") for f in frags]
        fulls = [f.get("full_dim") for f in frags]
        nfr = len(ncas)
        part.update({
            "backend": s.get("backend"),
            "shots_per_circuit": s.get("shots"),
            "n_circuits": len(hist),          # one glued job per macro cycle
            "shots_total": s.get("shots", 0) * len(hist),
            "sqd_batches": s.get("sqd_batches"),
            "sqd_samples_per_batch": s.get("sqd_samples_per_batch"),
            "sqd_iterations": s.get("sqd_iterations"),
            "carryover_eps": s.get("carryover_eps"),
            "subspace_fraction": min(sfs) if sfs else None,
            "subspace_fraction_mean": statistics.fmean(sfs) if sfs else None,
            "subspace_dim_json": json.dumps(dims), "full_dim_json": json.dumps(fulls),
            "valid_shot_fraction": statistics.fmean(vfs) if vfs else None,
            "n_diagonalisations": len(hist) * nfr * int(s.get("sqd_iterations", 0))
            * int(s.get("sqd_batches", 0)),
            "subspace_fraction_trace_json": json.dumps([
                [f.get("subspace_fraction") for f in h.get("fragments", [])]
                for h in hist]),
        })
    return part


# ═════════════════════════════════════════════════════════════════════════
# Collect: row.json files -> results.csv (+ LASSQD-vs-LASSCF join, aggregates)
# ═════════════════════════════════════════════════════════════════════════

LEAD_COLUMNS = [
    "status", "run_id", "config_id", "system_id", "study", "molecule", "basis",
    "route", "solver", "shots_per_circuit", "shots_total", "seed",
    "energy_Ha", "err_solver_kcal", "solver_ref_name", "err_vs_casci_kcal",
    "err_vs_fci_kcal", "err_vs_best_ref_kcal", "best_ref_name",
    "subspace_fraction", "subspace_dim", "full_dim", "valid_shot_fraction",
    "n_qubits", "circuit_depth_max", "n_diagonalisations", "t_step3_s",
]


def collect(out):
    rows = []
    for p in sorted((Path(out) / "runs").glob("*/row.json")):
        r = _read_json(p)
        if r is not None:
            rows.append(r)

    # LASSQD's own exact limit is LASSCF on the same system and fragments.
    las_ref = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("solver") == "lasscf":
            las_ref[(r["system_id"], r.get("las_fragments"),
                     r.get("orbital_optimization"))] = r
    for r in rows:
        if r.get("status") == "ok" and r.get("solver") == "lassqd":
            ref = las_ref.get((r["system_id"], r.get("las_fragments"),
                               r.get("orbital_optimization")))
            if ref is not None and r.get("energy_Ha") is not None:
                r["solver_ref_name"] = "LASSCF"
                r["solver_ref_energy_Ha"] = ref["energy_Ha"]
                r["err_solver_kcal"] = _kcal(r["energy_Ha"], ref["energy_Ha"])
                r["variational_ok"] = bool(r["energy_Ha"] >= ref["energy_Ha"] - 1e-6)

    _write_csv(Path(out) / "results.csv", rows, LEAD_COLUMNS)
    agg = aggregate(rows)
    _write_csv(Path(out) / "results_agg.csv", agg,
               ["config_id", "molecule", "basis", "solver", "shots_per_circuit",
                "n", "n_failed"])
    n_ok = sum(r.get("status") == "ok" for r in rows)
    print(f"[collect] {len(rows)} runs ({n_ok} ok, {len(rows) - n_ok} failed) "
          f"-> {Path(out) / 'results.csv'}")
    print(f"[collect] {len(agg)} configurations -> {Path(out) / 'results_agg.csv'}")
    return rows


def _write_csv(path, rows, lead):
    import csv

    keys = list(lead) + sorted({k for r in rows for k in r} - set(lead))
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    os.replace(tmp, path)


AGG_FIELDS = ("energy_Ha", "err_solver_kcal", "err_vs_casci_kcal",
              "err_vs_fci_kcal", "err_vs_best_ref_kcal", "subspace_fraction",
              "valid_shot_fraction", "t_step3_s")


def aggregate(rows):
    """Mean / std / min / max over seeds, per config_id (same params bar seed)."""
    groups = {}
    for r in rows:
        groups.setdefault(r.get("config_id"), []).append(r)
    out = []
    for cid, rs in groups.items():
        ok = [r for r in rs if r.get("status") == "ok"]
        head = ok[0] if ok else rs[0]
        a = {"config_id": cid, "n": len(ok), "n_failed": len(rs) - len(ok)}
        for k in ("molecule", "basis", "route", "solver", "shots_per_circuit",
                  "carryover_eps", "best_ref_name", "solver_ref_name", "study",
                  "system_id", *[k for k in head if k.startswith("tag_")]):
            a[k] = head.get(k)
        for f in AGG_FIELDS:
            vals = [float(r[f]) for r in ok
                    if r.get(f) is not None and not _isnan(r[f])]
            if vals:
                a[f"{f}_mean"] = statistics.fmean(vals)
                a[f"{f}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                a[f"{f}_min"] = min(vals)
                a[f"{f}_max"] = max(vals)
        a["seeds"] = json.dumps(sorted(r.get("seed") for r in ok
                                       if r.get("seed") is not None))
        out.append(a)
    return out


def _isnan(x):
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# ═════════════════════════════════════════════════════════════════════════
# Check: does the CSV carry what the figures need, and is it sane?
# ═════════════════════════════════════════════════════════════════════════

# E_FCI_Ha / E_CASCI_Ha are not required individually: FCI is impossible
# past ~16 orbitals (ScF), so "some exact reference exists" is the check.
REQUIRED_ALL = ["energy_Ha", "err_vs_best_ref_kcal", "best_ref_name",
                "E_best_ref_Ha", "n_active_el", "n_active_orb",
                "t_step3_s", "git_sha", "rerun_cmd", "settings_json",
                "params_json", "versions_json", "reproducibility"]
REQUIRED_BY_SOLVER = {
    "sqd": ["seed", "shots_per_circuit", "shots_total", "subspace_dim",
            "full_dim", "subspace_fraction", "valid_shot_fraction", "n_qubits",
            "circuit_depth_max", "n_diagonalisations", "err_solver_kcal",
            "t_sampling_s", "t_diag_s", "energy_trace_json"],
    "lassqd": ["seed", "shots_per_circuit", "shots_total", "subspace_fraction",
               "valid_shot_fraction", "n_qubits", "n_macro", "converged",
               "grad_norm", "carryover_eps", "err_solver_kcal",
               "energy_trace_json", "loc_sv_min"],
    "lasscf": ["n_macro", "converged", "grad_norm", "energy_trace_json",
               "loc_sv_min"],
    "gqe": ["seed", "subspace_fraction", "n_qubits", "err_solver_kcal",
            "n_epochs"],
}
REQUIRED_BY_SOLVER["skqd"] = REQUIRED_BY_SOLVER["sqd"]
REQUIRED_BY_SOLVER["sqdrift"] = REQUIRED_BY_SOLVER["sqd"]


def check(out):
    rows = collect(out)
    problems, notes = [], []
    failed = [r for r in rows if r.get("status") != "ok"]
    for r in failed:
        problems.append(f"run {r['run_id']} ({r.get('solver')}) FAILED: "
                        f"{(r.get('error') or '').strip().splitlines()[-1:]}")
    ok = [r for r in rows if r.get("status") == "ok"]

    # 1. Columns present and filled
    print("\n== 1. Required columns ==")
    for solver in sorted({r["solver"] for r in ok}):
        rs = [r for r in ok if r["solver"] == solver]
        need = REQUIRED_ALL + REQUIRED_BY_SOLVER.get(solver, [])
        missing = {k: sum(r.get(k) is None for r in rs) for k in need}
        missing = {k: v for k, v in missing.items() if v}
        status = "PASS" if not missing else "FAIL"
        print(f"  {solver:<8} {len(rs):>3} rows  {status}"
              + ("" if not missing else f"  empty: {missing}"))
        if missing:
            problems.append(f"{solver}: empty columns {missing}")

    # 2. Physics sanity
    print("\n== 2. Sanity ==")
    for r in ok:
        tag = f"{r['run_id']} {r['solver']}"
        fci, casci, hf = r.get("E_FCI_Ha"), r.get("E_CASCI_Ha"), r.get("E_HF_Ha")
        if fci is not None and casci is not None and casci < fci - 1e-8:
            problems.append(f"{tag}: CASCI {casci:.8f} below FCI {fci:.8f}")
        if fci is not None and hf is not None and hf < fci - 1e-8:
            problems.append(f"{tag}: HF below FCI")
        if r.get("variational_ok") is False:
            problems.append(f"{tag}: energy below its own exact limit "
                            f"({r.get('solver_ref_name')}) by "
                            f"{-r.get('err_solver_kcal', 0):.4f} kcal/mol")
        if r["route"] == "las" and fci is not None and r["energy_Ha"] < fci - 1e-6:
            problems.append(f"{tag}: LAS energy below full FCI")
        sf = r.get("subspace_fraction")
        if sf is not None and not 0 < sf <= 1 + 1e-12:
            problems.append(f"{tag}: subspace_fraction {sf} outside (0, 1]")
        if r.get("git_dirty"):
            notes.append(f"{tag}: git tree was dirty when this ran")
    for name in ("CCSD_T", "CCSD", "MP2"):
        below = {r["system_id"] for r in ok
                 if r.get(f"E_{name}_Ha") is not None and r.get("E_FCI_Ha") is not None
                 and r[f"E_{name}_Ha"] < r["E_FCI_Ha"] - 1e-6}
        if below:
            notes.append(f"{name} below FCI on {len(below)} system(s) -- not "
                         f"variational; expected at stretched geometry, and "
                         f"itself a result")

    # 3. Seeds actually vary something
    print("\n== 3. Seed repeats ==")
    groups = {}
    for r in ok:
        if r["solver"] in STOCHASTIC:
            groups.setdefault(r["config_id"], []).append(r)
    for cid, rs in sorted(groups.items(), key=lambda kv: (kv[1][0]["solver"],
                                                          str(kv[1][0].get("shots_per_circuit")))):
        es = [r["energy_Ha"] for r in rs]
        seeds = sorted(r.get("seed") for r in rs)
        spread = (max(es) - min(es)) * HARTREE_TO_KCAL if es else 0
        sfs = [r.get("subspace_fraction") or 0 for r in rs]
        p0 = json.loads(rs[0].get("params_json") or "{}")
        var = " ".join(f"{k.replace('lassqd_', '')}={p0[k]}" for k in
                       ("lassqd_carryover_eps", "lassqd_lucj_optimize",
                        "lassqd_lucj_pairs") if k in p0)
        msg = (f"  {rs[0]['solver']:<8} {var:<34} shots={rs[0].get('shots_per_circuit')!s:<7} "
               f"seeds={seeds}  spread={spread:.2e} kcal/mol  "
               f"subspace={min(sfs):.3f}-{max(sfs):.3f}")
        print(msg)
        if len(rs) > 1 and spread == 0 and min(sfs) < 1 - 1e-12:
            problems.append(f"{rs[0]['solver']} config {cid}: seeds give "
                            f"identical energies with an incomplete subspace "
                            f"-- the seed is not reaching the sampler")

    # 4. Summary table
    print("\n== 4. Summary (mean +- std over seeds) ==")
    print(f"  {'solver':<8} {'shots':>8} {'n':>3} {'err_solver kcal/mol':>22} "
          f"{'err_vs_best kcal/mol':>22} {'subspace':>9} {'t3 (s)':>8}")
    for a in sorted(aggregate(rows), key=lambda a: (a["solver"] or "",
                                                     a.get("shots_per_circuit") or 0)):
        def fmt(f):
            m_, s_ = a.get(f"{f}_mean"), a.get(f"{f}_std")
            return "-" if m_ is None else f"{m_:+.4f} +- {s_:.4f}"
        sf = a.get("subspace_fraction_mean")
        print(f"  {a['solver']:<8} {a.get('shots_per_circuit') or '-':>8} {a['n']:>3} "
              f"{fmt('err_solver_kcal'):>22} {fmt('err_vs_best_ref_kcal'):>22} "
              f"{'-' if sf is None else f'{sf:.3f}':>9} "
              f"{a.get('t_step3_s_mean', 0):>8.1f}")

    print("\n== Result ==")
    for n in notes:
        print(f"  note: {n}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    print("  CHECK PASSED" if not problems else f"  CHECK FAILED ({len(problems)} problem(s))")
    return 0 if not problems else 1


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════

def _fail_row(spec, error):
    return {"status": "failed", "run_id": spec["run_id"],
            "config_id": spec["config_id"], "system_id": spec["system_id"],
            "study": spec["study"], "solver": spec["params"]["solver"],
            "molecule": spec["params"].get("molecule"),
            "basis": spec["params"].get("basis"),
            "seed": spec["params"].get("seed", spec["params"].get(
                "lassqd_seed", spec["params"].get("gqe_seed"))),
            "shots_per_circuit": spec["params"].get(
                "shots", spec["params"].get("lassqd_shots")),
            **spec["tags"], "error": error,
            "params_json": _canon(spec["params"]),
            "rerun_cmd": rerun_command(spec["params"]),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname()}


def ensure_system(out, run, timeout):
    sid = run["system_id"]
    sdir = system_dir(out, sid)
    sdir.mkdir(parents=True, exist_ok=True)
    with file_lock(sdir / ".lock"):
        prep = _read_json(sdir / "prep.json")
        if prep is None or prep.get("status") != "ok":
            _write_json(sdir / "system.json", {"system_id": sid,
                                               "system_params": run["system_params"]})
            print(f"  [system {sid}] steps 1, 0 ...", flush=True)
            rc, dt = _spawn(["--_prep-system", str(out), sid],
                            sdir / "prep.log", timeout)
            prep = _read_json(sdir / "prep.json")
            if rc != 0 or prep is None:
                err = f"system prep rc={rc}\n{_log_tail(sdir / 'prep.log')}"
                _write_json(sdir / "prep.json", {"status": "failed", "error": err})
                return err
            print(f"  [system {sid}] ready ({dt:.0f} s)", flush=True)

        if run["params"]["solver"] in DMET_SOLVERS:
            ref = run["params"].get("dmet_reference", "casci")
            ddir = dmet_dir(out, sid, ref)
            ddir.mkdir(parents=True, exist_ok=True)
            dprep = _read_json(ddir / "prep.json")
            if dprep is None or dprep.get("status") != "ok":
                print(f"  [system {sid}] step 2 ({ref}) ...", flush=True)
                rc, dt = _spawn(["--_prep-dmet", str(out), sid, ref],
                                ddir / "prep.log", timeout)
                dprep = _read_json(ddir / "prep.json")
                if rc != 0 or dprep is None:
                    err = f"step 2 rc={rc}\n{_log_tail(ddir / 'prep.log')}"
                    _write_json(ddir / "prep.json", {"status": "failed", "error": err})
                    return err
    return None


def execute(out, runs, retry_failed, timeout):
    n = len(runs)
    for i, run in enumerate(runs, 1):
        rdir = run_dir(out, run["run_id"])
        prior = _read_json(rdir / "row.json")
        label = (f"[{i}/{n}] {run['run_id']} {run['params']['solver']:<7} "
                 f"shots={run['params'].get('shots', run['params'].get('lassqd_shots', '-'))} "
                 f"seed={run['params'].get('seed', run['params'].get('lassqd_seed', run['params'].get('gqe_seed', '-')))}")
        if prior and prior.get("status") == "ok":
            print(f"{label}  skip (done)")
            continue
        if prior and prior.get("status") == "failed" and not retry_failed:
            print(f"{label}  skip (failed before; --retry-failed to rerun)")
            continue
        if rdir.exists():
            shutil.rmtree(rdir)
        rdir.mkdir(parents=True)
        _write_json(rdir / "run.json", run)

        err = ensure_system(out, run, timeout)
        if err:
            _write_json(rdir / "row.json", _fail_row(run, err))
            print(f"{label}  FAILED in system prep (see systems/{run['system_id']}/)")
            continue
        rc, dt = _spawn(["--_run", str(out), run["run_id"]], rdir / "run.log", timeout)
        row = _read_json(rdir / "row.json")
        if rc != 0 or row is None:
            _write_json(rdir / "row.json",
                        _fail_row(run, f"rc={rc}\n{_log_tail(rdir / 'run.log')}"))
            print(f"{label}  FAILED rc={rc} ({dt:.0f} s) -> {rdir / 'run.log'}")
            continue
        e = row.get("err_solver_kcal")
        print(f"{label}  ok {dt:6.1f} s  E={row['energy_Ha']:.8f}  "
              f"err_solver={'-' if e is None else f'{e:+.4f}'} kcal/mol  "
              f"subspace={row.get('subspace_fraction')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("grid", nargs="?", help="grid JSON")
    ap.add_argument("--out", required=False, help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="list runs and exit")
    ap.add_argument("--shard", default=None, help="I/N: run every N-th run from I")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds per child (system prep or run)")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="collect, then verify columns and sanity")
    ap.add_argument("--_prep-system", nargs=2, dest="prep_system", help=argparse.SUPPRESS)
    ap.add_argument("--_prep-dmet", nargs=3, dest="prep_dmet", help=argparse.SUPPRESS)
    ap.add_argument("--_run", nargs=2, dest="child_run", help=argparse.SUPPRESS)
    ap.add_argument("--_row", nargs=2, dest="child_row", help=argparse.SUPPRESS)
    ap.add_argument("--rebuild-rows", action="store_true",
                    help="re-derive every ok row.json from results on disk "
                         "(adds new columns; no solver is re-run)")
    a = ap.parse_args(argv)

    try:
        if a.prep_system:
            child_prep_system(*a.prep_system)
            return 0
        if a.prep_dmet:
            child_prep_dmet(*a.prep_dmet)
            return 0
        if a.child_run:
            child_run(*a.child_run)
            return 0
        if a.child_row:
            child_run(*a.child_row, rebuild=True)
            return 0
    except Exception:
        traceback.print_exc()
        return 1

    if not a.out:
        ap.error("--out is required")
    out = Path(os.path.expanduser(a.out)).resolve()
    if a.collect_only:
        collect(out)
        return 0
    if a.rebuild_rows:
        done = sorted(p.parent.name for p in (out / "runs").glob("*/row.json")
                      if (_read_json(p) or {}).get("status") == "ok")
        print(f"[rebuild] {len(done)} ok runs")
        for i, rid in enumerate(done, 1):
            rc, dt = _spawn(["--_row", str(out), rid],
                            run_dir(out, rid) / "rebuild.log", a.timeout)
            print(f"  [{i}/{len(done)}] {rid}  {'ok' if rc == 0 else f'FAILED rc={rc}'}"
                  f"  ({dt:.0f} s)")
        collect(out)
        return 0
    if a.check:
        return check(out)
    if not a.grid:
        ap.error("grid JSON required")

    with open(a.grid) as fh:
        grid = json.load(fh)
    runs = expand_grid(grid, dests=_parser_dests())
    i, n = parse_shard(a.shard)
    mine = runs[i::n]
    systems = {r["system_id"] for r in runs}
    print(f"[benchmark] {grid.get('name', a.grid)}: {len(runs)} runs over "
          f"{len(systems)} system(s); this shard: {len(mine)}  -> {out}")

    if a.dry_run:
        for r in mine:
            p = r["params"]
            print(f"  {r['run_id']}  sys={r['system_id']}  {p['solver']:<7} "
                  f"shots={p.get('shots', p.get('lassqd_shots', '-'))!s:<7} "
                  f"seed={p.get('seed', p.get('lassqd_seed', p.get('gqe_seed', '-')))!s:<4} "
                  f"{json.dumps(r['tags']) if r['tags'] else ''}")
        print("\n  first run reproduces with:\n  " + rerun_command(mine[0]["params"]))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.grid, out / f"grid_{Path(a.grid).name}")
    execute(out, mine, a.retry_failed, a.timeout)
    collect(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
