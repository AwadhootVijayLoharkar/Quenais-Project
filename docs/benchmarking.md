# Benchmarking: running grids, reading results, making figures

Everything in the results chapter is produced by three tools:

| tool | what it does |
|---|---|
| `tools/run_benchmark.py` | runs a parameter grid; one CSV row per run; resumable |
| `tools/plot_benchmark.py` | figures (PNG + PDF) and `summary.md` tables from a results folder |
| `tools/gqe_vs_cipsi.py` | GQE vs oracle vs CIPSI at GQE's own budget, for plain `quenais-run` GQE folders |

Diagnostics used along the way: `tools/diagnose_lassqd_circuit.py` (where a
LASSQD circuit loses correlation), `tools/export_cas_hamiltonian.py` (plain CAS
Hamiltonian in step-2 format, no DMET).

---

## 1. Quick start

```bash
B=/home/loharkar/quenais-standalone/benchmark_results     # OUTSIDE the repo
python tools/run_benchmark.py benchmarks/h4_smoke.json --out $B/h4_smoke --dry-run
nohup python tools/run_benchmark.py benchmarks/h4_smoke.json --out $B/h4_smoke > $B/h4_smoke.log 2>&1 &
python tools/run_benchmark.py --out $B/h4_smoke --check
python tools/plot_benchmark.py $B/h4_smoke
cat $B/h4_smoke/figures/summary.md
```

Keep the output folder **outside** the repo, otherwise every row is marked
`git_dirty=True`.

## 2. Grid files (`benchmarks/*.json`)

```json
{
  "name": "example",
  "defaults": { "molecule": "N2", "basis": "sto-3g", "...": "..." },
  "studies": [
    {"name": "curve",
     "cases": [ {"geometry": "N 0 0 0; N 0 0 1.1", "tag_R": 1.1}, ... ],
     "sweep": {"solver": ["sqd", "lassqd"], "shots": [1000, 10000], "seed": [1, 2, 3]}}
  ]
}
```

- Any `quenais-run` flag, dashes → underscores (`avas_ao_labels`,
  `las_fragments`, `lassqd_carryover_eps`, `gqe_max_iters`, ...).
- `shots` and `seed` are generic: mapped to `--shots/--seed` (sqd, skqd,
  sqdrift), `--lassqd-shots/--lassqd-seed` (lassqd), `--gqe-seed` (gqe);
  dropped for lasscf.
- `tag_*` keys are labels copied to the CSV (`tag_R` = bond length, used by the
  curve plot); they are not part of the run identity.
- Flags that don't apply to a solver are dropped before hashing, so one grid can
  sweep all solvers and duplicates collapse (lasscf × 5 seeds = 1 run).
- `cases` is a zipped axis (e.g. geometry + its tag); `sweep` is a cartesian
  product.
- Grids that share `defaults` share systems: point a second grid at the same
  `--out` folder and steps 0–2 are reused (e.g. `n2_gqe.json` into
  `n2_curve/`).

Shipped grids: `h4_smoke`, `h4_lucj_pairs`, `h8_smoke`, `h8_lucj_opt`,
`gqe_smoke`, `gqe_h8`, `scf_quick`, `scf_carryover`, `n2_curve`, `n2_gqe`,
`h4_sqd_opt`, `h8_sqd_opt`, `n2_sqd_opt` (task 3a), `n2_gqe_tune` (task 3b).

**DMET+SQD with the optimised LUCJ** (`"solver": "sqd", "ansatz":
"lucj_opt"`): the step-2 embedding Hamiltonian is solved by the LASSQD
fragment solver as one fragment (`quenais/quantum/sqd_opt.py`). Tuning flags
(LASSQD defaults): `sqd_lucj_maxiter` 10, `sqd_lucj_pairs` heavy_hex,
`sqd_lucj_reps` 1, `sqd_batches` 15, `sqd_samples_per_batch` 170,
`sqd_iterations` 6, `sqd_carryover_eps` 1e-5. They are dropped for every
other solver/ansatz. Extra CSV columns: `lucj_energy_Ha`, `err_lucj_kcal`
(the circuit alone, no SQD), `lucj_off_hf_weight`, `n_2q_gates`;
`circuit_depth_max` is the depth in the {cx, rz, sx, x} basis.

A `null` in a sweep means "flag not given" (the default), so the run has the
same identity as a run in a grid that never mentioned the flag
(`n2_gqe_tune` reuses the finished default-setting runs of `n2_gqe`).

## 3. What the driver guarantees

- **Resumable.** Truth per run is `runs/<run_id>/row.json`. Re-running the same
  command skips finished runs; `--retry-failed` re-runs failed ones.
  `results.csv` is rebuilt from all row files.
- **Isolated.** Each run has its own project dir; steps 1 then 0 (so CASCI/FCI
  see the active space) run once per *system* in `systems/<id>/`, step 2 once
  per DMET reference in `systems/<id>/dmet_<ref>/`.
- **Crash-proof.** Each stage runs in a child process; a crash fails one run.
- **Re-runnable.** Every row carries `rerun_cmd` (the two `quenais-run`
  commands), the seed, `settings_json`, `params_json`, git SHA, package
  versions, thread settings, hostname, SLURM ids.

Layout:

```
<out>/results.csv        one row per run (~110 columns)
<out>/results_agg.csv    one row per configuration: mean/std/min/max over seeds
<out>/systems/<id>/      shared steps 0, 1 (+ dmet_<ref>/ step 2)
<out>/runs/<id>/         results/, run.log, run.json, row.json
<out>/figures/           after plot_benchmark.py
```

## 4. Key CSV columns

| column | meaning |
|---|---|
| `energy_Ha` | the solver's energy |
| `err_solver_kcal` | vs the solver's own exact limit: embedding FCI (DMET route), LASSCF (LASSQD) |
| `err_vs_casci_kcal`, `err_vs_fci_kcal`, `err_vs_best_ref_kcal` | vs molecular references |
| `w1` | weight of the largest exact CI amplitude of the embedding Hamiltonian |
| `err_oracle_kcal`, `err_cipsi_kcal` | best possible / CIPSI with the **same** number of determinants (DMET route) |
| `advantage_vs_cipsi_kcal` | `err_cipsi − err_solver`; > 0 means the solver beat CIPSI |
| `subspace_dim`, `full_dim`, `subspace_fraction` | diagonalised space vs full space |
| `valid_shot_fraction` | shots with the right electron numbers |
| `n_qubits`, `circuit_depth_max`, `transpiled_depth_max`, `n_diagonalisations` | cost |
| `n_macro`, `converged`, `grad_norm`, `loc_sv_min` | LAS convergence and fragment localisation |
| `energy_trace_json` | energy per iteration / macro cycle |
| `t_step3_s`, `t_sampling_s`, `t_diag_s` | wall times |
| `reproducibility` | stochastic / optimizer-dependent |

## 5. Checking and plotting

`--check` verifies required columns per solver, physics sanity (HF ≥ FCI,
CASCI ≥ FCI, variational vs own exact limit, subspace in (0,1]), and that seeds
change the result when the subspace is incomplete. Known false alarms:

- "LASSQD below LASSCF": LASSCF has several stationary points; LASSQD can land
  in a lower one (seen on N₂ 1.2 Å). Not a bug.
- "seeds identical with incomplete subspace" for single-determinant fragments
  (N₂ quartet fragments). Correct result, check too strict.
- "CCSD_T/CCSD/MP2 below FCI": non-variational methods at stretched geometry —
  a result, not an error.

`--rebuild-rows` re-derives every finished row from results on disk (adds new
columns, e.g. w₁/oracle/CIPSI) without re-running solvers; provenance columns
are kept from the original run.

`plot_benchmark.py <out> [<out> ...]` writes: `error_vs_shots`,
`subspace_vs_error`, `curve` (rows with `tag_R`), `las_convergence`,
`summary.md/.csv` (includes oracle/CIPSI/w₁ and non-parallelity errors).

## 6. Running on the erebos cluster

Put these in `~/.bashrc` (needed for GQE; harmless otherwise):

```bash
export B=/home/loharkar/quenais-standalone/benchmark_results
export GQE_QSCI_REPO_PATH=/home/loharkar/quenais-standalone/Quenais-Project/gqe-for-qsci
# mpi4py hangs at import (UCX netlink) on the cluster nodes; GQE is single-GPU
# and does not need UCX:
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=^ucx
export UCX_NET_DEVICES=lo
export PYTHONUNBUFFERED=1
```

and, once per checkout, `git config submodule.gqe-for-qsci.ignore dirty` (the
GQE patch modifies the submodule; without this every row is marked dirty).

Rules learned the hard way:

- Terminals opened before editing `.bashrc` don't have the variables; inside
  `nohup bash -c '...'` use full paths or export inside the quotes.
- `nohup ... > logfile 2>&1 &`: the log's folder must exist, and `--out` is a
  folder, not the log file.
- **GQE runs one at a time per checkout**: the trainer writes a config file
  into `gqe-for-qsci/`. Never start two drivers on the same `--out` without
  `--shard`.
- SLURM/parallel: `--shard I/N` per process, then `--collect-only`.
- `quenais-doctor` checks the whole environment; if it hangs at the MPI check,
  the UCX variables above are missing.

## 7. Cost reference (measured)

| run | time |
|---|---|
| SQD, H8 (16 q) / N₂ (20 q) | ~10 s |
| SKQD, H8 (16 q), 5 Krylov states | ~1 470 s, independent of shots |
| LASSQD optimised, H8 | ~60 s |
| LASSQD optimised, ScF | 600–950 s (carryover on); up to 2 500 s off |
| LASSQD optimised, N₂ doublet fragments | 140–1 500 s |
| GQE, 120 epochs, A100: H8 / N₂ DMET / N₂ CAS | ~860 s / ~720 s / ~720 s |
| GQE, 20 epochs, H4, RTX 4070 Ti | ~140 s |
