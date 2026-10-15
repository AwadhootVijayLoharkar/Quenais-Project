# QuEnAIS — benchmarking handoff (15 Oct 2026)

Paste this into a new chat together with the current repo zip. This chat is for
**running benchmarks and fixing code**. Thesis writing happens in a separate
chat (`THESIS_HANDOFF.md`).

## Who, deadline, working style

- Awadhoot Loharkar, Master's thesis, Fraunhofer ITWM (QuEnAIS). Submit
  20–24 Nov 2026 (hard deadline 1 Dec). Wants a paper out of it if possible.
- Edits on a Windows laptop (files copied into WSL) → commit → push →
  `git pull` on the cluster. Delete `*:Zone.Identifier` files before committing.
- Deliver code as a **zip mirroring the repo** plus exact commands. Grids as
  attached `.json` files, never long heredocs.
- **Answers short, simple, step by step, plain words.** Lead with what to run
  and what output to expect. No week-by-week plans: he is in "doing" mode.
  He pastes raw terminal output back.

## Read first

1. `docs/results_log.md` — every run, numbers, meaning. **§"Step 3 continued
   (10–15 Oct)" and "What it means (updated 15 Oct)" are the current state.**
2. `docs/benchmarking.md` — driver, grids, tools, locks, sampler/memory, cluster.
3. `docs/reproducibility.md`, `docs/Physics_documentation/`.

## Cluster facts

- `$B = ~/quenais-standalone/benchmark_results`. Folders in use: `h8_smoke`,
  `n2_curve` (all STO-3G N₂: SQD, GQE, LAS), `n2_dz` (cc-pVDZ), `fe2bpym`.
- Two A100 allocations, both on node **erebos03**, called **W1 (job 57092)**
  and **W2 (job 57094)**; the prompt shows `[W1 job …]`. Each window only
  sees its own processes (`ps` in W1 does not show W2's jobs). Allocations
  last about a week.
- Slurm sets `SLURM_CPUS_PER_TASK=1`; `quenais/_threads.py` then pins numpy
  to 1 thread unless `OMP_NUM_THREADS` is exported. For heavy PySCF work:
  `export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYSCF_MAX_MEMORY=100000` (OpenBLAS >1 thread inside OpenMP → floods
  "Detect OpenMP Loop" warnings and oversubscribes).
- `python tools/status.py` (not `tools/status.py`: no exec bit) → progress,
  ETA, STALE? of every `$B/*.log`.
- `python tools/show_rows.py $B/<folder> --where k=v … --cols …` (default
  filter solver=sqd ansatz=lucj_opt; `--where` with nothing = all rows;
  `--agg`, `--range`). Rows reach results.csv only at the end of a grid or
  with `python tools/run_benchmark.py --out $B/<folder> --collect-only`.
- Every run holds a lock (`<out>/locks/<id>.lock`); a second driver on the
  same folder skips it. Restart a killed grid with `--retry-failed`.
- `QUENAIS_SELECTION_MAX_DET=4000000` to get CIPSI/oracle columns above 2 M
  determinants (needed for cc-pVDZ).
- GQE: one run at a time per gqe-for-qsci checkout. After changing
  `quenais/patches/gqe_dmet_source.patch`: in `$GQE_QSCI_REPO_PATH`
  `git checkout -- gqe_qsci/factory.py gqe_qsci/gqe/sampler.py
  gqe_qsci/train_pipeline.py`, `rm .quenais_patch_applied`, then
  `quenais-gqe-setup --repo $GQE_QSCI_REPO_PATH`.
- DMRG (block2) does not work on this cluster; FCI/CASCI are the references.

## State of the code

`pytest -q -m "not slow"` → **401 passed** on the cluster. New since 8 Oct
(full list: results_log "Code changes" 9–12):

- `--ansatz lucj_opt` for DMET+SQD (`quenais/quantum/sqd_opt.py`): LASSQD's
  optimised LUCJ on the embedding Hamiltonian, cached per geometry, flags
  `--sqd-lucj-pairs/-reps/-maxiter`, `--sqd-batches/-samples-per-batch/
  -iterations/-carryover-eps`, `--sqd-explore-fraction/-scale`. Automatic
  sampler (statevector ≤ 28 qubits) and memory plan.
- GQE `--gqe-pool-angles`, `--gqe-pool-threshold`, determinant dumps,
  `tools/gqe_ensemble.py`.
- `tools/status.py`, `tools/show_rows.py`, run locks.

## Where we are (one screen)

| method | result | status |
|---|---|---|
| DMET+SQD, optimised LUCJ (3a) | N₂ STO-3G 1–3 kcal/mol at 10k shots, 0.2–0.5 at 100k; bad at 1.6–1.8 Å | done |
| SQD vs CIPSI at same size | CIPSI 10–1000× better (STO-3G), 5–25× (cc-pVDZ) | done — key finding |
| Exploration shots | always exact, because the subspace fills the whole space | done — "filling, not selection" |
| GQE (3b) | CCSD-angle plateau found; free angles 81→75 (1.8 Å), 40.5→28.7 (2.1 Å) | done, still worst solver |
| GQE+LAS | not worth it | **decided: skip, outlook only** |
| SKQD cheaper (3c) | – | dropped |
| N₂ cc-pVDZ | SQD/GQE/LAS rows in `n2_dz` | done except 2.1 Å exploration rows (check) |
| [Fe₂(bpym)]⁴⁺ LAS | restarted 13 Oct night with 24 threads, step 0 done | **running — check first** |

## Next, in order

1. **Collect** (no code):
   - Fe, in **W2**: `python tools/status.py`, `tail -5 $B/fe2bpym.log`,
     `grep -v OpenBLAS $B/fe2bpym/systems/*/prep.log | tail -5`,
     then `python tools/show_rows.py $B/fe2bpym --where --cols solver
     las_fragments seed energy_Ha err_solver_kcal n_macro t_step3_s`.
     Compare LASSCF with −3655.496377 Ha.
   - n2_dz: `python tools/show_rows.py $B/n2_dz --where solver=sqd tag_R=2.1
     --cols shots_per_circuit explore_fraction subspace_fraction
     err_solver_kcal err_cipsi_kcal t_step3_s`.
   - Put the numbers into results_log (§Fe and the cc-pVDZ table).
2. **Step 4 — determinant-list SQD (code, ~½ day).** New option
   `--sqd-subspace product|list` (default product). `list` keeps only the
   sampled (α,β) determinant pairs (after configuration recovery) instead of
   the α×β product, and diagonalises with `det_analysis.projected_energy`.
   Then error vs subspace size, directly comparable to the CIPSI column at
   equal size. Test: on H4 with all determinants sampled it must equal FCI;
   with few samples it must stay variational. Grid: N₂ STO-3G 1.1/1.8/2.5 Å,
   10k/100k, 3 seeds, into `n2_curve`. This answers "is quantum selection
   worse than CIPSI" without the product-space filling effect.
3. **Step 5 — noise model (code, ~½ day).** Aer sampler with a fake IBM
   backend noise model (`qiskit_ibm_runtime.fake_provider`), flag
   `--backend noisy`; one N₂ geometry, 10k shots, 3 seeds; report error and
   valid-shot fraction vs noiseless.
4. Thesis: results chapter from results_log (separate chat).
5. Optional, only with free GPU time: `benchmarks/n2_gqe_angles_ext.json`
   (free angles at 1.1/2.5 Å; needs a second gqe-for-qsci checkout
   `gqe-for-qsci-2` with `.git/info/exclude`).

## Hard constraints

1. Never copy code from `mrh` (GPL) or the LASSQD authors' unlicensed repo;
   never `import mrh`. Only numbers may be stored.
2. Never distribute a bundle containing block2 or PyCI (GPL-3.0).
3. IBM tokens from environment variables only.
