# Reproducibility

Every number reported from this package must be re-runnable and must say how
much it can vary. This page states the rules; `docs/benchmarking.md` shows the
tooling that enforces them.

## 1. Tiers

| tier | methods | what to report |
|---|---|---|
| deterministic | HF, MP2, CCSD, CCSD(T), CASCI, FCI, DMET embedding | one value; identical across runs on the same machine |
| optimizer-dependent | CASSCF, NEVPT2, LASSCF | one value **plus** the caveat that a different stationary point is possible (seen: LASSCF on N₂ at 1.2 Å) |
| stochastic | SQD, SKQD, SqDRIFT, GQE, LASSQD | mean ± std over ≥ 3 seeds (5 for headline claims); never a single run |

## 2. Seeds — what each one controls

| solver | flag | controls |
|---|---|---|
| sqd, skqd, sqdrift | `--seed` | simulator shot sampling, configuration recovery, SqDRIFT circuit draws. **Not** the ansatz angles (fixed design choice). |
| lassqd | `--lassqd-seed` | sampler, batch subsampling, recovery |
| gqe | `--gqe-seed` | the transformer trainer (`trainer.seed`) |

Same seed ⇒ bit-identical energy (verified: LASSQD H4 seeds 1/2 reproduce to 8
digits across independent runs).

## 3. The pinned-seed trap (GQE)

`gqe-for-qsci/configs/trainer/default.yaml` pins `seed: 32`. Without
`--gqe-seed`, every "repeat" is the same computation. The original N₂ 2.1 Å
headline (GQE 0.153 mHa vs CIPSI 1.640 mHa) was produced this way. A 5-seed
reproduction (Oct 2026, `docs/results_log.md` §6) gave 47.4, 5.28, 5.11, 5.27,
0.155 mHa: the headline was one favourable run in five.

## 4. Provenance

Every benchmark row records: git SHA and dirty flag, package versions, thread
settings, hostname, SLURM ids, timestamp, full settings (`settings_json`), grid
parameters (`params_json`) and `rerun_cmd`. `--rebuild-rows` keeps the
original provenance when adding columns later.

`git_dirty=True` from an applied GQE patch is suppressed with
`git config submodule.gqe-for-qsci.ignore dirty`; untracked grid files also
count as dirty, so commit grids before running.

## 5. Reporting checklist

1. State the tier.
2. Stochastic: n seeds, mean ± std, and the spread (max − min).
3. Name the reference: `err_solver` (own exact limit) vs `err_vs_fci`.
4. For selection methods: compare at the **same determinant budget** against
   the oracle bound and CIPSI, and give w₁.
5. Point to the CSV row (run_id) or the grid file.
