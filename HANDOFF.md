# QuEnAIS — benchmarking handoff (8 Oct 2026)

Paste this into a new chat together with the current repo zip. This chat is for
**running benchmarks and fixing code**. Thesis writing happens in a separate
chat (`THESIS_HANDOFF.md`).

## Who, deadline, working style

- Awadhoot Loharkar, Master's thesis, Fraunhofer ITWM (QuEnAIS). Submit
  20–24 Nov 2026 (hard deadline 1 Dec).
- Edits on a Windows laptop → commit → push → `git pull` on the cluster
  (erebos nodes, VS Code terminal). Grid JSON files are sometimes created
  on the cluster and committed from there; `git pull` on the laptop then.
- Deliver code as files in a tree mirroring the repo (or a zip of it), plus
  exact commands. Grids as attached `.json` files — **not** long heredocs to
  paste (pasting goes wrong).
- **Answers short, simple, step by step.** Lead with what to run and what
  output to expect. He pastes raw terminal output back.

## Read first

1. `docs/results_log.md` — everything run so far, the numbers, what they mean.
2. `docs/benchmarking.md` — driver, grids, CSV columns, plotting, cluster
   setup, pitfalls.
3. `docs/reproducibility.md` — tiers, seeds, reporting rules.
4. `docs/Physics_documentation/` — physics background (10 = LAS/LASSQD,
   06 = determinant selection, updated with the 5-seed GQE result).

## Cluster facts

- Results live outside the repo: `$B = ~/quenais-standalone/benchmark_results`
  (folders `h4_smoke`, `h4_lucj_pairs`, `h8_smoke`, `gqe_smoke`, `scf`,
  `n2_curve`, `headline`).
- GPUs: A100 nodes (erebos01, erebos05; shared, ask before assuming free),
  RTX 4070 Ti (makaria11, fine for ≤ ~28 qubits). Other GPUs are too old for
  CUDA-Q.
- `~/.bashrc` must export `B`, `GQE_QSCI_REPO_PATH`, the four Open MPI/UCX
  variables and `PYTHONUNBUFFERED` (see `docs/benchmarking.md` §6). Terminals
  opened before that don't have them; inside `nohup bash -c '…'` write paths
  and exports out in full.
- `git config submodule.gqe-for-qsci.ignore dirty` is set on the cluster
  checkout.
- DMRG (block2) does not work on this cluster (AMD + MKL); FCI/CASCI are the
  exact references.

## State of the code

All tests pass (`pytest -q -m "not slow"` → 379 passed). Since the previous
handoff: Qiskit step-3 fixes, `--seed`, benchmark driver with
w₁/oracle/CIPSI columns and `--rebuild-rows`, plotting, LASSQD circuit
diagnostic, `gqe_vs_cipsi.py`, unbuffered GQE output. Full list in
`docs/results_log.md`.

Known limitations (documented, not fixed):

- DMET+SQD samples a **random-angle** circuit and its recovery iterations are
  effectively one diagonalisation; it is a baseline, not a fair circuit
  comparison with LASSQD. Decision: leave as is, say so in the thesis.
- `--check` false alarms: LASSQD below LASSCF (multiple LASSCF solutions),
  single-determinant quartet fragments flagged as "seed not reaching sampler".
  Relax the check if convenient.
- LASSQD rows have no circuit-depth column.
- Use `--lassqd-lucj-optimize` for every LASSQD run (the unoptimised circuit
  keeps ~1.5 % of the correlation).
- GQE: one at a time per checkout; default 120 epochs, ngates 40.

## Headline status (most important)

The old abstract claimed DMET+GQE 0.153 mHa vs CIPSI 1.640 mHa at N₂ 2.1 Å.
Five-seed reproduction on the identical plain-CAS setup: 47.4, 5.28, 5.11,
5.27, **0.155** mHa. The claim was one favourable run in five; the median is
~3× worse than CIPSI. On the DMET-embedded N₂, GQE loses to CIPSI at every
geometry. Thesis is being reframed accordingly.

## The plan (Awadhoot's, agreed)

| step | status |
|---|---|
| 1. Test which method works where | mostly done: H4, H8, ScF, N₂ (STO-3G) |
| 2. Find where they have issues | done (table below) |
| **3. Optimise the methods** | **← start here** |
| 4. Bigger molecules, other bases, other parameters | after step 3 |

The research core is **DMET + quantum solvers (SQD, SKQD, GQE)**, plus LASSQD.
CIPSI/oracle are yardsticks, not the subject. Target: quantum solver within
chemical accuracy (1 kcal/mol) of the **exact embedding** answer.

| method | works on | issue found |
|---|---|---|
| DMET+SQD | H4 | random-angle circuit → N₂ 12–42 kcal/mol off |
| DMET+SKQD | H8 (1.0 kcal/mol) | ~1 470 s per run on 16 qubits, shot-independent cost |
| DMET+GQE | H8 (0.07) | N₂ (18–20 q) 46–95 off with default settings; seed-to-seed unreliable (1 in 5 good on plain CAS) |
| LASSQD | ScF, H8 exact; N₂ near dissociation | needs optimised LUCJ; fails where fragments are strongly bonded |

### Step 3 tasks, in order

**3a. DMET+SQD with an optimised circuit (code, ~1 day, CPU).** Reuse the
LASSQD machinery (`quenais/las/sqd_solver.py`: fragment mean field → LUCJ from
UCCSD → linear-method optimisation → sampling → SQD with K×d batching and
carryover) on the step-2 embedding Hamiltonian treated as ONE fragment
(h1s = [h1e, h1e], eri = h2e, nelec = (n_alpha, n_beta)). Energy = SQD energy +
ecore. Expose it as a new solver choice or an `--ansatz` option, record the same
CSV columns (subspace, valid shots, timings, seed). Validate: H4 and H8 must hit
the embedding FCI exactly; then run N₂ 1.1–2.5 Å, 3 seeds, 1k/10k shots.
Expected: large drop from 12–42 kcal/mol (LASSQD went 42 → 0 with this fix).
Watch: statevector LUCJ optimisation cost at 18–20 qubits (~10 orbitals) —
should be fine; check time.

**3b. GQE tuning on DMET N₂ (GPU, 2–3 days, in parallel with 3a).** Grid at
R = 1.8 and 2.1 Å into the existing `n2_curve` folder: `gqe_ngates` 40/80,
`gqe_max_iters` 120/300, seeds 1–3 (24 runs × ~12–30 min). Report error vs
embedding FCI per seed and best-of-3 (variational, so keeping the lowest
energy is legitimate). Operator-pool threshold (CCSD amplitudes are unreliable
at stretch: max|t₂| = 0.84 at 2.1 Å) is not a CLI flag yet — add
`--gqe-pool-threshold` if the sweep points there.

**3c. SKQD cheaper (optional).** Expose `skqd_krylov_dim`, `skqd_dt`,
`skqd_trotter_reps` as CLI flags; test fewer Krylov states on H8.

### Step 4 (after step 3)

N₂ in cc-pVDZ (DMET finally shrinks the problem: check `n_emb`; cap with
`cfg.dmet.max_embed_orbs` for GQE above ~30 qubits), Cu₂O₂ (needs geometry +
AVAS Cu 3d/O 2p; CASCI reference), Fe₂ (LASSCF reference −3655.496377 Ha),
cheap ablations (LASCI, bad fragmentation, AVAS vs APC).

## Hard constraints

1. Never copy code from `mrh` (GPL) or the LASSQD authors' unlicensed repo;
   never `import mrh`. Only numbers may be stored.
2. Never distribute a bundle containing block2 or PyCI (GPL-3.0).
3. IBM tokens from environment variables only.
