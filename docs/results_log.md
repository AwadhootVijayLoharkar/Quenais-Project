# Benchmark results log

Dated record of the October 2026 benchmark campaign: what was run, what came
out, what it means. Raw data: `results.csv` in each benchmark folder
(`~/quenais-standalone/benchmark_results/<name>/`); grids in `benchmarks/`;
how to run: `docs/benchmarking.md`.

## What was run

| date | grid | system | runs | purpose |
|---|---|---|---|---|
| 5 Oct | `h4_smoke` | H4 STO-3G full space | 13 | end-to-end validation, every solver hits its exact limit |
| 5 Oct | `h4_lucj_pairs` | H4 | 6 | why LASSQD sampled only the HF determinant |
| 5–6 Oct | `h8_smoke`, `h8_lucj_opt`, `gqe_h8` | H8 STO-3G, 16 qubits | 28 | first full solver comparison incl. GQE, oracle, CIPSI |
| 6–7 Oct | `scf_quick`, `scf_carryover` | ScF STO-3G (8e,9o) | 81 | LASSQD carryover study, 5 seeds |
| 6–8 Oct | `n2_curve`, `n2_gqe` | N₂ STO-3G (10e,8o), 1.0–2.5 Å | 117 | dissociation curve: DMET+SQD, DMET+GQE, LASSCF, LASSQD (two fragment spins) |
| 8 Oct | headline reproduction (`export_cas_hamiltonian.py` + `gqe_vs_cipsi.py`) | N₂ 2.1 Å plain CAS | 5 | re-test the thesis headline with independent seeds |
| 8–9 Oct | `h4_sqd_opt(_diag)`, `h8_sqd_opt(_d)`, `n2_sqd_opt` | H4, H8, N₂ STO-3G | ~150 | task 3a: DMET+SQD with the optimised LUCJ |
| 8–9 Oct | `n2_gqe_tune` | N₂ 1.8/2.1 Å | 24 | task 3b: GQE gates × epochs |
| 9–13 Oct | `h8_sqd_explore`, `n2_sqd_explore` | H8, N₂ STO-3G | 30 + 72 | exploration shots and 100k shots |
| 9–13 Oct | `n2_dz_cpu`, `n2_dz_gpu`, `n2_dz_sqd` | N₂ cc-pVDZ, 26 qubits | 16 + 12 | realistic basis: LAS, GQE, SQD |
| 13 Oct | `n2_gqe_ens` | N₂ 1.8/2.1 Å | 10 | GQE seeds 4–8 with determinant dumps |
| 13–14 Oct | `n2_gqe_angles` | N₂ 1.8/2.1 Å | 12 | GQE with free rotation angles instead of CCSD angles |
| 13– Oct | `fe2bpym_las` | [Fe₂(bpym)]⁴⁺ 6-31G | – | first large LAS system (running / see §Step 3 cont.) |

## Code changes made during the campaign

1. `--solver sqd|skqd|sqdrift` crashed at step 3 (settings read from the wrong
   object); no test covered it. Fixed + regression test.
2. SQD/SKQD/SqDRIFT shared one result file behind an existence-only cache, so a
   second solver silently returned the first one's result. Fixed.
3. `--shots` never reached SKQD/SqDRIFT. Fixed. Added `--seed`.
4. Alpha/beta bit order swapped in the Qiskit solvers (zero valid shots for
   open shells) and SqDRIFT injected the wrong "HF" determinant. Fixed.
5. SKQD with the default Bravyi–Kitaev mapping reads parity bits as
   occupations, so it samples the wrong determinants. Now requires JW.
6. **LASSQD's LUCJ circuit built from CCSD amplitudes keeps ~1.5 % of the
   correlation** (diagnosed, see §4). Fix: optimise the LUCJ parameters
   (`--lassqd-lucj-optimize`). All LASSQD results below use it unless marked.
7. Earlier GQE "repeats" were one run three times: the external trainer pins
   `seed: 32`. Fixed with `--gqe-seed`.
8. Cluster: CUDA-Q/mpi4py hung at import (UCX network layer); fixed with
   OpenMPI settings disabling UCX. Not a thesis point, just methods detail.
9. Task 3a: `--ansatz lucj_opt` for DMET+SQD -- the LASSQD circuit and SQD
   (UCCSD-initialised LUCJ, linear-method optimisation, K x d batches,
   carryover) on the embedding Hamiltonian as one fragment. Optimised LUCJ
   cached per geometry (`~/.cache/quenais/lucj`). Results: §Step 3 below.
10. SQD tuning flags: `--sqd-batches/-samples-per-batch/-iterations/
    -carryover-eps`, `--sqd-lucj-pairs heavy_hex|all`, `--sqd-lucj-reps`,
    `--sqd-lucj-maxiter` (0 = no optimisation), exploration shots
    `--sqd-explore-fraction/-scale` (part of the shot budget sampled from a
    random shallow circuit). Automatic sampler choice (statevector up to 28
    qubits, MPS above) and an automatic memory plan that switches the LUCJ
    optimisation off when it would not fit (N₂ cc-pVDZ needed ~74 GB).
11. GQE: `--gqe-pool-angles` (free rotation angles instead of 2·t_CCSD),
    `--gqe-pool-threshold` (also fixed: the pool silently ignored the
    threshold). The patched trainer dumps the best determinant set to
    `results/gqe_determinants.pkl`; `tools/gqe_ensemble.py` reads the dumps,
    re-diagonalises unions of seeds, and shows which excitation levels are
    captured.
12. Driver/tools: per-run lock files (two drivers on one folder are safe),
    `tools/status.py` (progress/ETA/STALE of every log),
    `tools/show_rows.py --agg/--range`, `QUENAIS_SELECTION_MAX_DET` (CIPSI/
    oracle columns above 2 M determinants).


## Results (all with seed repeats unless stated)

Errors in kcal/mol. "err_solver" = vs the solver's own exact limit (embedding
FCI for DMET solvers; LASSCF for LASSQD).

### 1 Validation (appendix material)

- H4/STO-3G, AVAS `H 1s` = full space: CASCI = FCI; SQD, SKQD, GQE, LASSQD all
  reproduce their exact limits (0.0000). Runs reproduce bit-for-bit for equal
  seeds.

### 2 LUCJ circuit diagnosis (H4, one 2-orbital fragment, first LAS cycle)

Weight off the HF determinant: exact **0.0262**.

| LUCJ variant | off-HF weight | E − E_FCI (mHa) |
|---|---|---|
| heavy-hex, 1 rep, not optimised (old default) | 0.0004 | 29.5 |
| all pairs, 4 reps, not optimised | 0.0267 | 0.004 |
| **any variant, optimised (linear method)** | **0.0262** | **0.000** |

Simulator sampling matched the statevector (0.00043 vs 0.00039), so the loss
is in the ansatz construction, not the sampling. Consequence on H8: LASSQD
without optimisation 42.5 ± 10.8 (1k shots) / 7.4 ± 7.0 (10k); with
optimisation **0.000 ± 0.000** at both.

### 3 H8 (linear, 1.0 Å, STO-3G, 16 qubits, 4 900 determinants, w₁ = 0.868)

| method | shots | error | oracle | CIPSI | subspace | time (s) |
|---|---|---|---|---|---|---|
| DMET+SQD | 1 000 | 17.57 ± 1.60 | 0.000 | 0.000 | 0.60 | 7 |
| DMET+SQD | 10 000 | 1.63 ± 2.77 | 0.000 | 0.000 | 0.96 | 8 |
| DMET+SKQD | 1 000 | 1.79 ± 0.24 | 0.024 | 0.025 | 0.33 | 1 468 |
| DMET+SKQD | 10 000 | 1.00 ± 0.34 | 0.001 | 0.001 | 0.46 | 1 471 |
| DMET+GQE | — | 0.072 ± 0.040 | 0.005 | 0.005 | 0.39 | 858 |
| LASSQD (opt) | 1k / 10k | 0.000 ± 0.000 | — | — | 1.00 | 57 |
| LASSCF vs FCI | — | +28.59 | | | | 45 |

Reading: GQE is the best *quantum* sampler, but **CIPSI at equal budget is
~14× better** and equals the oracle. Consistent with the thesis argument: w₁ =
0.868 is above the ≈0.74 threshold, so no headroom exists. SKQD finds the
important determinants (31 % subspace beats SQD's 69 %), but its cost is
independent of shots (circuit simulation dominates). LASSCF itself is 28.6
kcal/mol above FCI: four H₂ fragments ignore inter-fragment correlation.

### 4 ScF (STO-3G, AVAS Sc 3d/4s + F 2p → (8e,9o), fragments Sc(6o,1α1β) + F(3o,3α3β))

Energies (Ha): **CASSCF −850.293775 ≤ LASSCF −850.2864865100 ≤ CASCI
−850.286420**. LASSCF reproduces the earlier validated value exactly; it lies
4.57 kcal/mol above CASSCF (the inter-fragment correlation LAS omits) and
0.04 kcal/mol below CASCI (orbital relaxation; CASCI uses fixed step-1
orbitals). The strict bound is CASSCF ≤ LASSCF, not CASCI ≤ LASSCF.

**Carryover study** (LASSQD, optimised circuit, 5 seeds each, error vs LASSCF):

| shots | carryover on (ε = 1e-4, 1e-5, 1e-6 identical) | carryover off |
|---|---|---|
| 1 000 | 5.5 ± 10.8 | 10.4 ± 9.2 |
| 3 000 | **0.37 ± 0.34** | 5.0 ± 4.6 |
| 10 000 | **0.14 ± 0.32** | 2.3 ± 3.3 |
| 30 000 | **0.000 ± 0.000** | 0.60 ± 1.33 |

Wall time at 3 000 shots: ~730 s with carryover vs ~2 530 s without (more
macro cycles; energy oscillates). Findings: carryover cuts the error 10–20×
from 3 000 shots on and is exact at 30 000; the threshold ε does not matter;
at 1 000 shots neither works (minimum-shot limit). Reproduces the LASSQD
paper's claim on a new molecule, now with error bars. **Supersedes** the
handoff's earlier n=1 numbers (0.70 vs 1.18 kcal/mol, weak circuit).

### 5 N₂ dissociation (STO-3G, AVAS N 2s2p → (10e,8o), 1.0–2.5 Å)

w₁ along the curve: 0.938 (1.0) · 0.917 (1.1) · 0.889 (1.2) · 0.804 (1.4) ·
0.649 (1.6) · 0.426 (1.8) · 0.247 (2.0) · 0.192 (2.1) · 0.140 (2.25) ·
0.099 (2.5). DMET embedding: 20 qubits up to 2.0 Å, 18 qubits beyond (bath
size is adaptive).

- **LAS fragment spin must change along the curve** (one fragment per N, 4
  orbitals, 5 electrons each). Quartet fragments (2S = 3 each, opposite M_S):
  1320 kcal/mol above FCI at short range, **6.9 at 2.5 Å**. Doublet fragments:
  better at short range (34–270), worse at long range. LAS with atomic
  fragments cannot describe the triple bond near equilibrium (all bonding
  correlation is inter-fragment) and becomes good only at dissociation.
  Opposite-M_S quartets are not a pure singlet (spin contamination).
- **LASSCF has multiple solutions**: at 1.2 Å (doublets) LASSQD found a state
  30.5 kcal/mol below the LASSCF solution. Report as multiple stationary
  points, not an error.
- LASSQD (optimised, 10k shots) reproduces LASSCF to ≲0.1 kcal/mol at most
  points; exceptions at 1.6, 2.0, 2.25 Å (doublets) where seeds land in
  different solutions (e.g. 2.0 Å: 93.9 ± 81.3).
- **DMET+SQD**: 12–42 kcal/mol, growing with stretch; CIPSI at the same
  (large) budget is exact. Random-angle circuit = poor sampler.
- **DMET+GQE** (3 seeds, 120 epochs, default settings, QSCI cap 10 000): 46 ±
  0.8 (1.1 Å), 82 ± 0.5 (1.4), 82 ± 0.3 (1.8), **95 ± 0.04 (2.1)**, 78 ± 0.6
  (2.5), using only 5–8 % of the space. CIPSI at the same budget: 0.01–0.31.
  **GQE loses to CIPSI by >100× everywhere on the DMET-embedded N₂,
  including below the w₁ threshold.**
- Non-parallelity errors (kcal/mol): CASSCF 0.09, CASCI 0.21, DMET+SQD 29.7,
  DMET+GQE 49.2, LASSCF(doublet) 237, CCSD(T) 313, HF 435, LASSCF(quartet) 1314.
  CCSD(T), CCSD and MP2 fall below FCI at stretched geometries.


## Headline reproduction (N₂ 2.1 Å, plain CAS, 200-determinant budget)

Original setup exactly: plain
   CAS(10e,8o) Hamiltonian at N₂ 2.1 Å (16 qubits, 3 136 determinants,
   w₁ = 0.192), budget 200 determinants, 120 epochs, 5 independent seeds.
   Oracle and CIPSI reproduce the old values exactly (0.110 / 1.640 mHa),
   which confirms the setup is identical.

   | seed | GQE error (mHa) | beats CIPSI (1.640)? |
   |---|---|---|
   | 1 | 47.372 | no |
   | 2 | 5.283 | no |
   | 3 | 5.106 | no |
   | 4 | 5.267 | no |
   | 5 | **0.155** | yes (10.6×) |

   Mean 12.6 ± 19.5 mHa, median 5.27 mHa. **The old 0.153 mHa headline is a
   single favourable training run (1 of 5 seeds), not typical behaviour.**
   What survives, stated honestly:
   - GQE *can* find a near-optimal determinant set: the best seed lands
     0.045 mHa from the oracle and 10× below CIPSI at the same 200-determinant
     budget.
   - Training is unreliable: the typical (median) run is ~3× worse than CIPSI,
     and one run in five is ~30× worse.
   - Because QSCI energies are variational, "run k seeds, keep the lowest
     energy" is a legitimate protocol; best-of-5 still uses 200 determinants
     but costs 5× the training. State this as a protocol with its cost, not
     as the default result.
   - On the DMET-embedded N₂ (18–20 qubits, §4.5) GQE never approached CIPSI.

   The abstract and contribution 3 in the introduction must be rewritten
   accordingly. The w₁ framework itself stands, and is what makes this
   honest statement possible.

## Step 3 (9 Oct 2026)

### 3a DMET+SQD with the optimised LUCJ (`--ansatz lucj_opt`)

**H4** (8 qubits, 36 determinants). Code validated: 10k shots exact for
every setting. The LUCJ pair pattern decides the circuit quality
(1k shots, seed 1, error of the circuit alone / of SQD, kcal/mol; HF->FCI
correlation 42.6):

| pairs | reps | maxiter | err_lucj | err_SQD (1k) |
|---|---|---|---|---|
| heavy-hex | 1 | 10 | 23.5 | 25.1 |
| heavy-hex | 1 | 50 | 23.3 | 25.1 |
| heavy-hex | 2 | 10 | 1.9 | 0.000 |
| all | 1 | 10 | 1.0 | 0.000 |
| **all** | **2** | **10** | **0.005** | **0.000** |
| all | 2 | 50 | 0.000 | 0.000 |

Hardware-native (heavy-hex) pairs lose most of the correlation; more
optimiser iterations do not help. Default for DMET+SQD from here: all
pairs, 2 reps, 10 iterations.

**H8** (16 qubits, 4 900 determinants), 3 seeds:

| circuit | err_lucj | SQD 1k | SQD 10k | distinct valid samples (1k / 10k) | CX |
|---|---|---|---|---|---|
| random angles (old) | -- | 17.6 +- 1.6 | 1.6 +- 2.8 | | |
| heavy-hex LUCJ | 50.4 | 16.9-40.2 | 3.9-4.2 | | 208 |
| all-pairs LUCJ, 2 reps | 3.5 | 6.7-9.8 | 1.5-2.1 | 45-49 / 153-156 | 768 |

Batch size d (170, 1000, all samples in one batch) changes nothing: the
optimised circuit yields fewer than 170 distinct valid bitstrings even at
10k shots, so every batch already holds all of them. **The subspace is
limited by how concentrated the optimised state is, not by batching**: a
better circuit samples fewer distinct determinants (subspace 36-41 % at
10k vs 96 % for the random circuit). Cost: ~930 s per geometry for the
LUCJ optimisation (16 qubits), then 40-130 s per run with the cache.

### 3b GQE tuning, DMET-embedded N2 (STO-3G), 3 seeds, error vs embedding FCI (kcal/mol)

| R | 40 g / 120 ep | 40 g / 300 ep | 80 g / 120 ep | 80 g / 300 ep |
|---|---|---|---|---|
| 1.8 | 82.0-82.6 | 82.0-83.6 | 81.0-81.7 | 80.9-82.1 |
| 2.1 | 95.1 | 40.5, 40.5, 94.9 | 40.5 (x3) | 40.5 (x3) |

Cost relative to default: 80 gates x1.35, 300 epochs x2.5. At 2.1 A the
error jumps between two fixed values (40.4598 identical to 4 decimals in
8 runs) instead of improving gradually: GQE either reaches or misses one
group of determinants. Depth and training length are not the limit.
CIPSI at the same budget: 0.01-0.31. 3b closed: best setting 80 gates /
120 epochs; best-of-3 40.5 (2.1 A), 81.0 (1.8 A).

### N2 cc-pVDZ probe (1.1 A)

AVAS N 2s2p -> (10e, 8o) as in STO-3G. DMET embedding 8 impurity + 5 bath
= 13 orbitals = 26 qubits, (7a, 7b), 2 944 656 determinants (embedding
FCI feasible; selection metrics skipped above 2 M). HF -108.953796, CCSD
-109.267389, CASCI -109.034572 Ha.

## What it means — one paragraph (state of 9 Oct)

The package works end to end and its stochastic results are now reproducible
with error bars. On every system tested, **classical CIPSI at the same
determinant budget matched or beat every quantum sampler on average**. The
w₁/oracle/CIPSI framework is what exposes this, including that the earlier
GQE headline was one favourable run in five. GQE can, in a minority of
training runs, find a near-oracle determinant set where CIPSI is weak (low
w₁), which is the regime the thesis argues a quantum sampler would have to win.
On the LAS side, optimising the LUCJ circuit is essential, carryover makes
LASSQD reliable and faster from ~3 000 shots on, and LAS with atomic fragments
is only appropriate where inter-fragment correlation is weak (N₂ near
dissociation, with the fragment spin chosen to match).

## Step 3 continued (10–15 Oct 2026)

### 3a continued: N₂ STO-3G DMET+SQD with the optimised LUCJ (all pairs, 2 reps)

Error vs embedding FCI (kcal/mol), 3 seeds, one batch with all samples;
CIPSI = CIPSI with the same number of determinants; err_lucj = the
circuit's own energy, no SQD.

| R (Å) | err_lucj | SQD 1k | SQD 10k | SQD 100k | CIPSI @ 10k size |
|---|---|---|---|---|---|
| 1.1 | 4.3 | 5.8–7.9 | 1.2–3.4 | 0.27–0.53 | 0.004–0.024 |
| 1.2 | 5.8 | 7.3–10.0 | 1.9–3.5 | – | ~0.01 |
| 1.4 | 26.5 | 6.9–8.7 | 1.2–2.0 | – | ~0.0005 |
| 1.6 | 14.6 | 19.5–20.9 | 8.9–17.7 | 2.6–4.1 | 0.08–0.21 |
| 1.8 | 17.1 | 33–37 | 14.7–27.5 | 2.5–9.9 | 0.015–0.07 |
| 2.0 | 85.1 | 14–34 | 1.1–2.4 | – | 0.0001 |
| 2.25 | 89.9 | 70–152 | 1.0–1.4 | – | ~0 |
| 2.5 | 101.5 | 155 | 0.5 / 109 / 155 | 0.22–0.29 | 0.03–0.31 |

- 3a works: the old random-angle DMET+SQD was 12–42 kcal/mol off; with the
  optimised circuit and 10k shots most geometries are at 1–3 kcal/mol, with
  100k shots 0.2–0.5 at 1.1 and 2.5 Å. One-off cost ~2 500–2 900 s per geometry
  (LUCJ optimisation, cached), then 70–200 s per run.
- The hard region is 1.6–1.8 Å (intermediate correlation): 9–28 kcal/mol at 10k,
  still 2.5–10 at 100k.
- CIPSI with the same number of determinants is 10–1000× better everywhere.
  The sampled determinants are not the important ones; the energy improves
  only because more shots give a bigger subspace.
- At 2.5 Å/10k the seeds spread from 0.5 to 155: SQD sometimes misses a whole
  spin sector. Report the median and the spread, not the best.

### Exploration shots (H8, N₂ STO-3G)

Part of the shot budget (25 % or 50 %) sampled from a random shallow circuit.
H8 (4 900 determinants) and N₂ STO-3G (≤ 7 056 determinants): every run
reaches subspace = 100 % and error 0.0000, ~210 s (H8) / ~320 s (N₂) per run.

Reading: in SQD the subspace is the product of all sampled α-strings × all
sampled β-strings. A few thousand random strings already cover every string
of a small space, so the product is the full space and SQD becomes exact
diagonalisation. **That is filling, not better selection.** It only works
while the full space is small enough to diagonalise; it says nothing about
selection quality. Thesis wording: "exploration fills small spaces".

### N₂ cc-pVDZ (26 qubits, 2 944 656 determinants)

AVAS N 2s2p → embedding 13 orbitals (7α, 7β). LUCJ all pairs, 2 reps, **not
optimised** (`maxiter 0`; optimisation needs ~74 GB). Seed 1.

DMET+SQD, error vs embedding FCI (kcal/mol):

| R (Å) | shots | no exploration | subspace | CIPSI same size | 50 % exploration |
|---|---|---|---|---|---|
| 1.1 | 10k | 28.9 | 0.005 % (144 dets) | 5.4 | 0.0 (subspace 99.9 %, step 3 4 700 s, ~2.8 h wall) |
| 1.1 | 100k | 19.7 | 0.01 % | 1.5 | 0.0 (100 %, 4 700 s) |
| 1.6 | 10k | 70.9 | 0.02 % | 7.5 | 0.0 (100 %, 1 750 s) |
| 1.6 | 100k | 51.4 | 0.07 % | 2.0 | 0.0 (100 %, 1 750 s) |
| 2.1 | 10k | 94.1 | 0.02 % | 11.2 | 0.0 (100 %, 3 000 s) |
| 2.1 | 100k | 88.5 | 0.03 % | 7.9 | 0.0 (100 %, 3 000 s) |

The unoptimised circuit alone is 50.5 kcal/mol off at 1.1 Å. Without
exploration, sampling loses to CIPSI by 5–25× at the same size. With
exploration the product space again fills the whole 2.9 M space: exact, but
this is exact diagonalisation at 30 min – 3 h per run, not compression.
**Conclusion: the 13-orbital embedding is still small enough to brute-force;
SQD's selection quality is clearly below CIPSI.**

LAS / GQE on the same cc-pVDZ system (seed 1; err_vs_casci vs the CASCI
of the full active space):

| R (Å) | DMET+GQE (vs emb. FCI) | LASSCF frag A / B (vs CASCI) | LASSQD vs LASSCF |
|---|---|---|---|
| 1.1 | 54.6 (7 550 s) | 177 / 822 | 0.001 / 0.13 |
| 1.6 | 45.5 | 43 / 208 | 197 / 0.006 |
| 2.1 | 86.0 | 7.8 / 15.5 | 113 / 0.003 |

Fragments A = `0:4:3,2:1 1:4:2,3:1`, B = `0:4:4,1:3 1:4:1,4:3`. LAS with atomic
fragments is wrong near equilibrium (hundreds of kcal/mol) and only sensible
at 2.1 Å, as in STO-3G. LASSQD reproduces LASSCF where its circuit works
(B everywhere, A at 1.1 Å); with A at 1.6/2.1 Å it misses by 100–200.

### 3b continued: GQE seeds and free angles (N₂ STO-3G, 80 gates, 120 epochs)

Error vs embedding FCI (kcal/mol):

| R (Å) | CCSD angles, seeds 1–8 | free angles (−1, −0.4, −0.1, 0.1, 0.4, 1), seeds 4–6 | free angles + threshold 0.01 |
|---|---|---|---|
| 1.8 | 80.9–82.6, subspace 6–7 % | **75.03 (×3)**, subspace 25 % | 91.39 (×3), subspace 1.5 % |
| 2.1 | 40.46 (×11, identical) | **28.69 (×3)**, subspace 25 % | 40.46 (×3), subspace 5.6 % |

Cost: 1 000 s (CCSD angles); free angles 4 000–4 500 s at 1.8 Å, 2 250 s at 2.1 Å.

- The CCSD angles were the cause of the plateau (every seed landed on
  exactly 40.4598 at 2.1 Å). With free angles GQE improves by 6 kcal/mol
  (1.8 Å) and 12 kcal/mol (2.1 Å) and builds a 3–4× bigger subspace.
- The threshold of 0.01 removes useful operators: worse or no change. Drop it.
- Free angles give identical results across seeds, so more seeds add nothing.
- Still far from CIPSI at the same budget (0.01–0.3) and from DMET+SQD
  10k (1–28). **GQE stays the weakest solver here; 3b closed.**
- Decision: **no GQE+LAS integration** (it was only worth it if GQE improved
  substantially). Mention as outlook.

### [Fe₂(bpym)]⁴⁺ (first large LAS system)

6-31G, charge +4, singlet, AVAS Fe 3d; fragments `0:5:4,2:2 1:5:2,4:2`
(two Fe 3d, high-spin Fe(II) coupled to a singlet). Geometry from Hermes,
Pandharkar, Gagliardi, JCTC 16, 4923 (2020) SI Table S4
(`benchmarks/geometries/fe2bpym.xyz`). Reference LASSCF −3655.496377 Ha
(Wang et al. 2026). Grid: LASSCF + LASSQD (heavy-hex / all pairs) × seeds 1–2.

Status 15 Oct: the first attempt stalled in UHF for hours (1 thread, direct SCF);
restarted 13 Oct with 24 OpenMP threads, OpenBLAS 1 thread and
`PYSCF_MAX_MEMORY=100000`: system prep (UHF, AVAS, step 0) finished in
1 476 s. Then the log stayed at "ready", 0/5 runs, for 34 h (status STALE?).
To check: is the first run (LASSCF) still alive, or did it die/hang? Per-run
log: `$B/fe2bpym/runs/<id>/run.log`. Results: fill in.

## What it means (updated 15 Oct)

- **DMET+SQD needed the optimised circuit.** With it, N₂ STO-3G goes from
  12–42 kcal/mol to ~1–3 (10k shots), except the intermediate region
  1.6–1.8 Å.
- **At equal subspace size CIPSI wins everywhere**, by 1–3 orders of magnitude
  in STO-3G and 5–25× in cc-pVDZ. Sampling makes larger subspaces, not better
  ones.
- **Exploration shots make every tested system exact by filling the whole
  space.** This is a statement about small spaces (≤ 3 M determinants), not
  about quantum selection.
- **GQE**: the CCSD-angle plateau is understood and removed (free angles),
  but GQE is still 29–75 kcal/mol off on N₂. No GQE+LAS.
- **LASSQD** is exact relative to LASSCF when the fragment spin is chosen
  well, but LAS itself is only valid where fragments are weakly coupled
  (N₂ stretched, Fe₂ with separated metal centres).
- Regime map so far: weak/moderate correlation in small spaces → DMET+SQD
  (optimised) and CIPSI; separable strong correlation → LAS/LASSQD;
  intermediate bond breaking (N₂ 1.6–1.8 Å) → nothing quantum is good,
  CIPSI still is.

## Still open (15 Oct 2026)

- Fe₂(bpym): collect LASSCF/LASSQD numbers, compare with −3655.496377 Ha.
- Step 4 (code): determinant-list SQD — keep only the sampled (α,β) pairs
  instead of the α×β product, diagonalise with `det_analysis.projected_energy`,
  compare with CIPSI at the same size. Directly tests selection quality.
- Step 5 (code): noise model (Aer with a fake IBM backend) for one N₂
  geometry, to show how SQD degrades on hardware-like noise.
- Optional: `n2_gqe_angles_ext` (free angles at 1.1 and 2.5 Å; needs a second
  gqe-for-qsci checkout because GQE runs one at a time per checkout).
- `--check` false alarms listed in `docs/benchmarking.md` §5.
