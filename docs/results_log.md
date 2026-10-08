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

## What it means — one paragraph

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

## Still open (Oct 2026)

- N₂ in cc-pVDZ (realistic basis); Cu₂O₂ and Fe₂ (Fe₂ LASSCF reference
  −3655.496377 Ha, Wang et al. 2026).
- Cheap ablations: LASSCF vs LASCI, a deliberately bad fragmentation, AVAS vs APC.
- GQE: more seeds/epochs on the CAS Hamiltonian at several w₁ values, to turn
  "1 in 5" into a success rate with an uncertainty.
- `--check` false alarms listed in `docs/benchmarking.md` §5.
