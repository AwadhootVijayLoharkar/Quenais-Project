# QuEnAIS pipeline documentation

This folder explains the pipeline **stage by stage**, covering both the code
and the physics it implements. It is written for someone who has never seen
this project before and needs to understand *why* each stage exists, not just
how to call it.

If you want to *run* things, read the notebooks in `notebooks/` instead. This
folder is for understanding what the notebooks are doing.

## Reading order

| # | file | what it covers |
|---|---|---|
| — | this file | the whole pipeline in one page |
| 01 | [`01_configuration.md`](01_configuration.md) | `Config`, the five settings groups, the four ways to specify a molecule, and the caching rule |
| 02 | [`02_step0_classical.md`](02_step0_classical.md) | HF → MP2 → CCSD → CCSD(T) → CASSCF → NEVPT2, and why the answer key is softer than it looks |
| 03 | [`03_step1_active_space.md`](03_step1_active_space.md) | orbital entropy, the tier system, the four phases, and `force_active_space` |
| 04 | [`04_step2_embedding.md`](04_step2_embedding.md) | **the core of the method** — Schmidt decomposition, bath, the embedded Hamiltonian, `ecore`, `mu` |
| 05 | [`05_step3_quantum_solver.md`](05_step3_quantum_solver.md) | Jordan–Wigner, QSCI, the SQD family, GQE, operator pools |
| 06 | [`06_determinant_selection.md`](06_determinant_selection.md) | the oracle bound, CIPSI, w₁ — how to tell whether a quantum sampler helped at all |
| 07 | [`07_validation.md`](07_validation.md) | the checks, the three reproducibility tiers, golden data, `quenais-selftest`, `quenais-doctor` |
| 08 | [`08_glossary.md`](08_glossary.md) | every symbol and abbreviation used above |

Related documents outside this folder:

- `docs/limitations.md` — what this pipeline does *not* do reliably
- `docs/reproducibility.md` — the five failure modes and the practice that catches them
- `docs/gqe_integration.md`, `docs/gqe_setup.md` — the external GQE repo

---

## The problem in one paragraph

Solve the electronic Schrödinger equation exactly and you get the true energy
of a molecule. Doing so ("Full CI") costs a number of determinants that grows
combinatorially with the number of orbitals, so it is possible only for tiny
systems. Standard approximations (Hartree–Fock plus corrections) work well when
one electron configuration dominates the wavefunction — and fail for
**strongly correlated** systems, where several configurations are nearly
degenerate. Transition metals with partially filled d shells are the canonical
case, and they are the class this project targets.

## The strategy

Two independent reductions, applied in sequence:

1. **Shrink the system.** Density matrix embedding (DMET) replaces the whole
   molecule with a small *impurity + bath* problem that is exact with respect
   to a chosen reference state. Qubit count then scales with the fragment, not
   with the molecule.
2. **Shrink the diagonalisation.** Rather than diagonalising the full embedded
   Hilbert space, sample which determinants matter and diagonalise only in that
   subspace (QSCI). The sampling is where a quantum device is used.

Neither step is novel on its own. What this project measures is whether step 2
*actually helps* once step 1 has been done correctly — see
[`06_determinant_selection.md`](06_determinant_selection.md).

## The five stages

```
step 0  Classical      quenais/classical/runner.py       -> step0_classical.pkl
step 1  Active space   quenais/active_space/finder.py    -> step1_asf.pkl
step 2  Embedding      quenais/embedding/hamiltonian.py  -> step2_hamiltonian.pkl
step 3  Quantum solver quenais/quantum/{solver,gqe_runner}.py -> step3_results.pkl
step 4  Visualisation  quenais/visualization/plots.py    -> results_summary.csv, plots/
```

Run all of them with:

```bash
quenais-run --molecule LiH --basis sto-3g --steps 0 1 2 3 4 --project-dir ./lih_run
```

`quenais/cli.py`'s `run_step()` maps each integer to a module and calls its
`main(cfg, force=...)`. Step 3 is the exception: it goes through
`quenais.quantum.dispatch()`, which routes to the Qiskit stack (`sqd`, `skqd`,
`sqdrift`) or the CUDA-Q stack (`gqe`).

### What flows between stages

Each stage writes one pickle and the next stage reads it. The contracts:

| stage | needs from previous | produces (key items) |
|---|---|---|
| 0 | nothing | `methods{HF,MP2,CCSD,...}` each with `energy` + `tier` |
| 1 | nothing (reads step 0 only to *reuse* its active space, optional) | `mo_list`, `nel`, `mo_coeff`, `no_occ`, `dm_ao_{alpha,beta,total}_mp2` |
| 2 | step 1 — **hard requirement** | `h1e`, `h2e`, `ecore`, `mu`, `n_alpha`, `n_beta`, `n_emb`, `sv_all`, `embedded_scf_check` |
| 3 | step 1 and step 2 | `energy`, `spin_sq`, `iterations` (Qiskit) / `gqe_train.log` (GQE) |
| 4 | whatever exists | figures, `results_summary.csv`, `gqe_epoch_log.csv` |

**`step2_hamiltonian.pkl` is the interface file.** It is what the external GQE
trainer loads, what the determinant-selection tools read, and what
`tools/compare_pickles.py` diffs. If you keep one artefact from a run, keep
that one.

### Ordering constraints that are not obvious

- **Step 1 before step 0, if you want meaningful CASSCF/NEVPT2.** Those two
  methods reuse step 1's active space when it exists. On a first pass they fall
  back to a guess (`nel = min(nelectron, 10)`, `norb = min(nao//2, 8)`) and
  warn. The recommended order is steps 0 1 2, then re-run step 0 with `--force`.
- **Step 2 hard-fails without step 1.** `FileNotFoundError: Run step 1 first`.
  It also refuses a step 1 pickle built for a different molecule or basis.
- **Step 3 needs both 1 and 2.**

## The one check that matters

Everything downstream of step 2 is meaningless if the embedding Hamiltonian is
wrong, so step 2 ends with an independent test:

> Run a real, converged SCF on `h1e_emb` / `h2e_emb` / `ecore` alone. It must
> reproduce the **full molecule's** UHF energy.

Tolerance `2e-7` Ha (`EMBEDDED_SCF_VS_UHF_TOL`); validated at `1.3e-7` Ha on
ScH. It is stored in the step 2 pickle as `embedded_scf_check`.

This is deliberately *not* the `ecore` self-consistency identity that many DMET
codes report. `ecore` is **defined** as that difference, so the identity holds
by construction and can never fail — it tests arithmetic. From
`quenais/embedding/hamiltonian.py`:

> "The E_core self-consistency identity is deliberately NOT asserted here:
> E_core is defined as that difference, so the check is tautological and can
> never fail. The real check is verify_embedded_scf()."

A verification test that passes by construction is worse than no test: it buys
unjustified confidence.

## Why this codebase is written so defensively

Every bug in this project's history had the same signature: **right shape,
plausible magnitude, wrong value.** None of them crashed. None produced a
convergence failure. Examples, all real:

| bug | consequence | what caught it |
|---|---|---|
| bath built from sub-threshold singular values | ≈20 Ha error | N₂'s Schmidt spectrum |
| electron count from the active space, not the reference density | energy roughly doubled | independent embedded SCF |
| `dm_a_hf` from an assumed occupation rather than the real UHF density | corrupted `ecore` partition | embedded SCF |
| trainer config default not overridden | every run trained on N₂ while labelled LiH or ScH | identical epoch logs across molecules |
| operator pool built from individual Pauli strings | ≈50 % of samples discarded | particle-number audit |
| fixed trainer seed | three "repeats" identical to six figures | tracing why independent runs agreed |
| degenerate π orbitals fixed only up to a rotation | identical energies, different CI vectors | fingerprinting the CI vector |

That history is why you will find, throughout the code: hard guards instead of
warnings, tiered tolerances instead of one global tolerance, cache checks that
validate *contents* rather than existence, and unusually long docstrings
explaining why a line is the way it is. Those docstrings are the real
documentation — this folder summarises and connects them.

## Where to go next

Read [`04_step2_embedding.md`](04_step2_embedding.md) first if you want the
physics; it is the heart of the method. Read
[`01_configuration.md`](01_configuration.md) first if you want to run something.
