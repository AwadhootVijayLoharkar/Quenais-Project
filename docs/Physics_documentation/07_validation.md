# 07 — Validation, testing, and reproducibility

**Code:** `quenais/selftest.py`, `quenais/env_check.py`,
`quenais/provenance.py`, `quenais/_threads.py`, `tests/regression/`,
`tools/compare_pickles.py`

Every bug in this project's history had the same signature: **right shape,
plausible magnitude, wrong value.** None crashed. This document covers the
machinery built in response.

---

## 1. Two commands, two different questions

| command | question | speed |
|---|---|---|
| `quenais-doctor` | "will anything run at all?" | seconds |
| `quenais-selftest` | "does the physics reproduce?" | seconds (LiH) to minutes |

`install.sh` runs both.

### `quenais-doctor` — environment

> "Every check below corresponds to a failure that has already cost real time,
> and every one of them presents as something other than what it is."

That last clause is the point. Selected checks:

| check | the failure it prevents |
|---|---|
| **AVX-512** | PySCF's prebuilt `libcgto.so` uses AVX-512 **without runtime CPU dispatch** (numpy/OpenBLAS do dispatch; PySCF does not). On a CPU without it, the first two-electron integral call is `SIGILL` |
| **GPU compute capability** | cuQuantum needs cc ≥ 8.0 (Ampere). Below that: `RuntimeError: architecture mismatch` |
| **setuptools < 82** | setuptools ≥82 removed `pkg_resources`, which tequila imports at load time |
| **wandb credentials** | batch jobs cannot show a login prompt; the run dies with `UsageError: No API key configured` |
| **MPI consistency** | mpi4py and CUDA-Q's plugin must resolve to the **same** `libmpi`. A mismatch "shows up as hangs and corrupt gathers rather than a clean error" |
| **`lib.einsum`** | see below |
| **LLVM import order** | see below |
| **shim directory** | it must **not** contain `__init__.py` — it is a `sys.path` entry, not a package |

Two worth expanding because they are such good examples of the project's
failure mode.

**The einsum check.** NumPy 2.4.0 changed `einsum_path`'s contraction tuples
from 5 elements to 3; PySCF's `lib.einsum` unpacked 4 unconditionally. Why it
gets an explicit check:

> "two-operand contractions take a different code path, so HF, MP2 and CCSD all
> passed cleanly and the break only appeared inside ASF's DFUMP2
> natural-orbital step -- **four stages into a pipeline run, in a third-party
> library, with nothing in the message naming numpy.**"

**The LLVM order check.** torch bundles triton, which embeds its own LLVM;
cudaq embeds MLIR/LLVM. Both register the same global LLVM command-line
options, and the second to load aborts the interpreter.

> "torch first is fine; cudaq first is fatal... **This is an abort inside native
> code, not a Python exception -- there is no traceback and no try/except that
> can catch it.**"

Exit codes: `0` all clear (warnings allowed), `1` at least one FAIL. Every check
is individually wrapped so the doctor never crashes.

### `quenais-selftest` — physics

Runs the pipeline on LiH (default) into a temp directory and asserts ten
things: thread environment, reference availability, PySCF import, HF/MP2/CCSD
energies at their tier tolerances, active-space `mo_list` exactly, the presence
of `dm_ao_{alpha,beta}_mp2`, `n_bath` exactly, the embedded electron count
exactly, `ecore` at its own tolerance, and the embedded-SCF check within `2e-7`.

`--full` runs LiH, N₂ and ScH.

The closing logic is worth copying into other projects:

> `# Do not claim a clean bill of health when the physics never ran.`

If there were zero failures but some checks were *skipped* (e.g. PySCF absent),
it explicitly refuses to report success.

There is also a nice implementation detail — output is silenced at the **file
descriptor** level:

> "Two sources write past a Python-level redirect: PySCF's own logger, which
> holds its own stream reference, and the ASF library, whose internal CASCI
> prints from code we do not own. Silencing objects one at a time cannot cover
> third-party internals; dup2 on fd 1 covers everything."

## 2. The three reproducibility tiers

Introduced in [`02_step0_classical.md`](02_step0_classical.md); repeated here
because it governs the whole test suite.

```python
TOL = {DETERMINISTIC: 1e-9, OPTIMIZER_DEPENDENT: 2e-3, STOCHASTIC: 5e-2}
```

| tier | what it means | examples |
|---|---|---|
| `DETERMINISTIC` | reproduces to ~1e-10 across machines | HF, MP2, CCSD, CCSD(T), DMET+CASCI, `ecore`, `mu`, σ spectrum |
| `OPTIMIZER_DEPENDENT` | depends which valid solution the optimiser found | CASSCF, NEVPT2 |
| `STOCHASTIC` | varies run to run by design | DMET+GQE |

> "Every bug in this project's history was 'right shape, plausible magnitude,
> wrong value' -- so the DETERMINISTIC tier is the one that actually catches
> regressions, and it is deliberately tight."

Evidence for the middle tier: ScH CASSCF gave −752.680677 and −752.681604 on
identical input (0.93 mHa apart), with NEVPT2 moving 3.6 mHa with it. For the
third: ScH DMET+GQE has produced −752.668847, −752.678674 and −752.677509.

## 3. Golden data as bug tripwires

`tests/regression/golden/<system>/` holds the three stage pickles plus CSVs,
produced on an A100 / AMD EPYC 7402 run in July 2026.

What makes them useful is not the energies — it is the **structural** fields,
each of which guards a specific past bug:

| system | field | guards |
|---|---|---|
| LiH | `n_bath = 2` | "LiH DOES have a real bath" — the adaptive_bath fix |
| LiH | `n_alpha = 2` | the electron-count fix; the buggy path gives (1,1) and ~2× HF |
| N₂ | `n_bath = 0`, `max_abs_sv_all_below = 1e-8` | the fabricated-bath bug (~20 Ha) |
| ScH | `degenerate_pairs = [(11,12)]` | the split-degenerate-orbital bug |

The corresponding tests are named for the *signature*, not the code path:
`test_n2_has_no_bath` ("the fake-bath signature"),
`test_lih_electron_count_comes_from_reference_density` ("the 2×-HF signature"),
`test_sch_degenerate_pair_kept_together`.

The LiH electron-count test asserts not only `n_alpha == 2` but explicitly that
`n_alpha != naive_alpha` — "the buggy formula must not agree here". That is the
right way to write a regression test for a bug like this: assert the *wrong*
answer is not being produced, not just that the right one is.

## 4. Testing the tester

`test_reference_harness.py` exists because the comparator and the reference
table have to be trustworthy *before* anything is measured against them:

> "If this file fails, no other regression result means anything."

Three properties, and the third is the one that matters:

1. the table agrees with the golden pickles,
2. **no false alarms** — comparing a deepcopy to itself passes with zero SKIPs,
3. **no false negatives** — nine deliberate perturbations must each be caught:
   `ecore` drift 1e-4, electron count halved, bath count changed, `h1e` scaled
   by 1+1e-8, `ref_occ_alpha` dropped, `h2e` reshaped, molecule tag swapped,
   Schmidt spectrum zeroed, NaN introduced.

Note "`ref_occ_alpha` dropped" is in that list deliberately — a missing key
must be a failure, not a skip.

## 5. `compare_pickles.py` — the diffing tool

> "Every bug in this project's history produced output with the right shape, a
> plausible magnitude and the wrong value: a bath fabricated from numerical
> noise, an electron count taken from the wrong space, a cache entry belonging
> to a different molecule. **None of those crash, and none are visible in a
> summary table.**"

Four deliberate paranoia rules:

- a key present in golden but missing from the candidate is a **FAILURE**, not a
  skip — "that is how a dropped `ref_occ_alpha` would hide"
- an array whose **shape** changed is a failure *before* values are compared
- NaN or inf anywhere is a failure
- keys the comparator does not know how to compare are reported as **SKIPPED
  and counted**, never silently ignored

Per-key tolerances rather than one global number:

| tolerance | keys | why |
|---|---|---|
| `1e-10` | `sv`, `sv_all`, `sv_gap`, `sv2_cov` | singular values are small and clean |
| `1e-9` (default) | `h1e`, `h2e`, `deviation`, `no_occ`, densities | |
| `1e-8` | `ecore`, `mu`, `uhf_energy`, `mp2_energy`, `ref_occ_*` | "ScH carries ~750 Ha of magnitude, so 1e-9 relative is below double-precision noise for accumulated sums" |
| `1e-7` | `mo_coeff`, `mo_coeff_uhf` | "sign/phase of degenerate MOs is not unique" |

`INFORMATIONAL_KEYS = {"total_time", "provenance", "timestamp"}` are reported
but never failed on.

```bash
python tools/compare_pickles.py golden/LiH/step2_hamiltonian.pkl ./lih_run/results/step2_hamiltonian.pkl -v
```

Exit `0` if OK, `1` on mismatch.

## 6. Provenance

Every stage pickle carries a `provenance` block: quenais version, UTC
timestamp, git SHA + dirty flag, Python/platform/CPU, GPU, thread settings and
warnings, library versions, and the GQE repo's patch state.

> "A number without provenance costs a conversation to interpret and is
> sometimes uninterpretable after the fact: 'I got -752.68' does not say whether
> the GQE submodule was patched, whether OpenBLAS was oversubscribed, or which
> PySCF version produced it."

One rule governs the whole module:

> "**Nothing here may raise.** A provenance block that crashes the run it is
> describing would be worse than no provenance at all, so every probe is wrapped
> and degrades to None."

Library versions are read from `sys.modules` only, never imported:

> "importing cudaq or torch here would defeat the lazy-import discipline the
> package relies on. Report only what the process already loaded."

## 7. Thread control — why `_threads` must import first

`quenais/__init__.py` imports `quenais._threads` before anything else, and the
module has **no imports beyond `os`**.

The physics problem: block2 (used by the active-space finder) is
OpenMP-threaded, and OpenBLAS is pthread-threaded by default. When OpenBLAS is
called from inside a live OpenMP parallel region the two runtimes fight over
CPU affinity:

```
OpenBLAS Warning : Detect OpenMP Loop and this application may hang.
```

> "That is a genuine hang risk, not noise -- it is the same failure family behind
> the FeN6 DMRG hang. The fix is not rebuilding OpenBLAS; it is making sure only
> ONE of the two libraries spawns threads."

And why the import position is load-bearing:

> "OpenBLAS reads these variables once, lazily, the first time it needs a thread
> pool. **Setting them after NumPy's first import does not reliably take
> effect.**"

Hence `verify()` reports whether NumPy was imported first, and
`tests/conftest.py` imports quenais before anything else.

Settings: `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, `OMP_NUM_THREADS=<allocated CPUs>` — using
`setdefault`, so a deliberate user value survives.

Scheduler awareness matters on shared nodes: `SLURM_CPUS_PER_TASK`,
`SLURM_CPUS_ON_NODE`, `PBS_NP`, `NSLOTS`, `LSB_DJOB_NUMPROC` are read before
falling back to `os.cpu_count()`.

> "os.cpu_count() reports the whole machine, so a job allocated 8 cores would
> otherwise start ~95 OpenMP threads and fight every other job on the node."

When a scheduler told us the allocation, all those cores are used; when
guessing from `cpu_count`, one is left free.

> "The old unconditional cpu_count - 1 was wrong in both directions: it wasted a
> core inside a batch job, and grabbed ~95 threads on a login node."

## 8. Step 4's one correctness trap

`plots.true_embedding_casci()` exists because of a subtle mislabelling:

> "reference_density_info['e_cas'] is the small CASCI computed during phase B to
> help BUILD the reference density, **before Schmidt decomposition even runs.**
> It never reflects n_bath, mu, or the actual n_emb-orbital embedded solve, so it
> stayed frozen at the wrong value even after the electron-count and dm_a_hf
> fixes landed. Reporting it as 'DMET+CASCI' is wrong."

So the true value is recomputed through the adapter, and if that fails the bar
is **omitted** rather than substituted:

> "It is omitted rather than substituting the phase B reference-density CASCI,
> which is a different quantity and would look plausible while being wrong."

That is the general principle of this codebase in one sentence: **absence is
better than a plausible wrong number.**

## 9. The standing practice

> **Run every calculation twice. Diff the fingerprints, not just the energy.**

Failure mode 4 in `docs/reproducibility.md` produced **identical total energies
to 14 decimal places from different wavefunctions** — degenerate π orbitals
fixed only up to a rotation, because `symmetry=True` was not set. An
energy-only check would have passed it. The CI-vector fingerprint caught it.

A result that has not been run twice and fingerprinted is not a result yet — it
is a candidate for one of the five modes in `docs/reproducibility.md`.

## Next

[`08_glossary.md`](08_glossary.md) — every symbol and abbreviation used in
this folder.
