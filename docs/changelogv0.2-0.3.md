# Changelog — this session

In order. Each entry: why it happened, what was added, what it produced.

---

## 1. Independent LiH check

**Why.** Before measuring anything with the package, I needed to know the
package was right. Every conclusion afterwards would inherit that assumption.

**Added.** `lih_independent_fci.py` — a standalone LiH calculation in plain
numpy, sharing no code with QuEnAIS. It builds the many-body Hamiltonian by
applying creation/annihilation operators to bitstrings directly, so it can
only agree by being correct.

**Result.** −7.881246151999 against the package's −7.881246151999262.
Agreement to 15 decimals. Everything after this rests on it.

---

## 2. The oracle bound

**Why.** Your GQE error was 20.9 mHa at 2,439 determinants. Nobody could say
whether that was terrible or nearly optimal, because there was no ceiling to
compare against. That question had to be answerable before anything else.

**Added.** `quenais/quantum/det_analysis.py` — computes the best energy
obtainable from any N determinants. It refuses to run unless it first
reproduces your validated reference exactly.

**Result.** ScH needs only 144 determinants for chemical accuracy, out of
108,900. The best 2,439 give 0.0045 mHa. So GQE used 17× more determinants
than necessary and was still 4,600× worse than optimal. The error was never
about subspace size.

Also found: re-diagonalising barely helps beyond N≈50. Getting the right
determinants matters; getting their amplitudes right barely does.

---

## 3. The classical baseline

**Why.** "Better than GQE" is not a claim anyone accepts. A reviewer's first
question is "better than what classical method?"

**Added.** `quenais/quantum/det_expansion.py` — CIPSI, a 1973 selected-CI
algorithm, running on your embedded Hamiltonians with no quantum input.

**Result.** The session's turning point. CIPSI reached 0.0043 mHa on ScH where
GQE reached 20.9 — classical was 4,900× more accurate at identical cost.

**What this changed.** The neural sampler lost its justification. It would have
been built to beat a target already crushed by a fifty-year-old algorithm.

---

## 4. Diagnosis: ScH was the wrong test

**Why.** The CIPSI result needed explaining. Was the quantum method bad, or the
benchmark uninformative?

**Found.** ScH's top configuration carries 89.5% of the wavefunction. That is a
single-reference system — exactly what classical selected-CI was designed for
and tuned on for decades. No sampler could have shown anything there.

**Consequence.** This explains your whole project history. SQD → SKQD →
SqDRIFT → GQE were each replaced for failing on ScH. None of them could have
passed.

---

## 5. Finding a test that could discriminate

**Why.** We needed a system where correlation could be dialled up and down, so
we could see *where* classical selection starts to fail.

**Added.** `tools/run_correlation_scan.py` — stretches the N₂ bond across 8
geometries at fixed active space, computing the oracle and CIPSI at each.

**Result.** A threshold. Classical selection is optimal until the dominant
configuration drops below about 74%, then degrades sharply. Peak difficulty is
at *intermediate* stretching (2.1 Å), not full dissociation — the recoupling
region, which was not obvious in advance.

**This is the finding nobody has published.** It also tells you in advance
whether any future molecule is worth attempting.

---

## 6. Four rounds of fixing the scan

**Why.** The first three versions produced smooth, believable, wrong results.

| round | what was wrong | how we caught it |
|---|---|---|
| 1 | Active space of 4 orbitals = 36 determinants, with a 200-determinant budget. Every method got the whole space and looked identical. | Noticed the budget exceeded the space |
| 2 | 8 impurity orbitals in a 10-orbital basis left DMET no environment. Embedded SCF off by up to 0.73 Ha — while producing a beautiful monotonic curve. | The pipeline's own validation check |
| 3 | ARPACK seeded from a random vector; results changed between identical runs. | Ran it twice and diffed |
| 4 | Degenerate π orbitals fixed only up to a rotation, so each run silently used a different orbital basis. Energies matched to 14 decimals; wavefunctions didn't. | Fingerprints of the CI vector |

**Added along the way.** Trust checks in the data rather than in a log file, and
`civec_fp` / `order_fp` fingerprints that identify *which layer* an
irreproducibility lives in.

**Standing practice now:** run everything twice, diff the fingerprints.

---

## 7. Bridging to the quantum solvers

**Why.** The scan bypasses DMET and builds the CAS directly. Your quantum
solvers read a step-2 pickle. Without a bridge, quantum and classical would
have been compared on different Hamiltonians — meaningless.

**Added.** `tools/export_cas_hamiltonian.py` — writes the identical CAS
Hamiltonian in the format your pipeline expects.

**Result.** The first fair test your pipeline has ever had. At 2.1 Å with 200
determinants: oracle 0.110, **GQE 0.153**, CIPSI 1.640 mHa. Your quantum
pipeline beat classical by 10× and came within 0.04 mHa of the theoretical
ceiling.

---

## 8. Chasing the 1.8 Å failure

**Why.** At 1.8 Å GQE gave 78.9 mHa — 150× worse than classical, at a geometry
classical finds *easier* than 2.1 Å. Three runs gave identical results to six
figures, which looked like a systematic bug.

**Added.** Four diagnostic tools, plus a determinant dump patched into the
vendored GQE repo (`refine/pipeline.py`, environment-gated so it's inert by
default):

- `compare_gqe_determinants.py` — GQE's determinants vs the oracle's
- `inspect_gqe_determinants.py` — excitation-rank breakdown
- `expand_gqe_seed.py` — can classical expansion rescue a bad set?
- `test_symmetry_completion.py` — symmetry closure test

**What we learned.** GQE's maths is correct (re-solving its own determinants
reproduced 78.9371 exactly). Its choices are wrong — 76.9% of the wavefunction
captured versus 99.98%. Classical expansion improved it to 3.33 mHa but never
matched CIPSI.

**Three hypotheses, all wrong.** CCSD-derived operator pool quality (the
correlation ran backwards). Spatial symmetry breaking (no orbital symmetry
exists). Spin-exchange incompleteness (both halves of each pair were missing).

**The actual cause.** `gqe-for-qsci/configs/trainer/default.yaml` pins
`seed: 32`, and `GqeSettings.seed` defaults to `None`, which skips the
override. **Every "repeat" run was the same computation.** The apparent
deterministic failure was one run reported three times.

**Added.** A `--gqe-seed` flag for `cli.py` so repeats are actually
independent. *Verify this got applied.*

---

## 9. Dissociation curve

**Why.** You have the exact energies at 8 geometries but had never plotted them
as an actual potential energy curve — the standard way to show a molecule is
strongly correlated.

**Added.** `tools/run_dissociation.py` — all classical methods plus exact
across the bond scan.

**Result and two fixes.** The first run had only 5 points between 0.9 and 1.3 Å,
so the minimum landed on a grid point rather than its true position. And D_e was
computed as max − min, which used the repulsive wall instead of the dissociation
limit: 241 kcal/mol reported, **150 kcal/mol correct**. Both fixed; the grid now
has 31 points, dense through the bonding region.

Confirmed CCSD stops converging past 2.25 Å — which is itself the result.

---

## What changed strategically

Three things reversed direction during this session.

**The neural sampler was dropped.** Not because it's a bad idea, but because
CIPSI showed it had no measured justification yet, and there isn't time to
build and validate one in two months. It becomes future work — better
motivated than before, since you now know exactly what it would have to beat
and in which regime.

**The test system changed.** ScH can't distinguish selection methods. N₂ under
stretch can, and tells you where the boundary is.

**The thesis reframed.** From "I built a neural sampler" to "I built a pipeline
and measured where quantum simulation beats classical." Same subject, and you
have a result instead of a half-finished model.

---

## Open at handoff

1. Three-seed runs at 1.8 Å were in progress. First honest variance data.
2. GQE across all 8 geometries not yet run — that's the main figure.
3. Cr₂ active space indices in `run_correlation_scan.py` are a guess.
4. Verify `--gqe-seed` made it into `cli.py`.