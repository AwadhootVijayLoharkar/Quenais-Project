# 06 — Determinant selection: did the quantum sampler actually help?

**Code:** `quenais/quantum/det_analysis.py`, `quenais/quantum/det_expansion.py`,
`tools/run_stage0.py`, `tools/run_stage1.py`,
`tools/run_correlation_scan.py`, `tools/run_dissociation.py`

None of this is part of `quenais-run`. These are diagnostics that answer a
question the pipeline itself cannot: **given a fixed determinant budget, how
much of the remaining error is the sampler's fault?**

This is the study that produced the headline results of the thesis. It is also
the part most likely to be challenged, so it carries the most provenance
detail.

---

## 1. The question

Step 3 gives you a number: "DMET+GQE was 24 mHa above the exact embedded
answer." On its own that is uninterpretable. It could mean:

- the sampler did badly, or
- the problem was easy and *any* method would have done well, or
- the problem was hard and no method could have done better.

You cannot tell without two extra reference points, neither of which the
pipeline computes.

## 2. Three quantities

| quantity | what it is | code |
|---|---|---|
| **w₁** | \|c₀\|² of the largest CI amplitude — the correlation dial | `det_analysis.weight_curve()` |
| **oracle bound** | best energy obtainable from *any* N determinants | `det_analysis.oracle_curve()` |
| **CIPSI** | classical selected CI (1973), same Hamiltonian and budget | `det_expansion.cipsi_from_scratch()` |

### w₁ — the correlation coordinate

Diagonalise exactly, sort the CI amplitudes by weight, take the largest.

```
w₁ ≈ 0.9   one configuration dominates → single-reference
w₁ < 0.5   the state is genuinely multiconfigurational
```

This is the quantity that makes "strongly correlated" measurable rather than a
adjective. `weight_curve()` sorts with a **stable** sort on negated weights:

> "Symmetry-equivalent determinants carry exactly equal weight, so the ordering
> among them is not determined by the amplitudes. A stable sort... breaks those
> ties by determinant index, which is arbitrary but REPRODUCIBLE -- the previous
> `argsort(w)[::-1]` reversed an ascending sort, so tied groups came back in an
> order that depended on the sort's internal path and changed between runs."

### The oracle bound — a ceiling, not a method

Take the top-N determinants **by exact amplitude**, i.e. cheat by looking at
the answer first, then re-diagonalise in that subspace. No real method can know
this ranking in advance, which is the point: it is the best any N-determinant
method could possibly do.

Two properties worth stating:

**It is self-validating.** `run_gates()` refuses to report anything unless the
full determinant space first reproduces the package's own DMET+CASCI reference.

**It is conservative.** From the code:

> "Top-N by amplitude is not the energy-optimal N-determinant set, so
> E(top-N) >= E(optimal-N). The bound is therefore slightly pessimistic, which
> is the safe direction for a go/no-go: **the true headroom is at least what is
> measured here.**"

And every point is checked for variationality:

```
Projected energy fell BELOW the exact reference at N=...
This is not possible variationally -- the determinant indexing or the
reference is wrong. Stop and fix before reading any other number.
```

If you see that, the indexing is wrong and everything after it is noise. Do not
work around it.

### CIPSI — the classical baseline

Selected CI from 1973. Start from the lowest-diagonal determinant, repeatedly
estimate each candidate's perturbative importance, add the best ones, repeat.
Purely classical, no quantum input at all, running on the **identical** embedded
Hamiltonian with the **identical** budget.

This is the comparison a reviewer asks for first. "Better than classical" is not
a claim until it means better than this.

`det_expansion` also offers `expand_from_seed()`, which runs the same growth
starting from determinants a quantum sampler found — useful for asking whether
classical expansion can rescue a poor quantum proposal.

## 3. The budget guard

```python
if budget >= 0.5 * space.ndet:
    raise ValueError(...)   # "Every method would look identical."
```

This is a hard error, not a warning, because the failure it prevents is
invisible:

> "An early version of `run_correlation_scan.py` used a 4-orbital active space
> (36 determinants) with a 200-determinant selection budget. Every method --
> oracle, CIPSI, whatever else was being compared -- received the entire space,
> so every method's result was identical. **The plot looked like a valid
> comparison.**"

Practical consequence: N₂'s golden **(4e,4o)** space (36 determinants) is
useless for this study. The correlation scan uses the **full valence (10e,8o)**
space, MOs 2–9, 3136 determinants. Two different experiments:

| space | determinants | purpose |
|---|---|---|
| (4e,4o), MOs 5–8 | 36 | reproducing the golden regression numbers |
| (10e,8o), MOs 2–9 | 3136 | measuring selection quality |

The larger space is also the physically correct one: breaking a triple bond
needs all three bonding/antibonding pairs, or the stretched geometries are not
actually strongly correlated.

## 4. It deliberately does not use DMET

`run_correlation_scan.py` builds the CAS Hamiltonian directly with PySCF and
skips the embedding entirely. This is not laziness:

> "The first version ran the full DMET pipeline at each geometry and every point
> failed the embedded-SCF check by 0.3-0.7 Ha, equilibrium included. The cause
> was the active space, not the stretching: N2/STO-3G has only 10 orbitals, so
> an 8-orbital impurity plus 2 bath orbitals spans the whole molecule. DMET needs
> an environment to fold into e_core; with none left, the core potential and the
> electron count double-count and the embedding Hamiltonian is meaningless."

> "But this scan does not need DMET at all. The question being asked is about a
> **Hamiltonian**, not about an embedding."

Note the curve produced by the broken version was *smooth and monotonic across
geometries*. Nothing about its shape suggested a problem. It was caught by the
embedded-SCF check, not by looking at the plot.

## 5. Result 1 — ScH cannot discriminate

From `tests/regression/golden/ScH/`:

| quantity | value |
|---|---|
| w₁ | **0.895** |
| determinants for 90 % of the weight | 2 (of 108,900) |
| `n_chem` (determinants for chemical accuracy) | 144 (0.13 % of the space) |
| CIPSI error at N=2439 | **0.0043 mHa** |
| CIPSI gap to oracle at N=2439 | −0.0002 mHa |
| GQE error at the same budget (`headroom_mha`) | **20.90 mHa** |
| ratio | ≈ **4,880×** in favour of the 1973 classical method |

**Do not read this as "GQE is bad."** Read the gap-to-oracle column: CIPSI is
within 0.0002 mHa of the theoretical best possible answer. There is no room
left for *any* method to show an advantage.

w₁ = 0.895 means ScH is a **single-reference system by the correlation
measure**, despite being a transition-metal system with 22 qubits and 108,900
determinants. It looks hard and is not hard *in the way that distinguishes
selection methods*.

This reframes the project's whole solver history: SQD → SKQD → SqDRIFT → GQE
were each evaluated primarily on ScH, where no method could have shown a
measurable difference.

**Important caveat to state before someone else does:** w₁ = 0.895 is a
property of the chosen CAS(4e,6o), not of ScH in the abstract. A larger active
space would admit more configurations and could lower w₁. The conclusion holds
for every benchmark in this project, since they all used that space — but it is
not a claim about scandium hydride as a molecule.

## 6. Result 2 — bond breaking finds a threshold

Stretching N₂ tunes w₁ continuously. The active space is held **fixed** across
the scan so geometry is the only variable — otherwise you would be comparing
different embeddings rather than different correlation strengths.

N₂/STO-3G, (10e,8o), budget 200 of 3136:

| r (Å) | w₁ | CIPSI (mHa) | oracle (mHa) | gap |
|---|---|---|---|---|
| 1.0977 | 0.9173 | 0.1182 | 0.1200 | −0.0018 |
| 1.3000 | 0.8523 | 0.2981 | 0.2913 | +0.0069 |
| 1.5000 | 0.7373 | 0.4646 | 0.3743 | +0.0903 |
| 1.8000 | 0.4257 | 0.5186 | 0.3968 | +0.1218 |
| **2.1000** | **0.1916** | **1.6396** | **0.1101** | **+1.5295** |
| 2.4000 | 0.1112 | 0.9842 | 0.0405 | +0.9438 |
| 2.8000 | 0.0782 | 0.2394 | 0.0091 | +0.2303 |
| 3.2000 | 0.0677 | 0.0382 | 0.0060 | +0.0322 |

**The threshold is w₁ ≈ 0.74.** Above it, classical perturbative selection is
effectively optimal and nothing can beat it. Below it, a gap opens.

**The peak is at 2.1 Å, not at full dissociation.** This is the least intuitive
result in the study. At 2.8–3.2 Å the gap *shrinks again* — once the bond is
fully broken the wavefunction becomes a clean, near-degenerate combination that
CIPSI handles fine (`n_chem` drops to 23). The hard regime is **intermediate
stretching**, where the state is genuinely multiconfigurational but has no
simple structure.

## 7. Result 3 — the one geometry where quantum wins

At 2.1 Å, same 200-determinant budget:

| method | error vs exact CASCI | note |
|---|---|---|
| oracle bound | 0.110 mHa | ceiling, not achievable |
| **DMET + GQE** | **0.153 mHa** | this pipeline |
| CIPSI (1973) | 1.640 mHa | classical baseline |

GQE beats the classical baseline **10×** at equal cost, and lands 0.04 mHa from
the theoretical best possible answer.

Caveats to state out loud whenever quoting this:

1. **One geometry.** Of the eight scanned, this is the only one past the
   threshold where GQE has been run.
2. **Seed count.** Check how many independent `--gqe-seed` values back it before
   quoting a spread — see `docs/reproducibility.md` §5.
3. **The oracle is pessimistic**, so the headroom is a conservative bound.

## 8. The dissociation curve

`tools/run_dissociation.py` produces the standard strong-correlation figure:
every classical method against exact CASCI across bond length.

The thing to look for: **CCSD does not merely lose accuracy — it crosses
through the exact answer and ends up below it.** Coupled cluster is
non-variational, so nothing bounds it, and its amplitudes diverge: measured
`max|t₂| = 0.65` at 1.8 Å and **0.84** at 2.1 Å, where healthy is below 0.1.

That crossing is the cleanest available demonstration that a system is strongly
correlated, and it motivates everything downstream.

**Comparability caveat for the figure caption:** HF/MP2/CCSD/CCSD(T) are
all-electron; CASCI is within the active space with a frozen core. For
N₂/STO-3G with (10e,8o) the frozen 1s pair contributes almost no correlation,
so the comparison is meaningful — but these are not identical theory levels.

## 9. Reproducibility machinery specific to this study

**Fingerprints.** `measure()` records `civec_fp` and `order_fp` at every
geometry. If a re-run disagrees, diff those first:

| `civec_fp` | `order_fp` | energies | meaning |
|---|---|---|---|
| differ | — | — | the exact CI vector itself is not reproducible |
| same | differ | — | amplitudes near the cutoff are tied; ranking arbitrary, state fine |
| same | same | differ | the instability is downstream, in the subspace diagonalisation |

**ARPACK's random start vector.** `eigsh` seeds randomly when `v0` is not
given, so two runs of an identical calculation converge along different Krylov
paths and reorder near-equal amplitudes — an irreproducible *ranking* from a
perfectly correct energy. `v0` is now mandatory in `projected_energy`, fixed to
`np.full(n, 1/√n)`. **Do not remove it.**

**Tied groups at the cutoff.** Symmetry-equivalent determinants carry exactly
equal weight; taking some of a tied group and not the rest breaks the trial
space's symmetry and raises the energy for a reason unrelated to selection
quality. `measure()` extends the cut to whole tied groups — hence
`budget=200 (tied group -> using 201 dets)` in the scan output.

## 10. Re-run recipes

```bash
# Stage 0 -- weight curves + oracle bound (LiH seconds, ScH minutes)
python tools/run_stage0.py --system ScH --threads 24

# Stage 1 -- CIPSI at matched subspace sizes
python tools/run_stage1.py --system ScH --threads 24

# The bond-breaking study
python tools/run_correlation_scan.py --molecule N2 --threads 24

# The dissociation figure
python tools/run_dissociation.py --molecule N2 --out figs/
```

If ScH stage 0 takes hours, something is wrong — most likely `eigsh` got a bad
starting vector, or threads are not set. Expected: minutes.

## 11. Where this points next

Two extensions, both motivated by measurements above and neither attempted:

- **A learned proposal.** GQE reaches only ~2 % of ScH's determinant space. A
  model that proposes determinants *adjacent to the current support* — like
  CIPSI's perturbative expansion, but learned rather than fixed — could target
  the reachable subspace better than a static operator pool. Validated the same
  way everything else here was: against the oracle bound, not against wishful
  accuracy.
- **Multiple active spaces, stitched together.** DMET already fragments a
  molecule and the qubit count scales with fragment size. Running several
  fragments' Hamiltonians in parallel on different machines and recombining them
  breaks inter-fragment correlation; recovering it at the seams is the same
  class of problem as the learned proposal above.

## Next

[`07_validation.md`](07_validation.md) — the checks that keep all of this
honest.
