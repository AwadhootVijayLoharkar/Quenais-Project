# 02 — Step 0: classical reference methods

**Code:** `quenais/classical/runner.py` → `results/step0_classical.pkl`

These energies are the answer key everything else is validated against. From
the module docstring:

> "Every bug found during the N2 and LiH work was caught by a disagreement with
> a number produced here."

---

## 1. The physics: a ladder of approximations

The electronic Hamiltonian, in second quantisation:

```
H = Σ_pq h_pq a†_p a_q  +  ½ Σ_pqrs (pq|rs) a†_p a†_r a_s a_q
```

`h_pq` are one-electron integrals (kinetic + nuclear attraction), `(pq|rs)` are
two-electron repulsion integrals. Solving this exactly in a given orbital basis
is Full CI, and its cost is combinatorial. The classical methods are a ladder
of approximations that trade accuracy for cost.

### The methods, in order

**HF (Hartree–Fock)** — mean field. Each electron feels the *average* field of
the others. Recovers ≈99 % of the total energy. Everything missing from it is
called the **correlation energy**, and essentially all of chemistry lives
there: chemical accuracy is 1.6 mHa, a fraction of a percent of the correlation
energy, which is itself a fraction of a percent of the total.

Implemented as `scf.RHF` when `spin == 0`, else `scf.UHF`, with `max_cycle=400`
and `level_shift=0.3`. If DIIS does not converge it falls back to a
second-order Newton solver:

> "Level shifting gets most cases close but can stall just short of the
> tolerance."

If both fail you get a `RuntimeWarning` saying to treat *all* of step 0 as
unreliable — every method below builds on this reference.

**MP2** — second-order perturbation theory on top of HF. Cheap, and the first
thing that captures dynamic correlation. Fails when the HF reference is
qualitatively wrong, which is exactly the strongly correlated case.

**CCSD / CCSD(T)** — coupled cluster with single and double excitations, and
with a perturbative triples correction. CCSD(T) is the "gold standard" of
single-reference quantum chemistry.

**The catch that matters for this project:** coupled cluster is
**non-variational**. There is no bound stopping it from going *below* the true
energy. As a bond stretches, its amplitudes diverge and the CCSD curve crosses
through the exact answer. Measured on N₂/STO-3G: `max|t₂| = 0.65` at 1.8 Å and
`0.84` at 2.1 Å, where a healthy value is below 0.1. That crossing is the
cleanest available demonstration that a system is strongly correlated — see
`tools/run_dissociation.py` and
[`06_determinant_selection.md`](06_determinant_selection.md).

**CASSCF** — complete active space SCF. Picks a subset of orbitals (the active
space), does Full CI within it, and simultaneously optimises the orbitals. This
is a *multireference* method: it can describe several near-degenerate
configurations on an equal footing, which is what strong correlation requires.

**NEVPT2** — second-order perturbation theory built on the CASSCF reference,
adding the dynamic correlation CASSCF misses.

## 2. The three reproducibility tiers

This is the most important idea in step 0, and it propagates through the whole
package.

```python
METHOD_TIERS = {
    "HF": "deterministic",      "MP2": "deterministic",
    "CCSD": "deterministic",    "CCSD_T": "deterministic",
    "CASSCF": "optimizer-dependent",
    "NEVPT2": "optimizer-dependent",
}
```

> "CASSCF is an optimisation and can converge to different valid solutions: two
> runs of identical ScH input gave -752.680677 and -752.681604, 0.93 mHa apart.
> NEVPT2 is built on the CASSCF reference and moves with it (3.6 mHa on the same
> pair of runs). The single-determinant methods reproduce to ~1e-10 across
> machines."

The tolerances that go with the tiers (`tests/regression/reference_values.py`):

| tier | tolerance | methods |
|---|---|---|
| `DETERMINISTIC` | `1e-9` | HF, MP2, CCSD, CCSD(T), DMET+CASCI, `ecore`, `mu`, σ spectrum |
| `OPTIMIZER_DEPENDENT` | `2e-3` | CASSCF, NEVPT2 |
| `STOCHASTIC` | `5e-2` | DMET+GQE |

> "A single tolerance across all quantities is wrong, and asserting one would
> make the suite fail for reasons unrelated to the code."

> "Every bug in this project's history was 'right shape, plausible magnitude,
> wrong value' -- so the DETERMINISTIC tier is the one that actually catches
> regressions, and it is deliberately tight."

**Practical rule:** never write a regression test asserting CASSCF or NEVPT2 to
`1e-6`. It will pass on your machine and fail on someone else's, and you will
spend a day chasing a bug that is not there.

Every energy in `results_summary.csv` carries its tier in a `reproducibility`
column for exactly this reason — that CSV is the file a partner is most likely
to send back.

## 3. The active-space coupling — run step 1 first

CASSCF and NEVPT2 need an active space. When `step1_asf.pkl` exists and matches
the current molecule and basis, step 0 reuses it: `nel`, `n_active_orbs` and
`mo_list` (the last passed to `mcscf.addons.sort_mo` as a 0-based orbital
guess).

When it does not, there is a **fallback guess**:

```python
nel  = min(mol.nelectron, 10)
norb = min(mol.nao_nr() // 2, 8)
```

and a warning if `nel >= 2 * norb`:

> "Fallback active space (Ne, No) leaves no correlating degrees of freedom --
> CASSCF will trivially return the HF energy. Run step 1 first."

That is not a stylistic complaint. Cramming N electrons into N/2 orbitals means
every orbital is doubly occupied in every determinant, so the "Full CI" inside
that space has exactly one configuration and returns HF exactly.

**Recommended order:** run steps 0 1 2, then re-run step 0 with `--force` so
CASSCF and NEVPT2 use the real active space.

Step 0 distinguishes three cases and says which one it hit:

- valid cached step 1 → `"Step 1 loaded: (Ne, Norb)"`
- step 1 exists but for a different molecule/basis → ignored, fallback used
- step 1 absent → fallback used, with a note to run step 1 first

## 4. The diagnostic hidden in this stage

A well-chosen active space puts **CASSCF + NEVPT2 at or below CCSD(T)**. If
NEVPT2 lands *above* CCSD(T), the active space is too small — the
multireference treatment has nothing to work with.

On ScH, NEVPT2 lands **7.2 mHa above** CCSD(T) (`−752.702671` vs
`−752.709890`). That is how the d-block under-selection was found, and it is
why ScH's classical reference is softer than LiH's or N₂'s. From
`reference_values.py`:

> "do not build tight regression assertions on ScH CASSCF or NEVPT2. The DMET
> quantities on this system ARE reliable -- DMET+CASCI reproduced to 1e-10
> across two runs on different hardware."

This diagnostic is the main reason to bother running CASSCF and NEVPT2 at all
on a new system.

## 5. Output

```python
{
  "molecule": str, "basis": str, "total_time": float,
  "provenance": {...},
  "methods": {
     "HF":     {"energy", "converged", "tier"},
     "MP2":    {"energy", "e_corr", "success", "tier"},
     "CCSD":   {"energy", "e_corr", "success", "converged", "tier"},
     "CCSD_T": {"energy", "e_t_correction", "success", "tier"},
     "CASSCF": {"energy", "nel", "norb", "success", "converged", "tier"},
     "NEVPT2": {"energy", "success", "tier"},
  },
}
```

HF always runs regardless of `classical_methods`. NEVPT2 depends on the CASSCF
object, so requesting NEVPT2 without CASSCF gives
`"Skipped -- CASSCF not available"`. A failed method has `energy: None` and
prints as `FAILED` in the table rather than aborting the stage.

## 6. A rewrite worth knowing about

> "REWRITTEN in 0.2. The 0.1 version of this module was structurally broken: an
> indentation error left main() ending after the banner, with the entire
> run-and-save body absorbed into _run_nevpt2()'s scope. It parsed, imported and
> ran without error -- and did nothing, returning None and writing no pickle."

And a smaller one worth quoting because it is such a good example of the
project's failure mode:

> "The class is named `NEVPT`, exported as pyscf.mrpt.NEVPT. 'NEVPT2' is the
> method's name in the literature, not the class name -- asking for
> pyscf.mrpt.nevpt2.NEVPT2 raises AttributeError, which surfaced as a bare
> 'NEVPT2 FAILED' row in the results table with the real cause buried in a
> stderr warning. A wrong API name, not a convergence or memory problem."

## Next

[`03_step1_active_space.md`](03_step1_active_space.md) — choosing which
orbitals matter.
