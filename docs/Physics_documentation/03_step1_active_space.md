# 03 — Step 1: active-space construction

**Code:** `quenais/active_space/finder.py` → `results/step1_asf.pkl`

Which orbitals go into the impurity? Get this wrong and every number
downstream is wrong in a way that looks completely plausible.

---

## 1. The physics: entropy, not energy

An active space is a subset of molecular orbitals within which we will do an
exact (Full CI) treatment, freezing everything else at mean-field level. The
selection criterion is **not** orbital energy.

Consider a single orbital *i* and trace out everything else. The resulting
one-orbital reduced density matrix lives on a four-dimensional local Fock space
(empty, spin-up, spin-down, doubly occupied). Its von Neumann entropy is

```
S_i = -Σ_α w_α ln w_α
```

- `S_i = 0` — the orbital is in a **pure** state: always empty or always doubly
  occupied, in every determinant that matters. It carries no correlation.
- `S_i` large — the orbital's occupation is genuinely *undecided* across the
  wavefunction. That is exactly what multireference character means.

So: **energies order orbitals; only occupations reveal correlation.** An
orbital can sit near the HOMO and be uninteresting, or sit well below and carry
the entire multireference story. This is the single idea the stage rests on.

The automatic selector is **ASF** (`asf.wrapper.find_from_scf`), an external
package that ranks orbitals by entanglement entropy using a DMRG calculation
under the hood (hence the block2 dependency).

## 2. The tier system — how thresholds get chosen

Before selecting anything, `classify()` decides how hard the system looks:

```
has_tm                                              -> tier 3
contaminated  or  HOMO-LUMO gap < 1.0 eV            -> tier 2
otherwise                                           -> tier 1
```

- **`has_tm`** — any atom in a 44-element transition-metal set (Sc…Zn, Y…Cd,
  La/Hf…Hg, the lanthanides, Ac…Pu).
- **spin contamination** — for a singlet, `⟨S²⟩ > 0.05`; for an open shell,
  `⟨S²⟩ / S(S+1) > 1.3`, i.e. 30 % above the exact value. Contamination means
  the single-determinant reference is already struggling.
- **small HOMO–LUMO gap** — near-degeneracy between the frontier orbitals, the
  classic signature of static correlation.

The tier then picks ASF's parameters:

| tier | `entropy_threshold` | `max_norb` | `min_norb` |
|---|---|---|---|
| 1 | `0.05` | 12 | 2 |
| 2 | `0.02` | 14 | 2 |
| 3 | `0.005` | 16 | 4 |

Harder system → lower entropy threshold → more orbitals admitted. Sensible in
principle. In practice, **the thresholds are calibrated on main-group systems
and under-select for the d-block** — see §6.

## 3. The four phases

### Phase A — reference determinant and tier

UHF (`max_cycle=400`, `level_shift=0.5`), with a Newton fallback if DIIS
stalls. Then `classify()`.

### Phase B — MP2 density, then ASF

An MP2 one-particle density matrix is computed **spin-resolved** in the AO
basis. Both halves matter:

> "Spin resolution matters: the embedding's CASCI reference-density path needs
> alpha and beta separately, and reconstructing them from the total is not
> possible."

If MP2 fails it falls back to the UHF density with a warning, and `mp2_ok`
stays `False`.

Then either the forced path (§6) or `find_from_scf(...)`, which returns both
`mo_list` **and** `mo_coeff` — ASF's own natural-orbital coefficients, which
**replace the UHF ones and are what gets saved**. That substitution is the
source of the subtlety in §4.

If ASF returns nothing you get a `RuntimeError` naming the threshold and
suggesting `force_active_space`.

### Phase C — gap detection

`find_gap_cutoff` narrows ASF's candidate list by looking for the largest gap
in MP2 occupation deviation, bounded by `gap_min_norb`/`gap_max_norb`.

Phase C is the one part of this stage that can only ever make things *worse*,
and the code says so:

> "Phase C can only ever SHRINK ASF's candidate list, and it ranks by MP2
> occupation deviation -- a cruder signal than the entanglement entropy ASF used
> to pick those orbitals. On ScH it silently dropped MO 13 from ASF's own
> [10,11,12,13], and the resulting (2e,3o) CASSCF recovered barely half the
> correlation energy of plain CCSD. **Discarding orbitals a better metric
> deliberately selected must be visible.**"

So it warns loudly when it drops anything, and `cfg.asf.phase_c_enabled=False`
turns it off entirely.

### Phase D — Löwdin population

Symmetric orthogonalisation, then per-atom weights for each active MO, giving
`dominant_atoms`. Useful for sanity-checking a forced list: you can confirm the
orbitals you kept actually live on the metal.

## 4. The subtlety that caused a real bug: basis consistency

This is the most important paragraph in the module, and it generalises beyond
this project.

Occupation numbers must be computed in the **same basis** as the `mo_coeff`
that gets saved:

```python
dm_mo     = mo_coeff.T @ S @ dm_ao_total @ S @ mo_coeff
no_occ    = np.clip(np.diag(dm_mo), 0.0, 2.0)
deviation = np.minimum(no_occ, 2.0 - no_occ)
```

> "Occupations used to be computed once in the canonical UHF basis and then
> indexed with mo_list -- indices into ASF's own, different natural-orbital
> basis. Those are not the same orbitals in the same order, so the lookup
> quietly read values off the wrong orbitals."

And the part that explains a symptom nobody could otherwise diagnose:

> "It bites hardest in a degenerate subspace. The true, basis-independent
> occupations of a genuinely symmetric pair ARE equal, because the density
> restricted to that subspace is proportional to the identity -- but the
> canonical-basis array did not show that, because it was not the basis ASF was
> using. That is why N2's degenerate pi pair still got split even after the gap
> cutoff was made degeneracy-aware."

The consequence reaches into step 2: the embedding's CASCI reference density
fills core occupations from `step1["no_occ"]` indexed against
`step1["mo_coeff"]`. Different bases there means a wrong reference density,
which means a wrong bath.

**Rule to carry away:** an array of per-orbital quantities is meaningless
without knowing which basis its indices refer to. Two arrays of the same length
are not interchangeable.

## 5. Degeneracy — why the cutoff gets extended

Degenerate orbitals must be kept or dropped **as a group**. Splitting a
degenerate pair breaks the molecule's symmetry and leaves a physically
incomplete space.

`find_gap_cutoff` therefore extends the cut past the end of any near-degenerate
block:

```python
while k < n and abs(sorted_v[k-1] - sorted_v[k]) < degeneracy_tol:
    k += 1
```

> "Without this, N2's two pi orbitals -- identical entanglement entropy,
> S=0.246 each -- got split, one kept and its degenerate partner dropped,
> leaving a symmetry-broken and physically incomplete active space."

Two warnings can fire: one when the extension happens, and a second when it
pushes past `gap_max_norb`. Note **`k` is not clamped back** — the cap is
advisory, because splitting a degenerate group is considered worse than
exceeding the bound.

`reference_values.py` records ScH's pair explicitly as
`"degenerate_pairs": [(11, 12)]`, and `test_reference_harness.py` asserts
`(11 in mo_list) == (12 in mo_list)`.

## 6. `force_active_space` — the escape hatch you will need

**ASF under-selects for transition metals.** On ScH the automatic choice kept
4 orbitals but only **2 active electrons**, treating a genuinely valence orbital
as core. Every transition-metal result in this project uses a forced space.

```python
cfg.asf = AsfSettings(force_active_space=[9, 10, 11, 12, 13, 14])
```

```bash
quenais-run --molecule ScH --basis sto-3g --force-active-space 9 10 11 12 13 14 --steps 0 1 2
```

Four things to know:

1. **Indices are 0-based, in the UHF alpha-MO basis.** Not RHF, not 1-based.
2. **Forcing skips ASF and DMRG entirely** — so no block2 install is needed,
   and it is much faster. The code's justification:
   > "Worth keeping cheap: transition-metal systems currently need a forced
   > space, and requiring a working block2 install to bypass block2 would be
   > perverse."
   Phase C is skipped on this path too.
3. **Keep degenerate orbitals together** (ScH's 11/12).
4. **Only out-of-range indices are caught.** The validator raises
   `"force_active_space contains MO indices [...] that do not exist for
   <molecule>/<basis> (N orbitals). These are typically indices copied from a
   different molecule."` In-range-but-wrong indices raise nothing at all —
   check them against the deviation spectrum and `dominant_atoms`.

Confirm it took by looking for `Selection: forced (cfg.asf.force_active_space)`
in the output, or `step1["forced_active_space"]` programmatically.

Note that on the forced path `mo_coeff` saved in the pickle is the **UHF alpha**
basis, whereas on the ASF path it is ASF's natural-orbital basis. Per §4, that
distinction matters.

## 7. Counting the active electrons

```python
core_orbs = [i for i, occ in enumerate(mo_occ_total)
             if i not in active_set and occ > cfg.asf.core_occ_threshold]
nel = mol.nelectron - 2 * len(core_orbs)
```

`core_occ_threshold` defaults to `1.95` — effectively "doubly occupied". Then:
`nel <= 0` raises; `nel > 2 × n_orbitals` warns and caps; an odd count is
decremented; the floor is 2.

You do **not** specify the electron count for a forced space — it is derived.
That is why ScH's forced 6 orbitals give CAS(4e,6o) rather than (6e,6o).

## 8. Output contract

```
nel, mo_list, mo_coeff, n_active_orbs, no_occ, deviation, lowdin_weights,
dominant_atoms, tier, indicators, corr_strength, mol_info, uhf_energy,
mp2_energy, mp2_ok, mo_coeff_uhf, mo_energy, mo_occ, converged,
dm_ao_alpha_mp2, dm_ao_beta_mp2, dm_ao_total_mp2, forced_active_space,
gap_value, provenance
```

The three `dm_ao_*_mp2` keys are a hard requirement of step 2:

> "quenais.embedding requires dm_ao_alpha_mp2 / dm_ao_beta_mp2 /
> dm_ao_total_mp2 from it -- the CASCI reference-density path raises KeyError
> without them. The 0.1 package collapsed MP2 to a single total density and
> never spin-resolved it, which is why step 2 could not run in its default mode."

## Next

[`04_step2_embedding.md`](04_step2_embedding.md) — the heart of the method.
