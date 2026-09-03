# 09 — Replacing ASF: AVAS vs APC

Companion to [03 — Step 1: active space](03_step1_active_space.md) and
[limitations.md](../limitations.md). It answers one question: ASF
under-selects for the d-block and ScH has needed
`force_active_space=[9,10,11,12,13,14]` since 0.2 — can a PySCF selector
replace it, and which one.

**Short answer: AVAS.** APC is the more interesting algorithm and the
better general-purpose ASF substitute, but AVAS is the one whose mechanism
actually addresses the failure this project has.

Everything below about the two algorithms was read off the PySCF source,
not the abstracts. Nothing here has been executed — PySCF is not installed
in the environment this was written in — so the numbers in §7 are an
acceptance test to run, not results.

---

## 1. What the two methods actually are

They look interchangeable from the outside. They are not: they disagree on
what an active space *is*.

| | **AVAS** | **APC** |
|---|---|---|
| module | `pyscf.mcscf.avas` | `pyscf.mcscf.apc` |
| question it answers | "which MOs carry the character of *these atomic orbitals*?" | "which MOs are most strongly correlated, by a cheap estimator?" |
| input beyond `mf` | AO labels, e.g. `['Sc 3d','Sc 4s','H 1s']` | `max_size` (and optionally `n`) |
| ranking quantity | eigenvalues of the projector of the MOs onto a minimal-AO reference set | "APC entropy" from HF Fock + exchange coupling, `c = −K/(Δ + √(K²+Δ²))` |
| chemical input | **required** — you name the shell | none |
| needs DMRG / block2 | no | no |
| deterministic | yes (eigendecomposition) | yes |
| entry point | `avas.kernel(mf, aolabels, threshold=0.2, minao='minao', openshell_option=2, canonicalize=True, ncore=0)` | `apc.APC(mf, max_size=..., n=..., fixed=...).kernel()` |
| returns | `(ncas, nelecas, mo)` | `(ncas, nelecas, mo)` — *"following AVAS convention"* |
| UHF | `isinstance(mf, scf.uhf.UHF)` → "AVAS takes alpha orbitals only" | `isinstance(mf, scf.uhf.UHF)` → averaged F, summed K, summed occupations, alpha orbitals |

The last two rows are the load-bearing ones for us. **Both return the same
triple, in the same orbital ordering, and both accept the UHF object step 1
already builds in Phase A.** One adapter serves both, and adding the second
one later costs almost nothing. That is why the recommendation below is
"start with AVAS" rather than "AVAS instead of APC forever."

---

## 2. Why AVAS, and not APC, for this project

### 2.1 It targets the actual failure

The failure in `limitations.md` is not "ASF picked a slightly small space."
It is that ASF *classified MO 9 as inert core*, giving 2 active electrons
where ScH's valence count is 4, and then Phase C cut 4 orbitals to 3. The
resulting CASSCF(2e,3o) recovered −22.79 kcal/mol against HF where plain
CCSD recovered −44.01.

APC would attack that with a better correlation estimator. But it is still
an estimator with a cutoff — it ranks orbitals by a Fock/exchange coupling
score and keeps the top `max_size`. A compact 3d shell whose exchange
coupling to the occupied manifold is small can rank low under that score
exactly as it ranked low under ASF's entropy threshold. Different metric,
same failure shape.

AVAS does not have a failure of that shape available to it. You write
`'Sc 3d'` and the 3d manifold is in the space by construction. The
threshold in AVAS decides how many MOs carry that character, not whether
the character matters.

### 2.2 It keeps degenerate shells intact for free

`find_gap_cutoff()` carries a whole degeneracy-extension mechanism, and a
`RuntimeWarning`, because N₂'s two π orbitals (S = 0.246 each) were being
split across the cutoff — one kept, its partner dropped. AVAS projects onto
an entire AO shell: all five d functions, both π components. A symmetric
pair cannot be split, because the projector does not rank within a shell.
That is not a patch on the bug — it is a selection rule the bug cannot
occur under.

### 2.3 It is stable along a geometry scan, and `force_active_space` is not

This is the argument I would lead with, given where the project is going.

`force_active_space=[9,10,11,12,13,14]` is a list of **MO indices**. MO
indices are positions in an energy-ordered list. Stretch the Sc–H bond and
that list reorders: the orbital that is index 13 at 1.78 Å is not
necessarily the same physical orbital at 2.6 Å. A forced index list is
therefore only valid at the geometry it was calibrated at, and silently
wrong elsewhere — with no warning, because every index is still in range
and `finder.py` only validates `i >= n_ao`.

The N₂ bond-breaking scan in [06](06_determinant_selection.md) got away
with this because N₂ runs on automatic selection. Any ScH dissociation
curve, or any correlation scan on a transition metal, hits it directly.

AVAS re-derives the space from atomic character at every geometry. The
active space stays the *same physical object* along the curve even as the
indices move under it. For a project whose central result is a w₁ scan
across bond lengths, that is the difference between a scan you can publish
and one you have to caveat.

### 2.4 It removes a dependency rather than adding one

The ASF path needs block2, a `BLOCKEXE` wrapper, `MKL_THREADING_LAYER=GNU`,
`MKL_DEBUG_CPU_TYPE=5`, and a `_validate_block2_wrapper()` whose whole job
is to catch a wrapper pointing at a stale conda env. AVAS needs
`mol.intor` and an eigendecomposition. On a forced-space run today you
already skip block2 entirely — AVAS gets you the same freedom without
giving up automatic selection.

### 2.5 It is what the transition-metal literature actually uses

AVAS (Sayfutyarova, Sun, Chan, Knizia, 2017) was written for this problem
and is the default choice for TM active spaces in production CASSCF work.
APC's two cited papers are about π-orbital active spaces and large-scale
benchmarking of **vertical excitation energies** — organic chromophores.
I could not read either paper's benchmark tables (JCTC returned 403,
ChemRxiv served a CAPTCHA), so I cannot tell you APC has been shown to fail
on d-block systems. I can tell you nobody has published that it works, and
that its own module contains no transition-metal-specific handling.

---

## 3. Where APC still earns a place

Two cases, both real:

**Systems where you cannot name the shell.** AVAS's requirement is also its
cost — someone has to decide that ScH means `Sc 3d`, `Sc 4s`, `H 1s`. For a
CIF-loaded system nobody has looked at, or for a batch scan over a set of
candidate molecules, "no chemical input required" is worth something. APC
is the honest automatic fallback there, and a better one than ASF because
it needs no DMRG.

**Open-shell references.** APC gives singly-occupied orbitals an
artificially high entropy so they are always selected. Once you move to
ScO/TiO (flagged in `limitations.md` as the untested open-shell branch),
that is a property worth having. AVAS has its own open-shell handling via
`openshell_option`, so this is not decisive — but it is a reason to keep
APC on the roster rather than treat the choice as either/or.

The design in §5 therefore adds a *selector registry*, not a second
hardcoded branch. Both cost the same adapter.

---

## 4. Caveats — read before believing §7

**STO-3G weakens AVAS's discriminating power.** AVAS works by projecting
your MOs onto a minimal AO reference basis (`minao`). When the calculation
basis *is already minimal*, that projection is close to an identity and the
eigenvalue spectrum it thresholds is much less structured than it would be
in, say, cc-pVTZ. AVAS should still separate d-character from non-d
character — that is the part we need — but do not expect the
`threshold=0.2` default to be as meaningful as it is in the literature,
where AVAS is almost always run in a polarised basis. Expect to scan the
threshold on ScH and expect the result to be a plateau, not a sharp
optimum. If AVAS behaves badly here, the first thing to try is not a
different selector but the same selector in a larger basis.

**`nelecas` must come from the selector.** `count_active_electrons()`
computes the active electron count from `mf.mo_occ` — occupations in the
*canonical UHF basis* — indexed against `final_mo_list`. After AVAS or APC,
`final_mo_list` indexes the **selector's rotated basis**, not the UHF one.
Feeding one to the other is exactly the basis-mismatch class of bug that
`project_occupations()` exists to prevent, and it would be silent. Both
selectors return `nelecas` directly. Use it; do not recompute.

**Degenerate-orbital rotation stays arbitrary.** AVAS canonicalises within
the core/active/virtual blocks, so the orbitals it returns for a degenerate
shell are defined only up to a rotation within that shell. The CASCI energy
and the Schmidt spectrum are invariant to it; individual `mo_coeff`
columns, Löwdin weights and any per-orbital plot are not. Same gotcha the
project already documents for the N₂ π pair — it does not get worse, it
just does not go away.

**This changes step 1's output, so it changes everything downstream.** The
golden pickles in `tests/regression/golden/` were produced with the forced
space. A run under AVAS is a different active space and will not match
them, and *should* not. Treat AVAS as a new configuration with its own
golden data, not as a change that should reproduce the old numbers.

---

## 5. Integration design

The good news: `finder.py`'s output contract already accommodates a
selector that returns its own orbital basis. That is precisely what ASF
does today — `active_space.mo_coeff` is ASF's natural-orbital basis, not
the canonical UHF one, and `project_occupations()` was written for it.
AVAS and APC slot into the same seam.

### 5.1 Settings — `quenais/settings/asf.py`

```python
#: Which selector produces the active space.
#:   "asf"  -- ASF/DMRG entanglement entropy (default, today's behaviour)
#:   "avas" -- project onto named atomic valence orbitals (transition metals)
#:   "apc"  -- PySCF's ranked-orbital APC entropy (automatic, no AO labels)
#: force_active_space still overrides all three.
method: str = "asf"

#: AVAS: AO labels defining the valence shell, PySCF search_ao_label syntax
#: (['Sc 3d', 'Sc 4s', 'H 1s']). None -> derived from the elements present
#: by default_ao_labels(); pass explicitly for anything unusual.
avas_ao_labels: list | None = None

#: AVAS: projector eigenvalue above which an MO counts as carrying the
#: reference character. PySCF's default. In a minimal basis expect a
#: plateau rather than a sharp optimum -- see docs/.../09.
avas_threshold: float = 0.2
avas_minao: str = "minao"
avas_openshell_option: int = 2

#: APC: maximum active-space size, and the APC-n parameter (how many times
#: to strip the highest-entropy virtual before re-scoring; higher n favours
#: fewer doubly-occupied orbitals).
apc_max_size: int = 8
apc_n: int = 2
```

and in `validate()`:

```python
if self.method not in SELECTORS:
    raise ValueError(
        f"asf.method must be one of {sorted(SELECTORS)}, got {self.method!r}"
    )
if self.method == "avas" and self.avas_ao_labels is not None:
    if not all(isinstance(s, str) for s in self.avas_ao_labels):
        raise ValueError("avas_ao_labels must be strings, e.g. ['Sc 3d']")
if not 0.0 < self.avas_threshold < 1.0:
    raise ValueError(f"avas_threshold must lie in (0,1), got {self.avas_threshold}")
if self.apc_max_size < 1:
    raise ValueError(f"apc_max_size must be >= 1, got {self.apc_max_size}")
```

### 5.2 New module — `quenais/active_space/selectors.py`

A draft is in `selectors.py` alongside this document. Its shape:

```python
Selection = namedtuple("Selection", "mo_list mo_coeff nel meta")

def select_avas(mf, mol, cfg) -> Selection
def select_apc(mf, mol, cfg)  -> Selection
def default_ao_labels(mol, tiers) -> list[str]

SELECTORS = {"asf": ..., "avas": select_avas, "apc": select_apc}
```

The one piece of real logic is turning PySCF's return into our contract.
Both selectors return `mo = hstack((mofreeze, mocore, mocas, movir))`, so
the active orbitals are contiguous and start after the doubly-occupied
core:

```python
ncore = (mol.nelectron - nelecas) // 2
mo_list = list(range(ncore, ncore + ncas))
```

`nel = nelecas`, straight from the selector — see §4.

### 5.3 `finder.py` — the branch at line 413

Today `main()` is a two-way branch: `forced` or ASF. It becomes three-way,
with `forced` still winning:

```python
if forced:
    ...                                    # unchanged
elif cfg.asf.method in ("avas", "apc"):
    sel = SELECTORS[cfg.asf.method](mf, mol, cfg)
    final_mo_list, mo_coeff = sel.mo_list, sel.mo_coeff
    n_final, gap_val = len(final_mo_list), 0.0
    nel_from_selector = sel.nel                # <- do NOT recompute
    selection_meta = sel.meta
else:
    ...                                    # ASF path, unchanged
```

and then, at line 478:

```python
if nel_from_selector is not None:
    nel = nel_from_selector
    print(f"  Active electrons from {cfg.asf.method.upper()}: {nel}")
else:
    nel = count_active_electrons(mol, mf, final_mo_list, cfg)
```

**Phase C must not run on an AVAS/APC space.** Phase C narrows a candidate
list by MP2 occupation deviation, and its own docstring says it can only
shrink a selection chosen by a better metric — that is how ScH lost MO 13.
Applied to AVAS it would discard orbitals you asked for by name. It is
already skipped on the forced path; skip it here for the same reason.

`project_occupations()` still runs, unchanged and still necessary: it
recomputes occupations in the AVAS/APC basis so `no_occ` and `deviation`
stay consistent with the saved `mo_coeff`, which the embedding's CASCI
reference-density path depends on.

### 5.4 Pickle contract

Add, don't replace:

```python
"selection_method": "forced" | "asf" | "avas" | "apc",
"selection_meta": {...},   # AO labels + threshold, or max_size + n
```

Keep `forced_active_space: bool` as-is so existing readers and
`compare_pickles.py` do not break. `selection_meta` is what makes a run
reproducible from the pickle alone — with ASF the tier and thresholds were
enough; with AVAS the AO label list *is* the physics and belongs in
provenance.

### 5.5 CLI — `quenais/cli.py`

```python
parser.add_argument(
    "--active-space-method", default="asf", choices=["asf", "avas", "apc"],
    help="active-space selector. 'avas' projects onto named atomic valence "
         "orbitals and is the recommended choice for transition metals; "
         "'apc' is automatic and needs no AO labels. --force-active-space "
         "overrides all three.",
)
parser.add_argument(
    "--avas-ao-labels", nargs="+", default=None,
    help="AO labels for --active-space-method avas, e.g. Sc 3d 'Sc 4s'. "
         "Default: derived from the elements present.",
)
```

threaded into `AsfSettings(...)` at line 201. Note `--avas-ao-labels`
entries contain spaces, so they need quoting — worth saying in the help
text, because `Sc 3d` unquoted becomes two labels and
`mol.search_ao_label('Sc')` matches every Sc orbital including the core.

---

## 6. What this does *not* change

Steps 2–4 need no edits. `hamiltonian.py` takes `mo_list` + `mo_coeff` and
slices `C_imp = mo_coeff[:, mo_list]`; `dmet_lib.get_reference_density()`
transforms the MP2 densities into whatever basis `mo_coeff` is. Neither
cares where the orbitals came from, provided `mo_coeff` is orthonormal
under `S` — which both PySCF selectors guarantee, and which
`verify_embedded_scf()`'s 2e-7 check would catch if it were ever false.

That check is, incidentally, the reason this is a low-risk change: if an
AVAS space were handed downstream in an inconsistent basis, the embedded
SCF would not reproduce the full UHF energy and step 2 would say so.

---

## 7. Acceptance test — run this before writing any of §5

`probe_avas_apc.py` (alongside this doc) is standalone: it builds ScH,
runs UHF, calls AVAS across a threshold scan and APC across a size scan,
and for each candidate space reports orbital count, electron count,
Löwdin d-character on Sc, CASSCF, and NEVPT2 — against the forced
`[9,10,11,12,13,14]` space as the control.

```bash
python probe_avas_apc.py --molecule ScH --basis sto-3g
```

The project already owns the pass/fail criterion. From `limitations.md`:

> **How to tell it is happening:** a well-chosen active space should put
> CASSCF+NEVPT2 at or below CCSD(T). If NEVPT2 lands above CCSD(T), the
> active space is too small.

So:

| observation | conclusion |
|---|---|
| AVAS gives ≈(4e,6–7o) with the Sc 3d manifold whole, NEVPT2 ≤ CCSD(T) | integrate per §5; retire `force_active_space` for ScH |
| AVAS gives that space but NEVPT2 still ≈7 mHa above CCSD(T) | matches the forced space's own behaviour — AVAS is not worse, and ScH's classical reference is the soft part. Still integrate; keep the caveat |
| AVAS drops d orbitals at `threshold=0.2` but recovers them lower | integrate, and record the working threshold in `selection_meta` |
| AVAS cannot recover them at any threshold in STO-3G | re-run the probe in cc-pVDZ before concluding anything — see §4 |
| APC ≥ AVAS on all of the above | reconsider; APC needs no AO labels, which is worth a lot |

Compare on **energy and d-character, not on index equality**. AVAS returns a
rotated basis, so its `mo_list` will not read `[9,...,14]` even when it has
selected exactly the same physical space. Comparing index lists across
bases is the mistake `project_occupations()`'s docstring is about.

Second test, and the one that justifies the change on its own: run the
probe at 1.78, 2.2 and 2.6 Å. If AVAS returns the same physical space at
all three while the forced index list does not, that settles it.

---

## 8. Effort

| | |
|---|---|
| `selectors.py` + settings + validation | ~150 lines, draft already written |
| `finder.py` branch + skip Phase C + nel handling | ~25 lines |
| CLI + pickle keys + provenance | ~20 lines |
| tests: selector unit test, one regression pickle per method | the real work |
| **risk** | low — additive, default `method="asf"` keeps every current result bit-identical |

The honest cost is not the code, it is that ScH acquires a second golden
dataset and `07_validation.md`'s tripwire table grows a row.

---

## References

- AVAS: Sayfutyarova, Sun, Chan, Knizia, *Automated Construction of
  Molecular Active Spaces from Atomic Valence Orbitals*,
  [arXiv:1701.07862](https://arxiv.org/abs/1701.07862);
  [`pyscf/mcscf/avas.py`](https://pyscf.org/_modules/pyscf/mcscf/avas.html)
- APC: King & Head-Gordon, *The Ranked-Orbital Approach to Selecting Active
  Spaces*, doi:10.1021/acs.jctc.1c00037; benchmarking,
  doi:10.1021/acs.jctc.2c00630;
  [`pyscf/mcscf/apc.py`](https://pyscf.org/_modules/pyscf/mcscf/apc.html)
- [`pyscf.mcscf` API index](https://pyscf.org/pyscf_api_docs/pyscf.mcscf.html)