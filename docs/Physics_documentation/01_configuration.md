# 01 — Configuration, geometry, and caching

Everything the pipeline does is driven by one `Config` object. This document
covers what is in it, how a molecule gets specified, and the caching rule that
has bitten this project three separate times.

**Code:** `quenais/config.py`, `quenais/settings/`, `quenais/utils/geometry.py`,
`quenais/cli.py`

---

## 1. Why settings are grouped rather than flat

From `quenais/config.py`:

> "Grouping is deliberate. The flat version of this configuration grew to
> roughly 800 lines mixing paths, constants, CIF parsing, thresholds and
> training hyperparameters. Adding a nested group costs one line here; adding a
> flat field costs three, and it never stops."

So `Config` holds molecule identity, paths and the solver choice; everything
else lives in one of five dataclasses:

| group | class | governs |
|---|---|---|
| `cfg.asf` | `AsfSettings` | active-space selection (step 1) |
| `cfg.dmet` | `DmetSettings` | the embedding (step 2) |
| `cfg.qiskit` | `QiskitSolverSettings` | the SQD family (step 3) |
| `cfg.gqe` | `GqeSettings` | the CUDA-Q GQE solver (step 3) |
| `cfg.tiers` | `TierSettings` | how a system is classified for step 1 |

A useful property of this subpackage: **nothing in `quenais/settings/` imports
NumPy, PySCF, Qiskit or CUDA-Q.** That is what lets `--help` work on an install
with none of the heavy stack present.

## 2. The solver registry

```python
QISKIT_SOLVERS = ("sqd", "skqd", "sqdrift")
GQE_SOLVERS    = ("gqe",)
SOLVERS        = QISKIT_SOLVERS + GQE_SOLVERS
SOLVER_ALIASES = {"gqe_qsci": "gqe"}      # deprecated, Python API only
```

> "the single source of truth. Config.validate(), the CLI's --solver choices
> and quenais.quantum.dispatch all read these. Three hand-maintained lists is
> how a solver ends up accepted in one place and rejected in another."

`cfg.is_gqe` and `cfg.is_qiskit` are properties derived from it, and
`validate()` only validates the stack you actually selected — so a Qiskit-only
user is never blocked by a GQE setting they have not touched.

## 3. The four ways to specify a molecule

`load_geometry()` tries them in this exact order:

1. **`Config(geometry=...)`** — a PySCF-style string (`"Li 0 0 0; H 0 0 1.5949"`)
   or a list of `(symbol, (x, y, z))` tuples.
2. **`Config(xyz=...)`** — a path to an XYZ file.
3. **A built-in name** — `LiH`, `N2`, `ScH`, `H2O`.
4. **A CIF** at `<project_dir>/cif_files/<molecule>.cif`.

Passing both `geometry=` and `xyz=` raises:
`"pass either geometry= or xyz=, not both -- otherwise which one was actually used is a coin toss"`.

Units are **Ångström** throughout, matching PySCF's default.

### The built-in table

```python
BUILTIN_GEOMETRIES = {
    "ScH":  [("Sc", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7800))],
    "LiH":  [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))],
    "N2":   [("N",  (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
    "H2O":  [("O",  (0.0, 0.0, 0.1173)), ("H", (0.0, 0.7572, -0.4692)),
                                         ("H", (0.0, -0.7572, -0.4692))],
}
```

> "LiH, N2 and ScH are the systems the reference values in tests/regression/
> were produced with -- **do not change them without regenerating the golden
> data.**"

Note `Config`'s default molecule is `"TiO2"`, which is **not** in this table —
a default-constructed `Config().load_geometry()` therefore requires a CIF. The
CLI defaults differ deliberately (`--molecule LiH --basis sto-3g` vs `Config`'s
`TiO2` / `def2-svp`).

### Input robustness

`normalise_geometry()` accepts every shape people actually write —
`[("H", (0,0,0))]`, `[("H", [0,0,0])]`, `[("H", 0, 0, 0)]`, `[["H", 0, 0, 0]]` —
and returns the canonical form. `_clean_symbol()` turns `'FE'`, `'fe'` and
`'Fe1'` into `'Fe'`.

`parse_xyz()` treats a wrong atom count in line 1 as a **hard error**, not a
warning:

> "A wrong count in line 1 is a hard error -- it usually means the file was
> truncated, and silently reading fewer atoms than intended would produce a
> plausible wrong answer."

`write_xyz()` exists so you can record what was *actually* run. Use it. A
geometry retyped from a paper into a config and never echoed back is the
cheapest possible source of a wrong answer.

## 4. Derived paths

```
project_dir/
  results/
    step0_classical.pkl        cfg.step0_file
    step1_asf.pkl              cfg.step1_file
    step2_hamiltonian.pkl      cfg.step2_file
    step3_results.pkl          cfg.step3_file
    gqe_train.log              cfg.gqe_log_file
    results_summary.csv
    plots/                     cfg.plots_dir
  cif_files/                   cfg.cif_dir
```

## 5. The caching rule — read this one

Filenames are **fixed and shared across molecules**. A naive
`os.path.exists()` cache check therefore reuses the previous system's results
silently. From `config.py`:

> "That exact failure cost real debugging time three separate times in this
> project. The paths stay fixed on purpose -- the external gqe-for-qsci repo's
> molecule config references the step 2 path, so making the filenames
> molecule-specific would trade one stale-path bug for another. **Validating the
> contents is the fix that does not.**"

`cached_result_is_current(path)` is that fix. It returns `False` when the file
is absent, unreadable, carries no molecule tag, or carries a *different*
molecule or basis than the current config. Every stage calls it before reusing
a pickle, and `--force` / `force=True` bypasses the cache entirely.

Practical consequence: **if you change a setting and re-run, you will get the
old answer back unless you pass `--force`.** The cache validates *identity*
(molecule, basis), not *settings*.

## 6. Validation order

`validate()` runs, in order:

1. `spin >= 0`
2. `quantum_solver in SOLVERS`
3. `classical_methods` non-empty
4. `asf.validate()`, `dmet.validate()`, `tiers.validate()` — always
5. `qiskit.validate()` **only if** `is_qiskit`; `gqe.validate()` **only if** `is_gqe`

The CLI's `build_config()` ends with `cfg.validate().make_dirs().load_geometry()`
— in that order. So validation runs *before* directories exist and *before* the
geometry is resolved, which means an `--xyz` error surfaces after `results/`
has already been created. Harmless, but it surprises people.

## 7. Settings reference — the defaults that matter

Full field lists are in the dataclasses; these are the ones with physics
consequences.

### `AsfSettings` (step 1)

| field | default | note |
|---|---|---|
| `params` | per-tier entropy thresholds | tier 1 `0.05`, tier 2 `0.02`, tier 3 `0.005` |
| `gap_min_norb` / `gap_max_norb` | `2` / `16` | bounds on the gap cutoff |
| `gap_degeneracy_tol` | `1e-3` | keeps degenerate orbitals together |
| `core_occ_threshold` | `1.95` | occupation above which an orbital counts as core |
| `force_active_space` | `None` | explicit 0-based MO indices, **UHF alpha basis** |
| `phase_c_enabled` | `True` | Phase C can only ever *shrink* ASF's selection |

### `DmetSettings` (step 2)

| field | default | note |
|---|---|---|
| `bath_tolerance` | `1e-8` | **load-bearing** — see below |
| `max_embed_orbs` | `18` | each embedding orbital costs 2 qubits |
| `reference` | `"casci"` | or `"mp2"` (fast, unreliable where static correlation is strong) |
| `mu_correction` | `True` | provably inert for fixed-N solvers; kept for those that are not |
| `mu_search_range` | `"auto"` | a fixed bracket does not work — see below |

On `bath_tolerance`:

> "on N2's (4e,4o) space every Schmidt singular value comes back numerically
> zero (measured 5.4e-15) ... The correct answer there is zero bath orbitals.
> Manufacturing a bath from those values instead produces a badly
> non-orthonormal embedding basis and ~20 Ha errors."

On `max_embed_orbs = 18`:

> "18 is the validated value -- every reference number in
> tests/regression/reference_values.py was produced with it. (The 0.1 package
> shipped 24, which no validated run used.)"

On `mu_search_range = "auto"`:

> "A fixed guess such as (-5, 5) Ha fails to bracket once the core mean-field
> potential has shifted those eigenvalues -- on N2, four of eight embedding
> eigenvalues already sat below -5 Ha."

### `GqeSettings` (step 3, GQE only)

| field | default | note |
|---|---|---|
| `molecule_config` | `"dmet_embedding"` | **mandatory** — without it the trainer loads N₂ |
| `seed` | `None` | **`None` means "leave the repo's pinned seed=32 alone"**, not "randomise" |
| `max_iters` | `120` | training epochs; use `2` for a smoke test |
| `ngates` | `40` | circuit depth. On ScH: 10 stalled at HF, 20 plateaued, 40 recovered ~60 % |
| `num_samples` / `batch_size` | `100` / `100` | must be equal (online trainer) |
| `operator_pool_spec` | `"dmet_excitation"` | the only pools that work with an embedding |
| `qsci_max_dim` | `10000` | the repo default of 2000 pinned ScH's subspace from epoch 30 |
| `cudaq_target` | `$CUDAQ_DEFAULT_SIMULATOR` or `qpp-cpu` | selects the **circuit simulator** only |

The `seed` behaviour is the single most expensive default in the package to
misunderstand — see [`07_validation.md`](07_validation.md) and
`docs/reproducibility.md` §5.

`cudaq_target` is applied through the environment, not a Hydra key, because the
external repo never calls `cudaq.set_target()`. It also only governs circuit
simulation; the transformer trains on GPU via Lightning either way, which is
why a CPU-simulator run still logs `GPU available: True, used: True`.

### `TierSettings`

Holds a 44-element frozenset of transition-metal symbols and three thresholds
used to classify a system into tier 1/2/3. See
[`03_step1_active_space.md`](03_step1_active_space.md).

## 8. The CLI

`quenais/cli.py` maps flags onto the groups above. The module docstring records
why the GQE flags exist at all:

> "Until 0.3 this file built Config with asf=, dmet= and qiskit= but nothing for
> gqe=, so every `quenais-run --solver gqe` invocation silently used
> GqeSettings() defaults. There was no way to choose the simulator backend, the
> number of epochs, or the circuit depth from the command line at all."

Two CLI details worth knowing:

- `--gqe-num-samples` sets **both** `num_samples` and `batch_size`, because
  `validate()` rejects them differing.
- `--classical-methods` takes `CCSD_T`, not `CCSD(T)` — parentheses would need
  shell quoting.
- `cfg.tiers` is not CLI-configurable; it is always `TierSettings()`.

## Next

[`02_step0_classical.md`](02_step0_classical.md) — the classical reference
methods, and why the answer key is softer than it looks.
