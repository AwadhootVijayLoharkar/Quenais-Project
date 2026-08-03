# The GQE integration

How the package drives the external `gqe-for-qsci` trainer, and why each
piece is shaped the way it is.

## The problem

`gqe-for-qsci` builds its molecule from a geometry and picks an active
space as a subset of canonical MOs. A DMET embedding is neither: it is
impurity + bath orbitals from a Schmidt decomposition, a rotated basis
rather than a subset, with no geometry at all.

So three things are needed:

1. a molecule object built from `h1e` / `h2e` / `ecore` instead of atoms;
2. operator pools that derive excitations from the embedding's own CCSD
   amplitudes rather than from geometry;
3. a way to register those pools in upstream's factory.

## Where the code lives

The canonical implementations are in the package:

```
quenais/quantum/gqe_adapter.py    DMETEmbeddingMolecule, load_from_dmet_pickle
quenais/quantum/gqe_pools.py      DMETExcitationPool, DMETPauliEvolutionPool
```

Upstream's `factory.py` imports them by **top-level absolute name**
(`from dmet_excitation_pool import ...`), so they only need to be
*importable*, not physically present in the checkout. `_gqe_shims/`
contains two one-line re-export modules, and the runner prepends that
directory to `PYTHONPATH` for the training subprocess.

The result: one source of truth, nothing written into someone else's
repository, and the classes remain importable from the package for
testing.

`_gqe_shims/` deliberately has no `__init__.py`. It is a path entry, not a
subpackage.

## The patch

Three upstream files are modified. `quenais-gqe-setup` applies
`patches/gqe_dmet_source.patch`.

### `gqe_qsci/factory.py` — required

Registers `dmet_pauli_evolution` and `dmet_excitation` in
`create_operator_pool()`. Without it those specs are unknown and the run
fails at pool construction.

### `gqe_qsci/gqe/sampler.py` — required

Two identical one-line changes:

```python
-        if cudaq.mpi.is_initialized():
+        if self.mpi and cudaq.mpi.is_initialized():
```

An upstream bug: the config exposes `sampler.mpi`, but upstream takes the
MPI branch whenever MPI happens to be initialised by anything in the
process. Affects work distribution only — not circuit sampling, not the
QSCI subspace, no bearing on any energy.

Both call sites must be fixed. A test asserts the patch contains exactly
two of these.

### `gqe_qsci/train_pipeline.py` — required, and easy to dismiss

```python
-        self.metric_logger.log_result(self, log_inputs)
+        metrics = self.metric_logger.log_result(self, log_inputs)
+        print(f"[epoch {self.current_epoch}] {metrics}", flush=True)
```

Pure instrumentation, but **load-bearing**: these lines are the only thing
the visualisation stage can parse. Without them training runs to
completion, reports success, and produces nothing plottable. It looks like
a working run.

Two guards exist because of that: the runner raises if training finishes
having emitted zero epoch lines, and the log parser warns if a non-empty
log contains none.

### `configs/trainer/default.yaml` — NOT shipped

Carried leftover debugging values (`max_iters` 100→50, `load_checkpoint`
true→false, `target_var` 1e-5→1e-3). Baking them into the fork would
silently override the package's own settings, which are passed as Hydra
overrides. A test asserts the patch does not touch it.

## Regenerating the patch

Only from a checkout where the integration already works:

```bash
quenais-gqe-setup --repo /path/to/gqe-for-qsci --create-patch
```

`git diff` reports only **uncommitted** changes. If any of the three files
has been committed inside the submodule, its hunk silently disappears —
and a partial patch looks like a perfectly good file. That happened once
and produced a patch with zero hunks which overwrote the correct one.

`create_patch` now refuses to write unless all three files are present,
and leaves the existing patch untouched when they are not.

## Launching training

`gqe_runner.main()` does five things a bare `subprocess` call would not:

1. **Verifies the checkout is patched** before spending GPU time.
2. **Validates the step 2 pickle belongs to the current molecule.**
   Training on another system's embedding produces a plausible number with
   no meaning.
3. **Generates `configs/molecule/dmet_embedding.yaml`** at run time and
   also passes `molecule.step2_pickle_path` on the command line. The
   shipped version of that file had a hardcoded absolute path that
   silently pointed at a stale directory for an extended period.
4. **Prepends the shim directory to `PYTHONPATH`**, first, so a stale copy
   inside the checkout cannot shadow the package's implementation.
5. **Sets `CUDAQ_DEFAULT_SIMULATOR`.** The external repo never calls
   `cudaq.set_target()`, so there is no Hydra key for the backend.

## Two mandatory Hydra overrides

```
molecule=dmet_embedding
operator_pool.spec=dmet_excitation
```

Without the first, `configs/default.yaml` declares `molecule: n2` and
train.py trains on N₂ while the reported numbers claim to be yours. This
was silently the case for every run before it was added — the LiH and ScH
runs produced near-identical epoch logs with energies around −107 Ha,
which is N₂'s scale.

Without the second, the stock pools try to iterate `molecule.geometry`,
which is `None`.

Both are emitted unconditionally and asserted in `build_command`.

## Which pool

`dmet_excitation`, in almost all cases.

`DMETPauliEvolutionPool` appends every Pauli **term** of an excitation
generator as its own pool element, so each element is a single Pauli word.
Particle-number conservation is a property of the full **sum** of terms in
a JW-mapped fermionic excitation, never of one term alone — so that pool
cannot conserve electron number however its flags are set. Measured on
ScH: roughly half of every sample discarded as symmetry-violating.

`DMETExcitationPool` accumulates the terms into one operator per
excitation, which does conserve it.

Note that upstream's factory constructs `DMETExcitationPool` without
`remove_z_ladder` / `only_use_first_pauli`, so those settings are ignored
for this pool type.

## The excitation-generator convention

`excitation_generator_qubit_op()` builds the JW-mapped anti-Hermitian UCC
generator and multiplies by `-1j`. That factor reconciles it with
tequila's stored Hermitian form, `H = -i(T - T†)`.

This was validated against tequila's actual `make_excitation_gate()`
output on H₂/STO-3G: identical Pauli-string sets and a ratio of exactly
1.000000 for a single α excitation, a single β excitation, and an α–β
double.

`tests/test_gqe_adapter.py::test_excitation_generator_matches_tequila`
re-runs that check. A phase error here crashes nothing — it silently seeds
the pool with wrong-phase excitations.

## One cost to know about

`reference_keys` containing `R-CASCI` triggers a full FCI over the entire
embedding space, purely for logging, **before training starts**. Cost grows
combinatorially and becomes impractical above roughly 12–16 embedding
orbitals. If a run appears to hang at logger construction, drop it — it is
a reference value only, not used in training or diagonalisation.

The runner warns when `n_emb > 14` and an `R-CASCI` key is requested.
