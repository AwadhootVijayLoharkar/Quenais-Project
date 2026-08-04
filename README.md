# QuEnAIS — Quantum Embedding for Strongly Correlated Molecules

A Python package combining density matrix embedding theory (DMET) with
quantum solvers for strongly correlated molecules, in particular
transition-metal systems.

Two solver families are selectable by configuration:

| family | solvers | stack | how it runs |
|---|---|---|---|
| Qiskit | `sqd`, `skqd`, `sqdrift` | `quenais[qiskit]` | in-process |
| CUDA-Q | `gqe` | `quenais[cudaq]` + a patched `gqe-for-qsci` checkout | subprocess |

Neither stack is required for the other. The DMET pipeline itself needs
only PySCF.

Developed as part of Awadhoot Loharkar's Master's thesis at the Fraunhofer
Institute for Industrial Mathematics (ITWM), within the
[QuEnAIS](https://www.quantensysteme.info/projektatlas/projekte/q/quenais)
project — funded under the Eureka call "Applied Quantum Technologies",
coordinated by Fraunhofer ITWM with partners Cortex Discovery GmbH and
QunaSys ApS.

## Pipeline

| Step | Stage | Output |
|---|---|---|
| 0 | Classical references (HF, MP2, CCSD, CCSD(T), CASSCF, NEVPT2) | `step0_classical.pkl` |
| 1 | Active-space selection (ASF, entanglement entropy) | `step1_asf.pkl` |
| 2 | DMET embedding Hamiltonian (Schmidt decomposition) | `step2_hamiltonian.pkl` |
| 3 | Quantum solver (Qiskit in-process, or GQE by subprocess) | `step3_results.pkl` / `gqe_train.log` |
| 4 | Figures and CSV summaries | `results/plots/`, `results_summary.csv` |

Each stage exposes `main(cfg, force=False)`, caches to a pickle, and
validates that cache against the current molecule and basis before reusing
it.

## Requirements

- Python 3.10 or 3.11
- Linux or macOS. On Windows use WSL — `block2` has no working native
  Windows install path.
- PySCF. The default wheel is fine — see [Performance](#performance) if
  you care about speed on large systems.

## Install

```bash
git clone --recurse-submodules https://github.com/AwadhootVijayLoharkar/quenais.git
cd quenais

mamba env create -f environment.yml -p ./quenais-env
mamba activate ./quenais-env

pip install -e ".[qiskit]"        # or ".[cudaq]", or ".[all]"
```

To keep a PySCF you built yourself, install without dependencies so pip
does not replace it:

```bash
pip install -e . --no-deps
```

### Verify before trusting anything

```bash
quenais-selftest
```

Runs LiH end to end in seconds and checks each quantity against
known-good values, including the two silent physics bugs this package
exists to not have. Attach its output to any bug report — it identifies
which quantity drifted, which is most of the diagnosis.

```
[  ok  ] thread environment            OPENBLAS=1 OMP=24 (from SLURM_CPUS_PER_TASK)
[  ok  ] HF energy                     -7.862026959 Ha  (delta 5.3e-15)
[  ok  ] bath orbital count            2 (expected 2)
[  ok  ] embedded electron count       (2a, 2b) -- from the reference density
[  ok  ] embedded SCF vs full UHF      delta -3.02e-09 Ha  (tol 2e-07)
```

### The GQE solver needs one extra step

`gqe-for-qsci` is a git submodule pinned at `0a201ea`, and the DMET
integration requires three source edits to it. A plain
`git submodule update` gives a checkout that cannot run this pipeline.

```bash
quenais-gqe-setup --repo ./gqe-for-qsci
```

Idempotent, verifies the commit SHA before touching anything, and writes a
stamp so the runner can check readiness before spending GPU time. See
[docs/gqe_integration.md](docs/gqe_integration.md).

## Notebooks

| notebook | what it covers | needs |
|---|---|---|
| [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | LiH end to end, compared against the validated value | PySCF |
| [`02_dmet_internals.ipynb`](notebooks/02_dmet_internals.ipynb) | the Schmidt spectrum, why N₂ correctly gets no bath, reading the diagnostics | nothing — runs off the golden pickles |
| [`03_gqe_solver.ipynb`](notebooks/03_gqe_solver.ipynb) | configuring and running the CUDA-Q solver | `quenais[cudaq]` + patched submodule |

Start with 01. If you want the physics rather than the API, 02 runs in
seconds with no optional dependencies at all.

## Usage

```bash
# LiH, classical + active space + embedding
quenais-run --molecule LiH --basis sto-3g --steps 0 1 2

# your own molecule -- see "Bringing your own molecule" below
quenais-run --molecule BeH2 --basis sto-3g \
            --geometry "Be 0 0 0; H 0 0 1.33; H 0 0 -1.33" --steps 0 1 2

# transition metals need an explicit active space -- see docs/limitations.md
quenais-run --molecule ScH --basis sto-3g \
            --force-active-space 9 10 11 12 13 14

# with the GQE solver
quenais-run --molecule LiH --basis sto-3g --solver gqe

# with a Qiskit solver
quenais-run --molecule LiH --basis sto-3g --solver sqd --ansatz lucj
```

From Python:

```python
from quenais import Config
from quenais.settings import AsfSettings, DmetSettings

cfg = Config(
    molecule="ScH",
    basis="sto-3g",
    quantum_solver="gqe",
    asf=AsfSettings(force_active_space=[9, 10, 11, 12, 13, 14]),
    dmet=DmetSettings(reference="casci"),
).validate().make_dirs().load_geometry()

from quenais.classical import runner
from quenais.active_space import finder
from quenais.embedding import hamiltonian

runner.main(cfg)
finder.main(cfg)
step2 = hamiltonian.main(cfg)
print(step2["embedded_scf_check"])
```

Settings are grouped rather than flat: `cfg.dmet.bath_tolerance`,
`cfg.gqe.ngates`, `cfg.asf.gap_degeneracy_tol`.

## Bringing your own molecule

Nothing about the pipeline is specific to the bundled systems. Give it a
geometry and it runs.

```bash
# inline, straight out of a paper. Angstrom.
quenais-run --molecule BeH2 --basis sto-3g \
            --geometry "Be 0 0 0; H 0 0 1.33; H 0 0 -1.33" --steps 0 1 2

# from an XYZ file
quenais-run --molecule MyMol --basis 6-31g --xyz ./mymol.xyz --steps 0 1 2
```

```python
cfg = Config(molecule="BeH2", basis="sto-3g",
             geometry="Be 0 0 0; H 0 0 1.33; H 0 0 -1.33")
cfg = Config(molecule="MyMol", basis="6-31g", xyz="mymol.xyz")
cfg = Config(molecule="FeCO", geometry=[("Fe", (0, 0, 0)), ("C", (0, 0, 1.8))])
```

Geometry is resolved in this order, so explicit input always overrides a
bundled one:

| source | use it when |
|---|---|
| `geometry=` | pasting a geometry from a paper |
| `xyz=` | you have an XYZ file |
| built-in name | `LiH`, `N2`, `ScH`, `H2O` |
| `cif_files/<molecule>.cif` | crystallographic input |

`molecule` only names the cache files — any string works. Passing both
`geometry=` and `xyz=` is an error rather than a silent preference.

Two things to expect on an unfamiliar system:

- **Transition metals need `--force-active-space`.** Automatic selection
  under-selects for the d-block. Check by comparing NEVPT2 with CCSD(T):
  if NEVPT2 lands above it, the space is too small. See
  [docs/limitations.md](docs/limitations.md).
- **Only closed-shell (`spin=0`) systems are validated.** Open-shell runs
  exercise untested code paths.

[`notebooks/04_full_workflow.ipynb`](notebooks/04_full_workflow.ipynb) walks
the whole thing end to end on a non-bundled molecule with every tunable
parameter annotated.

## Validated reference values

`quenais-selftest` and the regression suite check against these. They come
from an A100 / EPYC 7402 run and are stored as golden pickles in
`tests/regression/golden/`.

**LiH / STO-3G, r = 1.5949 Å, (2e,2o), N_emb = 4**

| method | E (Ha) |
|---|---|
| RHF | −7.862026959 |
| CCSD | −7.882392917 |
| **DMET + CASCI** | **−7.881246152** |

**ScH / STO-3G, r = 1.78 Å, forced (4e,6o) MOs 9–14, N_emb = 11**

| method | E (Ha) |
|---|---|
| RHF | −752.638702408 |
| CCSD(T) | −752.709890151 |
| **DMET + CASCI** | **−752.699524181** |
| embedded SCF vs full UHF | agree to **3×10⁻⁹ Ha** |

**N₂ / STO-3G, (4e,4o)** — every Schmidt singular value is numerically
zero, so `n_bath = 0`. That is correct behaviour, not a failure; DMET+CASCI
agrees with CASSCF to 0.04 mHa.

Not every number reproduces to the same precision. `results_summary.csv`
labels each one `deterministic`, `optimizer-dependent` or `stochastic` —
see [docs/limitations.md](docs/limitations.md) before comparing results
across machines.

## Performance

**None of this changes the numbers.** The reference values below have been
reproduced to 5e-15 Ha on both a hand-built AVX-512 PySCF and a stock
manylinux wheel, on different CPUs and different PySCF versions. A faster
build is a speed choice, not a correctness one.

In rough order of payoff:

**Give it cores.** `OMP_NUM_THREADS` follows your scheduler's allocation
automatically — `--cpus-per-task=24` under SLURM is used without any
configuration. Without a scheduler it falls back to `cpu_count - 1`.
OpenBLAS is deliberately pinned to a single thread: block2 is
OpenMP-threaded, and two threading runtimes fighting over CPU affinity is a
genuine hang risk, not just a slowdown.

`quenais-selftest` prints what it decided:

```
[  ok  ] thread environment    OPENBLAS=1 OMP=24 (from SLURM_CPUS_PER_TASK)
```

**A GPU, for GQE.** Set `cfg.gqe.cudaq_target` to `"nvidia"` instead of the
`"qpp-cpu"` default. `"tensornet"` and `"tensornet-mps"` are also available;
the latter is approximate. This is applied through the
`CUDAQ_DEFAULT_SIMULATOR` environment variable, because the external
trainer never calls `cudaq.set_target()`.

**Build PySCF from source.** Only worth it for large active spaces, where
integral evaluation and DMRG dominate:

```bash
mamba activate ./quenais-env
pip uninstall pyscf -y
pip install --no-binary pyscf pyscf     # 20-30 min; needs cmake, gfortran, openblas
quenais-selftest                        # confirm the numbers did not move
```

`environment.yml` already provides the toolchain. Re-run the self-test
afterwards — not because a correct build would change anything, but
because a *broken* one might, and quietly.

**Cheaper GQE settings.** `reference_keys` containing `R-CASCI` triggers a
full FCI over the whole embedding space before training starts, purely for
logging. Intractable above roughly 12-16 embedding orbitals. Drop it if a
run appears to hang at logger construction.

### What does not help

Nothing in the pipeline is I/O bound at these sizes, and the caches are
already content-validated, so faster disks and warm caches make no
difference. For LiH and N2 none of the above is measurable — the whole
pipeline runs in seconds either way.

## Known limitations

Read [docs/limitations.md](docs/limitations.md) before trusting a result
from a system other than LiH or N₂. In short: ASF under-selects for
transition metals (use `--force-active-space`), CASSCF and NEVPT2 are not
reproducible to tight tolerance, only closed-shell systems are validated,
and GQE's accuracy on larger systems is bounded by sampling capacity.

## Tests

```bash
pytest -q                    # everything installed
pytest -q -m "not slow"      # skip the end-to-end runs
```

Tests requiring an optional stack are skipped, not failed, when it is
absent. The suite includes a comparator that diffs any stage output
against the golden pickles key by key — every bug in this project's
history produced the right shape, a plausible magnitude and the wrong
value, and that is the only thing that catches it.

## Project layout

```
quenais/
├── config.py            Config; settings groups live in settings/
├── _threads.py          thread-count guard, imported before NumPy
├── provenance.py        environment block stamped into every result
├── selftest.py          quenais-selftest
├── classical/runner.py  step 0
├── active_space/        step 1
├── embedding/           step 2 -- hamiltonian.py + side-effect-free dmet_lib.py
├── quantum/             step 3 -- solver.py (Qiskit), gqe_*.py (CUDA-Q)
│   └── _gqe_shims/      top-level module names the external repo imports
└── visualization/       step 4
patches/                 the gqe-for-qsci source patch
tests/regression/golden/ validated reference pickles
```

## Licensing

This package is Apache-2.0.

`gqe-for-qsci` is a third-party project with its own `LICENSE` and
`NOTICE`. The four DMET integration files are original work implementing
against its interfaces; `patches/gqe_dmet_source.patch` modifies upstream
source and redistributes its context lines. **Check upstream's `NOTICE`
for attribution terms that carry into derived work before publishing to
PyPI or a public repository.**
