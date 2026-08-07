# QuEnAIS — Quantum Embedding for Strongly Correlated Molecules

A Python package combining density matrix embedding theory (DMET) with
quantum solvers for strongly correlated molecules, in particular
transition-metal systems.

Two solver families are selectable by configuration:

| family | solvers | stack | how it runs |
|---|---|---|---|
| Qiskit | `sqd`, `skqd`, `sqdrift` | `quenais[qiskit]` | in-process |
| CUDA-Q | `gqe` | `quenais[cudaq]` + a patched `gqe-for-qsci` checkout | subprocess |

The two stacks stay independent at the *package* level — installing one
never drags in the other, and the DMET pipeline itself needs only PySCF.
`install.sh`, however, installs **both**: a partially-provisioned
environment is how this project repeatedly ended up with runs that
reported success and produced nothing. See [Install](#install).

Developed as part of Awadhoot Loharkar's Master's thesis at the Fraunhofer
Institute for Industrial Mathematics (ITWM), within the
[QuEnAIS](https://www.quantensysteme.info/projektatlas/projekte/q/quenais)
project — funded under the Eureka call "Applied Quantum Technologies",
coordinated by Fraunhofer ITWM with partners Cortex Discovery GmbH and
QunaSys ApS.

## Pipeline

| Step | Stage | Output |
|---|---|---|
| 0 | Classical references — `HF` and `MP2` by default; `CCSD`, `CCSD_T`, `CASSCF`, `NEVPT2` via `--classical-methods` | `step0_classical.pkl` |
| 1 | Active-space selection (ASF, entanglement entropy) | `step1_asf.pkl` |
| 2 | DMET embedding Hamiltonian (Schmidt decomposition) | `step2_hamiltonian.pkl` |
| 3 | Quantum solver (Qiskit in-process, or GQE by subprocess) | `step3_results.pkl` / `gqe_train.log` |
| 4 | Figures and CSV summaries | `results/plots/`, `results_summary.csv` |

Each stage exposes `main(cfg, force=False)`, caches to a pickle, and
validates that cache against the current molecule and basis before reusing
it.

## Requirements

- Python 3.10 or 3.11
- Linux, or WSL on Windows — `block2` has no working native Windows path.
  macOS runs the DMET pipeline but not the CUDA-Q solver.
- A compiler toolchain (`clang`, `cmake`, `make`, `gfortran`), Rust
  (`cargo`), and MPI (`mpicc`). `environment.yml` installs all of them.
- **AVX-512 is not required**, but its absence changes how PySCF must be
  installed — `install.sh` detects this and builds from source. See
  [docs/gqe_setup.md](docs/gqe_setup.md#cpu--avx-512-and-pyscf).
- A GPU is optional. Without one, or with a GPU below compute capability
  8.0, the installer selects CUDA-Q's `qpp-cpu` simulator automatically.

## Install

```bash
git clone --recurse-submodules https://github.com/AwadhootVijayLoharkar/quenais.git
cd quenais

mamba env create -f environment.yml -p ./quenais-env
mamba activate ./quenais-env

bash install.sh

# pick up the variables install.sh persisted into the env
mamba deactivate && mamba activate ./quenais-env
```

`install.sh` leaves the environment ready to run everything — both solver
families, including GQE. It is 16 steps and **every one is required**; the
script exits non-zero with a remediation message if any of them fails.

In particular it does four things a plain `pip install -e ".[all]"` does
not:

| | why |
|---|---|
| initialises **both** submodules and builds `theochem/pyci` from source | `gqe-for-qsci` needs it and it has no wheel; the `pyci` / `qc-PyCI` packages on PyPI are unrelated projects |
| installs `gqe-for-qsci` and **applies the source patch** | the DMET operator pools are unregistered without it, and an unpatched checkout trains to completion producing nothing parseable |
| compiles CUDA-Q's **MPI plugin** and persists `LD_LIBRARY_PATH`, `MPI_PATH`, `CUDAQ_DEFAULT_SIMULATOR`, `WANDB_MODE` into the env | CUDA-Q ships the plugin uncompiled; `CUDAQ_DEFAULT_SIMULATOR` is read at import, so setting it inside a script is too late |
| installs **PySCF last**, from source when this CPU lacks AVX-512 | the prebuilt wheel's `libcgto.so` SIGILLs on non-AVX-512 CPUs; installing it last stops an earlier pip step from replacing a good build |

### Overrides

```bash
QUENAIS_PYSCF_BUILD=source bash install.sh   # force the from-source build
QUENAIS_CUDAQ_TARGET=qpp-cpu bash install.sh # pin the simulator target
QUENAIS_SKIP_SELFTEST=1 bash install.sh      # CI only
```

There is no flag to skip the GQE stack. If you only want the Qiskit
solvers, `pip install -e ".[qiskit]"` still works and is unchanged — but
it is not what `install.sh` does, and `quenais-doctor` will report the
missing CUDA-Q pieces.

### Verify before trusting anything

Two checks, in order. The first asks "will anything run at all" and takes
a second; the second asks "does the physics reproduce". `install.sh` runs
both at the end.

```bash
quenais-doctor      # environment, hardware, submodule state
quenais-selftest    # LiH end to end against known-good values
```

`quenais-doctor` covers every failure this project has actually hit —
AVX-512 versus the PySCF build, GPU compute capability versus the CUDA-Q
target, whether mpi4py and CUDA-Q's plugin link the same `libmpi`, the
`setuptools<82` / `pkg_resources` ceiling, the numpy/pyscf `einsum` skew,
whether torch and cudaq can coexist in one process, whether
`gqe-for-qsci`'s declared dependencies are all present, and whether the
submodule is patched. Every one of those presents as something other than
what it is; the background on each is in
[docs/gqe_setup.md](docs/gqe_setup.md).

```
[  ok  ] CPU AVX-512                          absent, but PySCF was built from source
[  ok  ] GPU                                  NVIDIA A100-SXM4-40GB (cc 8.0)
[  ok  ] setuptools                           81.0.0, pkg_resources importable
[  ok  ] CUDAQ_DEFAULT_SIMULATOR              nvidia
[  ok  ] WANDB_MODE                           disabled
[  ok  ] MPI ABI                              both link libmpi.so.40.40.7
[  ok  ] required packages                    all 13 importable
[  ok  ] numpy/pyscf einsum                   pyscf 2.14.0 / numpy 2.4.6
[  ok  ] torch/cudaq coexistence              torch -> pytorch_lightning -> gqe_qsci -> cudaq
[  ok  ] gqe-for-qsci deps                    all 11 declared deps present
[  ok  ] gqe-for-qsci checkout                patched at /home/…/gqe-for-qsci
```

`quenais-selftest` runs LiH end to end in seconds and checks each quantity
against known-good values, including the two silent physics bugs this
package exists to not have. Attach its output to any bug report — it
identifies which quantity drifted, which is most of the diagnosis.

```
[  ok  ] thread environment            OPENBLAS=1 OMP=24 (from SLURM_CPUS_PER_TASK)
[  ok  ] HF energy                     -7.862026959 Ha  (delta 5.3e-15)
[  ok  ] bath orbital count            2 (expected 2)
[  ok  ] embedded electron count       (2a, 2b) -- from the reference density
[  ok  ] embedded SCF vs full UHF      delta -3.02e-09 Ha  (tol 2e-07)
```

### Re-applying the GQE patch

`install.sh` does this for you. You only need it by hand after a
`git submodule update`, which silently reverts the patch:

```bash
quenais-gqe-setup --repo ./gqe-for-qsci
```

Idempotent, verifies the pinned commit (`0a201ea`) before touching
anything, and writes a stamp the runner checks before spending GPU time.
`quenais-doctor` catches a reverted patch by comparing that stamp's hash
against the shipped patch.

See [docs/gqe_integration.md](docs/gqe_integration.md) for what the patch
does and [docs/gqe_setup.md](docs/gqe_setup.md) for everything about
getting the stack to run.

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

# a two-epoch GQE smoke test -- the real default is 120 iters, ngates 40
quenais-run --molecule LiH --basis sto-3g --solver gqe \
            --gqe-max-iters 2 --gqe-num-samples 10 --gqe-ngates 10

# with a Qiskit solver
quenais-run --molecule LiH --basis sto-3g --solver sqd --ansatz lucj
```

### Classical reference methods

Step 0 runs `HF` and `MP2` by default. The other four are opt-in:

```bash
quenais-run --molecule LiH --basis sto-3g --steps 0 \
            --classical-methods HF MP2 CCSD CCSD_T CASSCF NEVPT2
```

`CCSD_T`, not `CCSD(T)` — parentheses would need shell quoting.

**CASSCF and NEVPT2 need step 1 to have already run.** They reuse step 1's
active space, but step 0 runs *before* step 1, so on a first pass they
fall back to a guessed space and print `Step 1 not found -- CASSCF/NEVPT2
will use a fallback active space`. For meaningful numbers, go around
twice: `--steps 0 1 2` first, then `--steps 0 --force` with the methods
you want. Both are labelled `optimizer-dependent` in
`results_summary.csv` and do not reproduce to tight tolerance across
machines, which is why they are not on by default.

### Choosing the CUDA-Q backend

`--cudaq-target` selects the **circuit simulator** only. `install.sh`
picks it from the GPU's compute capability and persists it, so this is
usually already right:

```bash
quenais-run --molecule LiH --solver gqe --cudaq-target qpp-cpu   # force CPU
quenais-run --molecule LiH --solver gqe --cudaq-target nvidia    # needs cc >= 8.0
```

The transformer runs on the GPU through Lightning regardless of this
setting, so a `qpp-cpu` run still logs `GPU available: True, used: True`.
The line to read is the runner's own `backend :` field.

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

**None of this changes the numbers.** The reference values above have been
reproduced to 5e-15 Ha on both a hand-built AVX-512 PySCF and a stock
manylinux wheel, on different CPUs and different PySCF versions. Which
PySCF build you get is a speed choice, not a correctness one — *except* on
a CPU without AVX-512, where the wheel does not run at all. That is an
install-time concern, not a tuning one; see
[docs/gqe_setup.md](docs/gqe_setup.md#cpu--avx-512-and-pyscf).

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

**A GPU, for GQE.** `install.sh` picks the CUDA-Q target from the GPU's
compute capability and persists it, so this is usually already right. To
override, set `cfg.gqe.cudaq_target` — `"nvidia"` needs cc ≥ 8.0,
`"tensornet"` and `"tensornet-mps"` are also available, the latter
approximate. Applied through the `CUDAQ_DEFAULT_SIMULATOR` environment
variable, because the external trainer never calls `cudaq.set_target()`.

**Rebuild PySCF from source.** On an AVX-512 machine this is optional and
only worth it for large active spaces, where integral evaluation and DMRG
dominate:

```bash
mamba activate ./quenais-env
QUENAIS_PYSCF_BUILD=source bash install.sh   # or, by hand:
pip install "pyscf>=2.12" --no-binary pyscf --force-reinstall --no-deps
quenais-selftest                             # confirm the numbers did not move
```

`>=2.12` is a hard floor, not a preference: earlier releases break on
numpy ≥ 2.4, and `ffsim` requires it independently. See
[docs/gqe_setup.md](docs/gqe_setup.md#the-numpy-24--pyscf-einsum-break).
Allow up to an hour for the source build.

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
├── selftest.py          quenais-selftest   -- does the physics reproduce
├── env_check.py         quenais-doctor     -- will anything run at all
├── classical/runner.py  step 0
├── active_space/        step 1
├── embedding/           step 2 -- hamiltonian.py + side-effect-free dmet_lib.py
├── quantum/             step 3 -- solver.py (Qiskit), gqe_*.py (CUDA-Q)
│   └── _gqe_shims/      top-level module names the external repo imports
├── visualization/       step 4
└── patches/             the gqe-for-qsci source patch (package_data)
external/pyci/           submodule: theochem/pyci, built from source
gqe-for-qsci/            submodule: the external GQE trainer, patched
tests/regression/golden/ validated reference pickles
```

`patches/` lives *inside* the package rather than at the repo root because
`setup.cfg` ships it as `package_data`. At the root the glob matched
nothing, so any non-editable install carried no patch at all and
`quenais-gqe-setup` failed with "Patch not found".

## Licensing

This package is Apache-2.0.

`gqe-for-qsci` is a third-party project with its own `LICENSE` and
`NOTICE`. The four DMET integration files are original work implementing
against its interfaces; `quenais/patches/gqe_dmet_source.patch` modifies
upstream source and redistributes its context lines. **Check upstream's
`NOTICE` for attribution terms that carry into derived work before
publishing to PyPI or a public repository.**

`theochem/pyci` is likewise a third-party project carried as a submodule
with its own license — it is built from source, not vendored, so nothing
of it is redistributed here.