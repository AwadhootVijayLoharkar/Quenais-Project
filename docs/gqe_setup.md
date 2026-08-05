# GQE stack — setup, hardware, and every failure seen so far

`docs/gqe_integration.md` explains *how* the package drives `gqe-for-qsci`.
This file is about *getting it to run at all*: what install.sh does, why,
and what to do when a step fails.

Since 0.3 the GQE stack is **mandatory**. `install.sh` initialises the
submodules, builds `theochem/pyci`, installs the CUDA-Q stack, applies the
source patch, compiles the CUDA-Q MPI plugin, and exits non-zero if any of
that fails. There is no silent-skip path any more — the 0.2 script printed
a warning and exited 0 with an empty `gqe-for-qsci/`, which produced
installs that reported success and could not run the solver.

---

## Quick start

```bash
git clone --recurse-submodules <repo-url>
cd Quenais-Project
mamba env create -f environment.yml -p ./quenais-env
mamba activate ./quenais-env
bash install.sh
mamba deactivate && mamba activate ./quenais-env   # pick up persisted vars
quenais-doctor
```

If you already downloaded a zip rather than cloning, the submodules are
absent and cannot be fetched — `install.sh` fails at step 8 and says so.
Clone instead.

### Overrides

| Variable | Default | Effect |
|---|---|---|
| `QUENAIS_PYSCF_BUILD` | `auto` | `source` forces the from-source build, `wheel` forces the prebuilt one. `auto` decides from `/proc/cpuinfo`. |
| `QUENAIS_CUDAQ_TARGET` | `auto` | Pin the CUDA-Q simulator instead of deriving it from GPU compute capability. |
| `QUENAIS_SKIP_SELFTEST` | `0` | Skip the closing LiH physics check (CI only — never for a real install). |
| `QUENAIS_BLOCK2_WRAPPER` | `~/block2main_wrapper.sh` | Where to write the block2 wrapper. |

---

## Why the two submodules are submodules

Both `external/pyci` and `gqe-for-qsci` are git submodules rather than
clones or vendored copies, so upstream updates pull cleanly without merge
conflicts.

**`external/pyci` is `theochem/pyci`.** It is *not* the `pyci` or `qc-PyCI`
packages on PyPI — those are unrelated projects, and installing either
gives confusing import errors rather than a clean failure. It has no usable
wheel; `install.sh` runs `make && pip install --no-deps .` in the checkout.

**`gqe-for-qsci` is installed with `--no-deps`, deliberately.** Upstream
pins `qiskit==2.0.0`, but this pipeline needs 2.4.x for `qiskit-fermions`
and `qiskit-ibm-runtime`. `grep -r "import qiskit" gqe_qsci/` confirms
upstream never imports qiskit at all — the pin is dead. Its real runtime
dependencies (torch, cudaq, pytorch-lightning, hydra-core, wandb, tequila,
mpi4py) are declared in `quenais`'s `[cudaq]` extra instead, so
`pip install -e ".[all]"` covers them without the resolver ever seeing the
bad pin.

If you already downgraded qiskit: `pip install "qiskit>=2.4,<3"` to revert.

**A known-benign `pip check` complaint**: `gqe-for-qsci` wants
`scipy~=1.15.3`, conda-forge has `1.15.2`. Patch-level bugfix gap, safe to
ignore, and 1.15.3 is not on conda-forge for this setup anyway. Any *other*
`pip check` conflict should be investigated individually — don't assume
they are all this benign.

---

## The source patch

`quenais-gqe-setup --repo ./gqe-for-qsci` applies
`quenais/patches/gqe_dmet_source.patch` to three files. It is idempotent
and refuses to patch a commit other than the one it was written against
(`UPSTREAM_SHA` in `quenais/quantum/gqe_setup.py`) unless you pass
`--force`.

| File | What it does | Consequence if absent |
|---|---|---|
| `gqe_qsci/factory.py` | Registers `dmet_pauli_evolution` and `dmet_excitation` | Run fails immediately at pool construction |
| `gqe_qsci/gqe/sampler.py` | Guards both `cudaq.mpi.is_initialized()` checks with `self.mpi` | Single-process runs crash on an unpicklable `SampleResult` — see below |
| `gqe_qsci/train_pipeline.py` | Prints the per-epoch metrics dict | Training "succeeds" and produces nothing plottable |

`configs/trainer/default.yaml` is deliberately **not** patched — it carried
leftover debugging values that would silently override the package's own
Hydra overrides. A test asserts the patch does not touch it.

### The sampler.py bug in detail

`gqe_qsci/gqe/sampler.py` does `from mpi4py import MPI` at module load.
mpi4py calls `MPI_Init()` on import, unconditionally. Because mpi4py and
CUDA-Q's plugin link the *same* `libmpi.so` in this environment, that
auto-init flips the shared runtime's "initialized" flag to true — even for
a plain single-process `python train.py` with no `mpirun` anywhere.

`Sampler.run()` then checked `cudaq.mpi.is_initialized()` without also
checking `self.mpi` (the `sampler.mpi: false` config flag that is supposed
to gate this), took the gather branch, and called
`MPI.COMM_WORLD.allgather(res)` on a list of raw `cudaq.SampleResult`
objects. Those have no pickle support ([NVIDIA/cuda-quantum#1422], still
open) and `allgather` pickles even in a size-1 gather:

```
TypeError: cannot pickle 'cudaq...SampleResult' object
```

The fix is two lines — `self.mpi and` in front of both checks. A test
asserts the shipped patch contains exactly two of them.

**This is a local patch to a third-party submodule.** `git submodule
update` will silently wipe it. `verify_gqe_repo()` catches that on the next
run by comparing the `.quenais_patch_applied` stamp against the shipped
patch's sha256 — but it is still worth pointing the submodule at your own
fork if you update often.

### Regenerating

```bash
quenais-gqe-setup --repo /path/to/gqe-for-qsci --create-patch
```

Only from a checkout where the integration already works. `git diff`
reports **uncommitted** changes only, so if a file was committed inside the
submodule its hunk silently disappears. `create_patch` refuses to write
unless all three files are present.

[NVIDIA/cuda-quantum#1422]: https://github.com/NVIDIA/cuda-quantum/issues/1422

---

## Hardware

### CPU — AVX-512 and PySCF

PySCF ships a compiled `libcgto.so` with a **non-dispatching** AVX-512
codepath. numpy and OpenBLAS do runtime CPU dispatch; PySCF does not. On a
CPU without AVX-512 the first integral call is:

```
Illegal instruction (core dumped)
```

No traceback, no Python-level error. Root cause confirmed by a `gdb`
backtrace on `GTOint2c()`.

AVX-512 is per-chip, not per-vendor:

| CPU | Microarchitecture | AVX-512 | Result |
|---|---|---|---|
| AMD Threadripper 1920X | Zen 1 | No | SIGILL |
| AMD EPYC 7402 | Zen 2 | No | SIGILL (confirmed) |
| Intel Xeon Silver 4108 | Skylake-SP | Yes | Fine |
| AMD Genoa etc. | Zen 4+ | Yes | Should be fine (untested here) |
| Intel Alder Lake+ consumer | — | Disabled on consumer parts | Would crash |

`install.sh` step 15 checks `/proc/cpuinfo` and builds from source when the
flag is absent:

```bash
pip install pyscf==2.11.0 --no-binary pyscf --force-reinstall --no-deps
```

Check any new node before running anything:

```bash
grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u
```

Nothing returned means expect the SIGILL. `quenais-doctor` reports this as
a FAIL when the CPU lacks AVX-512 *and* PySCF looks like a manylinux wheel.

**Dependency safety**: `--no-deps` means pip touches nothing else, so
`pyscf-dmrgscf` and `openfermionpyscf` (PyPI) survive a rebuild of a
conda-forge pyscf intact — neither pins pyscf tightly enough to conflict.
Just don't run a plain `pip install pyscf` afterwards, which would put the
wheel back.

Note the wheel and a hand-built AVX-512 PySCF give identical energies to
5e-15 Ha. On an AVX-512 machine this is purely a speed choice; on a
non-AVX-512 machine it is the difference between running and not.

### GPU — compute capability

| GPU | Architecture | cc | cuQuantum/cuStateVec |
|---|---|---|---|
| TITAN V | Volta | 7.0 | **Unsupported** — `RuntimeError: architecture mismatch` |
| A100 SXM4-40GB | Ampere | 8.0 | Fully supported |

`install.sh` step 14 queries `nvidia-smi --query-gpu=compute_cap` and picks
`nvidia` at cc ≥ 8.0, `qpp-cpu` otherwise, then persists the choice:

```bash
conda env config vars set -p "$CONDA_PREFIX" CUDAQ_DEFAULT_SIMULATOR=qpp-cpu
```

`CUDAQ_DEFAULT_SIMULATOR` is read at CUDA-Q import time, so **it must be
set before Python starts** — setting it inside a script is too late. This
is why it is persisted into the env rather than exported once.

---

## MPI

CUDA-Q ships its distributed-interface plugin uncompiled. Without building
it:

```
RuntimeError: Unable to open distributed interface library
'...libcudaq_distributed_interface_mpi.so'
```

`install.sh` step 13 runs `activate_custom_mpi.sh` from
`site-packages/distributed_interfaces/` and persists `LD_LIBRARY_PATH` and
`MPI_PATH` with `conda env config vars set`, so they survive future
activations. A bare `export` does not.

mpi4py and the plugin must resolve to the **same** `libmpi.so`. Installing
both from conda-forge (as `environment.yml` does) is what guarantees it.
`quenais-doctor` checks this directly; by hand:

```bash
ldd $(python -c "from mpi4py import MPI; print(MPI.__file__)") | grep -i libmpi
ldd "$CONDA_PREFIX/lib/python3.11/site-packages/distributed_interfaces/libcudaq_distributed_interface_mpi.so" | grep -i libmpi
```

Both must point at the identical file inside the env. One from `/usr/lib`
and one from the env is a real ABI mismatch — fix it before debugging
anything else.

---

## Runtime environment

**setuptools < 82.** 82.0.0 (Feb 2026) removed `pkg_resources`; tequila and
several transitive deps still import it at load time, giving
`ModuleNotFoundError: No module named 'pkg_resources'`. The ceiling is now
in `setup.cfg`'s `install_requires` and `environment.yml`, and install.sh
pins it in step 2 before anything else runs. A residual deprecation
`UserWarning` afterwards is expected and harmless.

**`WANDB_MODE=offline`.** With no API key, wandb raises
`UsageError: No API key configured` in any non-interactive context — piped
through `tee`, run under `gdb`, or an sbatch job. It passes interactively,
which is why it tends to be discovered on the cluster. install.sh persists
`offline`. Use `disabled` to skip wandb entirely; confirmed not to affect
training behaviour or mask errors, so it's a clean way to rule wandb in or
out when debugging.

**Hydra key is `trainer.max_iters`, not `trainer.epochs`.** The latter does
not exist in the schema and Hydra refuses it in struct mode. `GqeSettings`
uses the right one; this only bites when overriding by hand.

---

## Quick reference — every failure seen so far

| # | Symptom | Root cause | Fix / where it's handled now |
|---|---|---|---|
| 1 | `ModuleNotFoundError: pkg_resources` | setuptools ≥82 removed it | pinned `<82` in setup.cfg, environment.yml, install.sh step 2 |
| 2 | Hydra `Could not override 'trainer.epochs'` | wrong config key | use `trainer.max_iters` |
| 3 | `UsageError: No API key configured` | non-interactive wandb login | `WANDB_MODE=offline`, persisted by install.sh step 14 |
| 4 | qiskit / scipy `pip check` conflicts | dead qiskit pin; patch-level scipy gap | `--no-deps` install of gqe-for-qsci; ignore scipy |
| 5 | `RuntimeError: architecture mismatch` | GPU below cc 8.0 | install.sh step 14 auto-selects `qpp-cpu` |
| 6 | `Illegal instruction (core dumped)` | PySCF's non-dispatching AVX-512 codepath | install.sh step 15 source build |
| 7 | `Unable to open distributed interface library` | CUDA-Q MPI plugin not compiled | install.sh step 13 |
| 8 | `cannot pickle 'cudaq...SampleResult'` | `sampler.py` ignored `self.mpi`; mpi4py's import-time `MPI_Init()` made the check spuriously true | source patch, install.sh step 12 |
| 9 | Training succeeds, nothing plottable | `train_pipeline.py` epoch prints missing | source patch, install.sh step 12 |
| 10 | Energies around −107 Ha for every molecule | `molecule=` override missing, so train.py ran N₂ | `build_command` asserts both mandatory overrides |
| 11 | `TypeError: 'NoneType' object is not iterable` | upstream geometry-based pool on a geometry-free embedding | `GqeSettings.validate()` rejects non-`dmet_*` pools |
| 12 | `quenais-gqe-setup`: "Patch not found" | `patches/` sat at the repo root while `package_data` looked under `quenais/` | patch moved to `quenais/patches/` |
| 13 | Run hangs at logger construction | `R-CASCI` reference key triggers full FCI over the embedding space | drop the key; the runner warns above `n_emb > 14` |
