# Licensing

**This package is Apache-2.0, and the default route through it uses only
Apache/BSD dependencies.** Two optional components pull in GPL-3.0
libraries. Nothing GPL is vendored, copied or redistributed here: those
libraries are fetched from their own upstreams by `install.sh` and
installed into your environment.

## What is in this repository

| Part | Licence |
|---|---|
| `quenais/` (all of it, including `quenais/las/`) | Apache-2.0 |
| `tests/`, `tools/`, `docs/`, notebooks | Apache-2.0 |
| `quenais/patches/gqe_dmet_source.patch` | modifies `gqe-for-qsci` source and carries its context lines -- see below |

`quenais/las/` (LASSCF and LASSQD) is original code written from the
published equations of Hermes & Gagliardi (JCTC 2019, 2020) and Wang et
al. (PNAS 2026). It contains **no code** from `mrh` (GPL-2.0-or-later) or
from the LASSQD authors' repository (no licence file), and neither is a
dependency, submodule or optional import. `mrh` may be run locally to
produce reference energies; only those numbers are stored, in
`tests/regression/golden/las_mrh_reference.json`. Keep it that way: never
add an `import mrh` anywhere in this repository.

## Dependencies

| Dependency | Licence | Needed for |
|---|---|---|
| PySCF | Apache-2.0 | everything |
| NumPy, SciPy, matplotlib, h5py, networkx | BSD-style | everything |
| Qiskit, qiskit-aer, qiskit-addon-sqd, qiskit-ibm-runtime | Apache-2.0 | `quenais[qiskit]`: sqd, skqd, sqdrift, lassqd |
| ffsim | Apache-2.0 | LUCJ circuits (lassqd) |
| OpenFermion | Apache-2.0 | qubit Hamiltonians |
| [Active Space Finder (HQS)](https://github.com/HQSquantumsimulations/ActiveSpaceFinder) | Apache-2.0 | step 1, `--active-space-method asf` |
| **[block2](https://github.com/block-hczhai/block2-preview)** | **GPL-3.0** | the DMRG backend ASF calls |
| **[theochem/PyCI](https://github.com/theochem/PyCI)** | **GPL-3.0** | built from source; required by `gqe-for-qsci` |
| CUDA-Q, torch, pytorch-lightning | Apache-2.0 / BSD | `quenais[cudaq]`: the GQE solver |
| `gqe-for-qsci` (submodule), tequila | see each project's own LICENSE/NOTICE | the GQE solver |

Verify the last row in your own checkout (`external/`, `gqe-for-qsci/`)
and record what you find here; their terms carry into anything derived
from them.

## The Apache-only path

Neither GPL dependency is needed for the pipeline's default route:

| Stage | Apache-only choice | Pulls in GPL |
|---|---|---|
| Step 1, active space | `--active-space-method avas` or `apc` (PySCF) | `asf` (block2) |
| Step 3, solver | `sqd`, `skqd`, `sqdrift`, `lasscf`, `lassqd` | `gqe` (PyCI) |

So `pip install -e ".[qiskit]"` plus AVAS plus any SQD/LAS solver is
Apache-2.0 end to end. `install.sh` installs the GPL pieces because it
provisions *every* stack; a Qiskit-only or LAS-only install does not need
them.

## Practical rules

1. **Ship source, not bundles.** Distributing this repository is fine.
   Distributing a Docker image, a packed conda environment or a wheel that
   *contains* block2 or PyCI makes that artifact a combined work under
   GPL-3.0, and it can no longer be labelled Apache-2.0.
2. **Results are not covered.** Energies, plots and CSVs produced by any of
   these programs are your data, whichever licence the program had.
3. **Before a public release or a PyPI upload**, check `gqe-for-qsci`'s
   NOTICE for attribution terms that carry into the patch and the four
   DMET integration files.
4. **Citing is separate from licensing.** Cite PySCF, Qiskit, ffsim,
   qiskit-addon-sqd, ASF, block2 and the LASSCF/LASSQD papers in any
   publication, regardless of licence.

None of this is legal advice. For a public release, or anything going to
the QuEnAIS partners, confirm with Fraunhofer's legal or
technology-transfer office.
