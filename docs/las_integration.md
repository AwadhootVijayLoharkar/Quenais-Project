# LAS solvers: LASSCF and LASSQD

`--solver lasscf` and `--solver lassqd` run the localized active space
method: the step-1 active space is split into fragments (typically one per
metal centre), each fragment is solved in the mean field of the others,
and the orbitals are optimised. `lasscf` solves each fragment exactly;
`lassqd` solves it with sample-based quantum diagonalization, following
Wang et al., *PNAS* **123**, e2603914123 (2026).

## Licensing: an independent implementation

`quenais/las/` is original Apache-2.0 code written from the published
equations and algorithms. It contains **no code** from

- `mrh` (GPL-2.0-or-later), the reference LASSCF implementation, or
- the LASSQD authors' repository, which carries no licence.

Neither is a dependency, submodule or optional import. Runtime dependencies
are PySCF, Qiskit, qiskit-aer, qiskit-addon-sqd and ffsim, all Apache-2.0.
`mrh` may be run **locally** to produce reference energies; only those
numbers are stored (`tests/regression/golden/las_mrh_reference.json`),
never code. Keep it that way: do not copy from either codebase, and do not
add an `import mrh` anywhere in this repository, including `tools/`.

Cite, in the thesis and any publication:

- M. R. Hermes, L. Gagliardi, *J. Chem. Theory Comput.* **15**, 972 (2019)
- M. R. Hermes et al., *J. Chem. Theory Comput.* **16**, 4923 (2020)
- Q. Wang et al., *Proc. Natl. Acad. Sci.* **123**, e2603914123 (2026)

## Where it sits in the pipeline

```
DMET route:  step 0 -> step 1 -> step 2 (DMET) -> step 3 sqd|gqe      -> step 4
LAS route:   step 0 -> step 1 --------------------> step 3 lasscf|lassqd -> step 4
```

LAS reads step 1 and **not** step 2. Results go to
`results/step3_lasscf.pkl` and `results/step3_lassqd.pkl`, so the exact
reference and the quantum run coexist; step 4 adds both to
`fig5_method_comparison.png` and `results_summary.csv`, draws
`fig6_las_convergence.png` (|E - E_LASSCF| per cycle, 1 kcal/mol line) and
writes `las_history.csv` (energies, gradients, SQD subspace sizes).

## Usage

```bash
# 1. active space -- AVAS names the shells, so it splits cleanly by atom
quenais-run --molecule ScF --basis sto-3g --steps 0 1 \
    --active-space-method avas --avas-ao-labels 'Sc 3d' 'Sc 4s' 'F 2p'

# 2. exact reference
quenais-run --molecule ScF --basis sto-3g --steps 3 \
    --active-space-method avas --avas-ao-labels 'Sc 3d' 'Sc 4s' 'F 2p' \
    --solver lasscf --las-fragments 'Sc:6:1,1' 'F:3:3,3'

# 3. LASSQD, then plots
quenais-run --molecule ScF --basis sto-3g --steps 3 4 \
    --active-space-method avas --avas-ao-labels 'Sc 3d' 'Sc 4s' 'F 2p' \
    --solver lassqd --las-fragments 'Sc:6:1,1' 'F:3:3,3'
```

Do not include step 2 in a LAS run; it is not used.

### Fragment specs

`ATOMS:NORB:NA,NB[:2S]`, quoted, one per fragment.

| part | meaning |
|---|---|
| `ATOMS` | `+`-joined atom indices (0-based) or element symbols (a symbol means every atom of that element): `Sc`, `0`, `0+1` |
| `NORB` | active orbitals on the fragment |
| `NA,NB` | alpha and beta electrons |
| `2S` | optional target spin, default `abs(NA-NB)` |

Checked before any work starts: orbitals and electrons add up to step 1's
active space, `sum(NA-NB)` equals `--spin`, fragments share no atoms, and
no fragment asks for more orbitals than its atoms have basis functions.
The log prints each fragment's localisation quality (smallest singular
value of the projection onto the fragment's atomic orbitals); below 0.5 it
warns -- the orbitals you asked for do not live on those atoms.

Antiferromagnetic Fe(III) dimer from the paper, (6e,5o) fragments:
`'0:5:4,2:2' '1:5:2,4:2'` (atoms 0 and 1 are the two Fe in `fefe.xyz`).

### Main options

| flag | default | |
|---|---|---|
| `--las-max-cycles` | 50 | macro cycles |
| `--las-no-orbital-opt` | off | LASCI: keep the localised step-1 orbitals |
| `--las-conv-tol` | 1e-8 / 1e-5 | energy convergence, lasscf / lassqd |
| `--las-density-fit` | off | for larger basis sets |
| `--lassqd-backend` | `aer_mps` | `aer_statevector`, `ibm` (+ `--lassqd-ibm-backend NAME`) |
| `--lassqd-shots` | 100000 | per macro cycle, all fragments in one job |
| `--lassqd-batches` / `--lassqd-samples-per-batch` | 15 / 170 | SQD K and d |
| `--lassqd-iterations` | 6 | configuration-recovery iterations |
| `--lassqd-carryover-eps` | 1e-5 | 0 = conventional SQD (no carryover) |
| `--lassqd-nroots` | 3 | Davidson roots per subspace |
| `--lassqd-lucj-optimize` | off | refine LUCJ with ffsim's linear method |

Everything else is in `quenais/settings/las.py` (`cfg.las.*`).

## What the code does

| file | |
|---|---|
| `energy.py` | LAS energy from fragment RDMs; fragment Hamiltonians (spin-resolved when the environment is open shell); analytic orbital gradient |
| `optimizer.py` | orbital rotation at fixed RDMs: moving-frame L-BFGS, diagonal-Hessian preconditioner, Armijo line search |
| `driver.py` | macro loop: build fragment Hamiltonians -> solve all fragments -> energy/gradient -> orbital step |
| `fragments.py` | spec parsing; localisation of the step-1 active orbitals by sequential SVD onto meta-Löwdin fragment AOs |
| `fci_solver.py` | exact fragments (PySCF `direct_uhf`, spin penalty) |
| `sqd_solver.py` | per fragment: RHF/ROHF basis -> UCCSD -> LUCJ (heavy-hex pairs); all fragment circuits in one job; SQD (post-selection / configuration recovery, K batches, selected CI with spin penalty); carryover of determinants with abs(c) > eps within and across macro cycles, transported by maximum orbital overlap |
| `sqd_bits.py` | bit-order conventions, subsampling, carryover bookkeeping |
| `stage.py` | step-3 entry point; reads step 1, writes the pickle |

Two things differ from a naive port and are deliberate:

- **Energy.** Always evaluated from the RDMs with the exact Hamiltonian, so
  every reported LASSQD energy is a true expectation value of a LAS state.
- **Open-shell environments.** A fragment next to an open-shell fragment
  has h_alpha != h_beta. PySCF's selected CI is spin restricted, so it gets
  the average and the difference is added exactly inside the subspace.
  Without this the Fe(III) dimer's fragments would be solved in the wrong
  Hamiltonian (a toy test showed ~5 mHa error).

## Validation

Runs anywhere (NumPy only), against brute-force Fock-space references that
share no code with `quenais.las`:

- `tests/test_las_core.py` -- energy equals the explicit product-state
  expectation value (closed and open shell); fragment Hamiltonians reproduce
  the total energy; orbital gradient vs finite differences; one fragment =
  CASCI; LASSCF converges to a stationary point above FCI.
- `tests/test_las_sqd_logic.py` -- the LASSQD solver with exact sampling:
  bit order, batching, carryover and its transport, RDM back-rotation;
  LASSQD reproduces LASSCF when SQD spans each fragment (closed and open
  shell); the spin-dependent subspace operator vs brute force.
- `tests/test_las_bits.py`, `tests/test_las_settings.py`.

Needs PySCF (and Qiskit for the last one):

- `tests/test_las_pyscf.py` -- RHF through the LAS energy; single fragment
  LASCI = CASCI and LASSCF = CASSCF; H4 two-fragment bounds; density
  fitting; selected CI with spin-dependent h = UHF FCI; LASSQD on H4 with
  `aer_statevector` = LASSCF.
- `tests/test_las_reference.py` -- against locally produced mrh numbers.

Suggested ladder for the thesis: H4/H8 (fill in the mrh references) ->
ScF (Sc 6o + F 3o, compare with DMET on the same AVAS space) -> stretched
N2 (carryover vs `--lassqd-carryover-eps 0`, plus `det_analysis` oracle
and CIPSI at matched subspace size) -> [Fe(H2O)4]2bpym4+ with (6e,5o)
fragments against the paper's LASSCF value, -3655.496377 Eh (Table 1).

## Limitations

- Inter-fragment correlation is mean field -- inherent to LAS.
- Orbital optimisation is first-order (L-BFGS). It converges to the same
  stationary points as a second-order LASSCF but can need more macro
  cycles; for `lassqd` each cycle is a sampling job.
- For `lassqd`, rotations inside a fragment are not optimised (as in the
  paper); for exact fragments they are redundant anyway.
- Only the lowest state per fragment; no LASSI.
- Carryover transport maps orbitals by maximum overlap; if the fragment
  basis rotates too much between cycles the carryover set is dropped for
  that cycle (logged).
