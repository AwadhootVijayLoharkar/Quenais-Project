# Known limitations

Status: the DMET pipeline (steps 0-2) and the GQE integration layer are
validated against the reference values in the README. The GQE solver has
not yet been run end to end THROUGH THE PACKAGE -- the adapter, pools,
runner and patch tooling are each tested, but the full
`quenais-run --solver gqe` path is the last unexercised link. Until it is,
keep `scripts/test_8/` as the oracle.


Read this before trusting a result from a system that is not LiH or N₂.

## Active-space selection under-selects for transition metals

**Status:** known, to be fixed in a later release.

ASF's entanglement-entropy thresholds are calibrated on main-group systems.
For d-block elements they select too few orbitals, and too few active
electrons.

Observed on ScH:

- ASF chose 4 orbitals but only 2 active electrons, treating MO 9 as inert
  core. For a 3d metal that leaves most of the interesting correlation
  outside the active space.
- Phase C gap detection then cut those 4 down to 3, second-guessing ASF's
  entropy selection with a cruder MP2 occupation-deviation metric.
- The resulting CASSCF(2e,3o) recovered −22.79 kcal/mol against HF, while
  plain CCSD recovered −44.01.

**Workaround:** set `force_active_space` explicitly. ScH ships with
`[9, 10, 11, 12, 13, 14]` — every orbital in ASF's own window with entropy
≥ 0.055, which is a clean break (MO 14: S=0.072 vs MO 15: S=0.023). MOs 9
and 10 are the occupied valence pair, giving 4 active electrons, matching
ScH's actual valence count. MOs 0–8 are genuine Sc core.

**How to tell it is happening:** a well-chosen active space should put
CASSCF+NEVPT2 at or below CCSD(T). If NEVPT2 lands above CCSD(T), the
active space is too small.

Even with the forced space, ScH's NEVPT2 (−752.702671) sits 7.2 mHa **above**
CCSD(T) (−752.709890). So ScH's classical reference is softer than LiH's or
N₂'s and should not be used as a tight answer key. The DMET quantities on
ScH are unaffected and reproduce to 1e-10 across machines.

## CASSCF and NEVPT2 are not reproducible to tight tolerance

CASSCF is an optimisation, and it can converge to different valid solutions
from run to run. Two runs of identical ScH input gave CASSCF −752.680677 and
−752.681604 (0.93 mHa apart); NEVPT2 is built on the CASSCF reference and
moved 3.6 mHa with it.

Neither is an error. But it means any claim of the form "DMET agrees with
NEVPT2 to X mHa" is solution-dependent. For ScH that agreement has been
observed at both 0.50 mHa and 3.15 mHa depending on which CASSCF solution
was found.

Quantities are labelled in `results_summary.csv` by trust tier:

| tier | reproducibility | members |
|---|---|---|
| `deterministic` | ~1e-10 across machines | HF, MP2, CCSD, CCSD(T), DMET+CASCI, `ecore`, `mu`, `n_bath`, `n_alpha`/`n_beta`, Schmidt spectrum |
| `optimizer-dependent` | ~1 mHa | CASSCF, NEVPT2 |
| `stochastic` | varies by design | DMET+GQE |

## Only closed-shell systems are validated

Unequal-spin code paths exist but are untested.
`chemical_potential_correction()` warns and no-ops for `n_alpha != n_beta`,
which is acceptable — μ is provably inert for fixed-particle-number solvers.
Open-shell transition-metal systems (TiO, ScO) will exercise untested
branches.

## GQE sampling capacity limits larger systems

GQE's accuracy is bounded by how much of the determinant space it can reach.
On ScH (22 qubits, C(11,4)² = 108,900 determinants) the recovered correlation
scaled with circuit depth and subspace cap:

| settings | error vs embedded CASCI | note |
|---|---|---|
| ngates=10, samples=10 | 60.4 mHa | stalled at HF |
| ngates=20, samples=100 | 36.8 mHa | subspace froze at 1116 |
| ngates=40, samples=100 | 24.1 mHa | subspace hit the 2000 cap |

LiH and N₂ (4 embedding orbitals each) converge to the embedded CASCI energy
exactly, so this is a capacity limit on larger systems, not a correctness
problem.

## `reference_keys` can make large systems intractable

Including `"R-CASCI"` in the GQE reference keys triggers a full FCI over the
entire embedding space purely for logging. Cost grows combinatorially and
becomes impractical above roughly 12–16 embedding orbitals. If a run hangs or
exhausts memory before training starts, drop `"R-CASCI"` — it is a reference
value only and is not used in training or diagonalisation.
