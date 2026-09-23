# 10 — The second route: LASSCF and LASSQD

Everything in 02–07 follows one route: pick an active space, cut out **one**
region with DMET, solve it. This file covers the other route the package
now offers — cut the active space into **several fragments**, solve each in
the mean field of the others, and optimise the orbitals. With an exact
fragment solver that is LASSCF; with a quantum-sampled one it is LASSQD,
following Wang et al., *PNAS* **123**, e2603914123 (2026).

Code: `quenais/las/`. Usage and options:
[`../las_integration.md`](../las_integration.md).

---

## 1. Why a second route exists

DMET answers "what does the rest of the molecule do to *this* region?" by
building bath orbitals from a Schmidt decomposition of a reference density
(04). It gives one embedded problem, whose size (impurity + bath) grows
with the impurity.

That is awkward for a molecule with two or more metal centres. The
chemistry lives in several places at once, and treating all of them as one
impurity makes the embedded problem large again — for ScF, 9 impurity + 6
bath = 15 orbitals, 30 qubits.

LAS asks a different question: **can the active space be written as a
product of small, spatially separate pieces?** For ScF the answer is Sc
(6 orbitals) and F (3 orbitals): two problems of 12 and 6 qubits instead of
one of 30. For the Fe dimer of the paper it is one 10-orbital fragment per
iron.

The price is stated up front: fragments talk to each other only through
their averaged densities.

---

## 2. The ansatz

A LAS wave function is an antisymmetrised product of fragment wave
functions on top of a doubly occupied core:

```
|LAS>  =  A[ |psi_1> |psi_2> ... |psi_K> ] ^ |core>
```

Each fragment has its **own fixed number of alpha and beta electrons**.
That is the whole approximation, and everything else follows from it:

- the 1-RDM is block diagonal — one block per fragment, plus the core;
- every two-electron term that couples two *different* blocks factorises
  exactly into products of those blocks' 1-RDMs;
- so the energy is the Hartree–Fock functional evaluated with the full
  block-diagonal density, **corrected** inside each fragment by the
  difference between its exact 2-RDM and the mean-field one:

```
E = E_mf[D]  +  sum_K ( E2_K[Gamma_K] - E2mf_K[D_K] )
```

This is exact for the ansatz — no further approximation is made in
evaluating the energy (`quenais/las/energy.py`, validated against explicit
Fock-space expectation values).

**What is lost:** correlation *between* fragments. Two electrons on
different metals cannot be correlated beyond what their average densities
convey. Charge cannot flow between fragments, because the particle numbers
are fixed. Both matter most exactly where fragments are strongly coupled.

---

## 3. The fragment problem

Differentiating the energy with respect to fragment K's state gives an
ordinary electronic-structure problem in K's orbitals:

```
h_K^sigma  =  C_K^T ( h + J[D_env] - K[D_env^sigma] ) C_K ,   D_env = D - D_K
```

with the fragment's own two-electron integrals. Two consequences worth
remembering:

1. **The fragment Hamiltonian is spin dependent** whenever the environment
   is open shell (h_alpha != h_beta). An antiferromagnetically coupled
   dimer is precisely this case: each iron sees the other's unpaired spins.
   The exact solver handles this natively; the SQD path adds the difference
   (h_alpha − h_beta)/2 back inside the subspace, because PySCF's selected
   CI is spin restricted.
2. **The spin state must be chosen, not inherited.** An M_S = 0 fragment
   sector contains singlets *and* triplets; a high-spin sector still
   contains higher-S states. Both solvers therefore carry a penalty
   `shift * (S^2 − S(S+1))^2` with the fragment's requested 2S.

---

## 4. The two loops

**Inner (LASCI).** Fragments are solved with the others' current densities,
repeatedly, until they stop changing. With a sampling solver each pass
costs a full job, so LASSQD does one pass per macro cycle — as in the
paper.

**Outer (orbital optimisation).** With the fragment RDMs held fixed, the
orbitals are rotated downhill. The gradient is analytic
(`las_energy_and_gradient`, checked against finite differences); the
optimiser is a moving-frame L-BFGS with a diagonal-Hessian preconditioner
and an Armijo line search. Redundant rotations are excluded: core–core,
virtual–virtual, and rotations *within* a fragment.

Converged LASSCF = fragments are eigenstates of their own Hamiltonians
**and** the orbital gradient vanishes. The log prints both: the energy
change per cycle and `|g_orb|`.

A LASSCF solution is a stationary point, not a unique answer. Different
starting orbitals can land on different ones; they can differ by several
mHa, and comparing energies across basins is meaningless.

---

## 5. What SQD changes (LASSQD)

Each macro cycle, for every fragment:

1. **Mean field inside the fragment.** RHF/ROHF on the fragment
   Hamiltonian. Its orbitals define the basis the circuit is written and
   measured in — the one where the HF determinant dominates, which is what
   makes sampling efficient.
2. **LUCJ circuit** built from fragment UCCSD amplitudes, with heavy-hex
   interaction pairs by default (2 qubits per orbital, Jordan–Wigner).
3. **One sampling job for all fragments**, placed side by side on disjoint
   qubits with one classical register each. Fragment circuits are
   independent, so nothing is lost by gluing them, and only one job is
   submitted.
4. **SQD** (05 covers the algorithm): post-select or repair bitstrings,
   split them into K batches of d configurations, diagonalise H in each
   batch's alpha × beta product space, keep the lowest.
5. **Carryover** (the paper's Algorithm 2): determinants whose |amplitude|
   exceeds eps are added to every batch of the next iteration *and* to the
   next macro cycle, transported into the new fragment basis by maximum
   orbital overlap.

Two properties are worth stating because they make the numbers
interpretable:

- **The energy is always an exact expectation value** of a LAS state,
  computed from the returned RDMs with the true Hamiltonian. Sampling
  decides *which* state, never how its energy is scored. So LASSQD sits
  above LASSCF within the same basin.
- **The subspace fraction is the diagnostic.** `subspace_dim / full_dim`,
  printed per fragment per cycle, says how much of the fragment's Hilbert
  space the sampled configurations spanned. If it is small and the energy
  is far from LASSCF, add shots before blaming the method.

### Why carryover matters

Without it, every macro cycle re-samples from scratch and forgets the
determinants it found last time. The subspace, and hence the energy,
fluctuates from cycle to cycle, and orbital optimisation is being fed a
moving target. On ScF (README) the no-carryover run swung by ~12 mHa
between consecutive cycles and its subspace jumped between 16 and 36 of
36 configurations, while the carryover run fell monotonically at a fixed
subspace of 25. This is the paper's Figure 2 reproduced on a different
molecule.

---

## 6. Choosing fragments

The fragment specification (`ATOMS:NORB:NA,NB[:2S]`) is a physical claim:
*these orbitals live on these atoms, and they hold this many electrons of
each spin.* Three checks, in order:

1. **Do the orbitals localise?** The log prints, per fragment, the smallest
   singular value of the projection onto that fragment's atomic orbitals.
   Above ~0.9 is clean (ScF: 0.955 and 0.975); below 0.5 the code warns.
   AVAS (09) is the intended source of the active space because it names
   shells rather than orbital indices.
2. **Do the counts add up?** Orbitals and electrons must sum to step 1's
   active space, and the fragment M_S values must sum to the molecule's.
   Checked before any work starts.
3. **Is the split chemically sensible?** For an ionic system, put the
   ligand's occupied valence orbitals in the ligand fragment (ScF: F 2p
   holds 6 electrons, Sc 3d/4s holds 2). For an antiferromagnetic dimer,
   give the two metals opposite M_S — e.g. `'0:5:4,2:2' '1:5:2,4:2'` — so
   the product state can represent the low-spin coupling.

---

## 7. How to read a LAS log

```
fragment Sc   atoms [0]  (6o, 1a+1b, 2S=0)  localisation: min singular value 0.955
  fragment 0: SQD dim 25 (69.4% of FCI), valid shots 100.0% (10 distinct)
LAS cycle   5  E = -850.2853740894  dE = -1.19e-04  |g_orb| = 5.07e-04
```

| field | healthy | if not |
|---|---|---|
| localisation singular value | > 0.9 | wrong atoms, or the active space does not factorise |
| valid shots | ~100% | the circuit is not conserving particle number — a bug, not noise |
| SQD dim / % of FCI | grows, then holds | too few shots, or eps too large |
| dE | falls monotonically (LASSCF); small oscillation is normal for LASSQD | oscillation of mHa size means carryover is off or shots are too few |
| \|g_orb\| | falls to ~1e-5 | optimiser stalling; check the line-search message |

---

## 8. Where this sits next to DMET

| | DMET (04) | LAS |
|---|---|---|
| regions | one impurity + bath | K fragments, no bath |
| coupling to the rest | bath orbitals from a Schmidt decomposition | averaged densities of the other fragments |
| orbitals | fixed after step 2 | optimised (LASSCF) |
| problem size | n_imp + n_bath in one solve | n_K per fragment, solved separately |
| charge flow | bath allows it | forbidden between fragments |
| best case | one localised region of interest | several separated centres |

Because both routes share step 0, step 1 and step 4, and can use the same
sampling solver, the package can compare them on identical footing — same
molecule, same active space, same shots. That comparison is the reason the
second route was added.
