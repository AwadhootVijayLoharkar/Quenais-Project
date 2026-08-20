# 04 — Step 2: density matrix embedding (DMET)

**Code:** `quenais/embedding/hamiltonian.py`, `quenais/embedding/dmet_lib.py`
→ `results/step2_hamiltonian.pkl`

This is the core of the method. It turns "solve this molecule" into "solve this
small embedded problem", and it does so *exactly* with respect to a chosen
reference state.

---

## 1. The physics: why a small bath is enough

Take a wavefunction and split the orbitals into an **impurity** A (the active
space) and an **environment** B. Any pure state admits a Schmidt decomposition:

```
|Ψ⟩ = Σ_i λ_i |a_i⟩ ⊗ |b_i⟩
```

The number of non-vanishing λ is bounded by the dimension of the **smaller**
subsystem — here, the impurity. This is a theorem, not an approximation.

The consequence is the whole reason DMET exists:

> **N_bath ≤ N_imp, independently of how large the molecule is.**

Environment states with λ = 0 do not appear in the expansion at all. So the
entire environment couples to the impurity through at most `N_imp` bath
orbitals, and those can be constructed exactly. Everything else is frozen at
mean-field level — again exactly, not approximately, because it is not
entangled with the impurity.

**Qubit count becomes a property of the fragment, not of the molecule.** ScH
needs 22 qubits because its impurity is 6 orbitals, not because ScH has 22
electrons. Embed the same 6-orbital impurity in a molecule ten times larger and
it still needs 22 qubits.

### The scope of "exact"

The bath is exact **with respect to the reference state used to build it** —
not with respect to the true FCI ground state of the molecule. A poor reference
gives a bath that faithfully represents a poor state.

This is worth stating before someone else does. It is also where the residual
ScH embedding error comes from: `reference = "casci"` means the bath is exact
for the CASCI reference, and that reference is itself limited by the active
space chosen in step 1.

## 2. The reference density

`get_reference_density()` builds the one-particle density matrix that the
Schmidt decomposition will act on. Two options:

- **`"mp2"`** — reuse step 1's MP2 1-RDM. Fast, and unreliable exactly where
  static correlation is strong.
- **`"casci"`** (default) — CASCI within the active space, screened by the core
  mean-field potential.

Two past bugs here are instructive.

**Screening.** The CASCI is run on a Hamiltonian that includes Coulomb and
exchange screening from the non-active electrons (`h1e_bare + 2·vj − vk`):

> "This used to be built from bare kinetic + nuclear attraction only, ignoring
> Coulomb/exchange screening from the non-active electrons. The embedding stage
> DOES add that potential when building h1e_emb, so the reference density and
> the embedding solver were being computed from two different effective
> Hamiltonians -- one screened, one not. That mismatch feeds straight into a
> badly chosen Schmidt bath."

**The MP2 overlay.** After the CASCI values are computed, they are written into
the active–active block of the *MP2* density rather than into a fresh diagonal:

> "The earlier version built a bare diagonal from no_occ for every non-active
> orbital and filled in only the active-active block, leaving every
> active/non-active CROSS TERM at exactly zero -- a block-diagonal density with
> no impurity-environment coupling at all. **The Schmidt decomposition exists
> precisely to extract that coupling, so it had nothing to find:** all singular
> values came back 0.0 and the resulting 'bath' was arbitrary noise."

> "Real correlated densities are not block-diagonal, so the MP2 density supplies
> the genuine active-core coupling while CASCI supplies the better active-space
> values."

This is why the `casci` path still requires `dm_ao_alpha_mp2` /
`dm_ao_beta_mp2` from step 1.

## 3. The Schmidt decomposition, concretely

Working in the **Löwdin (symmetrically orthogonalised)** basis, where overlap
is the identity:

```python
S_sqrt, S_invsqrt = lowdin_matrices(S)

C_imp  = mo_coeff[:, mo_list]          # impurity orbitals, AO basis
Q_imp  = S_sqrt @ C_imp                # impurity, Löwdin basis  (n_ao x n_imp)
dm_lo  = S_sqrt @ dm_ao_total @ S_sqrt # reference density, Löwdin basis

P_env  = np.eye(n_ao) - Q_imp @ Q_imp.T   # projector onto the environment
F      = P_env @ dm_lo @ Q_imp            # the coupling block  (n_ao x n_imp)

U_env, sv, _ = np.linalg.svd(F, full_matrices=True)
Q_bath = U_env[:, :n_bath]                # leading left singular vectors
```

Reading it back in words: **`F` is the environment-projected part of the
density's coupling to the impurity.** Its singular values `σ_i` measure how
strongly each environment direction is entangled with the impurity. The
corresponding left singular vectors *are* the bath orbitals, and they live in
the environment subspace by construction.

`P_env` is a genuine orthogonal projector because `Q_impᵀ Q_imp = C_impᵀ S C_imp = I`
— that is what the Löwdin transformation buys.

Since `F` has `n_imp` columns, `sv` has length `n_imp`, which is the Schmidt
bound made concrete.

Finally the embedding basis is assembled and transformed back to AOs:

```python
Q_emb = np.hstack([Q_imp, Q_bath])     # or just Q_imp if n_bath == 0
C_emb = S_invsqrt @ Q_emb              # so that C_embᵀ S C_emb = I
```

`n_emb = n_imp + n_bath`, and the qubit count is `2 × n_emb`.

## 4. Choosing the bath size — and why zero is a valid answer

`adaptive_bath(sv, n_imp, max_embed, bath_tol)` returns
`(n_bath, gap, sv2_coverage)`:

1. `max_bath = min(n_imp, max_embed - n_imp)` — the Schmidt bound, capped by
   the qubit budget.
2. Keep only `σ > bath_tol` (default `1e-8`).
3. **If nothing clears the tolerance, return 0.**
4. Otherwise take the union of two criteria: the largest gap in the retained
   spectrum, and the smallest count reaching 99.9 % of cumulative `σ²`.

Step 3 is the single most expensive bug this codebase has had:

> "ZERO IS A VALID ANSWER, and getting that wrong is the most expensive bug this
> code has had. The earlier version fell back to `sv[:max_bath]` -- taking the
> largest singular values regardless of whether any cleared the tolerance --
> whenever none did. That is not a safe fallback; **it manufactures a bath out of
> numerically meaningless near-zero values.**"

> "On N2's (4e,4o) active space every Schmidt singular value comes back exactly
> zero (measured max 5.4e-15): the active orbitals are already close to
> eigenvectors of the reference density, so there is no impurity-environment
> entanglement left to extract. The fallback still produced '4 bath orbitals',
> which gave a badly non-orthonormal embedding basis and roughly **20 Ha** of
> error."

The three validated systems:

| system | `n_imp` | `n_bath` | `n_emb` | qubits |
|---|---|---|---|---|
| LiH CAS(2e,2o) | 2 | 2 | 4 | 8 |
| ScH CAS(4e,6o) | 6 | 5 | 11 | 22 |
| N₂ CAS(4e,4o) | 4 | **0** | 4 | 8 |

N₂'s zero is *correct physics*, and the regression suite guards it with
`max_abs_sv_all_below: 1e-8`.

One nice detail: the orthogonality diagnostic checks only the bath vectors
actually kept, because

> "a column whose singular value is exactly 0.0 has a numerically arbitrary
> direction, free to overlap Q_imp, and is discarded."

## 5. The embedded electron count — bug #2

```python
dm_emb_alpha  = C_emb.T @ S @ dm_ao_alpha @ S @ C_emb
ref_occ_alpha = np.clip(np.diag(dm_emb_alpha), 0.0, 1.0)
n_alpha       = int(round(float(np.sum(ref_occ_alpha))))
```

The count comes from the **trace of the reference density in the embedding
basis**, not from the active-space electron count.

> "NOT from the active-space count. That is only correct when n_bath == 0; once
> bath orbitals exist the embedding holds whatever the reference density
> actually puts there. On LiH the active-space count gives (1, 1) while the
> reference-density trace gives (2.0000076, 2.0000076) -> (2, 2), and the wrong
> count roughly doubles the energy."

The stage prints both values side by side so the discrepancy is visible, and
`test_reference_harness.py` asserts explicitly that `n_alpha != naive_alpha` on
LiH — "the buggy formula must not agree here".

## 6. The embedded Hamiltonian

**Core potential.** Everything outside impurity+bath is frozen, but its
electrons still exert Coulomb and exchange on those inside. That is physics,
not bookkeeping:

```python
P_core_lo   = np.eye(n_ao) - Q_emb @ Q_emb.T     # outside impurity + bath
dm_core_*   = <project the spin densities through P_core_lo>
h1e_eff     = h1e_bare + (vj_a + vj_b) - 0.5 * (vk_a + vk_b)
```

**Integral transformation.**

```python
h1e_emb = C_emb.T @ h1e_eff @ C_emb
h2e_emb = symmetrize_h2e(ao2mo.kernel(mol, C_emb, compact=False))
```

`symmetrize_h2e` averages all 8 permutations of the ERI tensor, enforcing full
8-fold permutational symmetry exactly.

**`ecore`.** Defined as the residual between the true full-molecule UHF energy
and the HF energy evaluated inside the embedding space:

```
ecore ≡ E_UHF(full molecule) − E_HF-in-embedding[projected UHF density]
```

The density used is the **real projected UHF density**, which is bug #3:

> "This used to be a naive aufbau filling: occupy the first n_alpha columns of
> C_emb -- which are the impurity orbitals, then the bath orbitals -- regardless
> of h1e_emb's actual eigenvalue ordering. Since ecore is defined as
> mf.e_tot - e_hf_emb, that invented density directly determines how much of the
> true HF energy is folded into ecore rather than the embedding space.
> Projecting the real UHF solution removes the guess."

Because `ecore` is *defined* as that difference, an "`ecore` self-consistency
check" is tautological. See §8.

## 7. The chemical potential μ

```python
h1e_emb -= mu * I
ecore   += mu * (n_alpha + n_beta)
```

μ is chosen by bisection so the embedded particle number matches the reference.
For a fixed-N solver the two shifts cancel exactly — the total energy is
invariant, and this was confirmed bit-for-bit. It is kept on because it matters
for samplers that do not conserve N.

The bracket is derived from `h1e_emb`'s own eigenvalue spectrum rather than
hardcoded:

> "A fixed guess such as (-5, 5) Ha does not bracket the true chemical potential
> once the core mean-field potential has shifted those eigenvalues -- on N2, four
> of eight embedding eigenvalues already sat below -5 Ha while the target
> electron count was 3."

If bisection cannot bracket the target it warns and returns the Hamiltonian
**unshifted** rather than silently returning a wrong answer. Treat that warning
as a real diagnostic.

## 8. The check that can actually fail

```python
verify_embedded_scf(step2, tol=2e-7)
```

Build a fake, atom-less PySCF molecule with `nelectron = n_alpha + n_beta`.
Monkey-patch its integrals to be the embedding's: `get_hcore → h1e`,
`get_ovlp → I`, `_eri → h2e`, `energy_nuc → ecore`. Run a real, converged SCF.

**It must land on the full molecule's UHF energy.**

> "This is the single most diagnostic quantity in the pipeline. Unlike the E_core
> self-consistency identity, which is tautological, this one can actually fail:
> if a converged self-consistent HF on h1e_emb / h2e_emb / ecore does not land on
> the full molecule's UHF energy, the embedding Hamiltonian itself is wrong --
> independent of mu or the choice of reference density."

Tolerance `2e-7` Ha; validated at `1.3e-7` Ha on ScH. Because it runs *after*
the μ shift, passing also confirms the μ cancellation.

Stored in the pickle as `embedded_scf_check` =
`{e_scf_emb, e_uhf_full, delta, within_tol}`, deliberately written **before**
pickling "so its verdict is stored with the data it describes".

This one check independently caught three separate real bugs: the no-environment
case (0.73 Ha off, with a perfectly smooth curve), the electron-count bug, and a
mean-field density scaled by filling.

## 9. Output contract

```
schema_version, h1e, h2e, ecore, mu, n_emb, n_imp, n_bath, n_alpha, n_beta,
sv, sv_all, sv_gap, sv2_cov, uhf_energy, reference_density_info,
ref_occ_alpha, ref_occ_beta, mol_info, provenance, embedded_scf_check
```

`sv` is the kept spectrum; `sv_all` is the full one (length `n_imp`) — keep the
latter, it is what shows N₂'s zeros.

`STEP2_SCHEMA_VERSION = 2`:

> "Bumped when the step 2 pickle layout changes, so downstream consumers fail
> loudly on an old file instead of raising KeyError mid-run."

Step 2 refuses a step 1 pickle from a different molecule or basis:

> "silently building an embedding on another molecule's active space is exactly
> the failure this check exists to stop."

## Next

[`05_step3_quantum_solver.md`](05_step3_quantum_solver.md) — solving the
embedded problem.
