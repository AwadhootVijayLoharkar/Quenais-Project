# 05 — Step 3: the quantum solver

**Code:** `quenais/quantum/` → `results/step3_results.pkl` (Qiskit) or
`results/gqe_train.log` (GQE)

Step 2 handed us a small Hamiltonian `h1e`, `h2e`, `ecore` in an abstract
orbital basis. Now we solve it — and this is the only stage where a quantum
device appears.

---

## 1. Two stacks, one dispatcher

```python
QISKIT_SOLVERS = ("sqd", "skqd", "sqdrift")   # in-process, Qiskit
GQE_SOLVERS    = ("gqe",)                     # subprocess, CUDA-Q
```

`quenais.quantum.dispatch(cfg, force=...)` is the single routing point. It
imports lazily and rewrites `ImportError` into an actionable message
(`pip install 'quenais[qiskit]'` / `'quenais[cudaq]'`).

> "Nothing heavy is imported at module scope... That is what lets a Qiskit-only
> install import this package at all."

## 2. The physics: QSCI

The idea behind all four solvers is the same, and it is worth stating plainly
because it is *not* the usual "run VQE and read off the energy".

1. Prepare some trial state on the device.
2. **Measure it in the computational basis.** Each shot returns a bitstring,
   which is a determinant.
3. Keep the set of distinct determinants observed. Call it `S`.
4. Build the projected Hamiltonian `P_S H P_S` and **diagonalise it classically**.

The energy is the lowest eigenvalue of that projected matrix:

```
(P_S H_emb P_S) c = E c,     E ≥ E_0  for any S
```

Two properties follow, and they are the entire argument for this family of
methods on near-term hardware:

- **It is variational.** Failing to sample an important determinant enlarges the
  subspace error but can never push the energy *below* the truth. Contrast with
  coupled cluster, which has no such bound.
- **The device only supplies a proposal distribution.** No phase estimation, no
  high-precision expectation values. Noise degrades the proposal — it cannot
  break the bound.

So the quantum computer is being used as a *sampler*, and the eigenvalue
problem stays classical. Which raises the question the whole thesis is about:
is the quantum sampler's proposal actually better than a classical one? See
[`06_determinant_selection.md`](06_determinant_selection.md).

## 3. Mapping fermions to qubits

Electrons are fermions; qubits are not. The **Jordan–Wigner** transform bridges
them:

```
a†_p = ½ (X_p − i Y_p) ⊗ Z_{p−1} ⊗ ... ⊗ Z_0
```

One qubit per spin orbital, so `n_emb = 11` gives 22 qubits for ScH. The
trailing **Z-string** encodes fermionic anticommutation — the sign a fermionic
operator picks up from every orbital below it.

Two consequences that caused real bugs:

**Circuit depth depends on orbital ordering,** because the Z-string grows with
orbital index.

**Particle-number conservation belongs to the whole sum, never to one term.**
A mapped fermionic excitation is a *sum* of Pauli words. Individually, none of
them conserves electron number; together they do. Therefore:

> "DMETPauliEvolutionPool appends every Pauli TERM of an excitation generator as
> its own pool element, so each element is a single Pauli word. Particle number
> conservation is a property of the full SUM of terms in a JW-mapped fermionic
> excitation, never of one term alone, so that pool cannot conserve electron
> number however its flags are set. **Measured on ScH: roughly half of every
> sample was discarded as symmetry-violating.**"

Same reasoning applies to `remove_z_ladder`:

> "The Jordan-Wigner Z-ladder encodes fermionic anticommutation. Removing it
> makes exp(i*theta*P) no longer particle-number conserving, so sampled states
> leak into the wrong electron-number sector."

**Warning for anyone reading both stacks:** they use *different* spin-orbital
orderings. `solver.py` uses **blocked** (alpha `0..n_emb-1`, beta
`n_emb..2n_emb-1`); `gqe_pools.py` uses **interleaved** (alpha `2p`, beta
`2p+1`). Each is internally consistent, but bitstring conventions do not
transfer between them.

## 4. The Qiskit family: SQD, SKQD, SqDRIFT

All three end in the same classical loop, `iterative_solve`:

```python
recover_configurations(bsm, probs, avg_occs, num_elec_a, num_elec_b, ...)
solve_fermion(bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0)
energy = e_emb + ecore
```

seeded with the HF occupation pattern. They differ only in how the bitstrings
are produced.

**SQD** — one ansatz circuit, sampled at `n_shots` (default 8192).

**SKQD** — Krylov. Build `exp(−iHt)` as a Trotterised `PauliEvolutionGate`,
apply it `k = 0 … krylov_dim−1` times, sample each. Counts are **cumulative**
across `k`. This explores a Krylov subspace rather than relying on one ansatz.

**SqDRIFT** — qDRIFT randomised Trotterisation. `sqdrift_num_circuits` (70)
independent circuits, each with its own random operator ordering, all merged
into one count set. Uniquely, it calls `inject_hf_reference` to guarantee the
HF determinant is present.

### Ansatz choice matters more than it looks

| ansatz | particle number | note |
|---|---|---|
| `su2` | **not conserved** | "⚠ SU2 does not conserve particle number — ~40-60 % of shots will be filtered" |
| `lucj` (default) | conserved by construction | Givens rotations + Jastrow, built by hand |

After sampling, `filter_bitstrings` discards any row whose alpha or beta
population is wrong. If nothing survives:

```
RuntimeError: No valid bitstrings after particle-number filtering...
If using SU2: increase n_shots or switch to lucj.
```

Note the SU2 parameters are **random, not variationally optimised**
(`default_rng(42)`), so SU2 here is a sampling heuristic rather than a VQE
ansatz.

### Step 3 output (Qiskit path)

```
solver, ansatz, mapping, backend, energy, spin_sq,
uhf_energy, mp2_energy, iterations, mol_info
```

`iterations` is a list of per-iteration dicts. SQD/SqDRIFT rows carry
`iter, energy, e_emb, ecore, n_configs, vs_uhf, vs_mp2`; SKQD rows carry
`k, energy, n_configs, vs_uhf, vs_mp2`.

One asymmetry to be aware of: `solver.main` returns **`None`** on a cache hit
but a dict otherwise.

## 5. GQE — the generative eigensolver

GQE replaces continuous-parameter optimisation with a **generative model over
discrete gate sequences**. A transformer policy `π_θ` proposes sequences drawn
from an operator pool; gradients flow into the model, not into circuit angles.
Training targets `π_θ ∝ e^{−βE}`, so sequences that give low energy become more
likely.

```
|Ψ_θ⟩ = Π_k exp(−i t_k P_{j_k}) |Φ_HF⟩
```

Why GQE rather than VQE: a discrete pool, no angle optimisation, and the model
can be trained offline on previously evaluated sequences.

### Why an adapter was needed

This is the key architectural point of the whole integration:

> "PySCFMolecule builds its mean-field object from real atoms via
> gto.M(atom=...) and picks an active space as a SUBSET of canonical MOs. **A
> DMET embedding space -- impurity plus bath, from a Schmidt decomposition -- is
> a genuinely different orbital basis: a rotated combination, not a subset. It
> cannot be expressed that way.**"

The solution uses PySCF's custom-Hamiltonian pattern (the same one used for
FCIDUMP): hand `h1e_emb`/`h2e_emb` to PySCF as if they were AO integrals, run a
genuine SCF to find the HF-optimal orbitals *within* the embedding space, and
fold `ecore` in as the nuclear-repulsion constant.

> "mf.e_tot then comes out as the DMET-consistent total energy directly, because
> **E_total = ecore + ⟨psi|H_emb|psi⟩ holds for any psi in the embedding
> space** -- not only for the reference DMET used internally."

`DMETEmbeddingMolecule` exposes `norb`, `nelec`, `hf`, `mc`, `cas_hamiltonian`,
`compute_casci()`, `compute_ccsd()`, `ccsd_amplitude`, `casci_avg_occs()`, with
`geometry = None` and `basis = None` as deliberate placeholders — an embedding
has neither.

`load_from_dmet_pickle()` validates that `{h1e, h2e, ecore, n_alpha, n_beta}`
are all present and names the missing ones otherwise.

### The operator pool

Upstream pools rebuild the molecule from its geometry to derive CCSD
amplitudes. An embedding has no geometry, so:

> "operator_pool.py's loop over molecule.geometry dies with **'TypeError:
> 'NoneType' object is not iterable'**. Even if it did not, it would silently
> build gates for the wrong active space."

The replacements derive excitations from the **embedding's own CCSD
amplitudes** and work on abstract spin-orbital indices:

| spec | one pool element per | conserves N |
|---|---|---|
| `dmet_excitation` (default) | excitation (all its Pauli terms accumulated) | **yes** |
| `dmet_pauli_evolution` | individual Pauli term | no — see §3 |

The excitation generator itself:

```python
forward  = Π_i a†_{q_i} a_{p_i}
backward = Π_i a†_{p_i} a_{q_i}
return jordan_wigner(-1j * (forward - backward))
```

> "**THE -1j IS LOAD-BEARING.** (forward - backward) is the raw anti-Hermitian
> UCC generator; tequila stores its generator already in Hermitian form,
> H = -i(T - T†), ready for exp(-i*theta*H). Validation against tequila found a
> consistent ratio of exactly +i between the two... That validation passed on
> singles (both spins) and doubles, with identical Pauli-string sets and a ratio
> of exactly 1.000000 on every term. **A sign error would not crash anything, it
> would silently seed the pool with wrong-phase excitations.**"

Note: the file points at `tests/test_excitation_generator.py` as the guard for
this, and that file is **not present** in the current tree. If you touch the
generator, that validation needs reconstructing.

### The external repo

GQE runs as a **subprocess** against a checkout of `gqe-for-qsci` pinned at
commit `0a201ea`, with a patch applied to three files:

| file | change | why |
|---|---|---|
| `factory.py` | register the two DMET pools | otherwise they are unreachable |
| `gqe/sampler.py` | make the `sampler.mpi` flag authoritative | upstream takes the MPI branch whenever MPI happens to be initialised |
| `train_pipeline.py` | print `[epoch N] {metrics}` | **load-bearing instrumentation** — without it the run succeeds and produces nothing parseable |

`verify_gqe_repo()` checks the commit, the stamp file, the patch hash and the
three files' presence, and is called **before** launching training:

> "so a run that could only produce an unparseable log fails in a second rather
> than after hours of GPU time."

Patching is a deliberate explicit step, not a pip hook:

> "Patching a git submodule during `pip install` fails in wheel builds, in CI
> without submodules, and on every editable reinstall -- and **a half-applied
> patch is exactly the silent-wrong-value failure this project keeps hitting.**"

### The two mandatory Hydra overrides

`build_command()` asserts both are present before launching:

- **`molecule=dmet_embedding`** —
  > "configs/default.yaml declares `defaults: - molecule: n2`, so without this
  > override train.py loads N2 and never reads the DMET config. **That failure
  > is silent: the run succeeds**, and visualization reports (embedding CASCI) +
  > (N2's convergence error), which looks plausible and means nothing."
- **`operator_pool.spec=dmet_*`** — the stock pools need a geometry.

Plus `molecule.step2_pickle_path`, always passed explicitly:

> "Always pass it: a stale hardcoded path there once produced plausible but
> meaningless results for an extended period."

**Sanity check for any new run:** look at the *magnitude* of the energy in the
first few epochs. LiH ≈ −7.9, N₂ ≈ −107.6, ScH ≈ −752.7. If the scale is wrong,
the override did not take.

### Reading the epoch log

Produced by the patch, captured by `gqe_runner`, parsed by
`plots.parse_gqe_log` (with `ast.literal_eval`, never `eval`).

| column | meaning |
|---|---|
| `GQE-optimized(best_so_far)/energy - R-CASCI` | error vs the exact embedded answer — **the number that matters** |
| `.../num_sampled_basis` | distinct determinants found this epoch |
| `.../num_symmetry_preserving_basis` | how many survived the particle-number filter |
| `.../subspace_dim` | dimension actually diagonalised |
| `Global-refined`, `Local-refined` | post-hoc classical refinement of the sampled set |

Two failure signatures to check every time:

- **`num_symmetry_preserving_basis` ≪ `num_sampled_basis`** → wrong pool (§3).
- **`subspace_dim == qsci_max_dim` exactly** → the run terminated at the cap and
  had not converged. Any error quoted from it is an upper bound on a capacity
  limit, not a converged result. This is what happened on ScH at
  `n_g=40, N_s=100`: `subspace_dim = 2000` = the old default cap.

If the log is non-empty but yields zero parsed rows, that is the signature of a
missing `train_pipeline.py` patch hunk — and the parser says so explicitly,
because the training run itself will have looked entirely successful.

## 6. The seed trap

`gqe-for-qsci/configs/trainer/default.yaml` pins `seed: 32`. `GqeSettings.seed`
defaults to `None`, which by the override mechanism's design means *leave the
config's value alone*, **not** *randomise*.

So every GQE run launched without `--gqe-seed` trains on the identical seed and
reproduces the identical determinant set and energy. Three "repeat" runs at
N₂ 1.8 Å once gave 78.9 mHa to six figures each — which looks exactly like a
systematic solver bug, and cost hours and three wrong hypotheses before anyone
checked whether the runs were independent at all.

```bash
# WRONG - three bit-identical runs
for i in 1 2 3; do quenais-run --solver gqe --project-dir runs/rep$i ...; done

# RIGHT
for s in 101 102 103; do
  quenais-run --solver gqe --gqe-seed $s --project-dir runs/seed$s ...
done
```

## 7. Before trusting a GQE number

1. Was `--gqe-seed` passed, and varied across repeats?
2. Is the energy scale right for the molecule you think you ran?
3. Is `subspace_dim` below the cap?
4. Is `num_symmetry_preserving_basis ≈ num_sampled_basis`?
5. Does the same run, repeated, give a *different* answer? It should.

And the question none of those answer: **would a classical sampler have done
better?** That is [`06_determinant_selection.md`](06_determinant_selection.md).
