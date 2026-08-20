# 08 — Glossary

Every symbol, abbreviation and piece of jargon used in this folder.

---

## Methods and acronyms

| term | expansion | one line |
|---|---|---|
| **HF** | Hartree–Fock | Mean-field. Each electron feels the average field of the others. Recovers ≈99 % of the total energy |
| **RHF / UHF** | Restricted / Unrestricted HF | RHF forces paired spatial orbitals; UHF lets α and β differ |
| **MP2** | Møller–Plesset 2nd order | Cheapest correction to HF. Fails when HF is qualitatively wrong |
| **CCSD** | Coupled cluster, singles + doubles | Accurate for single-reference systems. **Non-variational** |
| **CCSD(T)** | CCSD + perturbative triples | "Gold standard" of single-reference quantum chemistry |
| **CASSCF** | Complete Active Space SCF | Full CI inside a chosen orbital subset, with orbital optimisation. Multireference |
| **CASCI** | Complete Active Space CI | Same, but orbitals held fixed |
| **NEVPT2** | n-Electron Valence PT2 | Perturbative dynamic correlation on top of CASSCF |
| **FCI** | Full Configuration Interaction | Exact within a one-particle basis. Cost is combinatorial |
| **DMRG** | Density Matrix Renormalization Group | Used inside ASF to estimate orbital entropies |
| **ASF** | Automated Selection of Active Spaces | The external package that ranks orbitals by entanglement entropy |
| **DMET** | Density Matrix Embedding Theory | The embedding method — see [04](04_step2_embedding.md) |
| **QSCI** | Quantum-Selected CI | Sample determinants on a device, diagonalise classically |
| **SQD / SKQD / SqDRIFT** | Sample-based Quantum Diagonalisation and variants | The Qiskit solver family |
| **GQE** | Generative Quantum Eigensolver | A transformer proposes gate sequences instead of optimising angles |
| **VQE** | Variational Quantum Eigensolver | The older approach GQE replaces; optimises circuit angles continuously |
| **CIPSI** | Configuration Interaction by Perturbative Selection, Iteratively | Classical selected CI, 1973. The baseline |
| **JW** | Jordan–Wigner | Fermion-to-qubit mapping |
| **BK** | Bravyi–Kitaev | Alternative fermion-to-qubit mapping |
| **LUCJ** | Local Unitary Cluster Jastrow | Particle-number-conserving ansatz. The default |
| **SU2** | `EfficientSU2` | Hardware-efficient ansatz. Does **not** conserve particle number |
| **MPS** | Matrix Product State | Tensor-network circuit simulation backend |
| **RDM / 1-RDM** | (One-particle) reduced density matrix | `γ_pq = ⟨a†_p a_q⟩` |
| **ERI** | Electron repulsion integrals | The `(pq\|rs)` two-electron tensor |
| **AO / MO** | Atomic / molecular orbital | Basis conventions; see the basis-consistency warning in [03](03_step1_active_space.md) |
| **SCF** | Self-consistent field | The iterative procedure that solves HF |
| **DIIS** | Direct Inversion in the Iterative Subspace | Standard SCF convergence accelerator |

## Symbols

| symbol | code name | meaning |
|---|---|---|
| `N_imp` | `n_imp` | impurity orbitals — the active space |
| `N_b` | `n_bath` | bath orbitals. **Bounded by `N_imp`** |
| `N_emb` | `n_emb` | `n_imp + n_bath`. Qubits = `2 × n_emb` |
| `N_q` | — | qubit count |
| `σ_i` | `sv`, `sv_all` | Schmidt singular values. `sv_all` is the full spectrum |
| `σ_tol` | `bath_tolerance` | retention threshold, default `1e-8` |
| `λ_i` | — | Schmidt coefficients in `\|Ψ⟩ = Σ λ_i \|a_i⟩⊗\|b_i⟩` |
| `S_i` | — | single-orbital von Neumann entropy |
| `w₁` | `dominant_weight` | \|c₀\|² of the largest CI amplitude — the correlation dial |
| `E_core` | `ecore` | frozen-core energy. **Defined** as `E_UHF(full) − E_HF-in-embedding` |
| `μ` | `mu` | chemical potential; shifts `h1e` and `ecore` in a cancelling pair |
| `v_core` | — | Coulomb + exchange potential from the frozen electrons |
| `h_pq` | `h1e`, `h1e_emb` | one-electron integrals |
| `(pq\|rs)` | `h2e`, `h2e_emb` | two-electron integrals |
| `C_emb` | `C_emb` | embedding basis coefficients, AO basis. `C_embᵀ S C_emb = I` |
| `S` | `S` | AO overlap matrix |
| `Δ_emb` | — | error from truncating to the embedding space alone |
| `Δ_solv` | — | error from the sampler alone, measured against the exact embedded answer |
| `N_chem` | `n_chem` | determinants needed to reach chemical accuracy |
| `n_g` | `ngates` | GQE circuit depth |
| `N_s` | `num_samples` | GQE samples per epoch |
| `t₁`, `t₂` | — | coupled-cluster amplitudes. `max\|t₂\| > 0.1` signals strong correlation |
| `⟨S²⟩` | `s2` | spin expectation; deviation from `S(S+1)` is spin contamination |

## Units and thresholds

| quantity | value |
|---|---|
| **Hartree (E_h)** | atomic unit of energy ≈ 27.211386 eV ≈ 627.509 kcal/mol |
| **mHa** | milli-Hartree, `1e-3` E_h. The working unit for errors here |
| **chemical accuracy** | **1.6 mHa** (≈1 kcal/mol) — the target for chemistry-relevant predictions |
| `EMBEDDED_SCF_VS_UHF_TOL` | `2e-7` Ha — the embedding's correctness check |
| `bath_tolerance` | `1e-8` — Schmidt singular-value retention |
| `core_occ_threshold` | `1.95` — occupation above which an orbital is core |
| `gap_degeneracy_tol` | `1e-3` — keeps degenerate orbitals together |
| w₁ threshold | **≈0.74** — below it, classical selection stops being optimal |

## Reproducibility tiers

| tier | tolerance | applies to |
|---|---|---|
| `DETERMINISTIC` | `1e-9` | HF, MP2, CCSD, CCSD(T), DMET+CASCI, `ecore`, `mu`, σ spectrum |
| `OPTIMIZER_DEPENDENT` | `2e-3` | CASSCF, NEVPT2 |
| `STOCHASTIC` | `5e-2` | DMET+GQE |

## The three validated systems

| | LiH | N₂ | ScH |
|---|---|---|---|
| geometry | 1.5949 Å | 1.0977 Å | 1.78 Å |
| basis | STO-3G | STO-3G | STO-3G |
| active space | (2e,2o) auto | (4e,4o) auto | **(4e,6o) forced**, MOs 9–14 |
| `n_imp` / `n_bath` / `n_emb` | 2 / 2 / 4 | 4 / **0** / 4 | 6 / 5 / 11 |
| qubits | 8 | 8 | 22 |
| determinants | 36 | 36 | 108,900 |
| w₁ | not yet measured | 0.92 @1.1 Å, 0.43 @1.8 Å, 0.19 @2.1 Å | 0.895 |
| role | complete-workflow benchmark | zero-bath edge case | transition-metal target |

## Files

| file | what it is |
|---|---|
| `step0_classical.pkl` | classical reference energies + tiers |
| `step1_asf.pkl` | active space, orbital coefficients, MP2 densities |
| `step2_hamiltonian.pkl` | **the interface file** — `h1e`, `h2e`, `ecore`, counts, σ spectrum |
| `step3_results.pkl` | quantum solver output (Qiskit path) |
| `gqe_train.log` | raw GQE training output (`[epoch N] {...}` lines) |
| `gqe_epoch_log.csv` | parsed version of the above |
| `results_summary.csv` | the human-readable table, with a `reproducibility` column |
| `.quenais_patch_applied` | stamp file in the GQE checkout holding the patch hash |

## Terms that mean something specific here

**Strong / static correlation** — several electron configurations are nearly
degenerate, so no single determinant dominates. HF is not merely inaccurate but
qualitatively wrong as a starting point. Measured by w₁.

**Dynamic correlation** — the short-range electron-electron avoidance that
perturbative methods (MP2, NEVPT2) capture well.

**Single-reference / multireference** — whether one determinant dominates.
Note ScH is *single-reference by w₁* (0.895) despite being a transition metal.

**Impurity** — the orbitals treated exactly. **Bath** — the exact, minimal
representation of everything entangled with them. **Environment** — the rest,
frozen at mean-field level.

**Oracle bound** — the best energy obtainable from any N determinants, computed
by cheating (ranking by exact amplitude). A ceiling, not a method.

**Variational** — an energy that is guaranteed to lie above the true ground
state. QSCI is variational; coupled cluster is not.

**Golden data** — the committed reference pickles in
`tests/regression/golden/`, produced on a validated run and used as the
regression baseline.

**Fingerprint** — `civec_fp` / `order_fp`, hashes of the CI vector and the
determinant ordering. They localise an irreproducibility to a layer, which an
energy comparison cannot.

**Tier** — two unrelated meanings, unfortunately. (1) A system's difficulty
class 1/2/3, which picks ASF's thresholds. (2) A quantity's reproducibility
class. Context distinguishes them.
