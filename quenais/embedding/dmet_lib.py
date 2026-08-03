"""
Reusable DMET helpers with no module-level side effects.

Kept separate from the stage module on purpose. In the original scripts
these functions lived inside DMET.py, which parsed argv and called
sys.exit(0) at module scope on a cache hit -- so `import DMET` from the GQE
bridge ran the whole cache-check-and-maybe-recompute path as an import side
effect and, on a cache hit, killed the importing process before any of its
own code ran. Anything importable must stay free of side effects.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "lowdin_matrices",
    "adaptive_bath",
    "symmetrize_h2e",
    "get_reference_density",
    "chemical_potential_correction",
    "embedding_consistency_score",
]


# ═════════════════════════════════════════════════════════════════════════
# Linear algebra
# ═════════════════════════════════════════════════════════════════════════

def lowdin_matrices(S):
    """Return (S^1/2, S^-1/2), dropping numerically null directions."""
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    sqrt_vals = np.sqrt(evals[mask])
    S_sqrt = (evecs[:, mask] * sqrt_vals) @ evecs[:, mask].T
    S_invsqrt = (evecs[:, mask] / sqrt_vals) @ evecs[:, mask].T
    return S_sqrt, S_invsqrt


def symmetrize_h2e(h2e):
    """Impose the full 8-fold permutational symmetry of the ERI tensor."""
    return (
        h2e
        + h2e.transpose(1, 0, 2, 3)
        + h2e.transpose(0, 1, 3, 2)
        + h2e.transpose(1, 0, 3, 2)
        + h2e.transpose(2, 3, 0, 1)
        + h2e.transpose(3, 2, 0, 1)
        + h2e.transpose(2, 3, 1, 0)
        + h2e.transpose(3, 2, 1, 0)
    ) / 8.0


# ═════════════════════════════════════════════════════════════════════════
# Bath selection
# ═════════════════════════════════════════════════════════════════════════

def adaptive_bath(sv, n_imp, max_embed, bath_tol):
    """
    Choose how many bath orbitals to keep from the Schmidt singular values.

    Returns (n_bath, gap, sv2_coverage).

    ZERO IS A VALID ANSWER, and getting that wrong is the most expensive
    bug this code has had. The earlier version fell back to `sv[:max_bath]`
    -- taking the largest singular values regardless of whether any cleared
    the tolerance -- whenever none did. That is not a safe fallback; it
    manufactures a bath out of numerically meaningless near-zero values.

    On N2's (4e,4o) active space every Schmidt singular value comes back
    exactly zero (measured max 5.4e-15): the active orbitals are already
    close to eigenvectors of the reference density, so there is no
    impurity-environment entanglement left to extract. The fallback still
    produced "4 bath orbitals", which gave a badly non-orthonormal
    embedding basis and roughly 20 Ha of error.

    Correct behaviour: if nothing clears bath_tol there is no bath. Return
    0 and let the embedding be the active space alone.
    """
    max_bath = min(n_imp, max(0, max_embed - n_imp))
    if max_bath == 0 or len(sv) == 0:
        return 0, 0.0, 0.0

    sv_arr = np.asarray(sv, dtype=float)
    sv_above = sv_arr[sv_arr > bath_tol]
    if len(sv_above) == 0:
        return 0, 0.0, 0.0

    sv_filtered = sv_above[:max_bath]
    n_avail = len(sv_filtered)
    if n_avail == 0:
        return 0, 0.0, 0.0

    best_gap, best_n = -1.0, 1
    for n in range(1, n_avail + 1):
        gap = sv_filtered[n - 1] - (sv_filtered[n] if n < n_avail else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, n

    sv2_total = float(np.sum(sv_filtered ** 2))
    if sv2_total < 1e-30:
        return 0, 0.0, 0.0

    cumulative, n_cov = 0.0, 0
    for i, s in enumerate(sv_filtered):
        cumulative += s * s
        n_cov = i + 1
        if cumulative / sv2_total >= 0.999:
            break

    n_bath = min(max(best_n, n_cov), max_bath)
    sv2_cov = float(np.sum(sv_filtered[:n_bath] ** 2) / sv2_total)
    return n_bath, float(best_gap), sv2_cov


# ═════════════════════════════════════════════════════════════════════════
# Reference density
# ═════════════════════════════════════════════════════════════════════════

def get_reference_density(mf, mol, step1, mo_list, mo_coeff, method):
    """
    Build the reference density the Schmidt decomposition acts on.

    Returns (dm_ao_total, dm_ao_alpha, dm_ao_beta, info).

    mo_list and mo_coeff are explicit parameters rather than module
    globals; the implicit-global version broke silently as soon as the
    function was reused outside its original script.
    """
    if method == "mp2":
        for key in ("dm_ao_total_mp2", "dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
            if key not in step1:
                raise KeyError(
                    f"step 1 pickle is missing {key!r}. Re-run step 1 with "
                    f"force=True."
                )
        return (step1["dm_ao_total_mp2"], step1["dm_ao_alpha_mp2"],
                step1["dm_ao_beta_mp2"],
                {"method": "mp2", "recomputed": False})

    if method != "casci":
        raise ValueError(
            f"Unknown DMET reference method {method!r}. Use 'mp2' or 'casci'."
        )

    from pyscf import ao2mo, fci
    from pyscf.scf import hf as pyscf_hf

    nel_active = step1["nel"]
    n_active = len(mo_list)
    n_alpha = nel_active // 2 + nel_active % 2
    n_beta = nel_active // 2

    # Screen the active-space one-electron Hamiltonian by the core
    # mean-field potential.
    #
    # This used to be built from bare kinetic + nuclear attraction only,
    # ignoring Coulomb/exchange screening from the non-active electrons.
    # The embedding stage DOES add that potential when building h1e_emb, so
    # the reference density and the embedding solver were being computed
    # from two different effective Hamiltonians -- one screened, one not.
    # That mismatch feeds straight into a badly chosen Schmidt bath.
    n_mo_total = mo_coeff.shape[1]
    no_occ = step1["no_occ"]
    active_set = set(mo_list)
    dm_core_mo = np.diag([
        no_occ[i] / 2.0 if i not in active_set else 0.0
        for i in range(n_mo_total)
    ])
    dm_core_ao = mo_coeff @ dm_core_mo @ mo_coeff.T   # alpha == beta

    h1e_bare = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    vj, vk = pyscf_hf.get_jk(mol, dm_core_ao, hermi=1)
    # dm_core_ao is the per-spin density and the core is closed shell, so
    # alpha and beta contributions are identical:
    #   (vj_a + vj_b) - 0.5 * (vk_a + vk_b)  ==  2*vj - vk
    h1e_screened = h1e_bare + 2.0 * vj - vk

    C_active = mo_coeff[:, mo_list]
    h1e_act = C_active.T @ h1e_screened @ C_active
    h2e_act = ao2mo.kernel(mol, C_active, compact=False).reshape(
        n_active, n_active, n_active, n_active
    )

    cisolver = fci.direct_spin1.FCI()
    cisolver.verbose = 0
    e_cas, civec = cisolver.kernel(h1e_act, h2e_act, n_active, (n_alpha, n_beta))
    dm_active_a, dm_active_b = cisolver.make_rdm1s(
        civec, n_active, (n_alpha, n_beta)
    )

    # Start from the FULL MP2 density and overwrite only the active-active
    # block with CASCI.
    #
    # The earlier version built a bare diagonal from no_occ for every
    # non-active orbital and filled in only the active-active block,
    # leaving every active/non-active CROSS TERM at exactly zero -- a
    # block-diagonal density with no impurity-environment coupling at all.
    # The Schmidt decomposition exists precisely to extract that coupling,
    # so it had nothing to find: all singular values came back 0.0 and the
    # resulting "bath" was arbitrary noise.
    #
    # Real correlated densities are not block-diagonal, so the MP2 density
    # supplies the genuine active-core coupling while CASCI supplies the
    # better active-space values.
    for key in ("dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
        if key not in step1:
            raise KeyError(
                f"step 1 pickle is missing {key!r} -- required even for the "
                f"'casci' reference, to preserve active-core coupling in the "
                f"reference density. Re-run step 1 with force=True."
            )

    S = mol.intor("int1e_ovlp")
    dm_full_a = mo_coeff.T @ S @ step1["dm_ao_alpha_mp2"] @ S @ mo_coeff
    dm_full_b = mo_coeff.T @ S @ step1["dm_ao_beta_mp2"] @ S @ mo_coeff

    non_active = [i for i in range(mo_coeff.shape[1]) if i not in active_set]
    cross_a = dm_full_a[np.ix_(mo_list, non_active)] if non_active else np.zeros(1)
    cross_b = dm_full_b[np.ix_(mo_list, non_active)] if non_active else np.zeros(1)
    max_cross = max(float(np.max(np.abs(cross_a))), float(np.max(np.abs(cross_b))))
    print(f"  [diag] MP2 density active<->non-active coupling: "
          f"max|.|={max_cross:.2e}")
    if max_cross < 1e-12 and non_active:
        # Expected for systems whose active orbitals are already close to
        # eigenvectors of the reference density -- N2/STO-3G (4e,4o) is the
        # known case, and its empty bath is correct, not a failure.
        #
        # Worth surfacing anyway: the same signature appears if the
        # reference density is ever rebuilt block-diagonally, which would
        # zero the coupling artificially on a system that does have a bath.
        warnings.warn(
            f"The reference density has essentially no active/non-active "
            f"coupling (max|.|={max_cross:.1e}), so the Schmidt "
            f"decomposition has nothing to extract and the bath will be "
            f"empty. This is expected when the active orbitals are already "
            f"near-eigenvectors of the reference density (N2/STO-3G is the "
            f"known case). Only a concern if you expected a bath for this "
            f"system.",
            RuntimeWarning,
        )

    for a_i, i in enumerate(mo_list):
        for a_j, j in enumerate(mo_list):
            dm_full_a[i, j] = dm_active_a[a_i, a_j]
            dm_full_b[i, j] = dm_active_b[a_i, a_j]

    dm_ao_alpha = mo_coeff @ dm_full_a @ mo_coeff.T
    dm_ao_beta = mo_coeff @ dm_full_b @ mo_coeff.T

    return (dm_ao_alpha + dm_ao_beta, dm_ao_alpha, dm_ao_beta,
            {"method": "casci", "e_cas": float(e_cas),
             "n_active": n_active, "nel_active": nel_active})


# ═════════════════════════════════════════════════════════════════════════
# Chemical potential
# ═════════════════════════════════════════════════════════════════════════

def chemical_potential_correction(h1e_emb, n_emb, n_alpha, n_beta,
                                  mu_range="auto", max_iter=60, tol=1e-10):
    """
    One-shot grand-canonical chemical-potential shift.

    mu_range="auto" derives the bisection bracket from h1e_emb's own
    eigenvalue spectrum. A fixed guess such as (-5, 5) Ha does not bracket
    the true chemical potential once the core mean-field potential has
    shifted those eigenvalues -- on N2, four of eight embedding eigenvalues
    already sat below -5 Ha while the target electron count was 3.

    If bisection still cannot bracket the target, this warns and returns
    the Hamiltonian unshifted rather than silently returning a wrong
    answer. Treat that warning as a real diagnostic.
    """
    evals = np.linalg.eigvalsh(h1e_emb)
    if isinstance(mu_range, str) and mu_range == "auto":
        margin = max(1.0, 0.1 * (evals.max() - evals.min()))
        mu_range = (float(evals.min()) - margin, float(evals.max()) + margin)

    if n_alpha != n_beta:
        warnings.warn(
            f"chemical_potential_correction assumes n_alpha == n_beta (got "
            f"{n_alpha}, {n_beta}); using n_alpha as the target and applying "
            f"the same shift to both spins. This path is untested.",
            RuntimeWarning,
        )
    target = n_alpha

    def n_below_zero(mu):
        return int(np.sum(np.linalg.eigvalsh(h1e_emb - mu * np.eye(n_emb)) < 0.0))

    lo, hi = mu_range
    n_lo, n_hi = n_below_zero(lo), n_below_zero(hi)
    if not (n_lo <= target <= n_hi):
        warnings.warn(
            f"The mu search range {mu_range} does not bracket the target "
            f"electron count {target} (n(mu={lo:.3f})={n_lo}, "
            f"n(mu={hi:.3f})={n_hi}), even with the auto-derived bracket. "
            f"This usually means n_alpha != n_beta for this embedding, or a "
            f"genuine degeneracy at the target count. Skipping the mu "
            f"correction.",
            RuntimeWarning,
        )
        return h1e_emb, 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if n_below_zero(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    mu = 0.5 * (lo + hi)
    return h1e_emb - mu * np.eye(n_emb), mu


# ═════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═════════════════════════════════════════════════════════════════════════

def embedding_consistency_score(step2_result, avg_occs, threshold=0.10):
    """
    Compare the occupations implied by the density that built the bath
    against whatever a solver later reports. Diagnostic only -- no loop, no
    rebuild.

    A large flagged mismatch means the reference density used to build the
    impurity+bath split does not match the embedding's own correlated
    solution. Read it as "do not trust this embedding yet", not as
    something to ignore.
    """
    ref_a = step2_result.get("ref_occ_alpha")
    ref_b = step2_result.get("ref_occ_beta")
    if ref_a is None or ref_b is None:
        raise KeyError(
            "step 2 pickle has no 'ref_occ_alpha'/'ref_occ_beta' -- re-run "
            "the embedding stage."
        )

    occ_a, occ_b = avg_occs
    mismatch_a = float(np.mean(np.abs(np.asarray(occ_a) - ref_a)))
    mismatch_b = float(np.mean(np.abs(np.asarray(occ_b) - ref_b)))
    score = 0.5 * (mismatch_a + mismatch_b)
    return {
        "mismatch_alpha": mismatch_a,
        "mismatch_beta": mismatch_b,
        "mismatch_score": score,
        "flag": score > threshold,
    }
