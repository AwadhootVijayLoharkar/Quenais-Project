"""
Regime diagnostics and solver recommendation for an embedding Hamiltonian.

Cheap, a-priori indicators of how strongly correlated an impurity problem
is, and a rule that maps them to a recommended solver route. The rule is
calibrated on our own data (N2 STO-3G / cc-pVDZ and the H10/H14 chains) and
is a first calibration, not a universal law; tools/regime_diagnostics.py
prints the diagnostics next to the measured GENIE/CIPSI error ratio so the
thresholds can be checked against every finished run.

Indicators (all from the embedded mean field, no exact solution needed):

  w1_est    estimated weight of the HF determinant from CCSD amplitudes,
            1 / (1 + |C1|^2 + |C2|^2) with C1 = t1 and C2 = t2 + t1 t1
            (spin-adapted closed-shell norms). Small = multireference.
  ccsd_ok   CCSD converged. Non-convergence is itself a strong-correlation
            signal.
  noon_frac sum over MP2 natural orbitals of min(n, 2 - n): the number of
            "fractionally occupied" electrons.
  noon_max  the largest min(n, 2 - n) of a single natural orbital (1 = one
            orbital exactly half filled).
  n_frac    number of natural orbitals with min(n, 2 - n) > 0.1 (CCSD
            natural orbitals when CCSD converges, else MP2).
  d_est     C(n_frac, n_frac/2)^2: rough number of determinants the
            strongly correlated orbitals span. COMPACTNESS, not strength:
            stretched N2 is strongly correlated but compact (6 orbitals,
            ~400 determinants); stretched H14 spreads over its whole
            embedding (10 orbitals, ~63 500). GENIE's failures follow
            compactness, not w1 (16-17 Oct data).
  w1_exact, ipr_exact, n99_exact  (optional, validation only) leading
            determinant weight, inverse participation ratio 1/sum c^4
            (effective number of determinants) and the number of
            determinants holding 99 % of the weight, from the exact state.

Only compute_diagnostics needs PySCF; everything else is NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Thresholds", "spin_adapted_hf_weight", "noon_metrics",
           "estimated_dets", "det_amplitudes", "amplitude_count",
           "recommend", "compute_diagnostics"]


@dataclass(frozen=True)
class Thresholds:
    """Provisional; recalibrate with tools/regime_diagnostics.py."""
    weak_w1: float = 0.90       # w1_est above this: weak correlation
    strong_w1: float = 0.60     # w1_est below this: strong correlation
    strong_noon: float = 0.40   # noon_max above this: strong correlation
    frac_occ: float = 0.10      # min(n, 2-n) above this: fractional orbital
    safety: float = 3.0         # n_est is within ~3x of n99: GENIE if 3 n_est <= budget


def spin_adapted_hf_weight(t1, t2):
    """
    HF weight 1 / (1 + |C1|^2 + |C2|^2) for closed-shell (R)CCSD amplitudes
    t1[i,a], t2[i,j,a,b] in intermediate normalisation, with C2 = t2 + t1 t1.

    Spin-adapted norms: singles 2 * sum t1^2 (alpha and beta);
    doubles sum_ijab C_ijab (2 C_ijab - C_ijba).
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    c1sq = 2.0 * float(np.sum(t1 * t1))
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    c2sq = float(np.sum(tau * (2.0 * tau - tau.transpose(0, 1, 3, 2))))
    return 1.0 / (1.0 + c1sq + max(c2sq, 0.0))


def noon_metrics(occupations):
    """(noon_frac, noon_max) from spin-summed occupations in [0, 2]."""
    n = np.clip(np.asarray(occupations, dtype=float), 0.0, 2.0)
    dev = np.minimum(n, 2.0 - n)
    return float(dev.sum()), float(dev.max(initial=0.0))


def estimated_dets(n_frac):
    """C(n, n/2)^2 for n fractionally occupied orbitals (0 -> 1)."""
    from math import comb
    n = int(n_frac)
    return comb(n, n // 2) ** 2 if n > 0 else 1


def det_amplitudes(t1, t2):
    """
    Determinant-level coefficients implied by closed-shell (R)CC/MP2
    amplitudes, as two flat arrays of |c|: singles (alpha and beta) and
    doubles (alpha-beta, alpha-alpha, beta-beta), with C2 = t2 + t1 t1.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    s = np.abs(t1).ravel()
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    ab = np.abs(tau).ravel()
    nocc, nvir = t1.shape
    anti = tau - tau.transpose(0, 1, 3, 2)
    io, jo = np.triu_indices(nocc, k=1)
    av, bv = np.triu_indices(nvir, k=1)
    ss = np.abs(anti[io[:, None], jo[:, None], av[None, :], bv[None, :]]).ravel()
    return np.concatenate([s, s]), np.concatenate([ab, ss, ss])


def _pairs_above(v, thr):
    """Number of unordered pairs k < l with v_k * v_l > thr (v >= 0)."""
    v = np.sort(np.asarray(v, dtype=float))
    v = v[v > 0]
    if v.size < 2:
        return 0
    total = 0
    for k, x in enumerate(v):
        # partners l > k with v_l > thr / x
        j = np.searchsorted(v, thr / x, side="right")
        total += max(0, v.size - max(j, k + 1))
    return int(total)


def amplitude_count(t1, t2, thr=0.01, ndet=None):
    """
    Estimated number of determinants with |c| > thr: the reference, singles,
    doubles, and (disconnected) quadruples approximated by products of two
    doubles. A cheap stand-in for n99 (the determinants holding 99 % of the
    exact state). Overlap between doubles is ignored, so the quadruple term
    is an upper estimate. Capped at ndet.
    """
    s, d = det_amplitudes(t1, t2)
    n = 1 + int(np.sum(s > thr)) + int(np.sum(d > thr)) + _pairs_above(d, thr)
    return min(n, int(ndet)) if ndet else n


def recommend(diag, thr=Thresholds(), fragments_separable=None, budget=None):
    """
    Map diagnostics to (regime, route, reason).

    diag: dict with w1_est, ccsd_ok, noon_max (missing keys are tolerated).
    fragments_separable: True/False if the user knows whether the system
    splits into weakly coupled fragments (LAS territory), None if unknown.
    budget: determinant budget of the quantum solver (e.g. the QSCI cap).
    With diag["n_est"] (or the cruder "d_est") compactness decides the route
    in every regime: GENIE if safety * n_est <= budget, GENIE with a deeper
    circuit / larger budget if n_est <= budget, otherwise not GENIE. The
    regime label (weak / moderate / strong, from w1 and NOONs) is reported
    alongside but no longer decides on its own.
    """
    w1 = diag.get("w1_est")
    ok = diag.get("ccsd_ok", True)
    nmax = diag.get("noon_max")

    strong = ((not ok) or (w1 is not None and w1 < thr.strong_w1)
              or (nmax is not None and nmax > thr.strong_noon))
    weak = (ok and w1 is not None and w1 >= thr.weak_w1
            and (nmax is None or nmax <= thr.strong_noon / 2))

    d_est = diag.get("n_est", diag.get("d_est"))
    label = "strong" if strong else "weak" if weak else "moderate"
    if budget is not None and d_est is not None:
        # Compactness decides the route in every regime (16-17 Oct data:
        # GENIE tracks n99 / budget, not correlation strength). n_est is
        # within ~3x of n99, hence the margin.
        if d_est * thr.safety <= budget:
            return (label, f"GENIE (compact: ~{d_est} determinants, budget {budget})",
                    f"{thr.safety:g} x n_est {d_est} <= {budget}")
        if d_est <= budget:
            return (label, f"GENIE (~{d_est} determinants close to budget {budget}); "
                    "use a deeper circuit or a larger budget",
                    f"n_est {d_est} <= {budget} < {thr.safety:g} x n_est")
    spread = budget is not None and d_est is not None   # here: n_est > budget
    if strong or spread:
        why = []
        if spread:
            why.append(f"spread: n_est {d_est} > budget {budget}")
        if not ok:
            why.append("CCSD did not converge")
        if w1 is not None and w1 < thr.strong_w1:
            why.append(f"w1_est {w1:.2f} < {thr.strong_w1}")
        if nmax is not None and nmax > thr.strong_noon:
            why.append(f"noon_max {nmax:.2f} > {thr.strong_noon}")
        if fragments_separable:
            route = "LASSQD (weakly coupled fragments)"
        elif fragments_separable is False:
            route = "GENIE-MR (not implemented): GENIE likely unreliable"
        else:
            route = ("LASSQD if the fragments are weakly coupled, else "
                     "GENIE-MR (not implemented): GENIE likely unreliable")
        return label, route, "; ".join(why)
    if weak:
        return ("weak", "GENIE (or classical CCSD(T) may already suffice)",
                f"w1_est {w1:.2f} >= {thr.weak_w1}")
    return ("moderate", "GENIE",
            "between the weak and strong thresholds"
            + (f" (w1_est {w1:.2f})" if w1 is not None else ""))


def compute_diagnostics(mol, exact=False, max_det=2_000_000):
    """
    Diagnostics for a DMETEmbeddingMolecule (quenais.quantum.gqe_adapter).
    Uses the embedded RHF (mol.hf); energies and occupations are basis
    invariant, so the GQE orbital basis does not matter here.
    """
    from pyscf import cc, mp

    mf = mol.hf
    out = {"norb": int(mol.norb), "nelec": list(mol.nelec)}

    pt = mp.MP2(mf)
    pt.verbose = 0
    pt.kernel()
    occ = np.linalg.eigvalsh(np.asarray(pt.make_rdm1()))
    out["noon_frac"], out["noon_max"] = noon_metrics(occ)
    from math import comb
    ndet = comb(mol.norb, mol.nelec[0]) * comb(mol.norb, mol.nelec[1])
    out["ndet"] = ndet
    t1_zero = np.zeros((pt.t2.shape[0], pt.t2.shape[2]))
    out["n_est_mp2"] = amplitude_count(t1_zero, pt.t2, 0.01, ndet)

    occ_used = occ
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.max_cycle = 200
    try:
        mycc.kernel()
        out["ccsd_ok"] = bool(mycc.converged)
        out["w1_est"] = spin_adapted_hf_weight(mycc.t1, mycc.t2)
        out["t1_diag"] = float(np.linalg.norm(mycc.t1)
                               / np.sqrt(max(1, mol.nelec[0])))
        out["n_est_cc"] = amplitude_count(mycc.t1, mycc.t2, 0.01, ndet)
        if mycc.converged:
            occ_used = np.linalg.eigvalsh(np.asarray(mycc.make_rdm1()))
    except Exception:              # noqa: BLE001 - divergence is a signal
        out["ccsd_ok"] = False
        out["w1_est"] = None
        out["t1_diag"] = None
        out["n_est_cc"] = None
    # the router's estimate: CCSD if it converged, else MP2
    out["n_est"] = (out["n_est_cc"] if out.get("ccsd_ok") and out.get("n_est_cc")
                    else out["n_est_mp2"])

    dev = np.minimum(np.clip(occ_used, 0, 2), 2 - np.clip(occ_used, 0, 2))
    out["n_frac"] = int(np.sum(dev > Thresholds().frac_occ))
    out["d_est"] = estimated_dets(out["n_frac"])

    if exact:
        if ndet <= max_det:
            from quenais.quantum.det_analysis import (casci_vector, n_for_weight,
                                                      weight_curve)
            _, flat, _ = casci_vector(mol)
            w = np.asarray(flat) ** 2
            out["w1_exact"] = float(np.max(w))
            out["ipr_exact"] = float(1.0 / np.sum(w * w))
            _, cum = weight_curve(flat)
            out["n99_exact"] = int(n_for_weight(cum, (0.99,))[0.99])
    return out
