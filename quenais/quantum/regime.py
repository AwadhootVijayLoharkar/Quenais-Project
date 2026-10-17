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
  w1_exact  (optional, for validation only) weight of the leading
            determinant of the exact embedding ground state.

Only compute_diagnostics needs PySCF; everything else is NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Thresholds", "spin_adapted_hf_weight", "noon_metrics",
           "recommend", "compute_diagnostics"]


@dataclass(frozen=True)
class Thresholds:
    """Provisional; recalibrate with tools/regime_diagnostics.py."""
    weak_w1: float = 0.90       # w1_est above this: weak correlation
    strong_w1: float = 0.60     # w1_est below this: strong correlation
    strong_noon: float = 0.40   # noon_max above this: strong correlation


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


def recommend(diag, thr=Thresholds(), fragments_separable=None):
    """
    Map diagnostics to (regime, route, reason).

    diag: dict with w1_est, ccsd_ok, noon_max (missing keys are tolerated).
    fragments_separable: True/False if the user knows whether the system
    splits into weakly coupled fragments (LAS territory), None if unknown.
    """
    w1 = diag.get("w1_est")
    ok = diag.get("ccsd_ok", True)
    nmax = diag.get("noon_max")

    strong = ((not ok) or (w1 is not None and w1 < thr.strong_w1)
              or (nmax is not None and nmax > thr.strong_noon))
    weak = (ok and w1 is not None and w1 >= thr.weak_w1
            and (nmax is None or nmax <= thr.strong_noon / 2))

    if strong:
        why = []
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
        return "strong", route, "; ".join(why)
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

    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.max_cycle = 200
    try:
        mycc.kernel()
        out["ccsd_ok"] = bool(mycc.converged)
        out["w1_est"] = spin_adapted_hf_weight(mycc.t1, mycc.t2)
        out["t1_diag"] = float(np.linalg.norm(mycc.t1)
                               / np.sqrt(max(1, mol.nelec[0])))
    except Exception:              # noqa: BLE001 - divergence is a signal
        out["ccsd_ok"] = False
        out["w1_est"] = None
        out["t1_diag"] = None

    if exact:
        from math import comb
        ndet = comb(mol.norb, mol.nelec[0]) * comb(mol.norb, mol.nelec[1])
        out["ndet"] = ndet
        if ndet <= max_det:
            from quenais.quantum.det_analysis import casci_vector
            _, flat, _ = casci_vector(mol)
            out["w1_exact"] = float(np.max(np.asarray(flat) ** 2))
    return out
