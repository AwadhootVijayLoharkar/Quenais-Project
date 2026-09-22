"""
Orbital optimisation at fixed fragment RDMs. Pure NumPy/SciPy.

Moving-frame L-BFGS: every step is a rotation exp(k) of the CURRENT
orbitals, the gradient is always evaluated at k = 0 of the current frame,
and (s, y) pairs from earlier frames are reused as if the frames coincided.
That transport error is second order in the step and is the standard
trade-off in orbital optimisers; the Armijo line search keeps every
accepted step downhill, so the worst case is slower convergence, never a
wrong answer.

The preconditioner is diagonal_hessian() from energy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

__all__ = ["OrbitalStepResult", "rotate", "optimize_orbitals"]


@dataclass
class OrbitalStepResult:
    C: np.ndarray
    energy: float
    grad_norm: float
    n_iter: int
    converged: bool
    energies: list = field(default_factory=list)


def rotate(C, mask, x):
    """C exp(k) with k antisymmetric, k[mask] = x."""
    k = np.zeros(mask.shape)
    k[mask] = x
    k = k - k.T
    return C @ expm(k)


def optimize_orbitals(energy_grad, C, mask, precond, *, max_iter=30, gtol=1e-5,
                      max_step=0.5, memory=8, log=None):
    """
    Minimise energy_grad(C) -> (E, g) over orbital rotations in ``mask``.

    energy_grad : callable, returns energy and the antisymmetric gradient
                  matrix of energy.las_energy_and_gradient
    mask        : boolean (nmo, nmo), lower triangle, rotations to optimise
    precond     : positive array shaped like mask (diagonal Hessian estimate)
    """
    h0 = np.asarray(precond)[mask]
    e, gmat = energy_grad(C)
    g = gmat[mask]
    s_hist, y_hist = [], []
    energies = [e]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        gnorm = float(np.linalg.norm(g))
        if gnorm < gtol:
            converged = True
            it -= 1
            break

        d = _two_loop(g, s_hist, y_hist, h0)
        if g @ d >= 0:                       # not a descent direction
            s_hist.clear(), y_hist.clear()
            d = -g / h0
        dn = np.linalg.norm(d)
        if dn > max_step:
            d *= max_step / dn

        alpha, accepted = 1.0, False
        slope = g @ d
        for _ in range(12):
            C_try = rotate(C, mask, alpha * d)
            e_try, gmat_try = energy_grad(C_try)
            if e_try <= e + 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            if log:
                log(f"      orbital line search stalled at |g| = {gnorm:.2e}")
            break

        g_new = gmat_try[mask]
        s, y = alpha * d, g_new - g
        if s @ y > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > memory:
                s_hist.pop(0), y_hist.pop(0)
        C, e, g = C_try, e_try, g_new
        energies.append(e)
    else:
        converged = float(np.linalg.norm(g)) < gtol

    return OrbitalStepResult(C=C, energy=float(e),
                             grad_norm=float(np.linalg.norm(g)),
                             n_iter=it, converged=converged, energies=energies)


def _two_loop(g, s_hist, y_hist, h0):
    q = g.copy()
    alphas = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        rho = 1.0 / (y @ s)
        a = rho * (s @ q)
        alphas.append((rho, a, s, y))
        q -= a * y
    if s_hist:
        s, y = s_hist[-1], y_hist[-1]
        gamma = (s @ y) / (y @ y)
        # Blend the diagonal estimate with the curvature just observed.
        r = q * np.minimum(1.0 / h0, 10.0 * abs(gamma))
    else:
        r = q / h0
    for rho, a, s, y in reversed(alphas):
        b = rho * (y @ r)
        r += (a - b) * s
    return -r
