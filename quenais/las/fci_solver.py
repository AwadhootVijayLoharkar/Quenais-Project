"""
Exact fragment solver (PySCF FCI) -- the classical LASSCF reference.

Uses PySCF's direct_uhf solver, so the spin-dependent effective one-electron
Hamiltonian of an open-shell environment is treated exactly. An optional
spin penalty (fix_spin_) steers each fragment to its requested 2S when a
different spin state with the same M_S would otherwise be lower.
"""

from __future__ import annotations

import numpy as np

from quenais.las.driver import FragmentResult

__all__ = ["FCIFragmentSolver", "spin_summed_2rdm"]


def spin_summed_2rdm(dm2s):
    aa, ab, bb = dm2s
    return aa + ab + ab.transpose(2, 3, 0, 1) + bb


class FCIFragmentSolver:
    stochastic = False
    name = "fci"

    def __init__(self, spin_penalty=True, spin_shift=0.2, conv_tol=1e-12,
                 max_cycle=200):
        self.spin_penalty = spin_penalty
        self.spin_shift = spin_shift
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self._ci = {}

    def solve(self, hams, cycle=0):
        from pyscf import fci
        from pyscf.fci import direct_spin1, direct_uhf

        out = []
        for ham in hams:
            n, nelec = ham.norb, tuple(ham.nelec)
            solver = direct_uhf.FCISolver()
            solver.conv_tol = self.conv_tol
            solver.max_cycle = self.max_cycle
            ss = 0.25 * ham.spin2s * (ham.spin2s + 2)
            # Always on: even a "high-spin" M_S sector contains higher-S
            # states, and an M_S = 0 sector contains triplets that can lie
            # below the singlet.
            if self.spin_penalty:
                solver = fci.addons.fix_spin_(solver, shift=self.spin_shift, ss=ss)
            h1 = (ham.h1s[0], ham.h1s[1])
            eri = (ham.eri, ham.eri, ham.eri)
            ci0 = self._ci.get(ham.index)
            if ci0 is not None and getattr(ci0, "shape", None) != _ci_shape(n, nelec):
                ci0 = None
            e, ci = solver.kernel(h1, eri, n, nelec, ci0=ci0)
            self._ci[ham.index] = ci
            (dma, dmb), dm2s = direct_spin1.make_rdm12s(ci, n, nelec)
            s2, mult = fci.spin_op.spin_square0(ci, n, nelec)
            out.append(FragmentResult(
                dm1s=np.stack([dma, dmb]),
                dm2=spin_summed_2rdm(dm2s),
                energy=float(e),
                info={"solver": "fci", "S^2": float(s2),
                      "target_S^2": float(ss), "dim": int(np.size(ci))},
            ))
        return out


def _ci_shape(n, nelec):
    from pyscf.fci import cistring

    return (cistring.num_strings(n, nelec[0]), cistring.num_strings(n, nelec[1]))
