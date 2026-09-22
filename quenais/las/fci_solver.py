"""
Exact fragment solver (PySCF FCI) -- the classical LASSCF reference.

Uses PySCF's direct_uhf solver, so the spin-dependent effective one-electron
Hamiltonian of an open-shell environment is treated exactly. A spin penalty
shift * (S^2 - S(S+1))^2 steers each fragment to its requested 2S when a
different spin state with the same M_S would otherwise be lower.

PySCF's own fci.addons.fix_spin_ raises NotImplementedError for direct_uhf,
so the penalty is added here (spin_penalized_uhf_solver) with the same
functional form fix_spin_ uses for the other solvers.
"""

from __future__ import annotations

import numpy as np

from quenais.las.driver import FragmentResult

__all__ = ["FCIFragmentSolver", "spin_summed_2rdm", "spin_penalized_uhf_solver"]


def spin_summed_2rdm(dm2s):
    aa, ab, bb = dm2s
    return aa + ab + ab.transpose(2, 3, 0, 1) + bb


def spin_penalized_uhf_solver(shift, ss):
    """
    direct_uhf FCI solver whose Hamiltonian gains shift * (S^2 - ss)^2.
    The penalty vanishes on states with S^2 = ss, so their energies are
    unchanged; states of other spin are pushed up.
    """
    from pyscf.fci import cistring, direct_uhf, spin_op

    class _PenalizedUHF(direct_uhf.FCISolver):
        # Below pspace_size PySCF diagonalises an explicitly built H and never
        # calls contract_2e, which would silently drop the penalty. PySCF's
        # own fix_spin_ forces Davidson for the same reason.
        davidson_only = True

        def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kw):
            ci1 = super().contract_2e(eri, fcivec, norb, nelec, link_index, **kw)
            if not shift:
                return ci1
            na, nb = nelec
            shape = (cistring.num_strings(norb, na), cistring.num_strings(norb, nb))
            c = np.asarray(fcivec).reshape(shape)
            v = spin_op.contract_ss(c, norb, nelec).reshape(shape) - ss * c
            pen = spin_op.contract_ss(v, norb, nelec).reshape(shape) - ss * v
            return ci1 + shift * pen.reshape(np.shape(ci1))

    return _PenalizedUHF()


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
            ss = 0.25 * ham.spin2s * (ham.spin2s + 2)
            # Always on: even a "high-spin" M_S sector contains higher-S
            # states, and an M_S = 0 sector contains triplets that can lie
            # below the singlet.
            if self.spin_penalty:
                solver = spin_penalized_uhf_solver(self.spin_shift, ss)
            else:
                solver = direct_uhf.FCISolver()
            solver.conv_tol = self.conv_tol
            solver.max_cycle = self.max_cycle
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