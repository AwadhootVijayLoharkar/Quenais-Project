"""
PySCF-backed integral provider for quenais.las.

The LAS core only needs four things from the integrals:

    hcore        (nao, nao)
    energy_nuc   float
    get_jk(dms)  dms (n, nao, nao) -> (vj, vk), each (n, nao, nao)
    ao2mo(c1, c2, c3, c4) -> (n1, n2, n3, n4) chemist's (ij|kl)

This module supplies them from a PySCF Mole, with or without density
fitting. tests/_las_toy.py supplies them from random arrays.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PyscfIntegrals"]


class PyscfIntegrals:
    def __init__(self, mol, density_fit=False, auxbasis=None):
        from pyscf import scf

        self.mol = mol
        mf = scf.RHF(mol)
        if density_fit:
            mf = mf.density_fit(auxbasis=auxbasis)
        self._mf = mf
        self.density_fit = bool(density_fit)
        self.hcore = mf.get_hcore()
        self.ovlp = mf.get_ovlp()
        self.energy_nuc = float(mol.energy_nuc())
        self.nao = mol.nao_nr()

    def get_jk(self, dms):
        dms = np.asarray(dms)
        vj, vk = self._mf.get_jk(self.mol, dms, hermi=1)
        return np.asarray(vj).reshape(dms.shape), np.asarray(vk).reshape(dms.shape)

    def ao2mo(self, c1, c2, c3, c4):
        shape = (c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1])
        if self.density_fit:
            eri = self._mf.with_df.ao2mo((c1, c2, c3, c4), compact=False)
        else:
            from pyscf import ao2mo

            eri = ao2mo.general(self.mol, (c1, c2, c3, c4), compact=False)
        return np.asarray(eri).reshape(shape)
