#!/usr/bin/env python3
"""
Find out why block2 kills the interpreter on this machine.

Run it from the repo root:

    python diagnose_block2.py            # block2 imported AFTER pyscf
    python diagnose_block2.py --first    # block2 imported BEFORE pyscf

Each stage prints "ok" before moving on, so whichever stage does NOT
print is the one that died. Run both orders: if --first survives and the
default does not, the cause is an OpenMP/MKL runtime clash between
block2 and PySCF, which is a known and fixable problem.

faulthandler is on, so a segfault prints a C-level traceback instead of
vanishing silently.
"""

import faulthandler
import os
import sys
import tempfile

faulthandler.enable()

BLOCK2_FIRST = "--first" in sys.argv

print(f"python      : {sys.version.split()[0]}")
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
print(f"MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS', '(unset)')}")
print(f"import order: {'block2 first' if BLOCK2_FIRST else 'pyscf first'}\n")


def stage(n, what):
    print(f"[{n}] {what} ...", end=" ", flush=True)


if BLOCK2_FIRST:
    stage(1, "import pyblock2.driver.core")
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    print("ok")

stage(2, "import pyscf and run RHF on H4")
from pyscf import gto, scf

mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.6; H 0 0 3.6",
            basis="sto-3g", verbose=0)
mf = scf.RHF(mol).run()
print(f"ok (E_RHF = {mf.e_tot:.8f})")

if not BLOCK2_FIRST:
    stage(3, "import pyblock2.driver.core")
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    print("ok")

stage(4, "build the active-space Hamiltonian with PySCF")
from pyscf import ao2mo, mcscf

mc = mcscf.CASCI(mf, 4, 4)
mc.verbose = 0
mc.mo_coeff = mf.mo_coeff
h1e, ecore = mc.get_h1eff()
g2e = ao2mo.restore(1, mc.get_h2eff(), 4)
print(f"ok (ecore = {ecore:.8f}, h1e {h1e.shape}, g2e {g2e.shape})")

scratch = tempfile.mkdtemp(prefix="block2_diag_")
stage(5, f"construct DMRGDriver (scratch {scratch})")
driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SU2, n_threads=None)
print("ok")

stage(6, "initialize_system(n_sites=4, n_elec=4, spin=0)")
driver.initialize_system(n_sites=4, n_elec=4, spin=0)
print("ok")

stage(7, "get_qc_mpo (reorder=fiedler)")
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, reorder="fiedler",
                        iprint=0)
print("ok")

stage(8, "get_random_mps(bond_dim=50)")
ket = driver.get_random_mps(tag="KET", bond_dim=50, nroots=1)
print("ok")

stage(9, "dmrg sweep")
energy = driver.dmrg(mpo, ket, n_sweeps=4, bond_dims=[50] * 4,
                     noises=[1e-4] * 4, thrds=[1e-8] * 4, tol=1e-8, iprint=0)
print(f"ok (E_DMRG = {float(energy):.8f})")

stage(10, "compare against FCI")
from pyscf import fci

e_fci = fci.FCI(mf).kernel()[0]
print(f"ok (E_FCI = {e_fci:.8f}, difference = {abs(energy - e_fci):.2e})")

print("\nAll stages completed. block2 works here; the crash is somewhere "
      "else in the path.")
