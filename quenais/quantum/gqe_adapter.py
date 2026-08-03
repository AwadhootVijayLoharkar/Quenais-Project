"""
DMET embedding presented as a molecule object for the external GQE trainer.

Drop-in replacement for gqe_qsci.molecule.PySCFMolecule, built from a DMET
embedding Hamiltonian (h1e, h2e, ecore, n_alpha, n_beta) rather than from a
geometry and a list of active orbitals.

WHY AN ADAPTER RATHER THAN A CONFIG CHANGE
------------------------------------------
PySCFMolecule builds its mean-field object from real atoms via gto.M(atom=...)
and picks an active space as a SUBSET of canonical MOs. A DMET embedding
space -- impurity plus bath, from a Schmidt decomposition -- is a genuinely
different orbital basis: a rotated combination, not a subset. It cannot be
expressed that way.

PySCF does support building a mean-field object directly from arbitrary
one- and two-body integrals (the "custom Hamiltonian" pattern used for
FCIDUMP and model Hamiltonians). This adapter uses exactly that: it hands
h1e_emb/h2e_emb to PySCF as if they were AO integrals, runs a genuine SCF
to find the HF-optimal orbitals WITHIN the embedding space, and folds ecore
in as the nuclear-repulsion constant. mf.e_tot then comes out as the
DMET-consistent total energy directly, because
E_total = ecore + <psi|H_emb|psi> holds for any psi in the embedding space
-- not only for the reference DMET used internally for its own bookkeeping.

Every heavy import is inside a function. This module must stay importable
on a Qiskit-only install.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Hamiltonian",
    "DMETEmbeddingMolecule",
    "load_from_dmet_pickle",
    "run_consistency_check",
]


@dataclass(frozen=True)
class Hamiltonian:
    h1: np.ndarray
    h2: np.ndarray
    e_core: float


class DMETEmbeddingMolecule:
    """
    Same public surface as gqe_qsci.molecule.PySCFMolecule:

      .cas_hamiltonian -> Hamiltonian(h1, h2, e_core)
      .norb, .nelec, .spin
      .hf              -- a real, converged PySCF mean-field object
      .compute_casci(), .compute_ccsd()
      .ccsd_amplitude  -- property, {"t1": ..., "t2": ...}
      .geometry, .basis, .active_indices

    geometry/basis/active_indices exist only so code that checks for their
    presence does not break. They are None or trivial. Do NOT feed them to
    tequila: it would reconstruct a Hamiltonian from a geometry that has
    nothing to do with h1e_emb/h2e_emb. Use DMETExcitationPool or
    DMETPauliEvolutionPool, never the stock ExcitationPool /
    PauliEvolutionPool, with this molecule type.
    """

    def __init__(self, h1e_emb, h2e_emb, ecore, n_alpha, n_beta,
                 num_threads=1, cache_key_extra="", cache_dir=None):
        from pyscf import lib, mcscf

        lib.num_threads(num_threads)

        n_emb = h1e_emb.shape[0]
        if h1e_emb.shape != (n_emb, n_emb):
            raise ValueError(f"h1e must be square, got {h1e_emb.shape}")
        if h2e_emb.shape != (n_emb,) * 4:
            raise ValueError(
                f"h2e must have shape {(n_emb,) * 4}, got {h2e_emb.shape}"
            )

        self.norb = n_emb
        self.nelec = (int(n_alpha), int(n_beta))
        self.spin = int(n_alpha) - int(n_beta)
        self._ecore = float(ecore)
        self._h1e_emb = np.asarray(h1e_emb)
        self._h2e_emb = np.asarray(h2e_emb)

        # Interface-compatibility placeholders only -- see the class
        # docstring. Never hand these to tequila.
        self.geometry = None
        self.basis = None
        self.active_indices = list(range(n_emb))

        self.mol = self._build_fake_mol()
        self.hf = self._run_embedding_scf()
        self.mc = mcscf.CASCI(self.hf, self.norb, self.nelec)
        self.mc.verbose = 0          # see the note in _run_embedding_scf
        self.mc.fcisolver.verbose = 0
        self.cas_hamiltonian = Hamiltonian(
            h1=self._h1e_emb, h2=self._h2e_emb, e_core=self._ecore
        )

        self._ccsd_amplitude = None
        self._cache_key = self._build_cache_key(cache_key_extra)
        # Default beside the caller rather than in the process cwd. The
        # trainer runs with the external repo as its working directory, so
        # a relative .cache/ there is a stale-cache vector across molecules.
        self._cache_dir = Path(cache_dir or Path(".cache") / "pyscf_dmet")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── PySCF custom-Hamiltonian setup ───────────────────────────────────
    def _build_fake_mol(self):
        from pyscf import gto

        mol = gto.M(verbose=0)
        mol.nelectron = sum(self.nelec)
        mol.spin = self.spin
        mol.incore_anyway = True
        mol.build(dump_input=False, parse_arg=False)
        return mol

    def _run_embedding_scf(self):
        from pyscf import ao2mo, scf

        n_emb = self.norb
        h1e = self._h1e_emb
        # 8-fold compact storage. h2e was already 8-fold symmetrised in the
        # embedding stage, so this is exact rather than an approximation.
        eri8 = ao2mo.restore(8, self._h2e_emb, n_emb)

        mf = scf.RHF(self.mol) if self.spin == 0 else scf.UHF(self.mol)
        mf.get_hcore = lambda *a, **k: h1e
        mf.get_ovlp = lambda *a, **k: np.eye(n_emb)
        mf._eri = eri8
        mf.energy_nuc = lambda *a, **k: self._ecore
        mf.max_cycle = 200
        mf.conv_tol = 1e-10
        # PySCF logs through its own stream, which contextlib.redirect_stdout
        # does not capture, so quiet the object rather than the stream.
        mf.verbose = 0
        mf.kernel()

        if not mf.converged:
            raise RuntimeError(
                "The embedding-space SCF did not converge. Check "
                "h1e_emb/h2e_emb for numerical problems before trusting "
                "anything downstream."
            )
        return mf

    def _build_cache_key(self, extra):
        payload = {
            "h1_hash": hashlib.sha256(self._h1e_emb.tobytes()).hexdigest(),
            "h2_hash": hashlib.sha256(self._h2e_emb.tobytes()).hexdigest(),
            "ecore": self._ecore,
            "nelec": self.nelec,
            "extra": extra,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    # ── Solvers ──────────────────────────────────────────────────────────
    @property
    def ccsd_amplitude(self):
        if self._ccsd_amplitude is None:
            self.compute_ccsd()
        return self._ccsd_amplitude

    def compute_casci(self):
        """
        Exact diagonalisation in the embedding space.

        The on-disk cache stores the CI vector as well as the energy. It
        used to store only the energy, so a cache MISS set
        _last_casci_civec while a cache HIT returned early without it --
        and casci_avg_occs()'s hasattr guard could not help, because
        calling compute_casci() again just hit the same cache. Every run
        after the first raised AttributeError.
        """
        cache_path = self._cache_dir / f"{self._cache_key}_casci.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                self._last_casci_civec = data["civec"]
                return float(data["energy"])

        e_fci, civec = self.mc.fcisolver.kernel(
            self.cas_hamiltonian.h1, self.cas_hamiltonian.h2, self.norb,
            self.nelec, ecore=self.cas_hamiltonian.e_core,
        )
        self._last_casci_civec = civec
        np.savez(cache_path, energy=e_fci, civec=np.asarray(civec))
        return float(e_fci)

    def compute_ccsd(self):
        from pyscf import cc

        cache_path = self._cache_dir / f"{self._cache_key}_ccsd.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=True) as data:
                self._ccsd_amplitude = {"t1": data["t1"], "t2": data["t2"]}
                return float(data["energy"])

        mycc = cc.RCCSD(self.hf) if self.spin == 0 else cc.UCCSD(self.hf)
        mycc.verbose = 0
        mycc.kernel()
        e_tot = self.hf.e_tot + mycc.e_corr

        # RCCSD gives single t1/t2 arrays directly. UCCSD returns tuples
        # ((t1a, t1b) / (t2aa, t2ab, t2bb)); only the closed-shell path is
        # validated -- see docs/limitations.md.
        t1, t2 = mycc.t1, mycc.t2
        self._ccsd_amplitude = {"t1": t1, "t2": t2}
        np.savez(
            cache_path,
            energy=e_tot,
            t1=np.asarray(t1, dtype=object) if self.spin != 0 else t1,
            t2=np.asarray(t2, dtype=object) if self.spin != 0 else t2,
        )
        return float(e_tot)

    def casci_avg_occs(self):
        if not hasattr(self, "_last_casci_civec"):
            self.compute_casci()
        dm_a, dm_b = self.mc.fcisolver.make_rdm1s(
            self._last_casci_civec, self.norb, self.nelec
        )
        return (np.clip(np.diag(dm_a), 0.0, 1.0),
                np.clip(np.diag(dm_b), 0.0, 1.0))


def load_from_dmet_pickle(step2_pickle_path, **kwargs):
    """
    Build a DMETEmbeddingMolecule from a step 2 pickle.

    This is the function the external trainer's Hydra config instantiates,
    so its signature is part of the integration contract. Hydra's
    instantiate() calls a plain function as happily as a constructor.
    """
    path = Path(step2_pickle_path)
    with open(path, "rb") as fh:
        step2 = pickle.load(fh)

    missing = {"h1e", "h2e", "ecore", "n_alpha", "n_beta"} - set(step2)
    if missing:
        raise KeyError(
            f"{path} is not a valid step 2 pickle -- missing "
            f"{sorted(missing)}. Re-run the embedding stage."
        )

    kwargs.setdefault("cache_dir", path.parent / ".cache" / "pyscf_dmet")
    mol = DMETEmbeddingMolecule(
        h1e_emb=step2["h1e"],
        h2e_emb=step2["h2e"],
        ecore=step2["ecore"],
        n_alpha=step2["n_alpha"],
        n_beta=step2["n_beta"],
        cache_key_extra=step2.get("mol_info", {}).get("molecule", ""),
        **kwargs,
    )
    mol._step2_result = step2
    return mol


def run_consistency_check(mol, threshold=0.10):
    """
    Compare the embedding's own CASCI occupations against the reference
    density that built the bath. Diagnostic only.
    """
    from quenais.embedding.dmet_lib import embedding_consistency_score

    occ_a, occ_b = mol.casci_avg_occs()
    result = embedding_consistency_score(
        mol._step2_result, (occ_a, occ_b), threshold=threshold
    )
    print(f"[Consistency check] mismatch_score="
          f"{result['mismatch_score']:.4f} (threshold={threshold})  "
          f"flagged={result['flag']}")
    return result
