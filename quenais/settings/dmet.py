"""
DMET embedding settings.

Several defaults here encode hard-won behaviour. Where a value looks
arbitrary, the docstring says what breaks if it is changed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DmetSettings", "REFERENCE_METHODS"]

#: Reference-density construction methods.
REFERENCE_METHODS = ("mp2", "casci")


@dataclass
class DmetSettings:
    """Schmidt decomposition, bath selection and the embedding Hamiltonian."""

    #: Singular values at or below this are numerical noise, not physics.
    #:
    #: Load-bearing: on N2's (4e,4o) space every Schmidt singular value comes
    #: back numerically zero (measured 5.4e-15), because the active orbitals
    #: are already close to eigenvectors of the reference density. The
    #: correct answer there is zero bath orbitals. Manufacturing a bath from
    #: those values instead produces a badly non-orthonormal embedding basis
    #: and ~20 Ha errors.
    bath_tolerance: float = 1e-8

    #: Warn if fewer bath orbitals than this were found. Zero is legitimate.
    min_bath_orbs: int = 0

    #: Ceiling on impurity + bath orbitals. Each embedding orbital costs two
    #: qubits, and the embedded CASCI cost grows combinatorially.
    #:
    #: 18 is the validated value -- every reference number in
    #: tests/regression/reference_values.py was produced with it. (The 0.1
    #: package shipped 24, which no validated run used.)
    max_embed_orbs: int = 18

    #: Reference density for the Schmidt decomposition.
    #:
    #: "mp2"   -- reuse step 1's MP2 1-RDM. Fast, and unreliable exactly
    #:            where static correlation is strong.
    #: "casci" -- CASCI within the ASF active space, screened by the core
    #:            mean-field potential. Recommended, and the default.
    reference: str = "casci"

    #: One-shot grand-canonical chemical-potential correction.
    #:
    #: Provably inert for total energies from fixed-particle-number solvers
    #: (the h1e - mu*I and ecore + mu*N shifts cancel exactly), and confirmed
    #: bit-for-bit. Kept on because it matters for solvers that do not fix N.
    mu_correction: bool = True

    #: "auto" derives the bisection bracket from h1e_emb's own eigenvalue
    #: spectrum. A fixed guess such as (-5, 5) Ha fails to bracket once the
    #: core mean-field potential has shifted those eigenvalues -- on N2, four
    #: of eight embedding eigenvalues already sat below -5 Ha.
    mu_search_range: str | tuple = "auto"
    mu_max_iter: int = 60
    mu_tol: float = 1e-10

    #: Flag the embedding when the solver's occupations diverge from the
    #: reference density's by more than this. Diagnostic only -- no loop,
    #: no rebuild.
    consistency_mismatch_threshold: float = 0.10

    def validate(self) -> "DmetSettings":
        if self.reference not in REFERENCE_METHODS:
            raise ValueError(
                f"dmet.reference must be one of {REFERENCE_METHODS}, "
                f"got {self.reference!r}"
            )
        if self.bath_tolerance <= 0:
            raise ValueError("bath_tolerance must be > 0")
        if self.min_bath_orbs < 0:
            raise ValueError("min_bath_orbs must be >= 0")
        if self.max_embed_orbs < 1:
            raise ValueError("max_embed_orbs must be >= 1")
        if self.mu_search_range != "auto":
            try:
                lo, hi = self.mu_search_range
            except (TypeError, ValueError):
                raise ValueError(
                    "mu_search_range must be 'auto' or a (lo, hi) pair, "
                    f"got {self.mu_search_range!r}"
                ) from None
            if lo >= hi:
                raise ValueError(f"mu_search_range lo >= hi: {self.mu_search_range}")
        if self.mu_max_iter < 1:
            raise ValueError("mu_max_iter must be >= 1")
        if self.mu_tol <= 0:
            raise ValueError("mu_tol must be > 0")
        if self.consistency_mismatch_threshold < 0:
            raise ValueError("consistency_mismatch_threshold must be >= 0")
        return self
