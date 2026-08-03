"""
DMET-aware operator pools for the external GQE trainer.

Replacements for gqe_qsci.gqe.operator_pool's PauliEvolutionPool and
ExcitationPool.

WHY THE STOCK POOLS CANNOT BE USED
----------------------------------
They build their UCCSD ansatz through tequila:
tq.Molecule(geometry=..., basis_set=..., active_orbitals=...) followed by
make_excitation_gate(). That reconstructs a Hamiltonian from real geometry,
which has nothing to do with the embedding-space Hamiltonian. A DMET
embedding has no geometry at all, so operator_pool.py's loop over
molecule.geometry dies with "TypeError: 'NoneType' object is not iterable".
Even if it did not, it would silently build gates for the wrong active
space.

These pools replace the gate-generator construction step only, using plain
OpenFermion on abstract spin-orbital indices -- no geometry, no basis set,
no tequila-built Hamiltonian. The amplitude screening logic is unchanged:
it only ever operated on CCSD t1/t2 amplitudes and orbital indices.

WHICH POOL TO USE
-----------------
DMETExcitationPool, in almost all cases.

DMETPauliEvolutionPool appends every Pauli TERM of an excitation generator
as its own pool element, so each element is a single Pauli word. Particle
number conservation is a property of the full SUM of terms in a JW-mapped
fermionic excitation, never of one term alone, so that pool cannot conserve
electron number however its flags are set. Measured on ScH: roughly half of
every sample was discarded as symmetry-violating.

DMETExcitationPool accumulates the terms into one operator per excitation,
which does conserve particle number.

LAZY CLASS CONSTRUCTION
-----------------------
The pool classes inherit from gqe_qsci's OperatorPool, so defining them at
module scope would import the external repo -- and cudaq with it -- at
import time. They are built on first attribute access instead, so
`import quenais.quantum.gqe_pools` stays free on a Qiskit-only install
while `from quenais.quantum.gqe_pools import DMETExcitationPool` works
exactly as normal.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "excitation_generator_qubit_op",
    "generate_excitations",
    "DMETUCCSDBasedPool",
    "DMETPauliEvolutionPool",
    "DMETExcitationPool",
]


# ═════════════════════════════════════════════════════════════════════════
# Generator construction -- needs OpenFermion only
# ═════════════════════════════════════════════════════════════════════════

def excitation_generator_qubit_op(indices):
    """
    Jordan-Wigner-mapped qubit operator for the antisymmetrised UCC
    excitation generator over a list of (occ, virt) spin-orbital pairs.

    For a single excitation p -> q:      G = a_q^dag a_p - a_p^dag a_q
    For simultaneous pairs (p_i -> q_i): G = prod_i(a_qi^dag a_pi) - h.c.

    THE -1j IS LOAD-BEARING. (forward - backward) is the raw anti-Hermitian
    UCC generator; tequila stores its generator already in Hermitian form,
    H = -i(T - T^dag), ready for exp(-i*theta*H). Validation against
    tequila found a consistent ratio of exactly +i between the two, so the
    factor below matches the convention exactly.

    That validation passed on singles (both spins) and doubles, with
    identical Pauli-string sets and a ratio of exactly 1.000000 on every
    term. Re-run tests/test_excitation_generator.py after any edit here --
    a sign error would not crash anything, it would silently seed the pool
    with wrong-phase excitations.

    Returns an openfermion.QubitOperator.
    """
    from openfermion import FermionOperator, jordan_wigner

    forward = FermionOperator.identity()
    backward = FermionOperator.identity()
    for p, q in indices:
        forward *= FermionOperator(((q, 1), (p, 0)))    # a_q^dag a_p
        backward *= FermionOperator(((p, 1), (q, 0)))   # a_p^dag a_q

    return jordan_wigner(-1j * (forward - backward))


def _qubit_op_terms_to_cudaq(qubit_op, remove_z_ladder=False):
    """
    Convert each term of an OpenFermion QubitOperator to a
    (cudaq unit-Pauli term, coefficient) pair.

    The identity term is skipped. A genuine excitation generator should not
    have one, but a numerical artifact producing a tiny one would otherwise
    mis-sign the pool.
    """
    from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin

    out = []
    for term, coeff in qubit_op.terms.items():
        if len(term) == 0:
            continue
        pauli = {idx: letter for idx, letter in term}
        if remove_z_ladder:
            pauli = {k: v for k, v in pauli.items() if v.lower() != "z"}
            if not pauli:
                continue
        cudaq_term = convert_pauli_to_cudaq_spin(pauli)
        if cudaq_term is None:
            continue
        out.append((cudaq_term, coeff))
    return out


def generate_excitations(ccsd_amplitude, threshold):
    """
    Screen CCSD amplitudes into a {spin-orbital index tuple: angle} map.

    Unchanged from the upstream logic -- it only ever operated on
    amplitudes and abstract orbital indices, never on geometry, so it is
    correct for an embedding space as-is.
    """
    from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes

    amplitudes_obj = ClosedShellAmplitudes(
        tIjAb=ccsd_amplitude["t2"], tIA=ccsd_amplitude["t1"]
    )
    all_amps = amplitudes_obj.make_parameter_dictionary(
        threshold=0.0, screening=False
    )
    amps = {k: v for k, v in all_amps.items()
            if not np.isclose(v, 0.0, atol=threshold)}
    amps = dict(sorted(amps.items(), key=lambda kv: np.fabs(kv[1]), reverse=True))

    indices = {}
    for key, t in amps.items():
        assert len(key) % 2 == 0
        angle = 2.0 * t
        if len(key) == 2:
            indices[(2 * key[0], 2 * key[1])] = angle
            indices[(2 * key[0] + 1, 2 * key[1] + 1)] = angle
        else:
            assert len(key) == 4
            idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
            indices[idx_abab] = angle
            if key[0] != key[2] and key[1] != key[3]:
                partner_t = all_amps.get((key[2], key[1], key[0], key[3]), 0.0)
                anglex = 2.0 * (t - partner_t)
                indices[(2 * key[0], 2 * key[1],
                         2 * key[2], 2 * key[3])] = anglex
                indices[(2 * key[0] + 1, 2 * key[1] + 1,
                         2 * key[2] + 1, 2 * key[3] + 1)] = anglex
    return indices


# ═════════════════════════════════════════════════════════════════════════
# Pool classes, built on first access
# ═════════════════════════════════════════════════════════════════════════

_CLASS_CACHE = {}


def _build_classes():
    if _CLASS_CACHE:
        return _CLASS_CACHE

    from abc import ABC
    from collections import Counter

    import cudaq
    from gqe_qsci.gqe.operator_pool import OperatorPool
    from gqe_qsci.gqe.utils import get_pauli_evolution_gate_count

    class DMETUCCSDBasedPool(OperatorPool, ABC):
        """Shared amplitude screening and generator construction."""

        def __init__(self, molecule, params, threshold=1e-8, **kwargs):
            super().__init__(molecule, params, threshold=threshold, **kwargs)

        def get_vocab_size(self):
            raise NotImplementedError

        def build_operator_pool(self):
            raise NotImplementedError

        def get_gate_count(self, seq):
            raise NotImplementedError

        def generate_excitations(self, threshold):
            return generate_excitations(self.molecule.ccsd_amplitude, threshold)

        def generate_excitation_generators(self, threshold):
            """(angle, qubit_op) per excitation, in place of a tequila circuit."""
            screened = self.generate_excitations(threshold=threshold)
            generators = []
            for idx, angle in screened.items():
                pairs = [(idx[2 * i], idx[2 * i + 1])
                         for i in range(len(idx) // 2)]
                generators.append((angle, excitation_generator_qubit_op(pairs)))
            return generators

        def _count_gates(self, seq):
            counts = Counter()
            for i in seq:
                for term in self.pool[i]:
                    counts.update(get_pauli_evolution_gate_count(
                        term.get_pauli_word(self.n_qubits)
                    ))
            return counts

    class DMETPauliEvolutionPool(DMETUCCSDBasedPool):
        """
        One pool element per Pauli TERM.

        Cannot conserve particle number -- see the module docstring.
        Prefer DMETExcitationPool.
        """

        def __init__(self, molecule, params, threshold=1e-8,
                     remove_z_ladder=False, only_use_first_pauli=False):
            super().__init__(molecule, params, threshold=threshold,
                             remove_z_ladder=remove_z_ladder,
                             only_use_first_pauli=only_use_first_pauli)

        def get_vocab_size(self):
            return len(self.pool)

        def build_operator_pool(self, threshold, remove_z_ladder=False,
                                only_use_first_pauli=False):
            generators = self.generate_excitation_generators(threshold=threshold)
            seen = set()
            pool = [self.get_identity_operator()]
            for angle, qubit_op in generators:
                for term, _coeff in _qubit_op_terms_to_cudaq(
                    qubit_op, remove_z_ladder=remove_z_ladder
                ):
                    if str(term) in seen:
                        continue
                    seen.add(str(term))
                    if self.params is None:
                        pool.append(angle * cudaq.SpinOperator(term))
                    else:
                        for p in self.params:
                            pool.append(p * cudaq.SpinOperator(term))
                    if only_use_first_pauli:
                        break
            return pool

        def get_gate_count(self, seq):
            return self._count_gates(seq)

    class DMETExcitationPool(DMETUCCSDBasedPool):
        """
        One pool element per EXCITATION, accumulating all its Pauli terms.

        The particle-number-conserving choice, and the default.

        Note that the external factory constructs this as
        DMETExcitationPool(molecule, params=..., threshold=...), so
        remove_z_ladder and only_use_first_pauli are not accepted here and
        those config fields are ignored for this pool type.
        """

        def __init__(self, molecule, params, threshold=1e-8):
            super().__init__(molecule, params)

        def get_vocab_size(self):
            return len(self.pool)

        def build_operator_pool(self, threshold):
            generators = self.generate_excitation_generators(threshold=threshold)
            pool = [self.get_identity_operator()]
            for angle, qubit_op in generators:
                operator = None
                for cudaq_term, coeff in _qubit_op_terms_to_cudaq(qubit_op):
                    weighted = cudaq_term * coeff
                    operator = weighted if operator is None else operator + weighted
                if operator is None:
                    continue
                if self.params is None:
                    pool.append(angle * cudaq.SpinOperator(operator))
                else:
                    for p in self.params:
                        pool.append(p * cudaq.SpinOperator(operator))
            return pool

        def get_gate_count(self, seq):
            return self._count_gates(seq)

    _CLASS_CACHE.update(
        DMETUCCSDBasedPool=DMETUCCSDBasedPool,
        DMETPauliEvolutionPool=DMETPauliEvolutionPool,
        DMETExcitationPool=DMETExcitationPool,
    )
    return _CLASS_CACHE


def __getattr__(name):
    if name in ("DMETUCCSDBasedPool", "DMETPauliEvolutionPool",
                "DMETExcitationPool"):
        return _build_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
